//! PostgreSQL COPY FROM STDIN sink.
//!
//! Writes parsed PubMed/OA structs directly to PG using text-format COPY,
//! bypassing Arrow/Parquet intermediate. Each method opens a COPY stream,
//! writes all rows, and finishes (commits).

use std::fmt::Write as FmtWrite;
use std::io::Write;

use postgres::{Client, NoTls};

use crate::build::oa_json::{OaAuthor, OaCitation, OaCrosswalk, OaTopic, OaWork};
use crate::parse::mesh::MeshEntry;
use crate::parse::xml::{Author, Chemical, Grant, MeshHeading, Paper};

/// Expected COPY columns per table — single source of truth for both
/// COPY statements and startup schema verification.
const EXPECTED_COLUMNS: &[(&str, &[&str])] = &[
    ("papers", &[
        "pmid", "doi", "pmc_id", "title", "abstract", "pub_year", "pub_date",
        "journal_title", "journal_issn", "journal_abbr", "volume", "issue", "pages",
        "language", "pub_type", "country", "medline_status", "created_date",
        "revised_date", "indexed_date",
    ]),
    ("authors", &[
        "pmid", "author_position", "last_name", "fore_name", "initials",
        "orcid", "affiliation", "is_collective",
    ]),
    ("mesh_headings", &[
        "pmid", "descriptor_ui", "descriptor_name", "qualifier_ui",
        "qualifier_name", "is_major_topic",
    ]),
    ("grants", &["pmid", "grant_id", "acronym", "agency", "country"]),
    ("chemicals", &["pmid", "registry_number", "substance_ui", "substance_name"]),
    ("works", &[
        "work_id", "work_id_int", "tier", "pmid", "doi", "title", "abstract",
        "pub_year", "pub_date", "type", "cited_by_count", "host_venue", "oa_status",
        "oa_url", "is_retracted", "updated_date", "pm_journal_abbr", "pm_country",
        "pm_medline_status", "pm_pub_type", "pm_created_date", "pm_revised_date",
        "pm_indexed_date",
    ]),
    ("work_authors", &[
        "work_id", "author_position", "display_name", "orcid",
        "institution_name", "institution_ror", "raw_affiliation",
    ]),
    ("work_topics", &[
        "work_id", "topic_id", "topic_name", "subfield", "field", "domain", "score",
    ]),
    ("work_citations", &["citing_id", "cited_id"]),
    ("id_crosswalk", &["work_id", "pmid"]),
    ("mesh_tree", &["descriptor_ui", "descriptor_name", "tree_number"]),
    ("_build_progress", &["source", "filename", "rows_inserted", "completed_at"]),
];

/// PostgreSQL COPY sink — wraps a single connection.
pub struct PgSink {
    client: Client,
}

impl PgSink {
    pub fn connect(conninfo: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let client = Client::connect(conninfo, NoTls)?;
        Ok(Self { client })
    }

    /// Verify that all expected tables and columns exist in the database.
    /// Fails fast with a clear error listing missing tables/columns.
    pub fn verify_schema(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Fetch all columns for our tables in one query
        let table_names: Vec<&str> = EXPECTED_COLUMNS.iter().map(|(t, _)| *t).collect();
        let rows = self.client.query(
            "SELECT table_name, column_name \
             FROM information_schema.columns \
             WHERE table_schema = 'public' AND table_name = ANY($1)",
            &[&table_names],
        )?;

        // Build actual columns map: table → {columns}
        let mut actual: std::collections::HashMap<String, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        for row in &rows {
            let table: String = row.get(0);
            let column: String = row.get(1);
            actual.entry(table).or_default().insert(column);
        }

        let mut errors: Vec<String> = Vec::new();

        for (table, expected_cols) in EXPECTED_COLUMNS {
            match actual.get(*table) {
                None => errors.push(format!("table '{}' not found", table)),
                Some(actual_cols) => {
                    let missing: Vec<&&str> = expected_cols
                        .iter()
                        .filter(|c| !actual_cols.contains(**c))
                        .collect();
                    if !missing.is_empty() {
                        errors.push(format!(
                            "table '{}': missing columns: {}",
                            table,
                            missing.iter().map(|c| format!("'{c}'")).collect::<Vec<_>>().join(", ")
                        ));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("schema verification failed:\n  {}", errors.join("\n  ")).into())
        }
    }

    /// Record a completed file in the _build_progress table.
    pub fn mark_progress(
        &mut self,
        source: &str,
        filename: &str,
        rows: i64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.client.execute(
            "INSERT INTO _build_progress (source, filename, rows_inserted) \
             VALUES ($1, $2, $3) \
             ON CONFLICT (source, filename) DO UPDATE SET \
               rows_inserted = EXCLUDED.rows_inserted, \
               completed_at = now()",
            &[&source, &filename, &rows],
        )?;
        Ok(())
    }

    /// Check if a file was already processed.
    /// Returns error on DB failure instead of silently returning false.
    pub fn is_done(
        &mut self,
        source: &str,
        filename: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let row = self.client.query_opt(
            "SELECT 1 FROM _build_progress WHERE source = $1 AND filename = $2",
            &[&source, &filename],
        )?;
        Ok(row.is_some())
    }

    /// Bulk-fetch all completed filenames for a source.
    pub fn done_filenames(
        &mut self,
        source: &str,
    ) -> Result<std::collections::HashSet<String>, Box<dyn std::error::Error>> {
        let rows = self.client.query(
            "SELECT filename FROM _build_progress WHERE source = $1",
            &[&source],
        )?;
        Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
    }

    /// Create all tables and indexes (idempotent — uses IF NOT EXISTS).
    pub fn init_schema(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.client.batch_execute(SCHEMA_DDL)?;
        Ok(())
    }

    /// Begin a transaction for atomic file-level writes.
    pub fn begin(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.client.execute("BEGIN", &[])?;
        Ok(())
    }

    /// Commit the current transaction.
    pub fn commit(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.client.execute("COMMIT", &[])?;
        Ok(())
    }

    /// Rollback the current transaction.
    pub fn rollback(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.client.execute("ROLLBACK", &[])?;
        Ok(())
    }

    // ── PubMed tables ──

    pub fn copy_papers(&mut self, papers: &[Paper]) -> Result<usize, Box<dyn std::error::Error>> {
        if papers.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY papers (pmid, doi, pmc_id, title, abstract, pub_year, pub_date, \
             journal_title, journal_issn, journal_abbr, volume, issue, pages, \
             language, pub_type, country, medline_status, created_date, revised_date, \
             indexed_date) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(4096);
        for p in papers {
            buf.clear();
            write_i32(&mut buf, p.pmid);
            tab(&mut buf);
            write_opt_text(&mut buf, p.doi.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.pmc_id.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.title.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.r#abstract.as_deref());
            tab(&mut buf);
            write_opt_i16(&mut buf, p.pub_year);
            tab(&mut buf);
            write_opt_text(&mut buf, p.pub_date.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.journal_title.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.journal_issn.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.journal_abbr.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.volume.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.issue.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.pages.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.language.as_deref());
            tab(&mut buf);
            write_text_array(&mut buf, &p.pub_type);
            tab(&mut buf);
            write_opt_text(&mut buf, p.country.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.medline_status.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.created_date.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.revised_date.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, p.indexed_date.as_deref());
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(papers.len())
    }

    pub fn copy_authors(
        &mut self,
        authors: &[Author],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if authors.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY authors (pmid, author_position, last_name, fore_name, initials, \
             orcid, affiliation, is_collective) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(512);
        for a in authors {
            buf.clear();
            write_i32(&mut buf, a.pmid);
            tab(&mut buf);
            write_i16(&mut buf, a.author_position);
            tab(&mut buf);
            write_opt_text(&mut buf, a.last_name.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.fore_name.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.initials.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.orcid.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.affiliation.as_deref());
            tab(&mut buf);
            write_bool(&mut buf, a.is_collective);
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(authors.len())
    }

    pub fn copy_mesh_headings(
        &mut self,
        headings: &[MeshHeading],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if headings.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY mesh_headings (pmid, descriptor_ui, descriptor_name, qualifier_ui, \
             qualifier_name, is_major_topic) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(256);
        for m in headings {
            buf.clear();
            write_i32(&mut buf, m.pmid);
            tab(&mut buf);
            write_text(&mut buf, &m.descriptor_ui);
            tab(&mut buf);
            write_text(&mut buf, &m.descriptor_name);
            tab(&mut buf);
            write_opt_text(&mut buf, m.qualifier_ui.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, m.qualifier_name.as_deref());
            tab(&mut buf);
            write_bool(&mut buf, m.is_major_topic);
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(headings.len())
    }

    pub fn copy_grants(
        &mut self,
        grants: &[Grant],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if grants.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY grants (pmid, grant_id, acronym, agency, country) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(256);
        for g in grants {
            buf.clear();
            write_i32(&mut buf, g.pmid);
            tab(&mut buf);
            write_opt_text(&mut buf, g.grant_id.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, g.acronym.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, g.agency.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, g.country.as_deref());
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(grants.len())
    }

    pub fn copy_chemicals(
        &mut self,
        chemicals: &[Chemical],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if chemicals.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY chemicals (pmid, registry_number, substance_ui, substance_name) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(256);
        for c in chemicals {
            buf.clear();
            write_i32(&mut buf, c.pmid);
            tab(&mut buf);
            write_opt_text(&mut buf, c.registry_number.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, c.substance_ui.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, c.substance_name.as_deref());
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(chemicals.len())
    }

    /// Soft-delete PMIDs in the papers table.
    /// Batch-delete PMIDs from papers + all child tables (authors, mesh, grants, chemicals).
    /// Used before re-inserting updated versions from PubMed update files.
    pub fn batch_delete_pmids(
        &mut self,
        pmids: &[i32],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if pmids.is_empty() {
            return Ok(0);
        }
        // Child tables first, then parent.
        for table in &["authors", "mesh_headings", "grants", "chemicals"] {
            self.client.execute(
                &format!("DELETE FROM {table} WHERE pmid = ANY($1)"),
                &[&pmids],
            )?;
        }
        let n = self.client.execute(
            "DELETE FROM papers WHERE pmid = ANY($1)",
            &[&pmids],
        )?;
        Ok(n)
    }

    pub fn soft_delete_pmids(
        &mut self,
        pmids: &[i32],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if pmids.is_empty() {
            return Ok(0);
        }
        let n = self.client.execute(
            "UPDATE papers SET is_deleted = TRUE, deleted_date = CURRENT_DATE \
             WHERE pmid = ANY($1)",
            &[&pmids],
        )?;
        Ok(n)
    }

    /// Delete existing OA works + child rows by work_id, for re-insert.
    pub fn delete_work_ids(
        &mut self,
        work_ids: &[&str],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if work_ids.is_empty() {
            return Ok(0);
        }
        for table in &["work_authors", "work_topics", "id_crosswalk"] {
            self.client.execute(
                &format!("DELETE FROM {table} WHERE work_id = ANY($1)"),
                &[&work_ids],
            )?;
        }
        // work_citations uses work_id_int (BIGINT), not work_id (TEXT)
        self.client.execute(
            "DELETE FROM work_citations WHERE citing_id IN \
             (SELECT work_id_int FROM works WHERE work_id = ANY($1))",
            &[&work_ids],
        )?;
        let n = self.client.execute(
            "DELETE FROM works WHERE work_id = ANY($1)",
            &[&work_ids],
        )?;
        Ok(n)
    }

    // ── OpenAlex tables ──

    pub fn copy_works(
        &mut self,
        works: &[OaWork],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if works.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY works (work_id, work_id_int, tier, pmid, doi, title, abstract, \
             pub_year, pub_date, type, cited_by_count, host_venue, oa_status, oa_url, \
             is_retracted, updated_date, pm_journal_abbr, pm_country, pm_medline_status, \
             pm_pub_type, pm_created_date, pm_revised_date, pm_indexed_date) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(2048);
        for w in works {
            buf.clear();
            write_text(&mut buf, &w.work_id);
            tab(&mut buf);
            write_i64(&mut buf, w.work_id_int);
            tab(&mut buf);
            write_text(&mut buf, w.tier.as_str());
            tab(&mut buf);
            write_opt_i32(&mut buf, w.pmid);
            tab(&mut buf);
            write_opt_text(&mut buf, w.doi.as_deref());
            tab(&mut buf);
            write_text(&mut buf, &w.title);
            tab(&mut buf);
            write_opt_text(&mut buf, w.abstract_text.as_deref());
            tab(&mut buf);
            write_opt_i16(&mut buf, w.pub_year);
            tab(&mut buf);
            write_opt_text(&mut buf, w.pub_date.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.work_type.as_deref());
            tab(&mut buf);
            write_opt_i32(&mut buf, w.cited_by_count);
            tab(&mut buf);
            write_opt_text(&mut buf, w.host_venue.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.oa_status.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.oa_url.as_deref());
            tab(&mut buf);
            write_bool(&mut buf, w.is_retracted);
            tab(&mut buf);
            write_opt_text(&mut buf, w.updated_date.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.pm_journal_abbr.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.pm_country.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.pm_medline_status.as_deref());
            tab(&mut buf);
            write_text_array(&mut buf, &w.pm_pub_type);
            tab(&mut buf);
            write_opt_text(&mut buf, w.pm_created_date.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.pm_revised_date.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, w.pm_indexed_date.as_deref());
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(works.len())
    }

    pub fn copy_work_authors(
        &mut self,
        authors: &[OaAuthor],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if authors.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY work_authors (work_id, author_position, display_name, orcid, \
             institution_name, institution_ror, raw_affiliation) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(512);
        for a in authors {
            buf.clear();
            write_text(&mut buf, &a.work_id);
            tab(&mut buf);
            write_i16(&mut buf, a.author_position);
            tab(&mut buf);
            write_opt_text(&mut buf, a.display_name.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.orcid.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.institution_name.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.institution_ror.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, a.raw_affiliation.as_deref());
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(authors.len())
    }

    pub fn copy_work_topics(
        &mut self,
        topics: &[OaTopic],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if topics.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY work_topics (work_id, topic_id, topic_name, subfield, field, \
             domain, score) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(512);
        for t in topics {
            buf.clear();
            write_text(&mut buf, &t.work_id);
            tab(&mut buf);
            write_text(&mut buf, &t.topic_id);
            tab(&mut buf);
            write_opt_text(&mut buf, t.topic_name.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, t.subfield.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, t.field.as_deref());
            tab(&mut buf);
            write_opt_text(&mut buf, t.domain.as_deref());
            tab(&mut buf);
            write_opt_f32(&mut buf, t.score);
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(topics.len())
    }

    pub fn copy_work_citations(
        &mut self,
        citations: &[OaCitation],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if citations.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY work_citations (citing_id, cited_id) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(64);
        for c in citations {
            buf.clear();
            write_i64(&mut buf, c.citing_id);
            tab(&mut buf);
            write_i64(&mut buf, c.cited_id);
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(citations.len())
    }

    pub fn copy_id_crosswalk(
        &mut self,
        crosswalk: &[OaCrosswalk],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if crosswalk.is_empty() {
            return Ok(0);
        }
        let mut writer = self.client.copy_in(
            "COPY id_crosswalk (work_id, pmid) FROM STDIN",
        )?;
        let mut buf = String::with_capacity(64);
        for c in crosswalk {
            buf.clear();
            write_text(&mut buf, &c.work_id);
            tab(&mut buf);
            write_i32(&mut buf, c.pmid);
            buf.push('\n');
            writer.write_all(buf.as_bytes())?;
        }
        writer.finish()?;
        Ok(crosswalk.len())
    }

    /// Execute a known SQL statement (DDL, TRUNCATE, DELETE, etc.).
    ///
    /// # Safety
    /// Callers must only pass hardcoded SQL — never user-supplied input.
    pub fn execute_static(&mut self, sql: &str) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(self.client.execute(sql, &[])?)
    }

    /// Execute a batch SQL statement (multiple statements separated by `;`).
    pub fn batch_execute(&mut self, sql: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.client.batch_execute(sql)?;
        Ok(())
    }

    /// Start a raw COPY FROM STDIN (for streaming binary/CSV data).
    pub fn copy_in_raw(
        &mut self,
        query: &str,
    ) -> Result<postgres::CopyInWriter<'_>, Box<dyn std::error::Error>> {
        Ok(self.client.copy_in(query)?)
    }

    /// Query a single i64 value (e.g. COUNT(*)).
    pub fn query_one_i64(&mut self, sql: &str) -> Result<i64, Box<dyn std::error::Error>> {
        let row = self.client.query_one(sql, &[])?;
        Ok(row.get::<_, i64>(0))
    }

    /// TRUNCATE all data tables + reset build progress.
    /// Use before full re-load after schema or pipeline changes.
    pub fn reset_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.client.batch_execute(
            "TRUNCATE papers, authors, mesh_headings, grants, chemicals, \
                      cited_by_clin, works, work_authors, work_topics, \
                      work_mesh, work_citations, id_crosswalk, mesh_tree, \
                      _build_progress CASCADE"
        )?;
        Ok(())
    }

    /// Enrich works with PubMed fields via SQL UPDATE.
    pub fn enrich_works_from_papers(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        let n = self.client.execute(
            "UPDATE works w SET \
                pm_journal_abbr = p.journal_abbr, \
                pm_country = p.country, \
                pm_medline_status = p.medline_status, \
                pm_pub_type = p.pub_type, \
                pm_created_date = p.created_date, \
                pm_revised_date = p.revised_date, \
                pm_indexed_date = p.indexed_date \
             FROM papers p \
             WHERE w.pmid = p.pmid AND w.pmid IS NOT NULL",
            &[],
        )?;
        Ok(n)
    }

    /// Generate work_mesh by joining mesh_headings with id_crosswalk.
    pub fn generate_work_mesh(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        let n = self.client.execute(
            "INSERT INTO work_mesh (work_id, descriptor_ui, descriptor_name, \
                qualifier_ui, qualifier_name, is_major_topic) \
             SELECT c.work_id, m.descriptor_ui, m.descriptor_name, \
                m.qualifier_ui, m.qualifier_name, m.is_major_topic \
             FROM mesh_headings m \
             INNER JOIN id_crosswalk c ON m.pmid = c.pmid",
            &[],
        )?;
        Ok(n)
    }

    /// Drop non-PK indexes for faster bulk load.
    pub fn drop_indexes(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.client.batch_execute(DROP_INDEXES_SQL)?;
        Ok(())
    }

    /// Create indexes after bulk load.
    /// Extracts CREATE INDEX statements from schema.sql and executes them.
    pub fn create_indexes(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for line in SCHEMA_DDL.lines() {
            let trimmed = line.trim();
            if trimmed.to_uppercase().starts_with("CREATE INDEX") {
                self.client.batch_execute(trimmed)?;
            }
        }
        Ok(())
    }

    /// VACUUM ANALYZE all tables.
    /// Each statement runs individually — VACUUM cannot run inside a transaction.
    pub fn vacuum(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for table in &[
            "papers", "authors", "mesh_headings", "grants", "chemicals",
            "cited_by_clin", "works", "work_authors", "work_topics",
            "work_mesh", "work_citations", "id_crosswalk", "mesh_tree",
            "_build_progress",
        ] {
            self.client.execute(&format!("VACUUM ANALYZE {table}"), &[])?;
        }
        Ok(())
    }

    /// Write MeSH descriptor tree entries via COPY.
    /// Replaces all rows atomically (BEGIN → DELETE → COPY → COMMIT).
    pub fn write_mesh_tree(&mut self, entries: &[MeshEntry]) -> Result<u64, Box<dyn std::error::Error>> {
        self.begin()?;
        let result = (|| -> Result<u64, Box<dyn std::error::Error>> {
            self.client.execute("DELETE FROM mesh_tree", &[])?;
            let mut writer = self.client.copy_in(
                "COPY mesh_tree (descriptor_ui, descriptor_name, tree_number) FROM STDIN"
            )?;
            let mut buf = String::with_capacity(256);
            for entry in entries {
                buf.clear();
                write_text(&mut buf, &entry.descriptor_ui);
                tab(&mut buf);
                write_text(&mut buf, &entry.descriptor_name);
                tab(&mut buf);
                write_text(&mut buf, &entry.tree_number);
                buf.push('\n');
                writer.write_all(buf.as_bytes())?;
            }
            writer.finish()?;
            Ok(entries.len() as u64)
        })();
        match result {
            Ok(n) => {
                self.commit()?;
                Ok(n)
            }
            Err(e) => {
                let _ = self.rollback();
                Err(e)
            }
        }
    }
}

// ── PG TEXT format COPY helpers ──

#[inline]
fn tab(buf: &mut String) {
    buf.push('\t');
}

#[inline]
fn write_i32(buf: &mut String, v: i32) {
    let _ = write!(buf, "{v}");
}

#[inline]
fn write_i64(buf: &mut String, v: i64) {
    let _ = write!(buf, "{v}");
}

#[inline]
fn write_i16(buf: &mut String, v: i16) {
    let _ = write!(buf, "{v}");
}

#[inline]
fn write_opt_i32(buf: &mut String, v: Option<i32>) {
    match v {
        Some(n) => write_i32(buf, n),
        None => buf.push_str("\\N"),
    }
}

#[inline]
fn write_opt_i16(buf: &mut String, v: Option<i16>) {
    match v {
        Some(n) => write_i16(buf, n),
        None => buf.push_str("\\N"),
    }
}

#[inline]
fn write_opt_f32(buf: &mut String, v: Option<f32>) {
    match v {
        Some(n) => {
            let _ = write!(buf, "{n}");
        }
        None => buf.push_str("\\N"),
    }
}

#[inline]
fn write_bool(buf: &mut String, v: bool) {
    buf.push(if v { 't' } else { 'f' });
}

/// Write a non-null text value, escaping backslashes, tabs, and newlines
/// per PG COPY text format.
#[inline]
fn write_text(buf: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '\\' => buf.push_str("\\\\"),
            '\t' => buf.push_str("\\t"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            _ => buf.push(ch),
        }
    }
}

#[inline]
fn write_opt_text(buf: &mut String, v: Option<&str>) {
    match v {
        Some(s) => write_text(buf, s),
        None => buf.push_str("\\N"),
    }
}

/// Write a TEXT[] value in PG array literal format: `{elem1,"elem 2",elem3}`
/// Empty vec → `\\N` (NULL).
///
/// Inside a COPY text row, the array literal is itself a field value.
/// PG COPY text-format field escaping applies *first* (backslash, tab, newline),
/// then within the array literal, element values must escape `"` and `\`.
/// We handle both layers here: tab → `\t`, newline → `\n`, carriage return → `\r`,
/// backslash → `\\`, double-quote → `\"`.
fn write_text_array(buf: &mut String, arr: &[String]) {
    if arr.is_empty() {
        buf.push_str("\\N");
        return;
    }
    buf.push('{');
    for (i, elem) in arr.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push('"');
        for ch in elem.chars() {
            match ch {
                '"' => buf.push_str("\\\""),
                '\\' => buf.push_str("\\\\"),
                '\t' => buf.push_str("\\t"),
                '\n' => buf.push_str("\\n"),
                '\r' => buf.push_str("\\r"),
                _ => buf.push(ch),
            }
        }
        buf.push('"');
    }
    buf.push('}');
}

/// Schema DDL — single source of truth: sql/schema.sql
const SCHEMA_DDL: &str = include_str!("../../../sql/schema.sql");

/// Drop non-PK indexes before bulk load.
const DROP_INDEXES_SQL: &str = include_str!("../../../sql/drop_indexes.sql");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_text_basic() {
        let mut buf = String::new();
        write_text(&mut buf, "hello world");
        assert_eq!(buf, "hello world");
    }

    #[test]
    fn test_write_text_escapes() {
        let mut buf = String::new();
        write_text(&mut buf, "a\\b\tc\nd\re");
        assert_eq!(buf, "a\\\\b\\tc\\nd\\re");
    }

    #[test]
    fn test_write_opt_text_none() {
        let mut buf = String::new();
        write_opt_text(&mut buf, None);
        assert_eq!(buf, "\\N");
    }

    #[test]
    fn test_write_opt_text_some() {
        let mut buf = String::new();
        write_opt_text(&mut buf, Some("foo\tbar"));
        assert_eq!(buf, "foo\\tbar");
    }

    #[test]
    fn test_write_text_array_empty() {
        let mut buf = String::new();
        write_text_array(&mut buf, &[]);
        assert_eq!(buf, "\\N");
    }

    #[test]
    fn test_write_text_array_basic() {
        let mut buf = String::new();
        write_text_array(&mut buf, &["Review".to_string(), "Journal Article".to_string()]);
        assert_eq!(buf, r#"{"Review","Journal Article"}"#);
    }

    #[test]
    fn test_write_text_array_escapes_quotes_and_backslash() {
        let mut buf = String::new();
        // "a\\b" in Rust = a\b (1 backslash) → escaped to a\\b in output
        write_text_array(&mut buf, &[r#"say "hello""#.to_string(), "a\\b".to_string()]);
        assert_eq!(buf, r#"{"say \"hello\"","a\\b"}"#);
    }

    #[test]
    fn test_write_text_array_escapes_tab_newline() {
        let mut buf = String::new();
        write_text_array(&mut buf, &["a\tb".to_string(), "c\nd".to_string()]);
        assert_eq!(buf, r#"{"a\tb","c\nd"}"#);
    }

    #[test]
    fn test_write_bool() {
        let mut buf = String::new();
        write_bool(&mut buf, true);
        write_bool(&mut buf, false);
        assert_eq!(buf, "tf");
    }

    #[test]
    fn test_write_i32() {
        let mut buf = String::new();
        write_i32(&mut buf, -42);
        assert_eq!(buf, "-42");
    }

    #[test]
    fn test_write_opt_i32() {
        let mut buf = String::new();
        write_opt_i32(&mut buf, Some(123));
        tab(&mut buf);
        write_opt_i32(&mut buf, None);
        assert_eq!(buf, "123\t\\N");
    }

    #[test]
    fn test_write_i64() {
        let mut buf = String::new();
        write_i64(&mut buf, 0);
        tab(&mut buf);
        write_i64(&mut buf, i64::MAX);
        tab(&mut buf);
        write_i64(&mut buf, i64::MIN);
        assert_eq!(buf, format!("0\t{}\t{}", i64::MAX, i64::MIN));
    }

    #[test]
    fn test_write_i16() {
        let mut buf = String::new();
        write_i16(&mut buf, 0);
        tab(&mut buf);
        write_i16(&mut buf, i16::MAX);
        tab(&mut buf);
        write_i16(&mut buf, i16::MIN);
        assert_eq!(buf, format!("0\t{}\t{}", i16::MAX, i16::MIN));
    }

    #[test]
    fn test_write_opt_i16() {
        let mut buf = String::new();
        write_opt_i16(&mut buf, Some(2024));
        tab(&mut buf);
        write_opt_i16(&mut buf, None);
        assert_eq!(buf, "2024\t\\N");
    }

    #[test]
    fn test_write_opt_f32_some() {
        let mut buf = String::new();
        write_opt_f32(&mut buf, Some(0.95));
        // f32 Display should produce a numeric string
        assert!(buf.starts_with("0.9"));
    }

    #[test]
    fn test_write_opt_f32_none() {
        let mut buf = String::new();
        write_opt_f32(&mut buf, None);
        assert_eq!(buf, "\\N");
    }

    #[test]
    fn test_write_opt_f32_nan() {
        let mut buf = String::new();
        write_opt_f32(&mut buf, Some(f32::NAN));
        assert_eq!(buf, "NaN");
    }

    #[test]
    fn test_write_opt_f32_infinity() {
        let mut buf = String::new();
        write_opt_f32(&mut buf, Some(f32::INFINITY));
        assert_eq!(buf, "inf");
    }

    /// Verify that EXPECTED_COLUMNS entries match the columns in schema.sql DDL.
    /// Catches drift between COPY column lists and the actual schema definition.
    #[test]
    fn test_expected_columns_match_schema_ddl() {
        // Parse CREATE TABLE blocks from schema DDL to extract column names
        let mut ddl_tables: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let mut current_table: Option<String> = None;

        for line in SCHEMA_DDL.lines() {
            let trimmed = line.trim();
            // Match: CREATE TABLE IF NOT EXISTS table_name (
            if let Some(rest) = trimmed
                .strip_prefix("CREATE TABLE IF NOT EXISTS ")
                .or_else(|| trimmed.strip_prefix("CREATE TABLE "))
            {
                let table = rest
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .trim_end_matches('(');
                current_table = Some(table.to_string());
                ddl_tables.entry(table.to_string()).or_default();
                continue;
            }

            if let Some(ref table) = current_table {
                // End of CREATE TABLE block
                if trimmed.starts_with(");") || trimmed == ")" {
                    current_table = None;
                    continue;
                }
                // Skip PRIMARY KEY constraints, empty lines, comments
                if trimmed.is_empty()
                    || trimmed.starts_with("--")
                    || trimmed.to_uppercase().starts_with("PRIMARY KEY")
                {
                    continue;
                }
                // Extract column name (first word)
                if let Some(col) = trimmed.split_whitespace().next() {
                    // Skip SQL keywords that appear as constraint lines
                    let upper = col.to_uppercase();
                    if upper == "CONSTRAINT" || upper == "UNIQUE" || upper == "CHECK" {
                        continue;
                    }
                    ddl_tables
                        .get_mut(table)
                        .unwrap()
                        .push(col.to_string());
                }
            }
        }

        for (table, expected_cols) in EXPECTED_COLUMNS {
            let ddl_cols = ddl_tables.get(*table).unwrap_or_else(|| {
                panic!("EXPECTED_COLUMNS references table '{}' not found in schema.sql", table)
            });
            for col in *expected_cols {
                assert!(
                    ddl_cols.contains(&col.to_string()),
                    "EXPECTED_COLUMNS: table '{}' column '{}' not found in schema.sql (available: {:?})",
                    table, col, ddl_cols
                );
            }
        }
    }
}
