//! iCite CSV → PostgreSQL pipeline.
//!
//! Streams CSV via PG COPY into a temp table, then UPDATEs papers/works
//! and regenerates cited_by_clin — all within a single transaction.

use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use crate::build::pg_sink::PgSink;

/// Stats from an iCite load.
pub struct IciteStats {
    pub papers_updated: u64,
    pub works_updated: u64,
    pub cited_by_clin_rows: i64,
    pub elapsed_secs: f64,
}

/// Load iCite CSV into PG: temp table → UPDATE papers → UPDATE works → cited_by_clin.
pub fn load_icite(
    pg_conninfo: &str,
    csv_path: &Path,
) -> Result<IciteStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let mut sink = PgSink::connect(pg_conninfo)?;

    sink.begin()?;
    let result = load_icite_inner(&mut sink, csv_path);
    match result {
        Ok(mut stats) => {
            sink.commit()?;
            stats.elapsed_secs = t0.elapsed().as_secs_f64();
            Ok(stats)
        }
        Err(e) => {
            let _ = sink.rollback();
            Err(e)
        }
    }
}

fn load_icite_inner(
    sink: &mut PgSink,
    csv_path: &Path,
) -> Result<IciteStats, Box<dyn std::error::Error>> {
    // 1. Create temp table and COPY CSV into it.
    sink.execute_static("DROP TABLE IF EXISTS _icite_tmp")?;
    sink.execute_static(
        "CREATE TEMPORARY TABLE _icite_tmp (
            pmid INTEGER,
            doi TEXT,
            title TEXT,
            authors TEXT,
            year SMALLINT,
            journal TEXT,
            is_research_article TEXT,
            citation_count INTEGER,
            field_citation_rate REAL,
            expected_citations_per_year REAL,
            citations_per_year REAL,
            relative_citation_ratio REAL,
            nih_percentile REAL,
            human REAL,
            animal REAL,
            molecular_cellular REAL,
            x_coord REAL,
            y_coord REAL,
            apt REAL,
            is_clinical TEXT,
            cited_by_clin TEXT,
            cited_by TEXT,
            \"references\" TEXT,
            provisional TEXT,
            last_modified TEXT
        )",
    )?;

    // Stream CSV file into PG via COPY.
    {
        let mut writer = sink.copy_in_raw(
            "COPY _icite_tmp FROM STDIN WITH (FORMAT csv, HEADER true)",
        )?;
        let mut file = std::fs::File::open(csv_path)?;
        let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8MB chunks
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            writer.write_all(&buf[..n])?;
        }
        writer.finish()?;
    }
    eprintln!("icite: CSV loaded into temp table");

    // 2. UPDATE papers with iCite metrics.
    let papers_updated = sink.execute_static(
        "UPDATE papers SET
            rcr = m.relative_citation_ratio,
            nih_percentile = m.nih_percentile,
            apt = m.apt,
            is_clinical = (m.is_clinical = 'Yes'),
            human = m.human,
            animal = m.animal,
            molecular_cellular = m.molecular_cellular,
            field_citation_rate = m.field_citation_rate
        FROM _icite_tmp m
        WHERE papers.pmid = m.pmid",
    )?;
    eprintln!("icite: UPDATE papers: {papers_updated} rows");

    // 3. UPDATE works (T1 — has PMID).
    let works_updated = sink.execute_static(
        "UPDATE works SET
            rcr = m.relative_citation_ratio,
            nih_percentile = m.nih_percentile,
            apt = m.apt,
            is_clinical = (m.is_clinical = 'Yes')
        FROM _icite_tmp m
        WHERE works.pmid = m.pmid",
    )?;
    eprintln!("icite: UPDATE works: {works_updated} rows");

    // 4. Regenerate cited_by_clin.
    sink.execute_static("TRUNCATE cited_by_clin")?;
    sink.execute_static(
        "INSERT INTO cited_by_clin (pmid, citing_pmid)
        SELECT m.pmid, unnest(
            string_to_array(m.cited_by_clin, ' ')
        )::integer
        FROM _icite_tmp m
        WHERE m.cited_by_clin IS NOT NULL AND m.cited_by_clin != ''",
    )?;
    let cbc_rows: i64 = sink.query_one_i64("SELECT count(*) FROM cited_by_clin")?;
    eprintln!("icite: cited_by_clin: {cbc_rows} rows");

    sink.execute_static("DROP TABLE IF EXISTS _icite_tmp")?;

    Ok(IciteStats {
        papers_updated,
        works_updated,
        cited_by_clin_rows: cbc_rows,
        elapsed_secs: 0.0, // filled by caller
    })
}
