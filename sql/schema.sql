-- Quarry schema DDL — single source of truth.
-- Loaded by: process-compose (PG init), quarry-parse db init, PGStore.init_schema()
-- All statements are idempotent (IF NOT EXISTS).

-- ── Extensions ──

CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ── Read-only role for MCP / ad-hoc queries ──

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'quarry_ro') THEN
    CREATE ROLE quarry_ro NOLOGIN;
  END IF;
END $$;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO quarry_ro;

-- ── PubMed tables ──

CREATE TABLE IF NOT EXISTS papers (
    pmid         INTEGER PRIMARY KEY,
    doi          TEXT,
    pmc_id       TEXT,
    title        TEXT,
    abstract     TEXT,
    pub_year     SMALLINT,
    pub_date     DATE,
    journal_title TEXT,
    journal_issn TEXT,
    journal_abbr TEXT,
    volume       TEXT,
    issue        TEXT,
    pages        TEXT,
    language     TEXT,
    pub_type     TEXT[],
    country      TEXT,
    medline_status TEXT,
    created_date DATE,
    revised_date DATE,
    indexed_date DATE,
    is_deleted   BOOLEAN DEFAULT FALSE,
    deleted_date DATE,
    rcr          REAL,
    nih_percentile REAL,
    apt          REAL,
    is_clinical  BOOLEAN,
    human        REAL,
    animal       REAL,
    molecular_cellular REAL,
    field_citation_rate REAL
);

CREATE TABLE IF NOT EXISTS cited_by_clin (
    pmid        INTEGER NOT NULL,
    citing_pmid INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cbc_pmid ON cited_by_clin(pmid);
CREATE INDEX IF NOT EXISTS idx_cbc_citing ON cited_by_clin(citing_pmid);

CREATE TABLE IF NOT EXISTS authors (
    pmid              INTEGER NOT NULL,
    author_position   SMALLINT,
    last_name         TEXT,
    fore_name         TEXT,
    initials          TEXT,
    orcid             TEXT,
    affiliation       TEXT,
    is_collective     BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_authors_pmid ON authors(pmid);

CREATE TABLE IF NOT EXISTS mesh_headings (
    pmid             INTEGER NOT NULL,
    descriptor_ui    TEXT NOT NULL,
    descriptor_name  TEXT NOT NULL,
    qualifier_ui     TEXT,
    qualifier_name   TEXT,
    is_major_topic   BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_mesh_pmid ON mesh_headings(pmid);
CREATE INDEX IF NOT EXISTS idx_mesh_descriptor ON mesh_headings(descriptor_ui);

CREATE TABLE IF NOT EXISTS mesh_tree (
    descriptor_ui    TEXT NOT NULL,
    descriptor_name  TEXT NOT NULL,
    tree_number      TEXT NOT NULL,
    PRIMARY KEY (descriptor_ui, tree_number)
);

-- Unified MeSH search index: all searchable terms for descriptors.
-- Combines three sources to enable synonym-based lookup:
--   1. Entry terms from NLM desc*.xml (ConceptList/Term/String) — e.g.,
--      "A-23187" → D000001 Calcimycin, "PTM" → D011499 (if NLM lists it)
--   2. Historical descriptors from work_mesh that are no longer in
--      the current MeSH vocabulary (NLM revises/removes descriptors
--      but paper annotations persist)
-- Replaces the previous mesh_descriptors table.
CREATE TABLE IF NOT EXISTS mesh_lookup (
    descriptor_ui    TEXT NOT NULL,
    descriptor_name  TEXT NOT NULL,
    term             TEXT NOT NULL,
    source           TEXT NOT NULL,  -- 'entry_term' | 'historical'
    has_tree         BOOLEAN NOT NULL DEFAULT true
);
CREATE INDEX IF NOT EXISTS idx_mesh_lookup_term ON mesh_lookup USING btree (term);
CREATE INDEX IF NOT EXISTS idx_mesh_lookup_desc ON mesh_lookup USING btree (descriptor_ui);

CREATE TABLE IF NOT EXISTS grants (
    pmid       INTEGER NOT NULL,
    grant_id   TEXT,
    acronym    TEXT,
    agency     TEXT,
    country    TEXT
);
CREATE INDEX IF NOT EXISTS idx_grants_pmid ON grants(pmid);

CREATE TABLE IF NOT EXISTS chemicals (
    pmid              INTEGER NOT NULL,
    registry_number   TEXT,
    substance_ui      TEXT,
    substance_name    TEXT
);
CREATE INDEX IF NOT EXISTS idx_chemicals_pmid ON chemicals(pmid);

-- ── OpenAlex tables ──

CREATE TABLE IF NOT EXISTS works (
    work_id        TEXT PRIMARY KEY,
    work_id_int    BIGINT NOT NULL,
    tier           TEXT NOT NULL,
    pmid           INTEGER,
    doi            TEXT,
    title          TEXT,
    abstract       TEXT,
    pub_year       SMALLINT,
    pub_date       DATE,
    type           TEXT,
    cited_by_count INTEGER,
    host_venue     TEXT,
    oa_status      TEXT,
    oa_url         TEXT,
    rcr            REAL,
    nih_percentile REAL,
    apt            REAL,
    is_clinical    BOOLEAN,
    is_retracted   BOOLEAN DEFAULT FALSE,
    updated_date   DATE,
    pm_journal_abbr    TEXT,
    pm_country         TEXT,
    pm_medline_status  TEXT,
    pm_pub_type        TEXT[],
    pm_created_date    DATE,
    pm_revised_date    DATE,
    pm_indexed_date    DATE,
    language           TEXT,
    fwci               REAL,
    citation_normalized_percentile REAL,
    cited_by_percentile_year_min   SMALLINT,
    cited_by_percentile_year_max   SMALLINT
);

CREATE TABLE IF NOT EXISTS work_counts_by_year (
    work_id        TEXT NOT NULL,
    year           SMALLINT NOT NULL,
    cited_by_count INTEGER NOT NULL,
    PRIMARY KEY (work_id, year)
);

CREATE TABLE IF NOT EXISTS work_authors (
    work_id          TEXT NOT NULL,
    author_position  SMALLINT,
    display_name     TEXT,
    orcid            TEXT,
    institution_name TEXT,
    institution_ror  TEXT,
    raw_affiliation  TEXT
);

CREATE TABLE IF NOT EXISTS work_topics (
    work_id    TEXT NOT NULL,
    topic_id   TEXT NOT NULL,
    topic_name TEXT,
    subfield   TEXT,
    field      TEXT,
    domain     TEXT,
    score      REAL,
    is_primary BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS work_mesh (
    work_id         TEXT NOT NULL,
    descriptor_ui   TEXT NOT NULL,
    descriptor_name TEXT NOT NULL,
    qualifier_ui    TEXT,
    qualifier_name  TEXT,
    is_major_topic  BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS work_citations (
    citing_id BIGINT NOT NULL,
    cited_id  BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS id_crosswalk (
    work_id TEXT PRIMARY KEY,
    pmid    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_works_work_id_int ON works(work_id_int);
CREATE INDEX IF NOT EXISTS idx_works_pmid ON works(pmid);
CREATE INDEX IF NOT EXISTS idx_works_doi ON works(doi);
CREATE INDEX IF NOT EXISTS idx_works_tier ON works(tier);
CREATE INDEX IF NOT EXISTS idx_works_pub_year ON works(pub_year);
CREATE INDEX IF NOT EXISTS idx_work_authors_wid ON work_authors(work_id);
CREATE INDEX IF NOT EXISTS idx_work_topics_wid ON work_topics(work_id);
CREATE INDEX IF NOT EXISTS idx_work_mesh_wid ON work_mesh(work_id);
CREATE INDEX IF NOT EXISTS idx_work_mesh_desc ON work_mesh(descriptor_ui);
CREATE INDEX IF NOT EXISTS idx_work_cit_citing ON work_citations(citing_id);
CREATE INDEX IF NOT EXISTS idx_work_cit_cited ON work_citations(cited_id);
CREATE INDEX IF NOT EXISTS idx_crosswalk_pmid ON id_crosswalk(pmid);

