/* ClickHouse schema for quarry ELT pipeline.
   Raw tables: ReplacingMergeTree for dedup (append-only, no DELETE needed).
   Export tables: created by ch_transform.sql after dedup + enrich. */

CREATE DATABASE IF NOT EXISTS quarry;

USE quarry;

/* OA raw tables */

CREATE TABLE IF NOT EXISTS oa_works (
    work_id        String,
    work_id_int    UInt64,
    tier           LowCardinality(String),
    pmid           Nullable(UInt32),
    doi            Nullable(String),
    title          String,
    abstract       Nullable(String),
    content_hash   Nullable(FixedString(32)),
    pub_year       Nullable(UInt16),
    pub_date       Nullable(Date32),
    type           Nullable(String),
    cited_by_count Nullable(UInt32),
    host_venue     Nullable(String),
    oa_status      Nullable(String),
    oa_url         Nullable(String),
    is_retracted   Bool DEFAULT false,
    updated_date   Date
) ENGINE = ReplacingMergeTree(updated_date)
ORDER BY (work_id)
PARTITION BY toYear(updated_date);

CREATE TABLE IF NOT EXISTS oa_work_authors (
    work_id          String,
    author_position  UInt16,
    display_name     Nullable(String),
    orcid            Nullable(String),
    institution_name Nullable(String),
    institution_ror  Nullable(String),
    raw_affiliation  Nullable(String),
    updated_date     Date
) ENGINE = ReplacingMergeTree(updated_date)
ORDER BY (work_id, author_position)
PARTITION BY toYear(updated_date);

CREATE TABLE IF NOT EXISTS oa_work_topics (
    work_id    String,
    topic_id   String,
    topic_name Nullable(String),
    subfield   Nullable(String),
    field      Nullable(String),
    domain     Nullable(String),
    score      Nullable(Float32),
    is_primary Bool DEFAULT false,
    updated_date Date
) ENGINE = ReplacingMergeTree(updated_date)
ORDER BY (work_id, topic_id)
PARTITION BY toYear(updated_date);

CREATE TABLE IF NOT EXISTS oa_work_citations (
    citing_id    UInt64,
    cited_id     UInt64,
    updated_date Date
) ENGINE = ReplacingMergeTree(updated_date)
ORDER BY (citing_id, cited_id)
PARTITION BY intDiv(citing_id, 100000000);

CREATE TABLE IF NOT EXISTS oa_id_crosswalk (
    work_id      String,
    pmid         UInt32,
    updated_date Date
) ENGINE = ReplacingMergeTree(updated_date)
ORDER BY (work_id);

/* PubMed raw tables */

CREATE TABLE IF NOT EXISTS pm_papers (
    pmid           UInt32,
    doi            Nullable(String),
    pmc_id         Nullable(String),
    title          Nullable(String),
    abstract       Nullable(String),
    pub_year       Nullable(UInt16),
    pub_date       Nullable(Date32),
    journal_title  Nullable(String),
    journal_issn   Nullable(String),
    journal_abbr   Nullable(String),
    volume         Nullable(String),
    issue          Nullable(String),
    pages          Nullable(String),
    language       Nullable(String),
    pub_type       Array(String),
    country        Nullable(String),
    medline_status Nullable(String),
    created_date   Nullable(Date),
    revised_date   Nullable(Date),
    indexed_date   Nullable(Date),
    is_deleted     Bool DEFAULT false,
    deleted_date   Nullable(Date),
    _version       UInt64 DEFAULT 0
) ENGINE = ReplacingMergeTree(_version)
ORDER BY (pmid);

CREATE TABLE IF NOT EXISTS pm_authors (
    pmid            UInt32,
    author_position UInt16,
    last_name       Nullable(String),
    fore_name       Nullable(String),
    initials        Nullable(String),
    orcid           Nullable(String),
    affiliation     Nullable(String),
    is_collective   Bool DEFAULT false,
    _version        UInt64 DEFAULT 0
) ENGINE = ReplacingMergeTree(_version)
ORDER BY (pmid, author_position);

CREATE TABLE IF NOT EXISTS pm_mesh_headings (
    pmid            UInt32,
    descriptor_ui   String,
    descriptor_name String,
    qualifier_ui    String DEFAULT '',
    qualifier_name  Nullable(String),
    is_major_topic  Bool DEFAULT false,
    _version        UInt64 DEFAULT 0
) ENGINE = ReplacingMergeTree(_version)
ORDER BY (pmid, descriptor_ui, qualifier_ui);

CREATE TABLE IF NOT EXISTS pm_grants (
    pmid     UInt32,
    grant_id String DEFAULT '',
    acronym  Nullable(String),
    agency   Nullable(String),
    country  Nullable(String),
    _version UInt64 DEFAULT 0
) ENGINE = ReplacingMergeTree(_version)
ORDER BY (pmid, grant_id);

CREATE TABLE IF NOT EXISTS pm_chemicals (
    pmid            UInt32,
    registry_number Nullable(String),
    substance_ui    String DEFAULT '',
    substance_name  Nullable(String),
    _version        UInt64 DEFAULT 0
) ENGINE = ReplacingMergeTree(_version)
ORDER BY (pmid, substance_ui);

CREATE TABLE IF NOT EXISTS pm_mesh_tree (
    descriptor_ui   String,
    descriptor_name String,
    tree_number     String
) ENGINE = ReplacingMergeTree()
ORDER BY (descriptor_ui, tree_number);

/* iCite raw table */

CREATE TABLE IF NOT EXISTS icite_raw (
    pmid                        UInt32,
    doi                         Nullable(String),
    title                       Nullable(String),
    authors                     Nullable(String),
    year                        Nullable(UInt16),
    journal                     Nullable(String),
    is_research_article         Nullable(String),
    citation_count              Nullable(UInt32),
    field_citation_rate         Nullable(Float32),
    expected_citations_per_year Nullable(Float32),
    citations_per_year          Nullable(Float32),
    relative_citation_ratio     Nullable(Float32),
    nih_percentile              Nullable(Float32),
    human                       Nullable(Float32),
    animal                      Nullable(Float32),
    molecular_cellular          Nullable(Float32),
    x_coord                     Nullable(Float32),
    y_coord                     Nullable(Float32),
    apt                         Nullable(Float32),
    is_clinical                 Nullable(String),
    cited_by_clin               Nullable(String),
    cited_by                    Nullable(String),
    `references`                Nullable(String),
    provisional                 Nullable(String),
    last_modified               Nullable(String)
) ENGINE = ReplacingMergeTree()
ORDER BY (pmid);
