-- Quarry ClickHouse schema
-- Run once to initialize tables

CREATE DATABASE IF NOT EXISTS quarry;

CREATE TABLE IF NOT EXISTS quarry.papers (
    openalex_id String,
    doi String DEFAULT '',
    pmid String DEFAULT '',
    title String,
    abstract String,
    pub_year UInt16,
    type LowCardinality(String),
    language LowCardinality(String),
    cited_by UInt32 DEFAULT 0,
    fwci Float32 DEFAULT 0,
    source_name String DEFAULT '',
    source_type LowCardinality(String) DEFAULT '',
    topic_name String DEFAULT '',
    field_name LowCardinality(String) DEFAULT '',
    domain_name LowCardinality(String) DEFAULT '',
    updated_date Date
) ENGINE = ReplacingMergeTree(updated_date)
ORDER BY openalex_id
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS quarry.authors (
    openalex_id String,
    author_position LowCardinality(String),
    author_id String DEFAULT '',
    author_name String DEFAULT '',
    orcid String DEFAULT ''
) ENGINE = MergeTree()
ORDER BY (openalex_id, author_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS quarry.edges (
    citing_id String,
    cited_id String
) ENGINE = MergeTree()
ORDER BY (citing_id, cited_id)
SETTINGS index_granularity = 8192;
