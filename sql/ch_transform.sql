/* ClickHouse transform: dedup raw tables → enriched export tables.
   Run after all raw data is loaded (oa, pubmed, icite).
   Export tables are consumed by pg_load (CH → PG COPY pipe). */

USE quarry;

/* 1. Force dedup on all raw tables */

OPTIMIZE TABLE oa_works FINAL;
OPTIMIZE TABLE oa_work_authors FINAL;
OPTIMIZE TABLE oa_work_topics FINAL;
OPTIMIZE TABLE oa_work_citations FINAL;
OPTIMIZE TABLE oa_id_crosswalk FINAL;
OPTIMIZE TABLE pm_papers FINAL;
OPTIMIZE TABLE pm_authors FINAL;
OPTIMIZE TABLE pm_mesh_headings FINAL;
OPTIMIZE TABLE pm_grants FINAL;
OPTIMIZE TABLE pm_chemicals FINAL;
OPTIMIZE TABLE pm_mesh_tree FINAL;
OPTIMIZE TABLE icite_raw FINAL;

/* 2. works_export: OA works enriched with PubMed + iCite fields */

CREATE OR REPLACE TABLE works_export
ENGINE = MergeTree()
ORDER BY work_id
AS
SELECT
    w.work_id,
    w.work_id_int,
    w.tier,
    w.pmid,
    w.doi,
    w.title,
    w.abstract,
    w.pub_year,
    w.pub_date,
    w.type,
    w.cited_by_count,
    w.host_venue,
    w.oa_status,
    w.oa_url,
    w.is_retracted,
    w.updated_date,
    /* PubMed enrichment */
    p.journal_abbr  AS pm_journal_abbr,
    p.country       AS pm_country,
    p.medline_status AS pm_medline_status,
    p.pub_type      AS pm_pub_type,
    p.created_date  AS pm_created_date,
    p.revised_date  AS pm_revised_date,
    p.indexed_date  AS pm_indexed_date,
    /* iCite enrichment */
    i.relative_citation_ratio AS rcr,
    i.nih_percentile,
    i.apt,
    if(i.is_clinical = 'Yes', true, false) AS is_clinical
FROM oa_works w
LEFT JOIN pm_papers p ON w.pmid = p.pmid AND w.pmid IS NOT NULL
LEFT JOIN icite_raw i ON w.pmid = i.pmid AND w.pmid IS NOT NULL;

/* 3. papers_export: PubMed papers enriched with iCite */

CREATE OR REPLACE TABLE papers_export
ENGINE = MergeTree()
ORDER BY pmid
AS
SELECT
    p.pmid,
    p.doi,
    p.pmc_id,
    p.title,
    p.abstract,
    p.pub_year,
    p.pub_date,
    p.journal_title,
    p.journal_issn,
    p.journal_abbr,
    p.volume,
    p.issue,
    p.pages,
    p.language,
    p.pub_type,
    p.country,
    p.medline_status,
    p.created_date,
    p.revised_date,
    p.indexed_date,
    p.is_deleted,
    p.deleted_date,
    /* iCite enrichment */
    i.relative_citation_ratio AS rcr,
    i.nih_percentile,
    i.apt,
    if(i.is_clinical = 'Yes', true, false) AS is_clinical,
    i.human,
    i.animal,
    i.molecular_cellular,
    i.field_citation_rate
FROM pm_papers p
LEFT JOIN icite_raw i ON p.pmid = i.pmid;

/* 4. work_mesh_export: crosswalk JOIN for OA work → MeSH mapping */

CREATE OR REPLACE TABLE work_mesh_export
ENGINE = MergeTree()
ORDER BY (work_id, descriptor_ui)
AS
SELECT
    c.work_id,
    m.descriptor_ui,
    m.descriptor_name,
    m.qualifier_ui,
    m.qualifier_name,
    m.is_major_topic
FROM pm_mesh_headings m
INNER JOIN oa_id_crosswalk c ON m.pmid = c.pmid;

/* 5. cited_by_clin_export: iCite clinical citation expansion */

CREATE OR REPLACE TABLE cited_by_clin_export
ENGINE = MergeTree()
ORDER BY (pmid, citing_pmid)
AS
SELECT
    pmid,
    toUInt32(citing) AS citing_pmid
FROM (
    SELECT pmid, cited_by_clin
    FROM icite_raw
    WHERE cited_by_clin IS NOT NULL AND cited_by_clin != ''
)
ARRAY JOIN splitByChar(' ', assumeNotNull(cited_by_clin)) AS citing
WHERE citing != '';
