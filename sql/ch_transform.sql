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
OPTIMIZE TABLE oa_counts_by_year FINAL;
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
    w.work_id       AS work_id,
    w.work_id_int   AS work_id_int,
    -- Tier reclassification: OA tier is based on OA abstract only;
    -- recompute after PM abstract fallback so T3→T1 when PM has abstract.
    multiIf(
        w.pmid IS NOT NULL AND coalesce(p.abstract, w.abstract) IS NOT NULL, 't1',
        coalesce(p.abstract, w.abstract) IS NOT NULL, 't2',
        w.pmid IS NOT NULL, 't3',
        't4'
    )                AS tier,
    w.pmid          AS pmid,
    w.doi           AS doi,
    w.title         AS title,
    coalesce(p.abstract, w.abstract) AS abstract,  -- PM full-text preferred over OA snippet
    w.pub_year      AS pub_year,
    w.pub_date      AS pub_date,
    w.type          AS type,
    w.cited_by_count AS cited_by_count,
    w.host_venue    AS host_venue,
    w.oa_status     AS oa_status,
    w.oa_url        AS oa_url,
    w.is_retracted  AS is_retracted,
    w.updated_date  AS updated_date,
    w.language      AS language,
    w.fwci          AS fwci,
    w.citation_normalized_percentile AS citation_normalized_percentile,
    w.cited_by_percentile_year_min   AS cited_by_percentile_year_min,
    w.cited_by_percentile_year_max   AS cited_by_percentile_year_max,
    /* PubMed enrichment */
    p.journal_abbr  AS pm_journal_abbr,
    p.country       AS pm_country,
    p.medline_status AS pm_medline_status,
    concat('{', arrayStringConcat(p.pub_type, ','), '}') AS pm_pub_type,
    p.created_date  AS pm_created_date,
    p.revised_date  AS pm_revised_date,
    p.indexed_date  AS pm_indexed_date,
    /* iCite enrichment */
    i.relative_citation_ratio AS rcr,
    i.nih_percentile AS nih_percentile,
    i.apt           AS apt,
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
    p.pmid          AS pmid,
    p.doi           AS doi,
    p.pmc_id        AS pmc_id,
    p.title         AS title,
    p.abstract      AS abstract,
    p.pub_year      AS pub_year,
    p.pub_date      AS pub_date,
    p.journal_title AS journal_title,
    p.journal_issn  AS journal_issn,
    p.journal_abbr  AS journal_abbr,
    p.volume        AS volume,
    p.issue         AS issue,
    p.pages         AS pages,
    p.language      AS language,
    concat('{', arrayStringConcat(p.pub_type, ','), '}') AS pub_type,
    p.country       AS country,
    p.medline_status AS medline_status,
    p.created_date  AS created_date,
    p.revised_date  AS revised_date,
    p.indexed_date  AS indexed_date,
    p.is_deleted    AS is_deleted,
    p.deleted_date  AS deleted_date,
    /* iCite enrichment */
    i.relative_citation_ratio AS rcr,
    i.nih_percentile AS nih_percentile,
    i.apt           AS apt,
    if(i.is_clinical = 'Yes', true, false) AS is_clinical,
    i.human         AS human,
    i.animal        AS animal,
    i.molecular_cellular AS molecular_cellular,
    i.field_citation_rate AS field_citation_rate
FROM pm_papers p
LEFT JOIN icite_raw i ON p.pmid = i.pmid;

/* 4. work_mesh_export: crosswalk JOIN for OA work → MeSH mapping */

CREATE OR REPLACE TABLE work_mesh_export
ENGINE = MergeTree()
ORDER BY (work_id, descriptor_ui)
AS
SELECT
    c.work_id           AS work_id,
    m.descriptor_ui     AS descriptor_ui,
    m.descriptor_name   AS descriptor_name,
    m.qualifier_ui      AS qualifier_ui,
    m.qualifier_name    AS qualifier_name,
    m.is_major_topic    AS is_major_topic
FROM pm_mesh_headings m
INNER JOIN oa_id_crosswalk c ON m.pmid = c.pmid;

/* 5. merged_citations: OA citations + iCite-derived citations (deduped).
      iCite references field contains space-separated PMIDs.
      Convert PMID → work_id_int via works_export (which has pmid column).
      ReplacingMergeTree deduplicates overlapping edges. */

CREATE OR REPLACE TABLE icite_citations
ENGINE = MergeTree()
ORDER BY (citing_id, cited_id)
AS
SELECT
    w_citing.work_id_int AS citing_id,
    w_cited.work_id_int  AS cited_id
FROM (
    SELECT pmid AS citing_pmid, toUInt32(ref) AS cited_pmid
    FROM icite_raw
    ARRAY JOIN splitByChar(' ', assumeNotNull(toString(`references`))) AS ref
    WHERE `references` IS NOT NULL AND `references` != ''
      AND ref != '' AND match(ref, '^\d+$')
) parsed
INNER JOIN works_export w_citing ON parsed.citing_pmid = w_citing.pmid
INNER JOIN works_export w_cited  ON parsed.cited_pmid  = w_cited.pmid
WHERE w_citing.work_id_int != w_cited.work_id_int;

CREATE OR REPLACE TABLE merged_citations
ENGINE = ReplacingMergeTree()
ORDER BY (citing_id, cited_id)
AS
SELECT citing_id, cited_id FROM oa_work_citations
UNION ALL
SELECT citing_id, cited_id FROM icite_citations;

OPTIMIZE TABLE merged_citations FINAL;

/* 6. cited_by_clin_export: iCite clinical citation expansion */

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
