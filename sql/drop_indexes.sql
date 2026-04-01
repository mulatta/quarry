-- Drop non-PK indexes before bulk load to eliminate WAL overhead.
-- Recreated by schema.sql (CREATE INDEX IF NOT EXISTS) after load completes.

-- PubMed
DROP INDEX IF EXISTS idx_cbc_pmid;
DROP INDEX IF EXISTS idx_cbc_citing;
DROP INDEX IF EXISTS idx_authors_pmid;
DROP INDEX IF EXISTS idx_mesh_pmid;
DROP INDEX IF EXISTS idx_mesh_descriptor;
DROP INDEX IF EXISTS idx_grants_pmid;
DROP INDEX IF EXISTS idx_chemicals_pmid;

-- OpenAlex
DROP INDEX IF EXISTS idx_works_work_id_int;
DROP INDEX IF EXISTS idx_works_pmid;
DROP INDEX IF EXISTS idx_works_doi;
DROP INDEX IF EXISTS idx_works_tier;
DROP INDEX IF EXISTS idx_works_pub_year;
DROP INDEX IF EXISTS idx_work_authors_wid;
DROP INDEX IF EXISTS idx_work_topics_wid;
DROP INDEX IF EXISTS idx_work_mesh_wid;
DROP INDEX IF EXISTS idx_work_mesh_desc;
DROP INDEX IF EXISTS idx_work_cit_citing;
DROP INDEX IF EXISTS idx_work_cit_cited;
DROP INDEX IF EXISTS idx_crosswalk_pmid;
