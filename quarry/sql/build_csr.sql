-- CSR build: node_map + dictionary + binary edge export
-- Memory budget: ~12GB (fits in 48GB)

-- Step 1: node_map table (~1GB)
DROP TABLE IF EXISTS quarry.node_map;
CREATE TABLE quarry.node_map (
    openalex_id String,
    node_idx UInt32
) ENGINE = MergeTree() ORDER BY openalex_id;

INSERT INTO quarry.node_map
SELECT openalex_id, rowNumberInAllBlocks() AS node_idx
FROM (SELECT openalex_id FROM quarry.papers ORDER BY openalex_id);

-- Step 2: dictionary (~8-10GB hash table)
DROP DICTIONARY IF EXISTS quarry.node_dict;
CREATE DICTIONARY quarry.node_dict (
    openalex_id String,
    node_idx UInt32
)
PRIMARY KEY openalex_id
SOURCE(CLICKHOUSE(TABLE 'node_map' DB 'quarry'))
LAYOUT(HASHED())
LIFETIME(0);
