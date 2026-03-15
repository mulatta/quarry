-- Export edges as binary uint32 pairs via dictionary lookup
-- Streaming: only dictionary stays in memory (~8-10GB)
-- Output: RowBinary (uint32, uint32) to stdout
SELECT
    dictGet('quarry.node_dict', 'node_idx', e.citing_id) AS src,
    dictGet('quarry.node_dict', 'node_idx', e.cited_id) AS dst
FROM quarry.edges e
WHERE dictHas('quarry.node_dict', e.citing_id)
  AND dictHas('quarry.node_dict', e.cited_id)
FORMAT RowBinary
