-- Export id_map as newline-separated text (node_idx order)
SELECT openalex_id FROM quarry.node_map ORDER BY node_idx FORMAT TSVRaw
