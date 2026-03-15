"""Dagster definitions entry point."""

import dagster as dg

from quarry.assets.csr import csr_graph
from quarry.assets.ingest import ch_authors, ch_edges, ch_papers

defs = dg.Definitions(
    assets=[ch_papers, ch_edges, ch_authors, csr_graph],
)
