"""Dagster job definitions for pipeline execution modes.

etl:        sync → parse → ch → parquet_export → r2_upload
serve:      pg_load + csr_graph + paper_embeddings (assumes parquet on disk)
embeddings: paper_embeddings only
full:       etl (without r2) + serve (single machine E2E)

Distributed workflow is sensor-driven, not a single job:
  weekly schedule → etl_on_batch → (sensor) r2_download → (sensor) serve
"""

from dagster import AssetSelection, define_asset_job

_sync = AssetSelection.assets(
    "oa_sync",
    "pubmed_baseline_sync",
    "pubmed_updates_sync",
    "mesh_descriptor_sync",
    "icite_metadata_sync",
)
_parse = AssetSelection.assets("oa_parse", "pm_parse", "mesh_stage")
_ch = AssetSelection.assets("ch_load", "ch_transform")
_parquet = AssetSelection.assets("parquet_export")
_pg = AssetSelection.assets("pg_load")
_csr = AssetSelection.assets("csr_graph")
_r2_up = AssetSelection.assets("r2_upload")
_r2_down = AssetSelection.assets("r2_download")
_embed = AssetSelection.assets("paper_embeddings")

# ETL → R2 (runs on Batch instance or locally with CH)
etl_job = define_asset_job(
    "etl",
    selection=_sync | _parse | _ch | _parquet | _r2_up,
)

# Serve: load parquet into serving layer (PG + CSR + embeddings)
serve_job = define_asset_job(
    "serve",
    selection=_pg | _csr | _embed,
)

# Embeddings only (reads parquet directly, no CH/PG needed)
embeddings_job = define_asset_job(
    "embeddings",
    selection=_embed,
)

# Full: single-machine E2E (no R2)
full_job = define_asset_job(
    "full",
    selection=_sync | _parse | _ch | _parquet | _pg | _csr | _embed,
)

# Sensor targets: single-asset jobs for chaining
r2_download_job = define_asset_job(
    "r2_download_job",
    selection=_r2_down,
)
