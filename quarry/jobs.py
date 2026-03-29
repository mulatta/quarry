"""Dagster job definitions for pipeline execution modes.

build:      sync → parse → ch_init → ch_load → ch_transform → parquet_export
load:       pg_load + csr_graph + paper_embeddings (parquet on disk assumed)
embeddings: paper_embeddings only
full:       build + load (single machine E2E, no R2)

R2 sync is sensor-driven (not part of any job):
  build done → r2_upload_sensor → r2_upload
  r2_upload done → distributed_r2_sync → r2_download (remote)
  r2_download done → distributed_serve → load (remote)
"""

from dagster import AssetSelection, define_asset_job

# ── Asset selections ──

_build = AssetSelection.assets(
    # sync
    "oa_sync",
    "pubmed_baseline_sync",
    "pubmed_updates_sync",
    "mesh_descriptor_sync",
    "icite_metadata_sync",
    # parse
    "oa_parse",
    "pm_parse",
    "mesh_stage",
    # CH
    "ch_init",
    "ch_load",
    "ch_transform",
    # export
    "parquet_export",
)

_load = AssetSelection.assets(
    "pg_load",
    "csr_graph",
    "paper_embeddings",
)

_embed = AssetSelection.assets("paper_embeddings")
_r2_up = AssetSelection.assets("r2_upload")
_r2_down = AssetSelection.assets("r2_download")

# ── Jobs ──

# DWH build: source sync → CH processing → parquet export
build_job = define_asset_job("build", selection=_build)

# Serving layer load: parquet → PG + CSR + LanceDB
load_job = define_asset_job("load", selection=_load)

# Embeddings only (GPU, reads parquet directly)
embeddings_job = define_asset_job("embeddings", selection=_embed)

# Full: single-machine E2E (build → load, no R2)
full_job = define_asset_job("full", selection=_build | _load)

# Sensor targets: single-asset jobs for R2 chain
r2_upload_job = define_asset_job("r2_upload_job", selection=_r2_up)
r2_download_job = define_asset_job("r2_download_job", selection=_r2_down)
