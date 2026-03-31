"""Unit tests for embedding parquet filter logic."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def works_parquet(tmp_path, monkeypatch):
    """Create a minimal works parquet with language + is_retracted columns."""
    works_dir = tmp_path / "export" / "works" / "tier=t1"
    works_dir.mkdir(parents=True)

    table = pa.table(
        {
            "work_id": ["W1", "W2", "W3", "W4", "W5"],
            "title": ["En paper", "Fr paper", "Null lang", "Retracted", "No abstract"],
            "abstract": ["abs1", "résumé", "abs3", "abs4", None],
            "type": ["article"] * 5,
            "language": ["en", "fr", None, "en", "en"],
            "is_retracted": [False, False, False, True, False],
        }
    )
    pq.write_table(table, works_dir / "000.parquet")

    monkeypatch.setattr("quarry.config.settings.parquet_dir", tmp_path / "export")
    monkeypatch.setattr(
        "quarry.config.settings.embed_allowed_types",
        ["article"],
    )
    return tmp_path


def test_language_filter(works_parquet):
    """Only English + null language pass the filter."""
    from quarry.etl.embeddings import _parquet_batches

    all_works = []
    for batch in _parquet_batches(batch_size=100):
        all_works.extend(batch)

    work_ids = {w["work_id"] for w in all_works}
    assert "W1" in work_ids, "English paper should pass"
    assert "W2" not in work_ids, "French paper should be filtered"
    assert "W3" in work_ids, "Null language should pass"
    assert "W4" not in work_ids, "Retracted paper should be filtered"
    assert "W5" not in work_ids, "No abstract should be filtered"


def test_no_language_column(tmp_path, monkeypatch):
    """Gracefully handle parquet without language column."""
    works_dir = tmp_path / "export" / "works" / "tier=t1"
    works_dir.mkdir(parents=True)

    # Parquet WITHOUT language column
    table = pa.table(
        {
            "work_id": ["W1", "W2"],
            "title": ["Paper 1", "Paper 2"],
            "abstract": ["abs1", "abs2"],
            "type": ["article", "article"],
            "is_retracted": [False, False],
        }
    )
    pq.write_table(table, works_dir / "000.parquet")

    monkeypatch.setattr("quarry.config.settings.parquet_dir", tmp_path / "export")
    monkeypatch.setattr(
        "quarry.config.settings.embed_allowed_types",
        ["article"],
    )

    from quarry.etl.embeddings import _parquet_batches

    all_works = []
    for batch in _parquet_batches(batch_size=100):
        all_works.extend(batch)

    assert len(all_works) == 2, "All works should pass without language filter"
