//! Manifest-diff based incremental sync.
//!
//! Compares old manifest snapshot against new live manifest to determine
//! which shards need downloading, re-downloading, or deletion.
//!
//! Also owns the [`ManifestSnapshot`] and [`ManifestEntry`] persistence types.

use std::fs;
use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::oa::{OAShard, TABLES, is_shard_complete};

// ============================================================
// Manifest snapshot (persisted after each sync)
// ============================================================

/// A snapshot of the remote manifest, saved after each successful sync.
/// Used to detect changed/removed shards via fingerprint comparison.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ManifestSnapshot {
    pub saved_at: String,
    pub entries: Vec<ManifestEntry>,
}

/// One entry in the manifest snapshot, keyed by URL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    pub url: String,
    pub shard_idx: usize,
    pub content_length: u64,
    pub record_count: u64,
    pub updated_date: Option<String>,
}

impl ManifestSnapshot {
    const FILENAME: &str = ".manifest.json";

    /// Load from `{output_dir}/.manifest.json`. Returns None on first run.
    pub fn load(output_dir: &Path) -> Option<Self> {
        let path = output_dir.join(Self::FILENAME);
        let text = fs::read_to_string(&path).ok()?;
        match sonic_rs::from_str(&text) {
            Ok(snap) => Some(snap),
            Err(e) => {
                tracing::warn!("Corrupt manifest snapshot {}: {e}", path.display());
                None
            }
        }
    }

    /// Atomic write: tmp → rename.
    pub fn save(&self, output_dir: &Path) -> anyhow::Result<()> {
        let path = output_dir.join(Self::FILENAME);
        let tmp = output_dir.join(format!("{}.tmp", Self::FILENAME));
        let json = sonic_rs::to_string_pretty(self)?;
        fs::write(&tmp, json)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Build a snapshot from live manifest shards (test helper).
    #[cfg(test)]
    pub fn from_shards(shards: &[OAShard]) -> Self {
        Self {
            saved_at: chrono::Utc::now().to_rfc3339(),
            entries: shards
                .iter()
                .map(|s| ManifestEntry {
                    url: s.url.clone(),
                    shard_idx: s.shard_idx,
                    content_length: s.content_length.unwrap_or(0),
                    record_count: s.record_count,
                    updated_date: s.updated_date.clone(),
                })
                .collect(),
        }
    }

    /// Build a snapshot excluding shards whose URL is in `exclude_urls`.
    /// Failed shards are excluded so they appear as "changed" on next run.
    pub fn from_shards_excluding(shards: &[OAShard], exclude_urls: &FxHashSet<&str>) -> Self {
        Self {
            saved_at: chrono::Utc::now().to_rfc3339(),
            entries: shards
                .iter()
                .filter(|s| !exclude_urls.contains(s.url.as_str()))
                .map(|s| ManifestEntry {
                    url: s.url.clone(),
                    shard_idx: s.shard_idx,
                    content_length: s.content_length.unwrap_or(0),
                    record_count: s.record_count,
                    updated_date: s.updated_date.clone(),
                })
                .collect(),
        }
    }
}

// ============================================================
// Types
// ============================================================

/// Result of comparing old manifest snapshot against new live manifest.
pub struct ManifestDiff {
    /// Shards that are new or whose fingerprint changed.
    pub changed: Vec<OAShard>,
    /// Shards that existed in old manifest but are gone from new.
    pub removed: Vec<RemovedShard>,
    /// Count of shards unchanged with valid local files.
    pub unchanged_ok: usize,
    /// Shards unchanged in manifest but local files are corrupt/missing.
    pub unchanged_missing: Vec<OAShard>,
}

/// A shard present in the old manifest but absent from the new one.
pub struct RemovedShard {
    pub shard_idx: usize,
}

/// Combined result: stabilized shards + their diff.
pub struct ManifestDiffResult {
    /// Shards with stabilized indices applied.
    pub shards: Vec<OAShard>,
    /// The diff between old and new manifests.
    pub diff: ManifestDiff,
}

// ============================================================
// Public API
// ============================================================

/// Stabilize shard indices and compute manifest diff in one step.
///
/// This enforces the correct ordering: indices must be stabilized
/// before diff can check local files by shard_idx.
pub fn compute_manifest_diff(
    mut new_shards: Vec<OAShard>,
    old_snapshot: Option<&ManifestSnapshot>,
    output_dir: &Path,
) -> ManifestDiffResult {
    stabilize_shard_indices(&mut new_shards, old_snapshot);
    let diff = diff_manifest(old_snapshot, &new_shards, output_dir);
    ManifestDiffResult {
        shards: new_shards,
        diff,
    }
}

/// Delete all parquet files (12 tables) for removed shards.
pub fn delete_shard_files(output_dir: &Path, removed: &[RemovedShard]) {
    for r in removed {
        let mut deleted = 0usize;
        for table in TABLES {
            let path = output_dir
                .join(table)
                .join(format!("shard_{:04}.parquet", r.shard_idx));
            if path.exists() {
                if let Err(e) = std::fs::remove_file(&path) {
                    tracing::warn!("Failed to delete {}: {e}", path.display());
                } else {
                    deleted += 1;
                }
            }
        }
        if deleted > 0 {
            tracing::info!(
                "Deleted {deleted} files for removed shard_{:04}",
                r.shard_idx
            );
        }
    }
}

// ============================================================
// Internal
// ============================================================

/// Compare old manifest snapshot against new live shards.
///
/// Fingerprint = `(content_length, record_count)` — unique per shard in OpenAlex.
/// - `old=None` (first run): all shards → changed.
/// - URL match + same fingerprint + local files valid → unchanged_ok.
/// - URL match + same fingerprint + local files invalid → unchanged_missing.
/// - URL match + different fingerprint → changed.
/// - URL in old but not in new → removed.
fn diff_manifest(
    old: Option<&ManifestSnapshot>,
    new_shards: &[OAShard],
    output_dir: &Path,
) -> ManifestDiff {
    let Some(old) = old else {
        return ManifestDiff {
            changed: new_shards.to_vec(),
            removed: Vec::new(),
            unchanged_ok: 0,
            unchanged_missing: Vec::new(),
        };
    };

    let old_map: FxHashMap<&str, &ManifestEntry> =
        old.entries.iter().map(|e| (e.url.as_str(), e)).collect();

    let mut changed = Vec::new();
    let mut unchanged_ok = 0usize;
    let mut unchanged_missing = Vec::new();
    let mut seen_urls: FxHashSet<&str> = FxHashSet::default();

    for shard in new_shards {
        seen_urls.insert(&shard.url);
        match old_map.get(shard.url.as_str()) {
            Some(old_entry)
                if old_entry.content_length == shard.content_length.unwrap_or(0)
                    && old_entry.record_count == shard.record_count =>
            {
                if is_shard_complete(output_dir, shard.shard_idx) {
                    unchanged_ok += 1;
                } else {
                    unchanged_missing.push(shard.clone());
                }
            }
            _ => {
                changed.push(shard.clone());
            }
        }
    }

    let removed: Vec<RemovedShard> = old
        .entries
        .iter()
        .filter(|e| !seen_urls.contains(e.url.as_str()))
        .map(|e| RemovedShard {
            shard_idx: e.shard_idx,
        })
        .collect();

    ManifestDiff {
        changed,
        removed,
        unchanged_ok,
        unchanged_missing,
    }
}

/// Re-assign shard_idx so existing URLs keep their old index.
/// New URLs get indices starting from `max_old_idx + 1`.
fn stabilize_shard_indices(new_shards: &mut [OAShard], old: Option<&ManifestSnapshot>) {
    let Some(old) = old else {
        return;
    };

    let url_to_idx: FxHashMap<&str, usize> = old
        .entries
        .iter()
        .map(|e| (e.url.as_str(), e.shard_idx))
        .collect();

    let max_old_idx = old.entries.iter().map(|e| e.shard_idx).max().unwrap_or(0);
    let mut next_idx = max_old_idx + 1;

    for shard in new_shards.iter_mut() {
        if let Some(&idx) = url_to_idx.get(shard.url.as_str()) {
            shard.shard_idx = idx;
        } else {
            shard.shard_idx = next_idx;
            next_idx += 1;
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn make_shards(n: usize) -> Vec<OAShard> {
        (0..n)
            .map(|i| OAShard {
                shard_idx: i,
                url: format!(
                    "https://example.com/updated_date=2025-06-{:02}/part.gz",
                    i + 1
                ),
                content_length: Some(1000),
                record_count: 100,
                updated_date: Some(format!("2025-06-{:02}", i + 1)),
            })
            .collect()
    }

    // ---- diff_manifest ----

    #[test]
    fn diff_first_run() {
        let shards = make_shards(5);
        let diff = diff_manifest(None, &shards, Path::new("/nonexistent"));
        assert_eq!(diff.changed.len(), 5);
        assert_eq!(diff.removed.len(), 0);
        assert_eq!(diff.unchanged_ok, 0);
        assert_eq!(diff.unchanged_missing.len(), 0);
    }

    #[test]
    fn diff_unchanged_missing() {
        let shards = make_shards(3);
        let snapshot = ManifestSnapshot::from_shards(&shards);
        let diff = diff_manifest(Some(&snapshot), &shards, Path::new("/nonexistent"));
        assert_eq!(diff.changed.len(), 0);
        assert_eq!(diff.unchanged_missing.len(), 3);
        assert_eq!(diff.unchanged_ok, 0);
        assert_eq!(diff.removed.len(), 0);
    }

    #[test]
    fn diff_changed_fingerprint() {
        let shards = make_shards(3);
        let mut old_shards = shards.clone();
        old_shards[1].record_count = 999;
        let snapshot = ManifestSnapshot::from_shards(&old_shards);
        let diff = diff_manifest(Some(&snapshot), &shards, Path::new("/nonexistent"));
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].shard_idx, 1);
        assert_eq!(diff.unchanged_missing.len(), 2);
        assert_eq!(diff.removed.len(), 0);
    }

    #[test]
    fn diff_changed_content_length() {
        let shards = make_shards(2);
        let mut old_shards = shards.clone();
        old_shards[0].content_length = Some(9999);
        let snapshot = ManifestSnapshot::from_shards(&old_shards);
        let diff = diff_manifest(Some(&snapshot), &shards, Path::new("/nonexistent"));
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].shard_idx, 0);
        assert_eq!(diff.unchanged_missing.len(), 1);
    }

    #[test]
    fn diff_removed() {
        let shards = make_shards(2);
        let old_shards = make_shards(3);
        let snapshot = ManifestSnapshot::from_shards(&old_shards);
        let diff = diff_manifest(Some(&snapshot), &shards, Path::new("/nonexistent"));
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].shard_idx, 2);
    }

    #[test]
    fn diff_new_url() {
        let mut shards = make_shards(2);
        let snapshot = ManifestSnapshot::from_shards(&shards);
        shards.push(OAShard {
            shard_idx: 2,
            url: "https://example.com/new_shard.gz".to_string(),
            content_length: Some(2000),
            record_count: 200,
            updated_date: Some("2025-07-01".to_string()),
        });
        let diff = diff_manifest(Some(&snapshot), &shards, Path::new("/nonexistent"));
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].url, "https://example.com/new_shard.gz");
    }

    #[test]
    fn diff_unchanged_ok_with_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let shards = make_shards(1);
        let snapshot = ManifestSnapshot::from_shards(&shards);

        for table in TABLES {
            let path = dir.path().join(table).join("shard_0000.parquet");
            write_minimal_parquet(&path);
        }

        let diff = diff_manifest(Some(&snapshot), &shards, dir.path());
        assert_eq!(diff.unchanged_ok, 1);
        assert_eq!(diff.changed.len(), 0);
        assert_eq!(diff.unchanged_missing.len(), 0);
    }

    #[test]
    fn diff_mixed() {
        let old_shards = make_shards(3);
        let snapshot = ManifestSnapshot::from_shards(&old_shards);
        let new_shards = vec![
            old_shards[0].clone(),
            OAShard {
                record_count: 999,
                ..old_shards[1].clone()
            },
        ];
        let diff = diff_manifest(Some(&snapshot), &new_shards, Path::new("/nonexistent"));
        assert_eq!(diff.unchanged_missing.len(), 1);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].shard_idx, 1);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].shard_idx, 2);
        assert_eq!(diff.unchanged_ok, 0);
    }

    // ---- stabilize_shard_indices ----

    #[test]
    fn stabilize_preserves_old() {
        let old_shards = make_shards(3);
        let snapshot = ManifestSnapshot::from_shards(&old_shards);
        let mut new_shards = vec![
            OAShard {
                shard_idx: 0,
                url: old_shards[2].url.clone(),
                content_length: Some(1000),
                record_count: 100,
                updated_date: old_shards[2].updated_date.clone(),
            },
            OAShard {
                shard_idx: 1,
                url: old_shards[0].url.clone(),
                content_length: Some(1000),
                record_count: 100,
                updated_date: old_shards[0].updated_date.clone(),
            },
            OAShard {
                shard_idx: 2,
                url: "https://example.com/new.gz".to_string(),
                content_length: Some(2000),
                record_count: 200,
                updated_date: Some("2025-07-01".to_string()),
            },
        ];
        stabilize_shard_indices(&mut new_shards, Some(&snapshot));
        assert_eq!(new_shards[0].shard_idx, 2);
        assert_eq!(new_shards[1].shard_idx, 0);
        assert_eq!(new_shards[2].shard_idx, 3);
    }

    #[test]
    fn stabilize_first_run() {
        let mut shards = make_shards(3);
        stabilize_shard_indices(&mut shards, None);
        assert_eq!(shards[0].shard_idx, 0);
        assert_eq!(shards[1].shard_idx, 1);
        assert_eq!(shards[2].shard_idx, 2);
    }

    #[test]
    fn stabilize_all_new_urls() {
        let old_shards = make_shards(3);
        let snapshot = ManifestSnapshot::from_shards(&old_shards);
        let mut new_shards = vec![
            OAShard {
                shard_idx: 0,
                url: "https://example.com/brand_new_a.gz".to_string(),
                content_length: Some(1000),
                record_count: 100,
                updated_date: None,
            },
            OAShard {
                shard_idx: 1,
                url: "https://example.com/brand_new_b.gz".to_string(),
                content_length: Some(1000),
                record_count: 100,
                updated_date: None,
            },
        ];
        stabilize_shard_indices(&mut new_shards, Some(&snapshot));
        assert_eq!(new_shards[0].shard_idx, 3);
        assert_eq!(new_shards[1].shard_idx, 4);
    }

    // ---- compute_manifest_diff (integrated) ----

    #[test]
    fn compute_diff_enforces_order() {
        // Verify that compute_manifest_diff stabilizes before diffing
        let old_shards = make_shards(2);
        let snapshot = ManifestSnapshot::from_shards(&old_shards);

        // New shard with URL of old shard[1] but listed first
        let new_shards = vec![OAShard {
            shard_idx: 0,
            url: old_shards[1].url.clone(),
            content_length: Some(1000),
            record_count: 100,
            updated_date: old_shards[1].updated_date.clone(),
        }];
        let result = compute_manifest_diff(new_shards, Some(&snapshot), Path::new("/nonexistent"));
        // shard_idx should be stabilized to 1 (from old)
        assert_eq!(result.shards[0].shard_idx, 1);
        // old shard[0] removed
        assert_eq!(result.diff.removed.len(), 1);
        assert_eq!(result.diff.removed[0].shard_idx, 0);
    }

    // ---- delete_shard_files ----

    #[test]
    fn delete_removes_parquets() {
        let dir = tempfile::TempDir::new().unwrap();
        for table in TABLES {
            let path = dir.path().join(table).join("shard_0005.parquet");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"fake").unwrap();
        }
        let removed = vec![RemovedShard { shard_idx: 5 }];
        delete_shard_files(dir.path(), &removed);
        for table in TABLES {
            let path = dir.path().join(table).join("shard_0005.parquet");
            assert!(
                !path.exists(),
                "should have been deleted: {}",
                path.display()
            );
        }
    }

    #[test]
    fn delete_nonexistent_is_noop() {
        let dir = tempfile::TempDir::new().unwrap();
        let removed = vec![RemovedShard { shard_idx: 99 }];
        delete_shard_files(dir.path(), &removed);
    }

    // ---- ManifestSnapshot persistence ----

    #[test]
    fn snapshot_save_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let shards = make_shards(3);
        let snap = ManifestSnapshot::from_shards(&shards);
        snap.save(dir.path()).unwrap();

        let loaded = ManifestSnapshot::load(dir.path()).unwrap();
        assert_eq!(loaded.entries.len(), 3);
        assert_eq!(loaded.entries[0].url, shards[0].url);
        assert_eq!(loaded.entries[2].record_count, 100);
    }

    #[test]
    fn snapshot_load_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(ManifestSnapshot::load(dir.path()).is_none());
    }

    #[test]
    fn snapshot_load_corrupt() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".manifest.json"), "NOT JSON {{{{").unwrap();
        assert!(ManifestSnapshot::load(dir.path()).is_none());
    }

    #[test]
    fn snapshot_from_shards_none_content_length() {
        let shards = vec![OAShard {
            shard_idx: 0,
            url: "https://example.com/part.gz".to_string(),
            content_length: None,
            record_count: 50,
            updated_date: None,
        }];
        let snap = ManifestSnapshot::from_shards(&shards);
        assert_eq!(snap.entries[0].content_length, 0);
    }

    // ---- helper ----

    fn write_minimal_parquet(path: &Path) {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1]))],
        )
        .unwrap();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
}
