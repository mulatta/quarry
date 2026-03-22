//! PubMed XML → PostgreSQL pipeline.
//!
//! Architecture (unified pattern with OA):
//!   1. Collect done filenames upfront (single query per source)
//!   2. Rayon parallel parse update XMLs → collect PMIDs + deduped FileData
//!   3. Batch DELETE existing update PMIDs from all tables (one-time)
//!   4. Crossbeam MPMC → N PG writer threads for updates
//!   5. Same pattern for baseline files (skip update PMIDs)
//!
//! Done-file checks happen BEFORE parsing (producer-side) to avoid wasting
//! CPU on already-processed files.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::build::config::BuildConfig;
use crate::build::pg_sink::PgSink;
use crate::parse::xml;

/// Stats from a PubMed build run.
pub struct PubmedBuildStats {
    pub num_papers: usize,
    pub num_authors: usize,
    pub num_mesh: usize,
    pub num_grants: usize,
    pub num_chemicals: usize,
    pub num_files_processed: usize,
    pub num_failed_files: usize,
    pub elapsed_secs: f64,
}

/// Parsed output from a single XML file, ready for PG writes.
struct FileData {
    result: xml::ParseResult,
    filename: String,
}

/// Per-thread write counters, merged at the end.
#[derive(Default)]
struct PubmedCounters {
    papers: usize,
    authors: usize,
    mesh: usize,
    grants: usize,
    chemicals: usize,
    failed_files: usize,
}

/// Write a single parsed file to PG within a transaction.
fn write_pubmed_file(
    sink: &mut PgSink,
    data: &FileData,
    source: &str,
    counters: &mut PubmedCounters,
) -> Result<(), Box<dyn std::error::Error>> {
    let pr = &data.result;
    let n_papers = pr.papers.len();

    sink.begin()?;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        sink.copy_papers(&pr.papers)?;
        sink.copy_authors(&pr.authors)?;
        sink.copy_mesh_headings(&pr.mesh_headings)?;
        sink.copy_grants(&pr.grants)?;
        sink.copy_chemicals(&pr.chemicals)?;
        if !pr.delete_pmids.is_empty() {
            sink.soft_delete_pmids(&pr.delete_pmids)?;
        }
        sink.mark_progress(source, &data.filename, n_papers as i64)?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            sink.commit()?;
            counters.papers += n_papers;
            counters.authors += pr.authors.len();
            counters.mesh += pr.mesh_headings.len();
            counters.grants += pr.grants.len();
            counters.chemicals += pr.chemicals.len();
            Ok(())
        }
        Err(e) => {
            eprintln!(
                "pubmed: WARN: write failed for {}: {}",
                data.filename,
                crate::err_chain(&*e)
            );
            sink.rollback()?;
            counters.failed_files += 1;
            Err(e)
        }
    }
}

type WriterHandles = Vec<std::thread::JoinHandle<Result<PubmedCounters, Box<dyn std::error::Error + Send + Sync>>>>;

/// Spawn N writer threads consuming from a crossbeam channel.
/// Returns join handles and an atomic counter for progress tracking.
fn spawn_writers(
    n_writers: usize,
    pg_conninfo: &str,
    write_rx: crossbeam_channel::Receiver<FileData>,
    source: &str,
    t0: Instant,
) -> (WriterHandles, Arc<AtomicUsize>) {
    let files_written = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..n_writers)
        .map(|i| {
            let rx = write_rx.clone();
            let conninfo = pg_conninfo.to_string();
            let written = Arc::clone(&files_written);
            let src = source.to_string();
            std::thread::Builder::new()
                .name(format!("pm-writer-{i}"))
                .spawn(move || -> Result<PubmedCounters, Box<dyn std::error::Error + Send + Sync>> {
                    let mut sink = PgSink::connect(&conninfo)
                        .map_err(|e| format!("pm-writer-{i} connect: {e}"))?;
                    let mut counters = PubmedCounters::default();
                    while let Ok(data) = rx.recv() {
                        if data.result.papers.is_empty() {
                            continue;
                        }
                        if let Err(e) = write_pubmed_file(&mut sink, &data, &src, &mut counters) {
                            eprintln!("pubmed: writer-{i} error for {}: {e}", data.filename);
                        }
                        let done = written.fetch_add(1, Ordering::Relaxed) + 1;
                        if done.is_multiple_of(100) {
                            let elapsed = t0.elapsed().as_secs_f64();
                            eprintln!(
                                "pubmed: {done} {src} files written, {} papers ({elapsed:.1}s)",
                                counters.papers,
                            );
                        }
                    }
                    Ok(counters)
                })
                .expect("failed to spawn pm-writer thread")
        })
        .collect();
    (handles, files_written)
}

/// Join writer threads and merge counters.
fn join_writers(
    handles: WriterHandles,
    base: &mut PubmedCounters,
) {
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(c)) => {
                base.papers += c.papers;
                base.authors += c.authors;
                base.mesh += c.mesh;
                base.grants += c.grants;
                base.chemicals += c.chemicals;
                base.failed_files += c.failed_files;
            }
            Ok(Err(e)) => eprintln!("pubmed: writer-{i} error: {e}"),
            Err(_) => eprintln!("pubmed: writer-{i} panicked"),
        }
    }
}

/// Build PubMed data from XML baseline + optional updates → PG directly.
pub fn build_pubmed(
    config: &BuildConfig,
    xml_dir: &Path,
    updates_dir: Option<&Path>,
    pg_conninfo: &str,
) -> Result<PubmedBuildStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let xml_files = find_xml_files(xml_dir)?;
    let num_baseline = xml_files.len();
    eprintln!(
        "pubmed: found {num_baseline} baseline XML files in {}",
        xml_dir.display()
    );

    let update_files = if let Some(udir) = updates_dir {
        let files = find_xml_files(udir)?;
        eprintln!(
            "pubmed: found {} update XML files in {}",
            files.len(),
            udir.display()
        );
        files
    } else {
        vec![]
    };
    let num_files = num_baseline + update_files.len();

    // Collect done filenames upfront (single query per source).
    let mut sink = PgSink::connect(pg_conninfo)?;
    let done_updates = sink.done_filenames("updates")?;
    let done_baseline = sink.done_filenames("baseline")?;
    eprintln!(
        "pubmed: done files — {} updates, {} baseline (skipping)",
        done_updates.len(),
        done_baseline.len()
    );

    let n_threads = config.effective_parse_threads();
    let n_writers = config.effective_pg_writer_threads();
    let chan_buf = if config.channel_buffer == 0 {
        n_threads * 2
    } else {
        config.channel_buffer.max(1)
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()?;

    let mut counters = PubmedCounters::default();

    // ── Phase 1: Updates ──────────────────────────────────────────────
    // 1. Parse ALL update files in parallel (including done — need PMIDs for baseline dedup)
    // 2. Batch DELETE all update PMIDs from DB (one-time, before any writes)
    // 3. Write undone update files via parallel writers
    let update_pmids: HashSet<i32> = if !update_files.is_empty() {
        // Step 1: Parse all update files with rayon.
        let mut parsed: Vec<(String, Option<xml::ParseResult>)> = pool.install(|| {
            update_files
                .par_iter()
                .map(|path| {
                    let filename = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    match xml::parse_file(path) {
                        Ok(pr) => (filename, Some(pr)),
                        Err(e) => {
                            eprintln!("pubmed: WARN: update parse {}: {e}", path.display());
                            (filename, None)
                        }
                    }
                })
                .collect()
        });

        // Sort descending (newest first) for correct dedup ordering.
        parsed.sort_by(|a, b| b.0.cmp(&a.0));

        // Collect all PMIDs (for baseline dedup) and dedup across update files.
        let mut all_pmids = HashSet::new();
        let mut undone_files: Vec<FileData> = Vec::new();

        for (filename, opt_pr) in &mut parsed {
            let pr = match opt_pr {
                Some(pr) => pr,
                None => {
                    counters.failed_files += 1;
                    continue;
                }
            };

            if done_updates.contains(filename.as_str()) {
                // Already written — just collect PMIDs for baseline dedup.
                for p in &pr.papers {
                    all_pmids.insert(p.pmid);
                }
                continue;
            }

            // Dedup: remove PMIDs already seen from newer files.
            pr.papers.retain(|p| !all_pmids.contains(&p.pmid));
            let mut file_pmids = HashSet::new();
            pr.papers.retain(|p| file_pmids.insert(p.pmid));
            pr.authors.retain(|a| file_pmids.contains(&a.pmid));
            pr.mesh_headings
                .retain(|m| file_pmids.contains(&m.pmid));
            pr.grants.retain(|g| file_pmids.contains(&g.pmid));
            pr.chemicals.retain(|c| file_pmids.contains(&c.pmid));
            all_pmids.extend(&file_pmids);

            undone_files.push(FileData {
                result: std::mem::take(opt_pr).unwrap(),
                filename: filename.clone(),
            });
        }

        // Step 2: Batch DELETE PMIDs that will be re-inserted (avoid PK conflicts).
        if !undone_files.is_empty() {
            let undone_pmids: Vec<i32> = undone_files
                .iter()
                .flat_map(|f| f.result.papers.iter().map(|p| p.pmid))
                .collect();
            if !undone_pmids.is_empty() {
                eprintln!(
                    "pubmed: batch DELETE {} PMIDs before update re-insert",
                    undone_pmids.len()
                );
                sink.begin()?;
                match sink.batch_delete_pmids(&undone_pmids) {
                    Ok(n) => {
                        sink.commit()?;
                        eprintln!("pubmed: deleted {n} existing papers");
                    }
                    Err(e) => {
                        eprintln!("pubmed: WARN: batch delete failed: {e}");
                        sink.rollback()?;
                    }
                }
            }

            // Step 3: Write undone update files via parallel writers.
            eprintln!(
                "pubmed: writing {} update files ({n_writers} writers)",
                undone_files.len()
            );
            let (write_tx, write_rx) = crossbeam_channel::bounded::<FileData>(chan_buf);
            let (handles, written_count) =
                spawn_writers(n_writers, pg_conninfo, write_rx.clone(), "updates", t0);
            drop(write_rx);

            for data in undone_files {
                if write_tx.send(data).is_err() {
                    eprintln!("pubmed: all update writers died");
                    break;
                }
            }
            drop(write_tx);
            join_writers(handles, &mut counters);

            let written = written_count.load(Ordering::Relaxed);
            eprintln!(
                "pubmed: phase 1 done — {} unique update PMIDs, {} papers written",
                all_pmids.len(),
                counters.papers,
            );
            let _ = written; // used in eprintln above via counters
        } else {
            eprintln!(
                "pubmed: phase 1 done — {} unique update PMIDs, 0 papers written (all done)",
                all_pmids.len(),
            );
        }
        all_pmids
    } else {
        HashSet::new()
    };

    // ── Phase 2: Baseline — rayon parse → crossbeam → N writers ──────
    let (write_tx, write_rx) = crossbeam_channel::bounded::<FileData>(chan_buf);
    let (handles, files_written) =
        spawn_writers(n_writers, pg_conninfo, write_rx.clone(), "baseline", t0);
    drop(write_rx);

    eprintln!("pubmed: phase 2 — {n_writers} writer threads, channel buffer {chan_buf}");

    // Producer: rayon parallel parse → crossbeam channel.
    let skip_pmids = Arc::new(update_pmids);
    let done_bl = Arc::new(done_baseline);
    let skipped_count = Arc::new(AtomicUsize::new(0));
    let skipped_clone = Arc::clone(&skipped_count);

    let producer = std::thread::spawn(move || {
        pool.install(|| {
            xml_files
                .par_iter()
                .for_each_with(write_tx, |tx, path| {
                    let filename = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();

                    // Skip already-done files BEFORE parsing.
                    if done_bl.contains(&filename) {
                        skipped_clone.fetch_add(1, Ordering::Relaxed);
                        return;
                    }

                    match xml::parse_file(path) {
                        Ok(mut pr) => {
                            if !skip_pmids.is_empty() {
                                pr.papers.retain(|p| !skip_pmids.contains(&p.pmid));
                                pr.authors.retain(|a| !skip_pmids.contains(&a.pmid));
                                pr.mesh_headings
                                    .retain(|m| !skip_pmids.contains(&m.pmid));
                                pr.grants.retain(|g| !skip_pmids.contains(&g.pmid));
                                pr.chemicals.retain(|c| !skip_pmids.contains(&c.pmid));
                            }
                            let _ = tx.send(FileData {
                                result: pr,
                                filename,
                            });
                        }
                        Err(e) => {
                            eprintln!("pubmed: WARN: {}: {e}", path.display());
                        }
                    }
                });
        });
    });

    producer
        .join()
        .map_err(|_| "rayon producer thread panicked")?;

    join_writers(handles, &mut counters);

    let elapsed = t0.elapsed().as_secs_f64();
    let written = files_written.load(Ordering::Relaxed);
    let skipped = skipped_count.load(Ordering::Relaxed);
    eprintln!(
        "pubmed: done — {} papers in {elapsed:.1}s ({n_writers} writers, {written} written, {skipped} skipped)",
        counters.papers,
    );

    Ok(PubmedBuildStats {
        num_papers: counters.papers,
        num_authors: counters.authors,
        num_mesh: counters.mesh,
        num_grants: counters.grants,
        num_chemicals: counters.chemicals,
        num_files_processed: num_files,
        num_failed_files: counters.failed_files,
        elapsed_secs: elapsed,
    })
}

/// Find all XML files (.xml.gz or .xml) in a directory, sorted.
fn find_xml_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            name.ends_with(".xml.gz") || name.ends_with(".xml")
        })
        .collect();
    files.sort();
    Ok(files)
}
