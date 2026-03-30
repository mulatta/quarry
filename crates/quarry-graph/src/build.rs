//! CSV → CSR build: parallel parse + HashMap ID mapping + forward/reverse CSR.
//!
//! Supports i64 node IDs (OpenAlex work_id_int, range ~10^10).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use memmap2::{Advice, Mmap};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn parse_line(line: &[u8]) -> Option<(i64, i64)> {
    if line.is_empty() || !line[0].is_ascii_digit() {
        return None;
    }
    let mut src: i64 = 0;
    let mut i = 0;
    while i < line.len() && line[i].is_ascii_digit() {
        src = src * 10 + (line[i] - b'0') as i64;
        i += 1;
    }
    if i >= line.len() || line[i] != b',' {
        return None;
    }
    i += 1;
    let mut dst: i64 = 0;
    while i < line.len() && line[i].is_ascii_digit() {
        dst = dst * 10 + (line[i] - b'0') as i64;
        i += 1;
    }
    Some((src, dst))
}

/// Parse all lines within a byte range [range_start, range_end).
/// If range_start > 0, skip to the first newline to align to a line boundary.
/// This eliminates the need for a pre-computed line offsets Vec.
fn parse_byte_range(data: &[u8], range_start: usize, range_end: usize) -> (Vec<i64>, Vec<i64>) {
    let mut srcs = Vec::new();
    let mut dsts = Vec::new();

    // Align: if we start mid-line, skip to the next newline (previous range owns it).
    // If we're at a line boundary (byte 0 or preceded by '\n'), start here.
    let mut pos = if range_start == 0 || data[range_start - 1] == b'\n' {
        range_start
    } else {
        match data[range_start..range_end.min(data.len())]
            .iter()
            .position(|&b| b == b'\n')
        {
            Some(p) => range_start + p + 1,
            None => return (srcs, dsts), // no newline in range
        }
    };

    let end = range_end.min(data.len());
    while pos < end {
        let line_end = data[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(data.len(), |p| pos + p);
        if let Some((s, d)) = parse_line(&data[pos..line_end]) {
            srcs.push(s);
            dsts.push(d);
        }
        pos = line_end + 1;
    }

    (srcs, dsts)
}

/// Extract CSR (indptr + indices) from sorted pairs. Does NOT sort — caller must sort first.
fn extract_csr(pairs: &[(u32, u32)], num_nodes: usize) -> (Vec<u64>, Vec<u32>) {
    let mut counts = vec![0u64; num_nodes];
    for &(n, _) in pairs {
        counts[n as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0u64);
    let mut cumsum = 0u64;
    for c in &counts {
        cumsum += c;
        indptr.push(cumsum);
    }
    let indices: Vec<u32> = pairs.iter().map(|&(_, nb)| nb).collect();
    (indptr, indices)
}

fn write_bin<T: Copy>(path: &Path, data: &[T]) -> std::io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
        )
    };
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
    w.write_all(bytes)?;
    w.flush()
}

fn chrono_free_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = dur.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    let mut y = 1970i32;
    let mut remaining = days as i64;
    loop {
        let year_days = if is_leap(y) { 366 } else { 365 };
        if remaining < year_days {
            break;
        }
        remaining -= year_days;
        y += 1;
    }
    let leap = is_leap(y);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut mo = 1u32;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        mo += 1;
    }
    let d = remaining + 1;
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn is_leap(y: i32) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// Core build logic. Uses HashMap for ID→index mapping (supports i64 IDs).
/// Returns (num_nodes, num_edges).
///
/// Memory optimizations applied:
/// - (E) Byte-range chunking instead of pre-computed line offsets (-24GB)
/// - (A) Segmented parallel parse to avoid holding all chunk results (-47GB)
/// - (B) madvise(SEQUENTIAL) + madvise(DONTNEED) to reduce RSS
/// - (C) id_set capacity num_edges/10 instead of num_edges/3 (-5GB)
/// - (D) Scoped fwd CSR to drop before reverse build (-13.5GB)
fn build_core(csv_path: &Path, graph_dir: &Path) -> Result<(usize, usize), String> {
    fs::create_dir_all(graph_dir).map_err(|e| e.to_string())?;

    let file = File::open(csv_path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let data_len = mmap.len();

    // (B) Hint kernel for sequential read-ahead
    let _ = mmap.advise(Advice::Sequential);

    // (E) Byte-range chunking: divide mmap into fixed-size ranges.
    // Each rayon task aligns to the next newline, so no pre-computed offsets needed.
    let byte_chunk_size: usize = 64 * 1024 * 1024; // 64 MB per chunk
    let num_chunks = if data_len == 0 {
        0
    } else {
        data_len.div_ceil(byte_chunk_size)
    };
    eprintln!("  {} bytes, {} byte-range chunks", data_len, num_chunks);

    // (A) Process chunks in segments to limit simultaneous intermediate Vecs.
    // Each segment: parse in parallel, then sequentially extend into src_raw/dst_raw.
    let segment_size = 48; // chunks per segment (~3GB of mmap per batch)
    let num_segments = if num_chunks == 0 {
        0
    } else {
        num_chunks.div_ceil(segment_size)
    };

    let mut src_raw: Vec<i64> = Vec::new();
    let mut dst_raw: Vec<i64> = Vec::new();

    for seg in 0..num_segments {
        let chunk_start = seg * segment_size;
        let chunk_end = ((seg + 1) * segment_size).min(num_chunks);

        let segment_results: Vec<(Vec<i64>, Vec<i64>)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let range_start = i * byte_chunk_size;
                let range_end = ((i + 1) * byte_chunk_size).min(data_len);
                parse_byte_range(&mmap, range_start, range_end)
            })
            .collect();

        for (s, d) in segment_results {
            src_raw.extend(s);
            dst_raw.extend(d);
        }
    }

    // (B) Release RSS pages before dropping the mmap
    // SAFETY: DontNeed only affects paging, we no longer read from mmap after this.
    #[cfg(not(target_os = "windows"))]
    unsafe {
        let _ = mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed);
    }
    drop(mmap);

    build_from_raw_edges(src_raw, dst_raw, graph_dir)
}

/// Shared build pipeline: raw i64 edge vecs → unique IDs → CSR files.
fn build_from_raw_edges(
    src_raw: Vec<i64>,
    dst_raw: Vec<i64>,
    graph_dir: &Path,
) -> Result<(usize, usize), String> {
    let num_edges = src_raw.len();

    eprintln!("  collecting unique node IDs...");
    let mut id_set = std::collections::HashSet::with_capacity(num_edges / 10);
    for &id in &src_raw {
        id_set.insert(id);
    }
    for &id in &dst_raw {
        id_set.insert(id);
    }
    let mut sorted_ids: Vec<i64> = id_set.into_iter().collect();
    sorted_ids.par_sort_unstable();
    let num_nodes = sorted_ids.len();

    assert!(
        num_nodes <= u32::MAX as usize,
        "node count {} exceeds u32::MAX — graph too large for u32 indices",
        num_nodes
    );
    eprintln!(
        "  mapping {} edges to u32 indices ({} nodes)...",
        num_edges, num_nodes
    );

    let src_idx: Vec<u32> = src_raw
        .par_iter()
        .map(|&p| {
            sorted_ids
                .binary_search(&p)
                .expect("unmapped source ID in edge list") as u32
        })
        .collect();
    drop(src_raw);
    let dst_idx: Vec<u32> = dst_raw
        .par_iter()
        .map(|&p| {
            sorted_ids
                .binary_search(&p)
                .expect("unmapped target ID in edge list") as u32
        })
        .collect();
    drop(dst_raw);

    let mut pairs: Vec<(u32, u32)> = src_idx
        .into_par_iter()
        .zip(dst_idx.into_par_iter())
        .collect();

    // Forward CSR
    {
        let fwd_dir = graph_dir.join("forward");
        fs::create_dir_all(&fwd_dir).map_err(|e| e.to_string())?;
        eprintln!("  building forward CSR ({} edges)...", num_edges);
        pairs.par_sort_unstable_by_key(|&(n, _)| n);
        let (fwd_indptr, fwd_indices) = extract_csr(&pairs, num_nodes);
        write_bin(&fwd_dir.join("indptr.bin"), &fwd_indptr).map_err(|e| e.to_string())?;
        write_bin(&fwd_dir.join("indices.bin"), &fwd_indices).map_err(|e| e.to_string())?;
    }

    // Reverse CSR: swap src↔dst in-place, re-sort, extract
    {
        let rev_dir = graph_dir.join("reverse");
        fs::create_dir_all(&rev_dir).map_err(|e| e.to_string())?;
        eprintln!("  building reverse CSR ({} edges)...", num_edges);
        pairs
            .par_iter_mut()
            .for_each(|(a, b)| std::mem::swap(a, b));
        pairs.par_sort_unstable_by_key(|&(n, _)| n);
        let (rev_indptr, rev_indices) = extract_csr(&pairs, num_nodes);
        write_bin(&rev_dir.join("indptr.bin"), &rev_indptr).map_err(|e| e.to_string())?;
        write_bin(&rev_dir.join("indices.bin"), &rev_indices).map_err(|e| e.to_string())?;
    }

    // id_map.bin
    {
        let mut w = BufWriter::new(
            File::create(graph_dir.join("id_map.bin")).map_err(|e| e.to_string())?,
        );
        for &id in &sorted_ids {
            writeln!(w, "{}", id).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())?;
    }

    let build_date = chrono_free_iso8601();
    let meta = serde_json::json!({
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "build_date": build_date,
    });
    fs::write(
        graph_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .map_err(|e| e.to_string())?;

    Ok((num_nodes, num_edges))
}

#[pyfunction]
pub fn build_from_csv(py: Python<'_>, csv_path: &str, graph_dir: &str) -> PyResult<PyObject> {
    let csv_path = csv_path.to_owned();
    let graph_dir = graph_dir.to_owned();

    let t0 = Instant::now();
    eprintln!("Reading {} ...", csv_path);

    let (num_nodes, num_edges) = py.allow_threads(|| {
        build_core(Path::new(&csv_path), Path::new(&graph_dir))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    })?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "CSR done in {:.0}s: {} nodes, {} edges",
        elapsed, num_nodes, num_edges
    );

    // Read build_date from the meta.json we just wrote
    let meta_str = std::fs::read_to_string(Path::new(&graph_dir).join("meta.json"))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("num_nodes", num_nodes)?;
    dict.set_item("num_edges", num_edges)?;
    dict.set_item(
        "build_date",
        meta["build_date"].as_str().unwrap_or(""),
    )?;
    dict.set_item("source", &csv_path)?;
    Ok(dict.into())
}

/// Build CSR from Parquet file directly (no intermediate CSV).
/// Expects two i64 columns: citing_id, cited_id.
fn build_from_parquet_core(pq_path: &Path, graph_dir: &Path) -> Result<(usize, usize), String> {
    use arrow::array::AsArray;
    use arrow::datatypes::UInt64Type;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fs::create_dir_all(graph_dir).map_err(|e| e.to_string())?;

    let file = File::open(pq_path).map_err(|e| e.to_string())?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| e.to_string())?
        .with_batch_size(1_000_000)
        .build()
        .map_err(|e| e.to_string())?;

    eprintln!("  reading parquet...");
    let mut src_raw: Vec<i64> = Vec::new();
    let mut dst_raw: Vec<i64> = Vec::new();

    for batch in reader {
        let batch = batch.map_err(|e| e.to_string())?;
        let src_col = batch.column(0).as_primitive::<UInt64Type>();
        let dst_col = batch.column(1).as_primitive::<UInt64Type>();
        // CSR pipeline uses i64 internally; u64 values fit since max work_id_int < i64::MAX
        src_raw.extend(src_col.values().iter().map(|&v| v as i64));
        dst_raw.extend(dst_col.values().iter().map(|&v| v as i64));
    }

    let num_edges = src_raw.len();
    eprintln!("  {} edges read from parquet", num_edges);

    // Reuse the same build pipeline: unique IDs → binary_search mapping → CSR
    build_from_raw_edges(src_raw, dst_raw, graph_dir)
}

#[pyfunction]
pub fn build_from_parquet(py: Python<'_>, pq_path: &str, graph_dir: &str) -> PyResult<PyObject> {
    let pq_path = pq_path.to_owned();
    let graph_dir = graph_dir.to_owned();

    let t0 = Instant::now();
    eprintln!("Building CSR from parquet: {}", pq_path);

    let (num_nodes, num_edges) = py.allow_threads(|| {
        build_from_parquet_core(Path::new(&pq_path), Path::new(&graph_dir))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    })?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "CSR done in {:.0}s: {} nodes, {} edges",
        elapsed, num_nodes, num_edges
    );

    let meta_str = std::fs::read_to_string(Path::new(&graph_dir).join("meta.json"))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("num_nodes", num_nodes)?;
    dict.set_item("num_edges", num_edges)?;
    dict.set_item("build_date", meta["build_date"].as_str().unwrap_or(""))?;
    dict.set_item("source", &pq_path)?;
    Ok(dict.into())
}

/// Pure-Rust entry point for tests.
#[cfg(test)]
pub(crate) fn build_from_csv_raw(
    csv_path: &Path,
    graph_dir: &Path,
) -> Result<(usize, usize), String> {
    build_core(csv_path, graph_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    pub(crate) fn create_test_csv(dir: &Path) -> std::path::PathBuf {
        let csv_path = dir.join("test.csv");
        let mut f = File::create(&csv_path).unwrap();
        writeln!(f, "citing,cited").unwrap();
        writeln!(f, "1,2").unwrap();
        writeln!(f, "1,3").unwrap();
        writeln!(f, "2,3").unwrap();
        writeln!(f, "3,4").unwrap();
        writeln!(f, "4,5").unwrap();
        csv_path
    }

    /// Large i64 IDs (simulating OpenAlex work_id_int ~10^10 range)
    pub(crate) fn create_large_id_csv(dir: &Path) -> std::path::PathBuf {
        let csv_path = dir.join("large.csv");
        let mut f = File::create(&csv_path).unwrap();
        writeln!(f, "src,dst").unwrap();
        writeln!(f, "2741809807,3141592653").unwrap();
        writeln!(f, "2741809807,2718281828").unwrap();
        writeln!(f, "3141592653,2718281828").unwrap();
        csv_path
    }

    #[test]
    fn test_parse_line() {
        assert_eq!(parse_line(b"123,456"), Some((123, 456)));
        assert_eq!(parse_line(b"citing,cited"), None);
        assert_eq!(parse_line(b""), None);
        assert_eq!(parse_line(b"123"), None); // no comma
        assert_eq!(parse_line(b",456"), None); // leading comma
    }

    #[test]
    fn test_parse_line_large_ids() {
        assert_eq!(
            parse_line(b"2741809807,3141592653"),
            Some((2741809807, 3141592653))
        );
        assert_eq!(
            parse_line(b"10000000000,20000000000"),
            Some((10000000000, 20000000000))
        );
    }

    #[test]
    fn test_build_csr_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = create_test_csv(dir.path());
        let graph_dir = dir.path().join("graph");

        let (num_nodes, num_edges) = build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        assert_eq!(num_nodes, 5);
        assert_eq!(num_edges, 5);

        assert!(graph_dir.join("forward/indptr.bin").exists());
        assert!(graph_dir.join("forward/indices.bin").exists());
        assert!(graph_dir.join("reverse/indptr.bin").exists());
        assert!(graph_dir.join("reverse/indices.bin").exists());
        assert!(graph_dir.join("id_map.bin").exists());
        assert!(graph_dir.join("meta.json").exists());

        // Verify meta.json content
        let meta_str = std::fs::read_to_string(graph_dir.join("meta.json")).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
        assert_eq!(meta["num_nodes"], 5);
        assert_eq!(meta["num_edges"], 5);
    }

    #[test]
    fn test_build_large_ids() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = create_large_id_csv(dir.path());
        let graph_dir = dir.path().join("graph");

        let (num_nodes, num_edges) = build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        assert_eq!(num_nodes, 3);
        assert_eq!(num_edges, 3);

        // Verify id_map contains sorted large IDs
        let id_map_str = std::fs::read_to_string(graph_dir.join("id_map.bin")).unwrap();
        let ids: Vec<i64> = id_map_str
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.parse::<i64>().unwrap())
            .collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "IDs must be sorted");
        assert!(ids.contains(&2741809807));
        assert!(ids.contains(&3141592653));
        assert!(ids.contains(&2718281828));
    }

    #[test]
    fn test_parse_byte_range_boundary() {
        // Lines at exact chunk boundaries must not be lost
        let data = b"1,2\n3,4\n5,6\n";
        // Chunk boundary at byte 4 (start of "3,4\n")
        let (s1, d1) = parse_byte_range(data, 0, 4);
        let (s2, d2) = parse_byte_range(data, 4, 8);
        let (s3, d3) = parse_byte_range(data, 8, data.len());
        assert_eq!(s1, vec![1]);
        assert_eq!(d1, vec![2]);
        assert_eq!(s2, vec![3]);
        assert_eq!(d2, vec![4]);
        assert_eq!(s3, vec![5]);
        assert_eq!(d3, vec![6]);
        // Total must match
        assert_eq!(s1.len() + s2.len() + s3.len(), 3);
    }

    #[test]
    fn test_parse_byte_range_mid_line() {
        // Chunk boundary splits a line at byte 2 (mid "1,2")
        let data = b"1,2\n3,4\n";
        let (s1, d1) = parse_byte_range(data, 0, 2);
        let (s2, d2) = parse_byte_range(data, 2, data.len());
        // Chunk 0 owns the full first line (finds \n past end)
        assert_eq!(s1, vec![1]);
        assert_eq!(d1, vec![2]);
        // Chunk 1 skips partial first line, parses second
        assert_eq!(s2, vec![3]);
        assert_eq!(d2, vec![4]);
    }

    #[test]
    fn test_build_empty_csv() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("empty.csv");
        {
            let mut f = File::create(&csv_path).unwrap();
            writeln!(f, "citing,cited").unwrap();
        }
        let graph_dir = dir.path().join("graph");
        let (num_nodes, num_edges) = build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        assert_eq!(num_nodes, 0);
        assert_eq!(num_edges, 0);
    }
}
