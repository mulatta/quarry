//! CSV → CSR build: parallel parse + HashMap ID mapping + forward/reverse CSR.
//!
//! Supports i64 node IDs (OpenAlex work_id_int, range ~10^10).

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;
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

fn line_offsets(data: &[u8]) -> Vec<usize> {
    let mut offsets = vec![0usize];
    for (i, &b) in data.iter().enumerate() {
        if b == b'\n' && i + 1 < data.len() {
            offsets.push(i + 1);
        }
    }
    offsets
}

fn parse_chunk(data: &[u8], offsets: &[usize], start: usize, end: usize) -> (Vec<i64>, Vec<i64>) {
    let cap = end - start;
    let mut srcs = Vec::with_capacity(cap);
    let mut dsts = Vec::with_capacity(cap);
    for &off in &offsets[start..end] {
        let line_end = data[off..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(data.len(), |p| off + p);
        if let Some((s, d)) = parse_line(&data[off..line_end]) {
            srcs.push(s);
            dsts.push(d);
        }
    }
    (srcs, dsts)
}

/// Build CSR via parallel sort.
fn build_csr(node_ids: &[u32], neighbor_ids: &[u32], num_nodes: usize) -> (Vec<u64>, Vec<u32>) {
    let mut pairs: Vec<(u32, u32)> = node_ids
        .par_iter()
        .zip(neighbor_ids.par_iter())
        .map(|(&n, &nb)| (n, nb))
        .collect();
    pairs.par_sort_unstable_by_key(|&(n, _)| n);

    let mut counts = vec![0u64; num_nodes];
    for &(n, _) in &pairs {
        counts[n as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0u64);
    let mut cumsum = 0u64;
    for c in &counts {
        cumsum += c;
        indptr.push(cumsum);
    }

    let indices: Vec<u32> = pairs.into_iter().map(|(_, nb)| nb).collect();
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
fn build_core(csv_path: &Path, graph_dir: &Path) -> Result<(usize, usize), String> {
    fs::create_dir_all(graph_dir).map_err(|e| e.to_string())?;

    let file = File::open(csv_path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;

    let offsets = line_offsets(&mmap);
    let num_lines = offsets.len();
    eprintln!(
        "  {} lines, offsets ready",
        num_lines,
    );

    let chunk_size = 2_000_000;
    let num_chunks = num_lines.div_ceil(chunk_size);

    let results: Vec<(Vec<i64>, Vec<i64>)> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(num_lines);
            parse_chunk(&mmap, &offsets, start, end)
        })
        .collect();

    let total_edges: usize = results.iter().map(|(s, _)| s.len()).sum();
    let mut src_raw: Vec<i64> = Vec::with_capacity(total_edges);
    let mut dst_raw: Vec<i64> = Vec::with_capacity(total_edges);
    for (s, d) in results {
        src_raw.extend(s);
        dst_raw.extend(d);
    }
    drop(mmap);
    let num_edges = src_raw.len();

    // Pass 1: collect all unique IDs → sorted Vec<i64> → HashMap<i64, u32>
    let mut id_set: HashMap<i64, ()> = HashMap::with_capacity(num_edges / 3);
    for &id in &src_raw {
        id_set.entry(id).or_insert(());
    }
    for &id in &dst_raw {
        id_set.entry(id).or_insert(());
    }
    let mut sorted_ids: Vec<i64> = id_set.into_keys().collect();
    sorted_ids.par_sort_unstable();
    let num_nodes = sorted_ids.len();

    let id_to_idx: HashMap<i64, u32> = sorted_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();

    // Pass 2: map edges (i64, i64) → (u32, u32)
    let src_idx: Vec<u32> = src_raw.par_iter().map(|&p| id_to_idx[&p]).collect();
    let dst_idx: Vec<u32> = dst_raw.par_iter().map(|&p| id_to_idx[&p]).collect();
    drop(src_raw);
    drop(dst_raw);
    drop(id_to_idx);

    // Forward CSR
    let fwd_dir = graph_dir.join("forward");
    fs::create_dir_all(&fwd_dir).map_err(|e| e.to_string())?;
    let (fwd_indptr, fwd_indices) = build_csr(&src_idx, &dst_idx, num_nodes);
    write_bin(&fwd_dir.join("indptr.bin"), &fwd_indptr).map_err(|e| e.to_string())?;
    write_bin(&fwd_dir.join("indices.bin"), &fwd_indices).map_err(|e| e.to_string())?;

    // Reverse CSR
    let rev_dir = graph_dir.join("reverse");
    fs::create_dir_all(&rev_dir).map_err(|e| e.to_string())?;
    let (rev_indptr, rev_indices) = build_csr(&dst_idx, &src_idx, num_nodes);
    write_bin(&rev_dir.join("indptr.bin"), &rev_indptr).map_err(|e| e.to_string())?;
    write_bin(&rev_dir.join("indices.bin"), &rev_indices).map_err(|e| e.to_string())?;

    // id_map.bin: one i64 per line (text format, sorted, binary-search ready)
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
        "source": csv_path.to_string_lossy(),
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
