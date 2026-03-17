use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn parse_line(line: &[u8]) -> Option<(i32, i32)> {
    if line.is_empty() || !line[0].is_ascii_digit() {
        return None;
    }
    let mut src: i32 = 0;
    let mut i = 0;
    while i < line.len() && line[i].is_ascii_digit() {
        src = src * 10 + (line[i] - b'0') as i32;
        i += 1;
    }
    if i >= line.len() || line[i] != b',' {
        return None;
    }
    i += 1;
    let mut dst: i32 = 0;
    while i < line.len() && line[i].is_ascii_digit() {
        dst = dst * 10 + (line[i] - b'0') as i32;
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

fn parse_chunk(data: &[u8], offsets: &[usize], start: usize, end: usize) -> (Vec<i32>, Vec<i32>) {
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
fn build_csr(
    node_ids: &[u32],
    neighbor_ids: &[u32],
    num_nodes: usize,
) -> (Vec<u64>, Vec<u32>) {
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
            data.len() * std::mem::size_of::<T>(),
        )
    };
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
    w.write_all(bytes)?;
    w.flush()
}

#[pyfunction]
fn build_csr_from_csv(csv_path: &str, csr_dir: &str) -> PyResult<PyObject> {
    let t0 = Instant::now();
    let csv_path = Path::new(csv_path);
    let csr_dir = Path::new(csr_dir);
    fs::create_dir_all(csr_dir).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    // Step 1: mmap CSV + parallel parse
    eprintln!("Reading {} ...", csv_path.display());
    let file =
        File::open(csv_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let offsets = line_offsets(&mmap);
    let num_lines = offsets.len();
    eprintln!(
        "  {} lines, offsets in {:.1}s",
        num_lines,
        t0.elapsed().as_secs_f64()
    );

    let chunk_size = 2_000_000;
    let num_chunks = (num_lines + chunk_size - 1) / chunk_size;

    let results: Vec<(Vec<i32>, Vec<i32>)> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(num_lines);
            parse_chunk(&mmap, &offsets, start, end)
        })
        .collect();

    let total_edges: usize = results.iter().map(|(s, _)| s.len()).sum();
    let mut src_raw: Vec<i32> = Vec::with_capacity(total_edges);
    let mut dst_raw: Vec<i32> = Vec::with_capacity(total_edges);
    for (s, d) in results {
        src_raw.extend(s);
        dst_raw.extend(d);
    }
    drop(mmap);

    let num_edges = src_raw.len();
    eprintln!("  {} edges in {:.1}s", num_edges, t0.elapsed().as_secs_f64());

    // Step 2: Build unique sorted PMIDs via bitset (5MB instead of 6.4GB clone+sort)
    eprintln!("Building node mapping ...");
    let t1 = Instant::now();

    let max_pmid = *src_raw.par_iter().max().unwrap_or(&0)
        .max(dst_raw.par_iter().max().unwrap_or(&0)) as usize;
    let bitset_words = (max_pmid + 64) / 64;
    let mut bitset = vec![0u64; bitset_words + 1];

    for &pmid in &src_raw {
        bitset[pmid as usize / 64] |= 1u64 << (pmid as usize % 64);
    }
    for &pmid in &dst_raw {
        bitset[pmid as usize / 64] |= 1u64 << (pmid as usize % 64);
    }

    let num_nodes: usize = bitset.iter().map(|w| w.count_ones() as usize).sum();
    let mut sorted_pmids: Vec<i32> = Vec::with_capacity(num_nodes);
    for (wi, &w) in bitset.iter().enumerate() {
        if w == 0 {
            continue;
        }
        let base = (wi * 64) as i32;
        for bit in 0..64 {
            if w & (1u64 << bit) != 0 {
                sorted_pmids.push(base + bit as i32);
            }
        }
    }

    // Rank lookup: word_rank[i] = popcount(bitset[0..i])
    let mut word_rank: Vec<u32> = Vec::with_capacity(bitset.len() + 1);
    word_rank.push(0u32);
    let mut cum = 0u32;
    for &w in &bitset {
        cum += w.count_ones();
        word_rank.push(cum);
    }

    eprintln!("  {} nodes in {:.1}s", num_nodes, t1.elapsed().as_secs_f64());

    // Step 3: Map PMIDs → node indices in-place via O(1) bitset rank
    eprintln!("Mapping edges to node indices ...");
    let t2 = Instant::now();

    src_raw.par_iter_mut().for_each(|pmid| {
        let p = *pmid as usize;
        let wi = p / 64;
        let bi = p % 64;
        let mask = (1u64 << bi) - 1;
        *pmid = (word_rank[wi] + (bitset[wi] & mask).count_ones()) as i32;
    });
    let src_idx: Vec<u32> = unsafe { std::mem::transmute::<Vec<i32>, Vec<u32>>(src_raw) };

    dst_raw.par_iter_mut().for_each(|pmid| {
        let p = *pmid as usize;
        let wi = p / 64;
        let bi = p % 64;
        let mask = (1u64 << bi) - 1;
        *pmid = (word_rank[wi] + (bitset[wi] & mask).count_ones()) as i32;
    });
    let dst_idx: Vec<u32> = unsafe { std::mem::transmute::<Vec<i32>, Vec<u32>>(dst_raw) };
    drop(bitset);
    drop(word_rank);

    eprintln!("  Mapped in {:.1}s", t2.elapsed().as_secs_f64());

    // Step 4: Build forward CSR
    eprintln!("Building forward CSR ...");
    let t3 = Instant::now();
    let fwd_dir = csr_dir.join("forward");
    fs::create_dir_all(&fwd_dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let (fwd_indptr, fwd_indices) = build_csr(&src_idx, &dst_idx, num_nodes);
    write_bin(&fwd_dir.join("indptr.bin"), &fwd_indptr)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    write_bin(&fwd_dir.join("indices.bin"), &fwd_indices)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    drop(fwd_indptr);
    drop(fwd_indices);
    eprintln!("  Forward CSR in {:.1}s", t3.elapsed().as_secs_f64());

    // Step 5: Build reverse CSR
    eprintln!("Building reverse CSR ...");
    let t4 = Instant::now();
    let rev_dir = csr_dir.join("reverse");
    fs::create_dir_all(&rev_dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let (rev_indptr, rev_indices) = build_csr(&dst_idx, &src_idx, num_nodes);
    write_bin(&rev_dir.join("indptr.bin"), &rev_indptr)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    write_bin(&rev_dir.join("indices.bin"), &rev_indices)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    drop(rev_indptr);
    drop(rev_indices);
    drop(src_idx);
    drop(dst_idx);
    eprintln!("  Reverse CSR in {:.1}s", t4.elapsed().as_secs_f64());

    // Step 6: Write id_map
    eprintln!("Writing id_map ...");
    let id_map_path = csr_dir.join("id_map.bin");
    {
        let mut w = BufWriter::new(
            File::create(&id_map_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?,
        );
        for &pmid in &sorted_pmids {
            writeln!(w, "{}", pmid)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        w.flush()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    let build_date = chrono_free_iso8601();
    let meta = serde_json::json!({
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "build_date": build_date,
        "source": csv_path.to_string_lossy(),
    });
    fs::write(
        csr_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "CSR done in {:.0}s: {} nodes, {} edges",
        elapsed, num_nodes, num_edges
    );

    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_nodes", num_nodes)?;
        dict.set_item("num_edges", num_edges)?;
        dict.set_item("build_date", &build_date)?;
        dict.set_item("source", csv_path.to_string_lossy().as_ref())?;
        Ok(dict.into())
    })
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

#[pymodule]
fn quarry_csr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_csr_from_csv, m)?)?;
    Ok(())
}
