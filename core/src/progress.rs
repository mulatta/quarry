//! Progress reporting for TTY and non-TTY environments.
//!
//! TTY mode: indicatif progress bars per worker (clear on completion).
//! Non-TTY mode: log-based output (no progress bars).
//!
//! All tracing output is routed through [`IndicatifMakeWriter`] so that log
//! lines are inserted above progress bars via `MultiProgress::println()`,
//! preventing garbled output when many workers run concurrently.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::io::{IsTerminal, Write};
use std::sync::Arc;
use tracing_subscriber::fmt::MakeWriter;

// ============================================================
// Abstract progress reporting
// ============================================================

/// Abstraction over progress reporting, decoupled from indicatif.
///
/// All pipeline/provider/retry code uses this trait instead of `ProgressBar`
/// directly, enabling library (PyO3) usage without a terminal.
pub trait ProgressReporter: Send + Sync {
    fn set_message(&self, msg: &str);
    fn set_length(&self, len: u64);
    fn set_position(&self, pos: u64);
    fn inc(&self, delta: u64);
    fn finish_and_clear(&self);
    fn finish_with_message(&self, msg: &str);

    /// Switch from indeterminate (pending) to determinate (bar) mode.
    ///
    /// Default implementation just sets the length. Indicatif overrides
    /// this to also switch the bar style from pending to bytes-bar.
    fn upgrade_to_determinate(&self, total: u64) {
        self.set_length(total);
    }
}

/// Indicatif-backed progress reporter.
pub struct IndicatifReporter(pub ProgressBar);

impl ProgressReporter for IndicatifReporter {
    fn set_message(&self, msg: &str) {
        self.0.set_message(msg.to_owned());
    }
    fn set_length(&self, len: u64) {
        self.0.set_length(len);
    }
    fn set_position(&self, pos: u64) {
        self.0.set_position(pos);
    }
    fn inc(&self, delta: u64) {
        self.0.inc(delta);
    }
    fn finish_and_clear(&self) {
        self.0.finish_and_clear();
    }
    fn finish_with_message(&self, msg: &str) {
        self.0.finish_with_message(msg.to_owned());
    }
    fn upgrade_to_determinate(&self, total: u64) {
        self.0.set_length(total);
        self.0.set_style(bar_style());
    }
}

/// No-op reporter for tests and non-TTY library use.
pub struct NoopReporter;

impl ProgressReporter for NoopReporter {
    fn set_message(&self, _msg: &str) {}
    fn set_length(&self, _len: u64) {}
    fn set_position(&self, _pos: u64) {}
    fn inc(&self, _delta: u64) {}
    fn finish_and_clear(&self) {}
    fn finish_with_message(&self, _msg: &str) {}
}

/// Global progress bar style with configurable unit label.
fn global_style(unit: &str) -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(&format!(
            "{{prefix:.bold}} {{bar:30.cyan/black.dim}} {{pos}}/{{len}} {unit} [{{elapsed_precise}}] {{wide_msg}}"
        ))
        .expect("invalid template")
        .progress_chars("=>-")
}

/// Per-worker progress bar style (bytes progress)
fn bar_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{prefix:<20.dim} {bar:30.green/black.dim} {binary_bytes:>7}/{binary_total_bytes:7} {eta:>4} {wide_msg:.dim}")
        .expect("invalid template")
        .progress_chars("--")
}

/// Pending style -- shown before total bytes are known
fn pending_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{prefix:<20.dim} {wide_msg:.dim}")
        .expect("invalid template")
}

/// Embed progress bar style (row count + throughput).
pub fn embed_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{prefix:.bold} {bar:30.cyan/black.dim} {pos}/{len} rows [{elapsed_precise}] {per_sec} ETA {eta}",
        )
        .expect("invalid template")
        .progress_chars("=>-")
}

/// Per-directory progress bar style (file count, used by push/pull)
fn dir_style(unit: &str) -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(&format!(
            "  {{prefix:<12.bold}} {{bar:25.cyan/black.dim}} {{pos}}/{{len}} {unit} [{{elapsed_precise}}]"
        ))
        .expect("invalid template")
        .progress_chars("=>-")
}

// ============================================================
// Tracing ↔ indicatif integration
// ============================================================

/// Writer that routes tracing output correctly for both TTY and non-TTY.
///
/// TTY: emits lines via `MultiProgress::println()` so logs appear above
/// progress bars without corruption.
/// Non-TTY: writes directly to stderr (indicatif's hidden draw target
/// may silently drop `println()` output).
pub struct IndicatifWriter {
    multi: Option<MultiProgress>,
    buf: Vec<u8>,
}

impl IndicatifWriter {
    fn emit_line(&self, line: &str) -> std::io::Result<()> {
        if let Some(multi) = &self.multi {
            multi.println(line).map_err(std::io::Error::other)
        } else {
            use std::io::Write;
            let mut stderr = std::io::stderr().lock();
            writeln!(stderr, "{line}")
        }
    }
}

impl Write for IndicatifWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.buf[..pos]).to_string();
            self.emit_line(&line)?;
            self.buf.drain(..=pos);
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.buf.is_empty() {
            let line = String::from_utf8_lossy(&self.buf).to_string();
            self.emit_line(&line)?;
            self.buf.clear();
        }
        Ok(())
    }
}

impl Drop for IndicatifWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Factory that produces [`IndicatifWriter`] instances for `tracing-subscriber`.
///
/// TTY: routes through `MultiProgress::println()` for progress bar compat.
/// Non-TTY: writes directly to stderr for reliable capture by job runners.
#[derive(Clone)]
pub struct IndicatifMakeWriter {
    multi: MultiProgress,
    is_tty: bool,
}

impl IndicatifMakeWriter {
    pub fn new(multi: MultiProgress) -> Self {
        Self {
            multi,
            is_tty: std::io::stderr().is_terminal(),
        }
    }
}

impl<'a> MakeWriter<'a> for IndicatifMakeWriter {
    type Writer = IndicatifWriter;

    fn make_writer(&'a self) -> Self::Writer {
        IndicatifWriter {
            multi: if self.is_tty {
                Some(self.multi.clone())
            } else {
                None
            },
            buf: Vec::new(),
        }
    }
}

// ============================================================
// Progress context
// ============================================================

/// Central progress context managing multi-progress bars.
pub struct ProgressContext {
    multi: MultiProgress,
    is_tty: bool,
    /// Top-level progress bar tracking overall shard completion.
    global_bar: ProgressBar,
}

impl ProgressContext {
    /// Access the underlying `MultiProgress` (for tracing writer integration).
    pub fn multi(&self) -> &MultiProgress {
        &self.multi
    }

    pub fn new() -> Self {
        let is_tty = std::io::stderr().is_terminal();
        let multi = MultiProgress::new();
        let global_bar = if is_tty {
            multi.add(ProgressBar::new(0))
        } else {
            ProgressBar::hidden()
        };
        Self {
            multi,
            is_tty,
            global_bar,
        }
    }

    /// Create from an existing `MultiProgress` (shared with tracing writer).
    pub fn with_multi(multi: MultiProgress) -> Self {
        let is_tty = std::io::stderr().is_terminal();
        let global_bar = if is_tty {
            multi.add(ProgressBar::new(0))
        } else {
            ProgressBar::hidden()
        };
        Self {
            multi,
            is_tty,
            global_bar,
        }
    }

    /// Initialize the global progress bar.
    /// Must be called before `shard_bar()` so it stays at the top.
    pub fn init_global(&self, total: u64, label: &str, unit: &str) {
        self.global_bar.set_length(total);
        self.global_bar.set_style(global_style(unit));
        self.global_bar.set_prefix(label.to_string());
    }

    /// Increment the global progress bar by one completed shard.
    pub fn inc_global(&self) {
        self.global_bar.inc(1);
    }

    /// Set a message on the global bar (e.g. summary stats).
    pub fn set_global_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.global_bar.set_message(msg);
    }

    /// Finish the global progress bar.
    pub fn finish_global(&self) {
        self.global_bar.finish();
    }

    /// Create per-worker progress bar (inserted below the global bar).
    pub fn shard_bar(&self, name: &str) -> Box<dyn ProgressReporter> {
        if !self.is_tty {
            return Box::new(NoopReporter);
        }

        let pb = self.multi.add(ProgressBar::new(0));
        pb.set_style(pending_style());
        let display = if name.len() > 20 { &name[..20] } else { name };
        pb.set_prefix(display.to_string());
        Box::new(IndicatifReporter(pb))
    }

    /// Create a per-directory progress bar (file count, `=>-` style).
    pub fn dir_bar(&self, name: &str, total: u64, unit: &str) -> Box<dyn ProgressReporter> {
        if !self.is_tty {
            return Box::new(NoopReporter);
        }

        let pb = self.multi.add(ProgressBar::new(total));
        pb.set_style(dir_style(unit));
        pb.set_prefix(name.to_string());
        Box::new(IndicatifReporter(pb))
    }

    /// Create a per-file transfer progress bar (bytes, `--` style).
    ///
    /// Returns `Arc` for use in async contexts (across `.await` points).
    /// Starts in pending mode; caller calls `upgrade_to_determinate(file_size)`.
    pub fn file_bar(&self, name: &str) -> Arc<dyn ProgressReporter> {
        if !self.is_tty {
            return Arc::new(NoopReporter);
        }

        let pb = self.multi.add(ProgressBar::new(0));
        pb.set_style(pending_style());
        let display = if name.len() > 20 { &name[..20] } else { name };
        pb.set_prefix(display.to_string());
        Arc::new(IndicatifReporter(pb))
    }
}

impl Default for ProgressContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper for `ProgressContext`.
pub type SharedProgress = Arc<ProgressContext>;

/// Format number with thousand separators.
pub fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_num_zero() {
        assert_eq!(fmt_num(0), "0");
    }

    #[test]
    fn fmt_num_small() {
        assert_eq!(fmt_num(1), "1");
        assert_eq!(fmt_num(12), "12");
        assert_eq!(fmt_num(123), "123");
    }

    #[test]
    fn fmt_num_thousands() {
        assert_eq!(fmt_num(1_000), "1,000");
        assert_eq!(fmt_num(1_234), "1,234");
        assert_eq!(fmt_num(12_345), "12,345");
        assert_eq!(fmt_num(123_456), "123,456");
    }

    #[test]
    fn fmt_num_millions() {
        assert_eq!(fmt_num(1_000_000), "1,000,000");
        assert_eq!(fmt_num(1_234_567), "1,234,567");
    }
}
