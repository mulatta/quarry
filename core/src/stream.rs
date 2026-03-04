//! HTTP streaming with gzip decompression and read timeout.
//!
//! Uses async reqwest internally with tokio::time::timeout for stall detection,
//! but presents a sync interface for compatibility with rayon workers.

use std::io::{self, BufReader, Read};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Context;
use std::time::Duration;

use flate2::read::GzDecoder;
use futures_util::StreamExt;
use tokio::io::{AsyncRead, ReadBuf};

/// Connect timeout
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Response timeout (connect + headers, not including body download)
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

// ============================================================
// HttpPool — shared HTTP client + tokio runtime
// ============================================================

/// Shared HTTP resources for parallel shard downloads.
///
/// Replaces the former global `OnceLock` statics, making the library
/// safe for multiple concurrent instances (e.g. via PyO3).
pub struct HttpPool {
    client: reqwest::Client,
    runtime: tokio::runtime::Runtime,
    /// Read timeout for stall detection (no data within this duration = stall)
    pub read_timeout: Duration,
    /// Maximum retry attempts for transient failures
    pub max_retries: u32,
}

impl HttpPool {
    /// Create a pool tuned for `concurrency` parallel streams.
    pub fn new(
        concurrency: usize,
        read_timeout: Duration,
        max_retries: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(concurrency)
            .build()?;

        let threads = (concurrency / 4).max(2);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .enable_all()
            .build()?;

        Ok(Self {
            client,
            runtime,
            read_timeout,
            max_retries,
        })
    }

    /// Tokio runtime handle (for block_on in sync contexts).
    pub fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }

    /// Reqwest client reference.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

/// Error types for stream operations
#[derive(Debug)]
pub enum StreamError {
    Http {
        status: Option<u16>,
        message: String,
    },
    Io(std::io::Error),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http {
                status: Some(s),
                message,
            } => write!(f, "HTTP {s}: {message}"),
            Self::Http {
                status: None,
                message,
            } => write!(f, "HTTP error: {message}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for StreamError {}

impl StreamError {
    /// Create HTTP error from reqwest error.
    /// Strips URL to avoid leaking signed URLs / tokens in logs.
    pub fn from_reqwest(e: &reqwest::Error) -> Self {
        let raw = e.to_string();
        let message = if let Some(start) = raw.find(" for url (") {
            let before = &raw[..start];
            let after = raw[start..]
                .find("): ")
                .map(|i| &raw[start + i + 3..])
                .unwrap_or("");
            if after.is_empty() {
                before.to_string()
            } else {
                format!("{before}: {after}")
            }
        } else {
            raw
        };
        Self::Http {
            status: e.status().map(|s| s.as_u16()),
            message,
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http { status, .. } => {
                // 400 = AWS STS token expired, 403 = URL signature expired, 410 = URL expired
                !matches!(status, Some(400 | 403 | 410))
            }
            Self::Io(e) => e.kind() != std::io::ErrorKind::StorageFull,
        }
    }
}

impl From<std::io::Error> for StreamError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Buffer size for gzip stream reader (256KB)
const GZIP_BUF_SIZE: usize = 256 * 1024;

/// Buffered reader over a gzipped HTTP response body with byte counting
pub type GzipReader = BufReader<GzDecoder<CountingReader<TimeoutReader>>>;

/// Shared byte counter for progress tracking
pub type ByteCounter = Arc<AtomicU64>;

/// HTTP GET -> gunzip -> buffered reader with byte counter
///
/// Returns (reader, byte_counter, total_bytes)
pub fn open_gzip_reader(
    pool: &HttpPool,
    url: &str,
) -> Result<(GzipReader, ByteCounter, Option<u64>), StreamError> {
    let url = url.to_string();
    let handle = pool.handle();
    let read_timeout = pool.read_timeout;

    let (reader, total_bytes) = handle.block_on(async {
        let response = tokio::time::timeout(RESPONSE_TIMEOUT, async {
            pool.client()
                .get(&url)
                .send()
                .await
                .and_then(|r| r.error_for_status())
                .map_err(|e| StreamError::from_reqwest(&e))
        })
        .await
        .map_err(|_| StreamError::Http {
            status: None,
            message: format!("response timeout ({RESPONSE_TIMEOUT:?})"),
        })??;

        let total_bytes = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok());

        let stream = response.bytes_stream();
        let async_reader = tokio_util::io::StreamReader::new(
            stream.map(|result| result.map_err(io::Error::other)),
        );

        Ok::<_, StreamError>((
            TimeoutReader::new(Box::pin(async_reader), handle.clone(), read_timeout),
            total_bytes,
        ))
    })?;

    let counter = Arc::new(AtomicU64::new(0));
    let counting_reader = CountingReader {
        inner: reader,
        count: counter.clone(),
    };
    let gz = GzDecoder::new(counting_reader);
    let buf = BufReader::with_capacity(GZIP_BUF_SIZE, gz);

    Ok((buf, counter, total_bytes))
}

/// Reader wrapper that tracks bytes read
pub struct CountingReader<R> {
    inner: R,
    count: Arc<AtomicU64>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

/// Async-to-sync bridge with read timeout.
///
/// Each read operation has a timeout -- if no data arrives within
/// the configured read_timeout, returns TimedOut error (triggers retry).
/// Captures the runtime handle and timeout at construction time.
pub struct TimeoutReader {
    inner: Pin<Box<dyn AsyncRead + Send + Sync>>,
    handle: tokio::runtime::Handle,
    read_timeout: Duration,
}

impl TimeoutReader {
    fn new(
        inner: Pin<Box<dyn AsyncRead + Send + Sync>>,
        handle: tokio::runtime::Handle,
        read_timeout: Duration,
    ) -> Self {
        Self {
            inner,
            handle,
            read_timeout,
        }
    }
}

impl Read for TimeoutReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read_timeout = self.read_timeout;
        self.handle.block_on(async {
            let read_future = async {
                let mut read_buf = ReadBuf::new(buf);
                std::future::poll_fn(|cx: &mut Context<'_>| {
                    Pin::as_mut(&mut self.inner).poll_read(cx, &mut read_buf)
                })
                .await?;
                Ok::<_, io::Error>(read_buf.filled().len())
            };

            match tokio::time::timeout(read_timeout, read_future).await {
                Ok(result) => result,
                Err(_) => Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("read timeout ({}s with no data)", read_timeout.as_secs()),
                )),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn http_err(status: u16) -> StreamError {
        StreamError::Http {
            status: Some(status),
            message: "test".to_string(),
        }
    }

    #[test]
    fn http_403_not_retryable() {
        assert!(!http_err(403).is_retryable());
    }

    #[test]
    fn http_410_not_retryable() {
        assert!(!http_err(410).is_retryable());
    }

    #[test]
    fn http_400_not_retryable() {
        assert!(!http_err(400).is_retryable());
    }

    #[test]
    fn http_500_retryable() {
        assert!(http_err(500).is_retryable());
    }

    #[test]
    fn http_429_retryable() {
        assert!(http_err(429).is_retryable());
    }

    #[test]
    fn io_timeout_retryable() {
        let err = StreamError::Io(io::Error::new(io::ErrorKind::TimedOut, "timeout"));
        assert!(err.is_retryable());
    }

    #[test]
    fn io_storage_full_not_retryable() {
        let err = StreamError::Io(io::Error::new(io::ErrorKind::StorageFull, "disk full"));
        assert!(!err.is_retryable());
    }

    #[test]
    fn display_http_with_status() {
        let err = http_err(404);
        assert_eq!(format!("{err}"), "HTTP 404: test");
    }

    #[test]
    fn display_http_without_status() {
        let err = StreamError::Http {
            status: None,
            message: "timeout".to_string(),
        };
        assert_eq!(format!("{err}"), "HTTP error: timeout");
    }
}
