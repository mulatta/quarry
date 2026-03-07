//! Retry with exponential backoff for shard processing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::ShardError;
use crate::progress::ProgressReporter;

/// Exponential backoff: 2^attempt seconds (2s, 4s, 8s, ...)
pub const fn backoff_duration(attempt: u32) -> Duration {
    Duration::from_secs(2u64.pow(attempt))
}

/// Retry a fallible shard operation with exponential backoff.
///
/// On retryable errors, logs the failure, updates the progress bar, sleeps,
/// and retries up to `max_retries` (from global [`HttpConfig`]).
/// Checks `cancelled` before each retry attempt.
pub fn retry_with_backoff<T>(
    shard_label: &str,
    pb: &dyn ProgressReporter,
    max_retries: u32,
    cancelled: &Arc<AtomicBool>,
    mut attempt_fn: impl FnMut() -> Result<T, ShardError>,
) -> Result<T, ShardError> {
    let mut attempt = 0u32;
    loop {
        match attempt_fn() {
            Ok(v) => return Ok(v),
            Err(ShardError::Cancelled) => return Err(ShardError::Cancelled),
            Err(e) if attempt < max_retries && e.is_retryable() => {
                if cancelled.load(Ordering::Relaxed) {
                    return Err(ShardError::Cancelled);
                }
                attempt += 1;
                pb.set_message(&format!("retry {attempt}/{max_retries}..."));
                tracing::debug!(
                    "{shard_label}: attempt {attempt}/{max_retries} failed: {e}, retrying..."
                );
                std::thread::sleep(backoff_duration(attempt));
            }
            Err(e) => {
                // Log at warn level; the caller (provider.rs) logs at error level
                tracing::warn!("{shard_label}: failed permanently: {e}");
                return Err(e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::NoopReporter;
    use crate::stream::StreamError;

    fn not_cancelled() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[test]
    fn backoff_exponential() {
        assert_eq!(backoff_duration(1), Duration::from_secs(2));
        assert_eq!(backoff_duration(2), Duration::from_secs(4));
        assert_eq!(backoff_duration(3), Duration::from_secs(8));
    }

    #[test]
    fn retry_succeeds_first_try() {
        let pb = NoopReporter;
        let cancelled = not_cancelled();
        let mut calls = 0u32;
        let result = retry_with_backoff("test", &pb, 3, &cancelled, || {
            calls += 1;
            Ok::<_, ShardError>(42)
        });
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls, 1);
    }

    #[test]
    fn retry_non_retryable_fails_immediately() {
        let pb = NoopReporter;
        let cancelled = not_cancelled();
        let mut calls = 0u32;
        let result = retry_with_backoff("test", &pb, 3, &cancelled, || {
            calls += 1;
            // HTTP 403 is non-retryable
            Err::<(), _>(ShardError::Stream(StreamError::Http {
                status: Some(403),
                message: "forbidden".to_string(),
            }))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1, "should not retry non-retryable errors");
    }

    #[test]
    fn retry_cancelled_stops_immediately() {
        let pb = NoopReporter;
        let cancelled = Arc::new(AtomicBool::new(true));
        let mut calls = 0u32;
        let result = retry_with_backoff("test", &pb, 3, &cancelled, || {
            calls += 1;
            // Retryable error, but cancelled flag is set
            Err::<(), _>(ShardError::Stream(StreamError::Http {
                status: Some(500),
                message: "server error".to_string(),
            }))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1, "should not retry when cancelled");
    }
}
