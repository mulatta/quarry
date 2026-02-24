//! Retry with exponential backoff for shard processing.

use std::time::Duration;

use indicatif::ProgressBar;

use crate::error::ShardError;
use crate::stream::http_config;

/// Exponential backoff: 2^attempt seconds (2s, 4s, 8s, ...)
pub const fn backoff_duration(attempt: u32) -> Duration {
    Duration::from_secs(2u64.pow(attempt))
}

/// Retry a fallible shard operation with exponential backoff.
///
/// On retryable errors, logs the failure, updates the progress bar, sleeps,
/// and retries up to `max_retries` (from global [`HttpConfig`]).
pub fn retry_with_backoff<T>(
    shard_label: &str,
    pb: &ProgressBar,
    mut attempt_fn: impl FnMut() -> Result<T, ShardError>,
) -> Result<T, ShardError> {
    let max_retries = http_config().max_retries;
    let mut attempt = 0u32;
    loop {
        match attempt_fn() {
            Ok(v) => return Ok(v),
            Err(e) if attempt < max_retries && e.is_retryable() => {
                attempt += 1;
                pb.set_message(format!("retry {attempt}/{max_retries}..."));
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
    use crate::stream::StreamError;

    #[test]
    fn backoff_exponential() {
        assert_eq!(backoff_duration(1), Duration::from_secs(2));
        assert_eq!(backoff_duration(2), Duration::from_secs(4));
        assert_eq!(backoff_duration(3), Duration::from_secs(8));
    }

    #[test]
    fn retry_succeeds_first_try() {
        let pb = ProgressBar::hidden();
        let mut calls = 0u32;
        let result = retry_with_backoff("test", &pb, || {
            calls += 1;
            Ok::<_, ShardError>(42)
        });
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls, 1);
    }

    #[test]
    fn retry_non_retryable_fails_immediately() {
        let pb = ProgressBar::hidden();
        let mut calls = 0u32;
        let result = retry_with_backoff("test", &pb, || {
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
}
