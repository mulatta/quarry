//! Common error type for shard processing pipelines

use crate::stream::StreamError;

/// Error from processing a single data shard (download + transform).
#[derive(Debug)]
pub enum ShardError {
    Stream(StreamError),
    Io(std::io::Error),
    Arrow(arrow::error::ArrowError),
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stream(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "IO: {e}"),
            Self::Arrow(e) => write!(f, "Arrow: {e}"),
        }
    }
}

impl std::error::Error for ShardError {}

impl ShardError {
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Stream(e) => e.is_retryable(),
            Self::Io(e) => e.kind() != std::io::ErrorKind::StorageFull,
            // Arrow schema/data errors are never retryable
            Self::Arrow(_) => false,
        }
    }
}

impl From<arrow::error::ArrowError> for ShardError {
    fn from(e: arrow::error::ArrowError) -> Self {
        Self::Arrow(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    fn http_err(status: u16) -> StreamError {
        StreamError::Http {
            status: Some(status),
            message: "test".to_string(),
        }
    }

    #[test]
    fn shard_error_stream_403_not_retryable() {
        let err = ShardError::Stream(http_err(403));
        assert!(!err.is_retryable());
    }

    #[test]
    fn shard_error_stream_500_retryable() {
        let err = ShardError::Stream(http_err(500));
        assert!(err.is_retryable());
    }

    #[test]
    fn shard_error_io_storage_full_not_retryable() {
        let err = ShardError::Io(std::io::Error::new(ErrorKind::StorageFull, "disk full"));
        assert!(!err.is_retryable());
    }

    #[test]
    fn shard_error_io_other_retryable() {
        let err = ShardError::Io(std::io::Error::new(ErrorKind::BrokenPipe, "pipe"));
        assert!(err.is_retryable());
    }
}
