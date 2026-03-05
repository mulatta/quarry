//! Batch accumulator trait for gzip shard processing.

use arrow::array::RecordBatch;
use arrow::error::ArrowError;

/// Default batch size for flushing accumulated rows into a `RecordBatch`.
pub const DEFAULT_BATCH_SIZE: usize = 8192;

/// Accumulator trait for batch processing of parsed rows into Arrow `RecordBatch`.
pub trait Accumulator {
    type Row;

    /// Push a row into the accumulator
    fn push(&mut self, row: Self::Row);

    /// Number of rows currently buffered
    fn len(&self) -> usize;

    /// Whether the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if buffer is full and should be flushed
    fn is_full(&self) -> bool {
        self.len() >= DEFAULT_BATCH_SIZE
    }

    /// Take buffered rows as a RecordBatch, resetting internal state
    fn take_batch(&mut self) -> Result<RecordBatch, ArrowError>;
}
