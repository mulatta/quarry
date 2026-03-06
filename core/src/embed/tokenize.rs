//! Batch tokenization with manual truncation and padding.
//!
//! Mirrors the Python `tokenize_batch` reference implementation:
//! disable built-in padding/truncation, truncate to min(max_length, batch_max),
//! preserve trailing special token (e.g. [SEP]), pad to batch max.

use anyhow::Result;
use tokenizers::Tokenizer;

/// Tokenized batch in row-major flat layout (batch_size * seq_len).
pub(crate) struct BatchEncoding {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Tokenize texts with manual truncation and padding.
///
/// The tokenizer must have padding and truncation disabled beforehand.
pub(crate) fn tokenize_batch(
    tokenizer: &Tokenizer,
    texts: &[String],
    max_length: usize,
) -> Result<BatchEncoding> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let seq_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0)
        .min(max_length);

    // Detect trailing special token for proper truncation
    let sep_id = encodings.first().and_then(|enc| {
        let mask = enc.get_special_tokens_mask();
        let ids = enc.get_ids();
        if !mask.is_empty() && *mask.last().unwrap() == 1 {
            ids.last().copied()
        } else {
            None
        }
    });

    let batch_size = texts.len();
    let cap = batch_size * seq_len;
    let mut input_ids = Vec::with_capacity(cap);
    let mut attention_mask = Vec::with_capacity(cap);
    let mut token_type_ids = Vec::with_capacity(cap);

    for enc in &encodings {
        let ids = enc.get_ids();
        let mask = enc.get_attention_mask();
        let types = enc.get_type_ids();
        let take = ids.len().min(seq_len);

        // Copy with truncation (u32 → i64 for ONNX)
        for i in 0..take {
            input_ids.push(ids[i] as i64);
            attention_mask.push(mask[i] as i64);
            token_type_ids.push(types[i] as i64);
        }

        // Preserve trailing special token if truncated
        if let Some(sep) = sep_id
            && ids.len() > seq_len
        {
            let last = input_ids.len() - 1;
            if input_ids[last] != sep as i64 {
                input_ids[last] = sep as i64;
                attention_mask[last] = 1;
                token_type_ids[last] = 0;
            }
        }

        // Pad to seq_len
        let pad = seq_len - take;
        input_ids.extend(std::iter::repeat_n(0i64, pad));
        attention_mask.extend(std::iter::repeat_n(0i64, pad));
        token_type_ids.extend(std::iter::repeat_n(0i64, pad));
    }

    Ok(BatchEncoding {
        input_ids,
        attention_mask,
        token_type_ids,
        batch_size,
        seq_len,
    })
}
