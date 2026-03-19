"""OpenAlex utility functions for abstract reconstruction and ID conversion.

DuckDB handles all JSON parsing via httpfs. These functions cover what SQL cannot:
- abstract_inverted_index: JSON keys are words, values are position arrays
- work_id string → int conversion
"""


def reconstruct_abstract(inv_idx: dict[str, list[int]]) -> str:
    """Convert OpenAlex abstract_inverted_index to plaintext.

    OA stores abstracts as {word: [pos1, pos2, ...]} — inverted index format.
    Reconstruct by sorting all (position, word) pairs and joining.

    Returns empty string if input is empty or None.
    """
    if not inv_idx:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inv_idx.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(word for _, word in pairs)


def work_id_to_int(work_id: str) -> int:
    """Extract integer from OpenAlex work ID.

    'W2741809807' → 2741809807
    'https://openalex.org/W2741809807' → 2741809807
    """
    # Strip URL prefix if present
    if "/" in work_id:
        work_id = work_id.rsplit("/", 1)[-1]
    # Strip 'W' prefix
    if work_id.startswith("W"):
        work_id = work_id[1:]
    return int(work_id)
