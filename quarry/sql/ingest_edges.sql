INSERT INTO quarry.edges
SELECT
    replaceOne(id, 'https://openalex.org/', '') AS citing_id,
    replaceOne(ref, 'https://openalex.org/', '') AS cited_id
FROM file(
    '{path}',
    'JSONEachRow',
    $$
    id String,
    display_name String,
    abstract_inverted_index Map(String, Array(UInt32)),
    type Nullable(String),
    is_retracted Bool,
    is_paratext Bool,
    language Nullable(String),
    referenced_works Array(String),
    updated_date Date
    $$
) AS t
ARRAY JOIN referenced_works AS ref
WHERE type IN (
    'article', 'review', 'preprint', 'book-chapter',
    'letter', 'editorial', 'dataset', 'report', 'dissertation', 'book'
)
  AND is_retracted = false
  AND is_paratext = false
  AND language = 'en'
  AND mapKeys(abstract_inverted_index) != []
  AND display_name != ''
