INSERT INTO quarry.authors
SELECT
    replaceOne(id, 'https://openalex.org/', '') AS openalex_id,
    coalesce(a.author_position, '') AS author_position,
    coalesce(replaceOne(a.author.id, 'https://openalex.org/', ''), '') AS author_id,
    coalesce(a.author.display_name, '') AS author_name,
    coalesce(a.author.orcid, '') AS orcid
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
    authorships Array(Tuple(
        author_position Nullable(String),
        author Tuple(id Nullable(String), display_name Nullable(String), orcid Nullable(String))
    )),
    updated_date Date
    $$
) AS t
ARRAY JOIN authorships AS a
WHERE type IN (
    'article', 'review', 'preprint', 'book-chapter',
    'letter', 'editorial', 'dataset', 'report', 'dissertation', 'book'
)
  AND is_retracted = false
  AND is_paratext = false
  AND language = 'en'
  AND mapKeys(abstract_inverted_index) != []
  AND display_name != ''
