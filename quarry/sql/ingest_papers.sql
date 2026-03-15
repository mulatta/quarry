INSERT INTO quarry.papers
SELECT
    replaceOne(id, 'https://openalex.org/', '') AS openalex_id,
    coalesce(doi, '') AS doi,
    coalesce(ids.pmid, '') AS pmid,
    display_name AS title,
    arrayStringConcat(
        arrayMap(
            x -> x.2,
            arraySort(
                x -> x.1,
                arrayFlatten(
                    arrayMap(
                        kv -> arrayMap(pos -> tuple(pos, kv.1), kv.2),
                        CAST(abstract_inverted_index, 'Array(Tuple(String, Array(UInt32)))')
                    )
                )
            )
        ),
        ' '
    ) AS abstract,
    toUInt16(coalesce(publication_year, 0)) AS pub_year,
    coalesce(type, '') AS type,
    coalesce(language, '') AS language,
    toUInt32(coalesce(cited_by_count, 0)) AS cited_by,
    toFloat32(coalesce(fwci, 0)) AS fwci,
    coalesce(primary_location.source.display_name, '') AS source_name,
    coalesce(primary_location.source.type, '') AS source_type,
    coalesce(primary_topic.display_name, '') AS topic_name,
    coalesce(primary_topic.field.display_name, '') AS field_name,
    coalesce(primary_topic.domain.display_name, '') AS domain_name,
    updated_date
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
    publication_year Nullable(UInt16),
    cited_by_count Nullable(UInt32),
    fwci Nullable(Float32),
    doi Nullable(String),
    ids Tuple(pmid Nullable(String)),
    primary_location Tuple(source Tuple(display_name Nullable(String), type Nullable(String))),
    primary_topic Tuple(display_name Nullable(String), field Tuple(display_name Nullable(String)), domain Tuple(display_name Nullable(String))),
    referenced_works Array(String),
    updated_date Date
    $$
)
WHERE type IN (
    'article', 'review', 'preprint', 'book-chapter',
    'letter', 'editorial', 'dataset', 'report', 'dissertation', 'book'
)
  AND is_retracted = false
  AND is_paratext = false
  AND language = 'en'
  AND mapKeys(abstract_inverted_index) != []
  AND display_name != ''
