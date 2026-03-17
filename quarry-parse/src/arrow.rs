//! Convert parsed structs → Arrow RecordBatches for zero-copy transfer to Python.

use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::xml::{Author, Chemical, Grant, MeshHeading, Paper, ParseResult};

pub fn papers_to_batch(papers: &[Paper]) -> RecordBatch {
    let pmid = Int32Array::from(papers.iter().map(|p| p.pmid).collect::<Vec<_>>());
    let doi = StringArray::from(papers.iter().map(|p| p.doi.as_deref()).collect::<Vec<_>>());
    let pmc_id = StringArray::from(papers.iter().map(|p| p.pmc_id.as_deref()).collect::<Vec<_>>());
    let title = StringArray::from(papers.iter().map(|p| p.title.as_deref()).collect::<Vec<_>>());
    let abs = StringArray::from(
        papers
            .iter()
            .map(|p| p.r#abstract.as_deref())
            .collect::<Vec<_>>(),
    );
    let pub_year = Int16Array::from(papers.iter().map(|p| p.pub_year).collect::<Vec<_>>());
    let pub_date = StringArray::from(
        papers
            .iter()
            .map(|p| p.pub_date.as_deref())
            .collect::<Vec<_>>(),
    );
    let journal_title = StringArray::from(
        papers
            .iter()
            .map(|p| p.journal_title.as_deref())
            .collect::<Vec<_>>(),
    );
    let journal_issn = StringArray::from(
        papers
            .iter()
            .map(|p| p.journal_issn.as_deref())
            .collect::<Vec<_>>(),
    );
    let journal_abbr = StringArray::from(
        papers
            .iter()
            .map(|p| p.journal_abbr.as_deref())
            .collect::<Vec<_>>(),
    );
    let volume = StringArray::from(
        papers
            .iter()
            .map(|p| p.volume.as_deref())
            .collect::<Vec<_>>(),
    );
    let issue = StringArray::from(papers.iter().map(|p| p.issue.as_deref()).collect::<Vec<_>>());
    let pages = StringArray::from(papers.iter().map(|p| p.pages.as_deref()).collect::<Vec<_>>());
    let language = StringArray::from(
        papers
            .iter()
            .map(|p| p.language.as_deref())
            .collect::<Vec<_>>(),
    );

    // pub_type: list of strings
    let mut pt_builder = ListBuilder::new(StringBuilder::new());
    for p in papers {
        if p.pub_type.is_empty() {
            pt_builder.append(false);
        } else {
            let values = pt_builder.values();
            for t in &p.pub_type {
                values.append_value(t);
            }
            pt_builder.append(true);
        }
    }
    let pub_type = pt_builder.finish();

    let country = StringArray::from(
        papers
            .iter()
            .map(|p| p.country.as_deref())
            .collect::<Vec<_>>(),
    );
    let medline_status = StringArray::from(
        papers
            .iter()
            .map(|p| p.medline_status.as_deref())
            .collect::<Vec<_>>(),
    );
    let created_date = StringArray::from(
        papers
            .iter()
            .map(|p| p.created_date.as_deref())
            .collect::<Vec<_>>(),
    );
    let revised_date = StringArray::from(
        papers
            .iter()
            .map(|p| p.revised_date.as_deref())
            .collect::<Vec<_>>(),
    );
    let indexed_date = StringArray::from(
        papers
            .iter()
            .map(|p| p.indexed_date.as_deref())
            .collect::<Vec<_>>(),
    );

    let schema = Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("doi", DataType::Utf8, true),
        Field::new("pmc_id", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, true),
        Field::new("abstract", DataType::Utf8, true),
        Field::new("pub_year", DataType::Int16, true),
        Field::new("pub_date", DataType::Utf8, true),
        Field::new("journal_title", DataType::Utf8, true),
        Field::new("journal_issn", DataType::Utf8, true),
        Field::new("journal_abbr", DataType::Utf8, true),
        Field::new("volume", DataType::Utf8, true),
        Field::new("issue", DataType::Utf8, true),
        Field::new("pages", DataType::Utf8, true),
        Field::new("language", DataType::Utf8, true),
        Field::new(
            "pub_type",
            DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
            true,
        ),
        Field::new("country", DataType::Utf8, true),
        Field::new("medline_status", DataType::Utf8, true),
        Field::new("created_date", DataType::Utf8, true),
        Field::new("revised_date", DataType::Utf8, true),
        Field::new("indexed_date", DataType::Utf8, true),
    ]);

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(pmid),
            Arc::new(doi),
            Arc::new(pmc_id),
            Arc::new(title),
            Arc::new(abs),
            Arc::new(pub_year),
            Arc::new(pub_date),
            Arc::new(journal_title),
            Arc::new(journal_issn),
            Arc::new(journal_abbr),
            Arc::new(volume),
            Arc::new(issue),
            Arc::new(pages),
            Arc::new(language),
            Arc::new(pub_type),
            Arc::new(country),
            Arc::new(medline_status),
            Arc::new(created_date),
            Arc::new(revised_date),
            Arc::new(indexed_date),
        ],
    )
    .expect("papers schema mismatch")
}

pub fn authors_to_batch(authors: &[Author]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("author_position", DataType::Int16, false),
        Field::new("last_name", DataType::Utf8, true),
        Field::new("fore_name", DataType::Utf8, true),
        Field::new("initials", DataType::Utf8, true),
        Field::new("orcid", DataType::Utf8, true),
        Field::new("affiliation", DataType::Utf8, true),
        Field::new("is_collective", DataType::Boolean, false),
    ]);

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(authors.iter().map(|a| a.pmid).collect::<Vec<_>>())),
            Arc::new(Int16Array::from(
                authors.iter().map(|a| a.author_position).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                authors.iter().map(|a| a.last_name.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                authors.iter().map(|a| a.fore_name.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                authors.iter().map(|a| a.initials.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                authors.iter().map(|a| a.orcid.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                authors.iter().map(|a| a.affiliation.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                authors.iter().map(|a| a.is_collective).collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("authors schema mismatch")
}

pub fn mesh_to_batch(headings: &[MeshHeading]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("qualifier_ui", DataType::Utf8, true),
        Field::new("qualifier_name", DataType::Utf8, true),
        Field::new("is_major_topic", DataType::Boolean, false),
    ]);

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(headings.iter().map(|m| m.pmid).collect::<Vec<_>>())),
            Arc::new(StringArray::from(
                headings.iter().map(|m| m.descriptor_ui.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                headings.iter().map(|m| m.descriptor_name.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                headings
                    .iter()
                    .map(|m| m.qualifier_ui.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                headings
                    .iter()
                    .map(|m| m.qualifier_name.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                headings.iter().map(|m| m.is_major_topic).collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("mesh schema mismatch")
}

pub fn grants_to_batch(grants: &[Grant]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("grant_id", DataType::Utf8, true),
        Field::new("acronym", DataType::Utf8, true),
        Field::new("agency", DataType::Utf8, true),
        Field::new("country", DataType::Utf8, true),
    ]);

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(grants.iter().map(|g| g.pmid).collect::<Vec<_>>())),
            Arc::new(StringArray::from(
                grants.iter().map(|g| g.grant_id.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                grants.iter().map(|g| g.acronym.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                grants.iter().map(|g| g.agency.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                grants.iter().map(|g| g.country.as_deref()).collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("grants schema mismatch")
}

pub fn chemicals_to_batch(chemicals: &[Chemical]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("registry_number", DataType::Utf8, true),
        Field::new("substance_ui", DataType::Utf8, true),
        Field::new("substance_name", DataType::Utf8, true),
    ]);

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(
                chemicals.iter().map(|c| c.pmid).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                chemicals
                    .iter()
                    .map(|c| c.registry_number.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                chemicals
                    .iter()
                    .map(|c| c.substance_ui.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                chemicals
                    .iter()
                    .map(|c| c.substance_name.as_deref())
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("chemicals schema mismatch")
}

/// Convert full ParseResult into a tuple of RecordBatches + delete_pmids.
pub fn result_to_batches(
    result: &ParseResult,
) -> (
    RecordBatch,
    RecordBatch,
    RecordBatch,
    RecordBatch,
    RecordBatch,
) {
    (
        papers_to_batch(&result.papers),
        authors_to_batch(&result.authors),
        mesh_to_batch(&result.mesh_headings),
        grants_to_batch(&result.grants),
        chemicals_to_batch(&result.chemicals),
    )
}
