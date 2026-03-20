//! Convert parsed structs → Arrow RecordBatches for zero-copy transfer to Python.

use arrow::array::*;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::schema;
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

    RecordBatch::try_new(
        Arc::new(schema::papers_schema()),
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
    RecordBatch::try_new(
        Arc::new(schema::authors_schema()),
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
    RecordBatch::try_new(
        Arc::new(schema::mesh_headings_schema()),
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
    RecordBatch::try_new(
        Arc::new(schema::grants_schema()),
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
    RecordBatch::try_new(
        Arc::new(schema::chemicals_schema()),
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

/// Convert full ParseResult into a tuple of RecordBatches.
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
