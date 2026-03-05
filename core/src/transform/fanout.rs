//! Fan-out accumulators: WorksKeys + 10 child tables.
//!
//! Each accumulator collects rows from the parent `WorkRow` and produces
//! Arrow `RecordBatch`es for a single child table.

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;

use crate::accumulator::DEFAULT_BATCH_SIZE;
use crate::id::normalize_doi;
use crate::schema;

use super::model::*;
use super::works::{build_list_list_string_array, build_list_string_array, opt_vec_to_opt};
use super::{extract_pmcid, extract_pmid, strip_oa_prefix};

/// Convert a column of `Arc<str>` to a StringArray, draining the Vec.
fn arc_str_col_to_array(col: &mut Vec<Option<Arc<str>>>) -> ArrayRef {
    let data = std::mem::take(col);
    let mut builder = StringBuilder::with_capacity(data.len(), data.len() * 32);
    for item in &data {
        match item {
            Some(s) => builder.append_value(s.as_ref()),
            None => builder.append_null(),
        }
    }
    Arc::new(builder.finish())
}

/// Declare a fan-out accumulator struct with `Default`, `new()`, `len()`, `is_full()`.
///
/// `push_from_work()` and `take_batch()` must be implemented manually in a
/// separate `impl` block because their logic differs per table.
macro_rules! fanout_accumulator {
    (
        $name:ident, $schema_fn:path,
        { $first:ident: $first_ty:ty $(, $field:ident: $fty:ty)* $(,)? }
    ) => {
        pub struct $name {
            schema: Arc<Schema>,
            $first: Vec<$first_ty>,
            $( $field: Vec<$fty>, )*
        }

        impl Default for $name {
            fn default() -> Self { Self::new() }
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    schema: $schema_fn().clone(),
                    $first: Vec::with_capacity(DEFAULT_BATCH_SIZE),
                    $( $field: Vec::with_capacity(DEFAULT_BATCH_SIZE), )*
                }
            }

            pub fn len(&self) -> usize {
                self.$first.len()
            }

            pub fn is_empty(&self) -> bool {
                self.$first.is_empty()
            }

            pub fn is_full(&self) -> bool {
                self.$first.len() >= DEFAULT_BATCH_SIZE
            }
        }
    };
}

// ============================================================
// WorksKeys — lookup keys extracted from the main works table
// ============================================================

fanout_accumulator!(WorksKeysAccumulator, schema::works_keys, {
    work_id: Option<String>,
    doi_norm: Option<String>,
    pmid: Option<String>,
    pmcid: Option<String>,
});

impl WorksKeysAccumulator {
    pub fn push_from_work(&mut self, _work_id: &Arc<str>, row: &WorkRow) {
        self.work_id
            .push(row.id.as_deref().map(|s| strip_oa_prefix(s).to_string()));
        self.doi_norm.push(
            row.doi
                .as_deref()
                .and_then(normalize_doi)
                .map(|b| b.into_string()),
        );
        self.pmid.push(
            row.ids
                .as_ref()
                .and_then(|ids| ids.pmid.as_deref())
                .and_then(extract_pmid)
                .map(|s| s.to_string()),
        );
        self.pmcid.push(
            row.ids
                .as_ref()
                .and_then(|ids| ids.pmcid.as_deref())
                .and_then(extract_pmcid)
                .map(|s| s.to_string()),
        );
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(std::mem::take(&mut self.work_id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.doi_norm))),
            Arc::new(StringArray::from(std::mem::take(&mut self.pmid))),
            Arc::new(StringArray::from(std::mem::take(&mut self.pmcid))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

// ============================================================
// Fan-out child tables
// ============================================================

fanout_accumulator!(CitationsAccumulator, schema::citations, {
    work_id: Option<Arc<str>>,
    referenced_work_id: Option<String>,
});

impl CitationsAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for rw in &row.referenced_works {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.referenced_work_id
                .push(Some(strip_oa_prefix(rw).to_string()));
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.referenced_work_id,
            ))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(CountsByYearAccumulator, schema::work_counts_by_year, {
    work_id: Option<Arc<str>>,
    year: Option<i32>,
    cited_by_count: Option<i32>,
});

impl CountsByYearAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for cby in &row.counts_by_year {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.year.push(cby.year);
            self.cited_by_count.push(cby.cited_by_count);
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(Int32Array::from(std::mem::take(&mut self.year))),
            Arc::new(Int32Array::from(std::mem::take(&mut self.cited_by_count))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(KeywordsAccumulator, schema::work_keywords, {
    work_id: Option<Arc<str>>,
    keyword_id: Option<String>,
    display_name: Option<String>,
    score: Option<f64>,
});

impl KeywordsAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for kw in &row.keywords {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.keyword_id.push(kw.id.clone());
            self.display_name.push(kw.display_name.clone());
            self.score.push(kw.score);
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(&mut self.keyword_id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.display_name))),
            Arc::new(Float64Array::from(std::mem::take(&mut self.score))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(SdgsAccumulator, schema::work_sdgs, {
    work_id: Option<Arc<str>>,
    sdg_id: Option<String>,
    display_name: Option<String>,
    score: Option<f64>,
});

impl SdgsAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for sdg in &row.sustainable_development_goals {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.sdg_id.push(sdg.id.clone());
            self.display_name.push(sdg.display_name.clone());
            self.score.push(sdg.score);
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(&mut self.sdg_id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.display_name))),
            Arc::new(Float64Array::from(std::mem::take(&mut self.score))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(FundersAccumulator, schema::work_funders, {
    work_id: Option<Arc<str>>,
    funder_id: Option<String>,
    display_name: Option<String>,
    ror: Option<String>,
});

impl FundersAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for f in &row.funders {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.funder_id.push(f.id.clone());
            self.display_name.push(f.display_name.clone());
            self.ror.push(f.ror.clone());
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(&mut self.funder_id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.display_name))),
            Arc::new(StringArray::from(std::mem::take(&mut self.ror))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(MeshAccumulator, schema::work_mesh, {
    work_id: Option<Arc<str>>,
    descriptor_ui: Option<String>,
    descriptor_name: Option<String>,
    qualifier_ui: Option<String>,
    qualifier_name: Option<String>,
    is_major_topic: Option<bool>,
});

impl MeshAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for m in &row.mesh {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.descriptor_ui.push(m.descriptor_ui.clone());
            self.descriptor_name.push(m.descriptor_name.clone());
            self.qualifier_ui.push(m.qualifier_ui.clone());
            self.qualifier_name.push(m.qualifier_name.clone());
            self.is_major_topic.push(m.is_major_topic);
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(&mut self.descriptor_ui))),
            Arc::new(StringArray::from(std::mem::take(&mut self.descriptor_name))),
            Arc::new(StringArray::from(std::mem::take(&mut self.qualifier_ui))),
            Arc::new(StringArray::from(std::mem::take(&mut self.qualifier_name))),
            Arc::new(BooleanArray::from(std::mem::take(&mut self.is_major_topic))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(AwardsAccumulator, schema::work_awards, {
    work_id: Option<Arc<str>>,
    id: Option<String>,
    display_name: Option<String>,
    funder_award_id: Option<String>,
    funder_id: Option<String>,
    funder_display_name: Option<String>,
});

impl AwardsAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for a in &row.awards {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.id.push(a.id.clone());
            self.display_name.push(a.display_name.clone());
            self.funder_award_id.push(a.funder_award_id.clone());
            self.funder_id.push(a.funder_id.clone());
            self.funder_display_name.push(a.funder_display_name.clone());
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(&mut self.id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.display_name))),
            Arc::new(StringArray::from(std::mem::take(&mut self.funder_award_id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.funder_id))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.funder_display_name,
            ))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(TopicsAccumulator, schema::work_topics, {
    work_id: Option<Arc<str>>,
    topic_id: Option<String>,
    display_name: Option<String>,
    score: Option<f64>,
    subfield_id: Option<String>,
    subfield_display_name: Option<String>,
    field_id: Option<String>,
    field_display_name: Option<String>,
    domain_id: Option<String>,
    domain_display_name: Option<String>,
});

impl TopicsAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for t in &row.topics {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.topic_id.push(t.id.clone());
            self.display_name.push(t.display_name.clone());
            self.score.push(t.score);
            self.subfield_id
                .push(t.subfield.as_ref().and_then(|s| s.id.clone()));
            self.subfield_display_name
                .push(t.subfield.as_ref().and_then(|s| s.display_name.clone()));
            self.field_id
                .push(t.field.as_ref().and_then(|s| s.id.clone()));
            self.field_display_name
                .push(t.field.as_ref().and_then(|s| s.display_name.clone()));
            self.domain_id
                .push(t.domain.as_ref().and_then(|s| s.id.clone()));
            self.domain_display_name
                .push(t.domain.as_ref().and_then(|s| s.display_name.clone()));
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(&mut self.topic_id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.display_name))),
            Arc::new(Float64Array::from(std::mem::take(&mut self.score))),
            Arc::new(StringArray::from(std::mem::take(&mut self.subfield_id))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.subfield_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(&mut self.field_id))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.field_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(&mut self.domain_id))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.domain_display_name,
            ))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(LocationsAccumulator, schema::work_locations, {
    work_id: Option<Arc<str>>,
    is_oa: Option<bool>,
    landing_page_url: Option<String>,
    pdf_url: Option<String>,
    license: Option<String>,
    version: Option<String>,
    source_id: Option<String>,
    source_display_name: Option<String>,
    source_issn_l: Option<String>,
    source_issn: Option<Vec<String>>,
    source_host_organization: Option<String>,
    source_type: Option<String>,
});

impl LocationsAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for loc in &row.locations {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.is_oa.push(loc.is_oa);
            self.landing_page_url.push(loc.landing_page_url.clone());
            self.pdf_url.push(loc.pdf_url.clone());
            self.license.push(loc.license.clone());
            self.version.push(loc.version.clone());
            let src = loc.source.as_ref();
            self.source_id.push(src.and_then(|s| s.id.clone()));
            self.source_display_name
                .push(src.and_then(|s| s.display_name.clone()));
            self.source_issn_l.push(src.and_then(|s| s.issn_l.clone()));
            self.source_issn.push(src.and_then(|s| s.issn.clone()));
            self.source_host_organization
                .push(src.and_then(|s| s.host_organization.clone()));
            self.source_type
                .push(src.and_then(|s| s.source_type.clone()));
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(BooleanArray::from(std::mem::take(&mut self.is_oa))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.landing_page_url,
            ))),
            Arc::new(StringArray::from(std::mem::take(&mut self.pdf_url))),
            Arc::new(StringArray::from(std::mem::take(&mut self.license))),
            Arc::new(StringArray::from(std::mem::take(&mut self.version))),
            Arc::new(StringArray::from(std::mem::take(&mut self.source_id))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.source_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(&mut self.source_issn_l))),
            build_list_string_array(&std::mem::take(&mut self.source_issn)),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.source_host_organization,
            ))),
            Arc::new(StringArray::from(std::mem::take(&mut self.source_type))),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

fanout_accumulator!(AuthorshipsAccumulator, schema::work_authorships, {
    work_id: Option<Arc<str>>,
    author_position: Option<String>,
    author_id: Option<String>,
    author_display_name: Option<String>,
    author_orcid: Option<String>,
    raw_author_name: Option<String>,
    is_corresponding: Option<bool>,
    countries: Option<Vec<String>>,
    raw_affiliation_strings: Option<Vec<String>>,
    institution_ids: Option<Vec<String>>,
    institution_display_names: Option<Vec<String>>,
    institution_rors: Option<Vec<String>>,
    institution_country_codes: Option<Vec<String>>,
    institution_types: Option<Vec<String>>,
    institution_lineages: Option<Vec<Vec<String>>>,
    affiliation_raw_strings: Option<Vec<String>>,
    affiliation_institution_ids: Option<Vec<Vec<String>>>,
});

impl AuthorshipsAccumulator {
    pub fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
        for a in &row.authorships {
            self.work_id.push(Some(Arc::clone(work_id)));
            self.author_position.push(a.author_position.clone());

            let author = a.author.as_ref();
            self.author_id.push(author.and_then(|au| au.id.clone()));
            self.author_display_name
                .push(author.and_then(|au| au.display_name.clone()));
            self.author_orcid
                .push(author.and_then(|au| au.orcid.clone()));
            self.raw_author_name.push(a.raw_author_name.clone());
            self.is_corresponding.push(a.is_corresponding);

            self.countries.push(opt_vec_to_opt(a.countries.clone()));
            self.raw_affiliation_strings
                .push(opt_vec_to_opt(a.raw_affiliation_strings.clone()));

            // institutions: parallel lists from institutions[]
            let insts = &a.institutions;
            if insts.is_empty() {
                self.institution_ids.push(None);
                self.institution_display_names.push(None);
                self.institution_rors.push(None);
                self.institution_country_codes.push(None);
                self.institution_types.push(None);
                self.institution_lineages.push(None);
            } else {
                // Use unwrap_or_default to preserve positional alignment across parallel lists.
                // filter_map would drop nulls, causing index mismatch between lists.
                self.institution_ids.push(Some(
                    insts
                        .iter()
                        .map(|i| i.id.clone().unwrap_or_default())
                        .collect(),
                ));
                self.institution_display_names.push(Some(
                    insts
                        .iter()
                        .map(|i| i.display_name.clone().unwrap_or_default())
                        .collect(),
                ));
                self.institution_rors.push(Some(
                    insts
                        .iter()
                        .map(|i| i.ror.clone().unwrap_or_default())
                        .collect(),
                ));
                self.institution_country_codes.push(Some(
                    insts
                        .iter()
                        .map(|i| i.country_code.clone().unwrap_or_default())
                        .collect(),
                ));
                self.institution_types.push(Some(
                    insts
                        .iter()
                        .map(|i| i.institution_type.clone().unwrap_or_default())
                        .collect(),
                ));
                self.institution_lineages
                    .push(Some(insts.iter().map(|i| i.lineage.clone()).collect()));
            }

            // affiliations: parallel lists from affiliations[]
            let affs = &a.affiliations;
            if affs.is_empty() {
                self.affiliation_raw_strings.push(None);
                self.affiliation_institution_ids.push(None);
            } else {
                self.affiliation_raw_strings.push(Some(
                    affs.iter()
                        .map(|af| af.raw_affiliation_string.clone().unwrap_or_default())
                        .collect(),
                ));
                self.affiliation_institution_ids.push(Some(
                    affs.iter()
                        .map(|af| af.institution_ids.iter().flatten().cloned().collect())
                        .collect(),
                ));
            }
        }
    }

    pub fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays: Vec<ArrayRef> = vec![
            arc_str_col_to_array(&mut self.work_id),
            Arc::new(StringArray::from(std::mem::take(&mut self.author_position))),
            Arc::new(StringArray::from(std::mem::take(&mut self.author_id))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.author_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(&mut self.author_orcid))),
            Arc::new(StringArray::from(std::mem::take(&mut self.raw_author_name))),
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.is_corresponding,
            ))),
            build_list_string_array(&std::mem::take(&mut self.countries)),
            build_list_string_array(&std::mem::take(&mut self.raw_affiliation_strings)),
            build_list_string_array(&std::mem::take(&mut self.institution_ids)),
            build_list_string_array(&std::mem::take(&mut self.institution_display_names)),
            build_list_string_array(&std::mem::take(&mut self.institution_rors)),
            build_list_string_array(&std::mem::take(&mut self.institution_country_codes)),
            build_list_string_array(&std::mem::take(&mut self.institution_types)),
            build_list_list_string_array(&std::mem::take(&mut self.institution_lineages)),
            build_list_string_array(&std::mem::take(&mut self.affiliation_raw_strings)),
            build_list_list_string_array(&std::mem::take(&mut self.affiliation_institution_ids)),
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::works::WorksAccumulator;
    use crate::accumulator::Accumulator;

    #[test]
    fn works_accumulator_basic() {
        let mut acc = WorksAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{
                "id": "https://openalex.org/W123",
                "display_name": "Test Paper",
                "title": "Test Paper",
                "doi": "https://doi.org/10.1234/TEST",
                "publication_year": 2023,
                "cited_by_count": 42,
                "open_access": {"is_oa": true, "oa_status": "gold", "oa_url": "https://example.com"},
                "fwci": 1.5,
                "biblio": {"volume": "10", "issue": "2"}
            }"#,
        )
        .unwrap();
        acc.push(row);
        let batch = acc.take_batch().unwrap();

        assert_eq!(batch.num_columns(), 84);
        assert_eq!(batch.num_rows(), 1);

        let id_col = batch
            .column_by_name("work_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(id_col.value(0), "W123");

        let doi_col = batch
            .column_by_name("doi_norm")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(doi_col.value(0), "10.1234/test");

        let fwci_col = batch
            .column_by_name("fwci")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((fwci_col.value(0) - 1.5).abs() < f64::EPSILON);

        let vol = batch
            .column_by_name("biblio_volume")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(vol.value(0), "10");
    }

    #[test]
    fn works_accumulator_abstract_decode() {
        let mut acc = WorksAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","abstract_inverted_index":{"Despite":[0],"growing":[1],"interest":[2]}}"#,
        ).unwrap();
        acc.push(row);
        let batch = acc.take_batch().unwrap();
        let abs_col = batch
            .column_by_name("abstract_text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(abs_col.value(0), "Despite growing interest");
    }

    #[test]
    fn works_keys_accumulator_4_cols() {
        let mut keys_acc = WorksKeysAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1234/TEST",
                "ids": {
                    "pmid": "https://pubmed.ncbi.nlm.nih.gov/99999",
                    "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123"
                }
            }"#,
        )
        .unwrap();
        keys_acc.push_from_work(&Arc::from("W123"), &row);
        let batch = keys_acc.take_batch().unwrap();

        assert_eq!(batch.num_columns(), 4);
        assert_eq!(batch.num_rows(), 1);

        let pmcid = batch
            .column_by_name("pmcid")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(pmcid.value(0), "PMC123");
    }

    #[test]
    fn citations_accumulator() {
        let mut acc = CitationsAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","referenced_works":["https://openalex.org/W2","https://openalex.org/W3"]}"#,
        ).unwrap();
        acc.push_from_work(&Arc::from("W1"), &row);
        let batch = acc.take_batch().unwrap();

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 2);
        let ref_col = batch
            .column_by_name("referenced_work_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ref_col.value(0), "W2");
        assert_eq!(ref_col.value(1), "W3");
    }

    #[test]
    fn counts_by_year_accumulator() {
        let mut acc = CountsByYearAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","counts_by_year":[{"year":2023,"cited_by_count":10},{"year":2022,"cited_by_count":5}]}"#,
        ).unwrap();
        acc.push_from_work(&Arc::from("W1"), &row);
        let batch = acc.take_batch().unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn keywords_accumulator() {
        let mut acc = KeywordsAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","keywords":[{"id":"k1","display_name":"ML","score":0.9}]}"#,
        ).unwrap();
        acc.push_from_work(&Arc::from("W1"), &row);
        let batch = acc.take_batch().unwrap();
        assert_eq!(batch.num_columns(), 4);
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn mesh_accumulator() {
        let mut acc = MeshAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","mesh":[{"descriptor_ui":"D001","descriptor_name":"Test","is_major_topic":true}]}"#,
        ).unwrap();
        acc.push_from_work(&Arc::from("W1"), &row);
        let batch = acc.take_batch().unwrap();
        assert_eq!(batch.num_columns(), 6);
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn authorships_accumulator() {
        let mut acc = AuthorshipsAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","authorships":[{
                "author_position":"first",
                "author":{"id":"https://openalex.org/A1","display_name":"Alice","orcid":"0000-0001"},
                "institutions":[{"id":"https://openalex.org/I1","display_name":"MIT","ror":"https://ror.org/042nb2s44","country_code":"US","type":"education","lineage":["https://openalex.org/I1"]}],
                "is_corresponding":true,
                "raw_author_name":"Alice Smith",
                "countries":["US"],
                "raw_affiliation_strings":["MIT"],
                "affiliations":[{"raw_affiliation_string":"MIT","institution_ids":["https://openalex.org/I1"]}]
            }]}"#,
        ).unwrap();
        acc.push_from_work(&Arc::from("W1"), &row);
        let batch = acc.take_batch().unwrap();
        assert_eq!(batch.num_columns(), 17);
        assert_eq!(batch.num_rows(), 1);

        let pos = batch
            .column_by_name("author_position")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(pos.value(0), "first");
    }

    #[test]
    fn topics_accumulator() {
        let mut acc = TopicsAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","topics":[{
                "id":"https://openalex.org/T1","display_name":"ML","score":0.99,
                "subfield":{"id":"S1","display_name":"AI"},
                "field":{"id":"F1","display_name":"CS"},
                "domain":{"id":"D1","display_name":"Science"}
            }]}"#,
        )
        .unwrap();
        acc.push_from_work(&Arc::from("W1"), &row);
        let batch = acc.take_batch().unwrap();
        assert_eq!(batch.num_columns(), 10);
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn locations_accumulator() {
        let mut acc = LocationsAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(
            r#"{"id":"https://openalex.org/W1","locations":[{
                "is_oa":true,"landing_page_url":"https://example.com","pdf_url":"https://example.com/pdf",
                "license":"cc-by","version":"publishedVersion",
                "source":{"id":"S1","display_name":"Nature","issn_l":"0028-0836","issn":["0028-0836","1476-4687"],"host_organization":"Springer","type":"journal"}
            }]}"#,
        ).unwrap();
        acc.push_from_work(&Arc::from("W1"), &row);
        let batch = acc.take_batch().unwrap();
        assert_eq!(batch.num_columns(), 12);
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn mag_as_number() {
        let row: WorkRow =
            sonic_rs::from_str(r#"{"id":"https://openalex.org/W1","ids":{"mag":2741809807}}"#)
                .unwrap();
        assert_eq!(row.ids.as_ref().unwrap().mag.as_deref(), Some("2741809807"));
    }

    #[test]
    fn works_schema_matches_accumulator() {
        let mut acc = WorksAccumulator::new();
        let row: WorkRow = sonic_rs::from_str(r#"{"id":"https://openalex.org/W1"}"#).unwrap();
        acc.push(row);
        let batch = acc.take_batch().unwrap();
        assert_eq!(
            batch.schema().fields().len(),
            schema::works().fields().len()
        );
        for (i, field) in schema::works().fields().iter().enumerate() {
            assert_eq!(
                batch.schema().field(i).name(),
                field.name(),
                "column {i} name mismatch"
            );
        }
    }
}
