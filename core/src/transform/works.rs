//! Works accumulator (82-column fact table) and arrow helper functions.

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::Schema;

use crate::accumulator::{Accumulator, DEFAULT_BATCH_SIZE};
use crate::id::normalize_doi;
use crate::schema;

use super::model::*;
use super::{decode_inverted_index, extract_pmcid, extract_pmid, strip_html_tags, strip_oa_prefix};

// ============================================================
// Arrow helpers
// ============================================================

/// Build List<Utf8> array from Vec<Option<Vec<String>>>.
pub(crate) fn build_list_string_array(data: &[Option<Vec<String>>]) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in data {
        match row {
            Some(items) => {
                for item in items {
                    builder.values().append_value(item);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

/// Build List<List<Utf8>> array from Vec<Option<Vec<Vec<String>>>>.
pub(crate) fn build_list_list_string_array(data: &[Option<Vec<Vec<String>>>]) -> ArrayRef {
    let mut builder = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
    for row in data {
        match row {
            Some(outer) => {
                for inner in outer {
                    for s in inner {
                        builder.values().values().append_value(s);
                    }
                    builder.values().append(true);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

/// Helper: flatten a Location into (is_oa, landing_page_url, pdf_url, license, version,
/// source_id, source_display_name, source_issn_l, source_issn, source_host_org, source_type)
#[allow(clippy::type_complexity)]
pub(crate) fn flatten_location(
    loc: Option<&Location>,
) -> (
    Option<bool>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<Vec<String>>,
    Option<String>,
    Option<String>,
) {
    match loc {
        Some(l) => {
            let src = l.source.as_ref();
            (
                l.is_oa,
                l.landing_page_url.clone(),
                l.pdf_url.clone(),
                l.license.clone(),
                l.version.clone(),
                src.and_then(|s| s.id.clone()),
                src.and_then(|s| s.display_name.clone()),
                src.and_then(|s| s.issn_l.clone()),
                src.and_then(|s| s.issn.clone()),
                src.and_then(|s| s.host_organization.clone()),
                src.and_then(|s| s.source_type.clone()),
            )
        }
        None => (
            None, None, None, None, None, None, None, None, None, None, None,
        ),
    }
}

/// Helper: flatten a Topic into (id, display_name, score, subfield_id, subfield_dn, field_id, field_dn, domain_id, domain_dn)
#[allow(clippy::type_complexity)]
pub(crate) fn flatten_topic(
    t: Option<&Topic>,
) -> (
    Option<String>,
    Option<String>,
    Option<f64>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    match t {
        Some(t) => (
            t.id.clone(),
            t.display_name.clone(),
            t.score,
            t.subfield.as_ref().and_then(|s| s.id.clone()),
            t.subfield.as_ref().and_then(|s| s.display_name.clone()),
            t.field.as_ref().and_then(|s| s.id.clone()),
            t.field.as_ref().and_then(|s| s.display_name.clone()),
            t.domain.as_ref().and_then(|s| s.id.clone()),
            t.domain.as_ref().and_then(|s| s.display_name.clone()),
        ),
        None => (None, None, None, None, None, None, None, None, None),
    }
}

pub(crate) fn opt_vec_to_opt(v: Vec<String>) -> Option<Vec<String>> {
    if v.is_empty() { None } else { Some(v) }
}

// ============================================================
// Works accumulator (84 cols)
// ============================================================

pub struct WorksAccumulator {
    schema: Arc<Schema>,
    /// Shard-level updated_date (same for all rows in a shard).
    shard_updated_date: Option<String>,
    /// Shard index for provenance tracking (replaces tracking table).
    oa_shard_idx: u16,
    // identity
    work_id: Vec<Option<String>>,
    doi: Vec<Option<String>>,
    doi_norm: Vec<Option<String>>,
    title: Vec<Option<String>>,
    abstract_text: Vec<Option<String>>,
    content_hash: Vec<Option<String>>,
    // dates
    publication_date: Vec<Option<String>>,
    publication_year: Vec<Option<i32>>,
    created_date: Vec<Option<String>>,
    updated_date: Vec<Option<String>>,
    // classification
    language: Vec<Option<String>>,
    work_type: Vec<Option<String>>,
    // counts
    cited_by_count: Vec<Option<i32>>,
    referenced_works_count: Vec<Option<i32>>,
    countries_distinct_count: Vec<Option<i32>>,
    institutions_distinct_count: Vec<Option<i32>>,
    locations_count: Vec<Option<i32>>,
    // metrics
    fwci: Vec<Option<f64>>,
    // flags
    is_retracted: Vec<Option<bool>>,
    is_paratext: Vec<Option<bool>>,
    // open access
    is_oa: Vec<Option<bool>>,
    oa_status: Vec<Option<String>>,
    oa_url: Vec<Option<String>>,
    any_repository_has_fulltext: Vec<Option<bool>>,
    // citation percentiles
    citation_normalized_percentile: Vec<Option<f64>>,
    is_in_top_1_pct: Vec<Option<bool>>,
    is_in_top_10_pct: Vec<Option<bool>>,
    cited_by_percentile_min: Vec<Option<f64>>,
    cited_by_percentile_max: Vec<Option<f64>>,
    // has_content
    has_content_pdf: Vec<Option<bool>>,
    has_content_grobid_xml: Vec<Option<bool>>,
    // biblio
    biblio_volume: Vec<Option<String>>,
    biblio_issue: Vec<Option<String>>,
    biblio_first_page: Vec<Option<String>>,
    biblio_last_page: Vec<Option<String>>,
    // apc_list
    apc_list_value: Vec<Option<i32>>,
    apc_list_currency: Vec<Option<String>>,
    apc_list_value_usd: Vec<Option<i32>>,
    apc_list_provenance: Vec<Option<String>>,
    // apc_paid
    apc_paid_value: Vec<Option<i32>>,
    apc_paid_currency: Vec<Option<String>>,
    apc_paid_value_usd: Vec<Option<i32>>,
    apc_paid_provenance: Vec<Option<String>>,
    // primary_topic (9)
    primary_topic_id: Vec<Option<String>>,
    primary_topic_display_name: Vec<Option<String>>,
    primary_topic_score: Vec<Option<f64>>,
    primary_topic_subfield_id: Vec<Option<String>>,
    primary_topic_subfield_display_name: Vec<Option<String>>,
    primary_topic_field_id: Vec<Option<String>>,
    primary_topic_field_display_name: Vec<Option<String>>,
    primary_topic_domain_id: Vec<Option<String>>,
    primary_topic_domain_display_name: Vec<Option<String>>,
    // primary_location (11)
    primary_location_is_oa: Vec<Option<bool>>,
    primary_location_landing_page_url: Vec<Option<String>>,
    primary_location_pdf_url: Vec<Option<String>>,
    primary_location_license: Vec<Option<String>>,
    primary_location_version: Vec<Option<String>>,
    primary_location_source_id: Vec<Option<String>>,
    primary_location_source_display_name: Vec<Option<String>>,
    primary_location_source_issn_l: Vec<Option<String>>,
    primary_location_source_issn: Vec<Option<Vec<String>>>,
    primary_location_source_host_organization: Vec<Option<String>>,
    primary_location_source_type: Vec<Option<String>>,
    // best_oa_location (11)
    best_oa_location_is_oa: Vec<Option<bool>>,
    best_oa_location_landing_page_url: Vec<Option<String>>,
    best_oa_location_pdf_url: Vec<Option<String>>,
    best_oa_location_license: Vec<Option<String>>,
    best_oa_location_version: Vec<Option<String>>,
    best_oa_location_source_id: Vec<Option<String>>,
    best_oa_location_source_display_name: Vec<Option<String>>,
    best_oa_location_source_issn_l: Vec<Option<String>>,
    best_oa_location_source_issn: Vec<Option<Vec<String>>>,
    best_oa_location_source_host_organization: Vec<Option<String>>,
    best_oa_location_source_type: Vec<Option<String>>,
    // external ids
    ids_pmid: Vec<Option<String>>,
    ids_pmcid: Vec<Option<String>>,
    // list columns
    corresponding_author_ids: Vec<Option<Vec<String>>>,
    corresponding_institution_ids: Vec<Option<Vec<String>>>,
    indexed_in: Vec<Option<Vec<String>>>,
    related_works: Vec<Option<Vec<String>>>,
}

impl Default for WorksAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! vec_cap {
    () => {
        Vec::with_capacity(DEFAULT_BATCH_SIZE)
    };
}

impl WorksAccumulator {
    pub fn new() -> Self {
        Self::with_shard_metadata(None, 0)
    }

    pub fn with_shard_metadata(shard_updated_date: Option<String>, oa_shard_idx: u16) -> Self {
        Self {
            schema: schema::works().clone(),
            shard_updated_date,
            oa_shard_idx,
            work_id: vec_cap!(),
            doi: vec_cap!(),
            doi_norm: vec_cap!(),
            title: vec_cap!(),
            abstract_text: vec_cap!(),
            content_hash: vec_cap!(),
            publication_date: vec_cap!(),
            publication_year: vec_cap!(),
            created_date: vec_cap!(),
            updated_date: vec_cap!(),
            language: vec_cap!(),
            work_type: vec_cap!(),
            cited_by_count: vec_cap!(),
            referenced_works_count: vec_cap!(),
            countries_distinct_count: vec_cap!(),
            institutions_distinct_count: vec_cap!(),
            locations_count: vec_cap!(),
            fwci: vec_cap!(),
            is_retracted: vec_cap!(),
            is_paratext: vec_cap!(),
            is_oa: vec_cap!(),
            oa_status: vec_cap!(),
            oa_url: vec_cap!(),
            any_repository_has_fulltext: vec_cap!(),
            citation_normalized_percentile: vec_cap!(),
            is_in_top_1_pct: vec_cap!(),
            is_in_top_10_pct: vec_cap!(),
            cited_by_percentile_min: vec_cap!(),
            cited_by_percentile_max: vec_cap!(),
            has_content_pdf: vec_cap!(),
            has_content_grobid_xml: vec_cap!(),
            biblio_volume: vec_cap!(),
            biblio_issue: vec_cap!(),
            biblio_first_page: vec_cap!(),
            biblio_last_page: vec_cap!(),
            apc_list_value: vec_cap!(),
            apc_list_currency: vec_cap!(),
            apc_list_value_usd: vec_cap!(),
            apc_list_provenance: vec_cap!(),
            apc_paid_value: vec_cap!(),
            apc_paid_currency: vec_cap!(),
            apc_paid_value_usd: vec_cap!(),
            apc_paid_provenance: vec_cap!(),
            primary_topic_id: vec_cap!(),
            primary_topic_display_name: vec_cap!(),
            primary_topic_score: vec_cap!(),
            primary_topic_subfield_id: vec_cap!(),
            primary_topic_subfield_display_name: vec_cap!(),
            primary_topic_field_id: vec_cap!(),
            primary_topic_field_display_name: vec_cap!(),
            primary_topic_domain_id: vec_cap!(),
            primary_topic_domain_display_name: vec_cap!(),
            primary_location_is_oa: vec_cap!(),
            primary_location_landing_page_url: vec_cap!(),
            primary_location_pdf_url: vec_cap!(),
            primary_location_license: vec_cap!(),
            primary_location_version: vec_cap!(),
            primary_location_source_id: vec_cap!(),
            primary_location_source_display_name: vec_cap!(),
            primary_location_source_issn_l: vec_cap!(),
            primary_location_source_issn: vec_cap!(),
            primary_location_source_host_organization: vec_cap!(),
            primary_location_source_type: vec_cap!(),
            best_oa_location_is_oa: vec_cap!(),
            best_oa_location_landing_page_url: vec_cap!(),
            best_oa_location_pdf_url: vec_cap!(),
            best_oa_location_license: vec_cap!(),
            best_oa_location_version: vec_cap!(),
            best_oa_location_source_id: vec_cap!(),
            best_oa_location_source_display_name: vec_cap!(),
            best_oa_location_source_issn_l: vec_cap!(),
            best_oa_location_source_issn: vec_cap!(),
            best_oa_location_source_host_organization: vec_cap!(),
            best_oa_location_source_type: vec_cap!(),
            ids_pmid: vec_cap!(),
            ids_pmcid: vec_cap!(),
            corresponding_author_ids: vec_cap!(),
            corresponding_institution_ids: vec_cap!(),
            indexed_in: vec_cap!(),
            related_works: vec_cap!(),
        }
    }
}

impl Accumulator for WorksAccumulator {
    type Row = WorkRow;

    fn push(&mut self, row: WorkRow) {
        // identity
        self.work_id
            .push(row.id.as_deref().map(|s| strip_oa_prefix(s).to_string()));
        self.doi.push(row.doi.clone());
        self.doi_norm.push(
            row.doi
                .as_deref()
                .and_then(normalize_doi)
                .map(|b| b.into_string()),
        );
        // Decode abstract and strip inline HTML tags (e.g. <italic>, <jats:p>)
        let abstract_text = row
            .abstract_inverted_index
            .as_ref()
            .map(decode_inverted_index)
            .map(|s| strip_html_tags(&s))
            .filter(|s| !s.is_empty());

        // blake3(title + \0 + abstract) for downstream embedding change detection
        let mut hasher = blake3::Hasher::new();
        hasher.update(row.title.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(abstract_text.as_deref().unwrap_or("").as_bytes());
        self.content_hash
            .push(Some(hasher.finalize().to_hex().to_string()));

        self.title.push(row.title.map(|s| strip_html_tags(&s)));
        self.abstract_text.push(abstract_text);

        // dates
        self.publication_date.push(row.publication_date);
        self.publication_year.push(row.publication_year);
        self.created_date.push(row.created_date);
        self.updated_date.push(row.updated_date);

        // classification
        self.language.push(row.language);
        self.work_type.push(row.work_type);

        // counts
        self.cited_by_count.push(row.cited_by_count);
        self.referenced_works_count.push(row.referenced_works_count);
        self.countries_distinct_count
            .push(row.countries_distinct_count);
        self.institutions_distinct_count
            .push(row.institutions_distinct_count);
        self.locations_count.push(row.locations_count);

        // metrics
        self.fwci.push(row.fwci);

        // flags
        self.is_retracted.push(row.is_retracted);
        self.is_paratext.push(row.is_paratext);

        // open access
        let oa = row.open_access.as_ref();
        self.is_oa.push(oa.and_then(|o| o.is_oa));
        self.oa_status.push(oa.and_then(|o| o.oa_status.clone()));
        self.oa_url.push(oa.and_then(|o| o.oa_url.clone()));
        self.any_repository_has_fulltext
            .push(oa.and_then(|o| o.any_repository_has_fulltext));

        // citation percentiles
        let cnp = row.citation_normalized_percentile.as_ref();
        self.citation_normalized_percentile
            .push(cnp.and_then(|c| c.value));
        self.is_in_top_1_pct
            .push(cnp.and_then(|c| c.is_in_top_1_percent));
        self.is_in_top_10_pct
            .push(cnp.and_then(|c| c.is_in_top_10_percent));
        let pct = row.cited_by_percentile_year.as_ref();
        self.cited_by_percentile_min.push(pct.and_then(|p| p.min));
        self.cited_by_percentile_max.push(pct.and_then(|p| p.max));

        // has_content
        let hc = row.has_content.as_ref();
        self.has_content_pdf
            .push(hc.and_then(|h| h.pdf).or(row.has_fulltext));
        self.has_content_grobid_xml
            .push(hc.and_then(|h| h.grobid_xml));

        // biblio
        let bib = row.biblio.as_ref();
        self.biblio_volume.push(bib.and_then(|b| b.volume.clone()));
        self.biblio_issue.push(bib.and_then(|b| b.issue.clone()));
        self.biblio_first_page
            .push(bib.and_then(|b| b.first_page.clone()));
        self.biblio_last_page
            .push(bib.and_then(|b| b.last_page.clone()));

        // apc_list
        let apc_l = row.apc_list.as_ref();
        self.apc_list_value.push(apc_l.and_then(|a| a.value));
        self.apc_list_currency
            .push(apc_l.and_then(|a| a.currency.clone()));
        self.apc_list_value_usd
            .push(apc_l.and_then(|a| a.value_usd));
        self.apc_list_provenance
            .push(apc_l.and_then(|a| a.provenance.clone()));

        // apc_paid
        let apc_p = row.apc_paid.as_ref();
        self.apc_paid_value.push(apc_p.and_then(|a| a.value));
        self.apc_paid_currency
            .push(apc_p.and_then(|a| a.currency.clone()));
        self.apc_paid_value_usd
            .push(apc_p.and_then(|a| a.value_usd));
        self.apc_paid_provenance
            .push(apc_p.and_then(|a| a.provenance.clone()));

        // primary_topic
        let (pt_id, pt_dn, pt_score, pt_sf_id, pt_sf_dn, pt_f_id, pt_f_dn, pt_d_id, pt_d_dn) =
            flatten_topic(row.primary_topic.as_ref());
        self.primary_topic_id.push(pt_id);
        self.primary_topic_display_name.push(pt_dn);
        self.primary_topic_score.push(pt_score);
        self.primary_topic_subfield_id.push(pt_sf_id);
        self.primary_topic_subfield_display_name.push(pt_sf_dn);
        self.primary_topic_field_id.push(pt_f_id);
        self.primary_topic_field_display_name.push(pt_f_dn);
        self.primary_topic_domain_id.push(pt_d_id);
        self.primary_topic_domain_display_name.push(pt_d_dn);

        // primary_location
        let (
            pl_oa,
            pl_lp,
            pl_pdf,
            pl_lic,
            pl_ver,
            pl_sid,
            pl_sdn,
            pl_issn_l,
            pl_issn,
            pl_ho,
            pl_st,
        ) = flatten_location(row.primary_location.as_ref());
        self.primary_location_is_oa.push(pl_oa);
        self.primary_location_landing_page_url.push(pl_lp);
        self.primary_location_pdf_url.push(pl_pdf);
        self.primary_location_license.push(pl_lic);
        self.primary_location_version.push(pl_ver);
        self.primary_location_source_id.push(pl_sid);
        self.primary_location_source_display_name.push(pl_sdn);
        self.primary_location_source_issn_l.push(pl_issn_l);
        self.primary_location_source_issn.push(pl_issn);
        self.primary_location_source_host_organization.push(pl_ho);
        self.primary_location_source_type.push(pl_st);

        // best_oa_location
        let (
            bo_oa,
            bo_lp,
            bo_pdf,
            bo_lic,
            bo_ver,
            bo_sid,
            bo_sdn,
            bo_issn_l,
            bo_issn,
            bo_ho,
            bo_st,
        ) = flatten_location(row.best_oa_location.as_ref());
        self.best_oa_location_is_oa.push(bo_oa);
        self.best_oa_location_landing_page_url.push(bo_lp);
        self.best_oa_location_pdf_url.push(bo_pdf);
        self.best_oa_location_license.push(bo_lic);
        self.best_oa_location_version.push(bo_ver);
        self.best_oa_location_source_id.push(bo_sid);
        self.best_oa_location_source_display_name.push(bo_sdn);
        self.best_oa_location_source_issn_l.push(bo_issn_l);
        self.best_oa_location_source_issn.push(bo_issn);
        self.best_oa_location_source_host_organization.push(bo_ho);
        self.best_oa_location_source_type.push(bo_st);

        // external ids
        let ids = row.ids.as_ref();
        self.ids_pmid.push(
            ids.and_then(|i| i.pmid.as_deref())
                .and_then(extract_pmid)
                .map(|s| s.to_string()),
        );
        self.ids_pmcid.push(
            ids.and_then(|i| i.pmcid.as_deref())
                .and_then(extract_pmcid)
                .map(|s| s.to_string()),
        );

        // list columns
        self.corresponding_author_ids
            .push(opt_vec_to_opt(row.corresponding_author_ids));
        self.corresponding_institution_ids
            .push(opt_vec_to_opt(row.corresponding_institution_ids));
        self.indexed_in.push(opt_vec_to_opt(row.indexed_in));
        self.related_works.push(opt_vec_to_opt(row.related_works));

        // (shard_updated_date is repeated in take_batch, no per-row clone needed)
    }

    fn len(&self) -> usize {
        self.work_id.len()
    }

    fn take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let n = self.len();
        let arrays: Vec<ArrayRef> = vec![
            // identity (5)
            Arc::new(StringArray::from(std::mem::take(&mut self.work_id))),
            Arc::new(StringArray::from(std::mem::take(&mut self.doi))),
            Arc::new(StringArray::from(std::mem::take(&mut self.doi_norm))),
            Arc::new(StringArray::from(std::mem::take(&mut self.title))),
            Arc::new(StringArray::from(std::mem::take(&mut self.abstract_text))),
            Arc::new(StringArray::from(std::mem::take(&mut self.content_hash))),
            // dates (4)
            Arc::new(StringArray::from(std::mem::take(
                &mut self.publication_date,
            ))),
            Arc::new(Int32Array::from(std::mem::take(&mut self.publication_year))),
            Arc::new(StringArray::from(std::mem::take(&mut self.created_date))),
            Arc::new(StringArray::from(std::mem::take(&mut self.updated_date))),
            // classification (2)
            Arc::new(StringArray::from(std::mem::take(&mut self.language))),
            Arc::new(StringArray::from(std::mem::take(&mut self.work_type))),
            // counts (5)
            Arc::new(Int32Array::from(std::mem::take(&mut self.cited_by_count))),
            Arc::new(Int32Array::from(std::mem::take(
                &mut self.referenced_works_count,
            ))),
            Arc::new(Int32Array::from(std::mem::take(
                &mut self.countries_distinct_count,
            ))),
            Arc::new(Int32Array::from(std::mem::take(
                &mut self.institutions_distinct_count,
            ))),
            Arc::new(Int32Array::from(std::mem::take(&mut self.locations_count))),
            // metrics (1)
            Arc::new(Float64Array::from(std::mem::take(&mut self.fwci))),
            // flags (2)
            Arc::new(BooleanArray::from(std::mem::take(&mut self.is_retracted))),
            Arc::new(BooleanArray::from(std::mem::take(&mut self.is_paratext))),
            // open access (4)
            Arc::new(BooleanArray::from(std::mem::take(&mut self.is_oa))),
            Arc::new(StringArray::from(std::mem::take(&mut self.oa_status))),
            Arc::new(StringArray::from(std::mem::take(&mut self.oa_url))),
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.any_repository_has_fulltext,
            ))),
            // citation percentiles (5)
            Arc::new(Float64Array::from(std::mem::take(
                &mut self.citation_normalized_percentile,
            ))),
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.is_in_top_1_pct,
            ))),
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.is_in_top_10_pct,
            ))),
            Arc::new(Float64Array::from(std::mem::take(
                &mut self.cited_by_percentile_min,
            ))),
            Arc::new(Float64Array::from(std::mem::take(
                &mut self.cited_by_percentile_max,
            ))),
            // has_content (2)
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.has_content_pdf,
            ))),
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.has_content_grobid_xml,
            ))),
            // biblio (4)
            Arc::new(StringArray::from(std::mem::take(&mut self.biblio_volume))),
            Arc::new(StringArray::from(std::mem::take(&mut self.biblio_issue))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.biblio_first_page,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.biblio_last_page,
            ))),
            // apc_list (4)
            Arc::new(Int32Array::from(std::mem::take(&mut self.apc_list_value))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.apc_list_currency,
            ))),
            Arc::new(Int32Array::from(std::mem::take(
                &mut self.apc_list_value_usd,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.apc_list_provenance,
            ))),
            // apc_paid (4)
            Arc::new(Int32Array::from(std::mem::take(&mut self.apc_paid_value))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.apc_paid_currency,
            ))),
            Arc::new(Int32Array::from(std::mem::take(
                &mut self.apc_paid_value_usd,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.apc_paid_provenance,
            ))),
            // primary_topic (9)
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_id,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_display_name,
            ))),
            Arc::new(Float64Array::from(std::mem::take(
                &mut self.primary_topic_score,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_subfield_id,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_subfield_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_field_id,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_field_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_domain_id,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_topic_domain_display_name,
            ))),
            // primary_location (11)
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.primary_location_is_oa,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_landing_page_url,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_pdf_url,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_license,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_version,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_source_id,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_source_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_source_issn_l,
            ))),
            build_list_string_array(&std::mem::take(&mut self.primary_location_source_issn)),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_source_host_organization,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.primary_location_source_type,
            ))),
            // best_oa_location (11)
            Arc::new(BooleanArray::from(std::mem::take(
                &mut self.best_oa_location_is_oa,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_landing_page_url,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_pdf_url,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_license,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_version,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_source_id,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_source_display_name,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_source_issn_l,
            ))),
            build_list_string_array(&std::mem::take(&mut self.best_oa_location_source_issn)),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_source_host_organization,
            ))),
            Arc::new(StringArray::from(std::mem::take(
                &mut self.best_oa_location_source_type,
            ))),
            // external ids (2)
            Arc::new(StringArray::from(std::mem::take(&mut self.ids_pmid))),
            Arc::new(StringArray::from(std::mem::take(&mut self.ids_pmcid))),
            // list columns (4)
            build_list_string_array(&std::mem::take(&mut self.corresponding_author_ids)),
            build_list_string_array(&std::mem::take(&mut self.corresponding_institution_ids)),
            build_list_string_array(&std::mem::take(&mut self.indexed_in)),
            build_list_string_array(&std::mem::take(&mut self.related_works)),
            // shard metadata (repeated constants — no per-row clone needed)
            Arc::new(StringArray::from(vec![
                self.shard_updated_date.as_deref();
                n
            ])) as ArrayRef,
            Arc::new(UInt16Array::from(vec![self.oa_shard_idx; n])) as ArrayRef,
        ];
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{Location, Source, Topic, TopicClassification, WorkRow};
    use super::*;
    use crate::accumulator::Accumulator;
    use arrow::array::AsArray;

    fn minimal_work() -> WorkRow {
        sonic_rs::from_str(
            r#"{
            "id": "https://openalex.org/W999",
            "doi": "https://doi.org/10.1234/test",
            "title": "Test Title",
            "publication_year": 2024,
            "language": "en",
            "type": "article"
        }"#,
        )
        .unwrap()
    }

    fn work_with_abstract() -> WorkRow {
        sonic_rs::from_str(
            r#"{
            "id": "https://openalex.org/W888",
            "title": "Paper A",
            "abstract_inverted_index": {"hello": [0], "world": [1]},
            "publication_year": 2023
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn build_list_string_array_some_and_none() {
        let data = vec![
            Some(vec!["a".to_string(), "b".to_string()]),
            None,
            Some(vec!["c".to_string()]),
        ];
        let arr = build_list_string_array(&data);
        assert_eq!(arr.len(), 3);
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn build_list_list_string_array_nested() {
        let data = vec![
            Some(vec![
                vec!["a".to_string()],
                vec!["b".to_string(), "c".to_string()],
            ]),
            None,
        ];
        let arr = build_list_list_string_array(&data);
        assert_eq!(arr.len(), 2);
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
    }

    #[test]
    fn flatten_location_none_returns_all_none() {
        let result = flatten_location(None);
        assert!(result.0.is_none());
        assert!(result.1.is_none());
        assert!(result.10.is_none());
    }

    #[test]
    fn flatten_location_with_source() {
        let loc = Location {
            is_oa: Some(true),
            landing_page_url: Some("https://example.com".into()),
            pdf_url: None,
            license: Some("cc-by".into()),
            version: Some("publishedVersion".into()),
            source: Some(Source {
                id: Some("S1".into()),
                display_name: Some("Nature".into()),
                issn_l: Some("0028-0836".into()),
                issn: Some(vec!["0028-0836".into()]),
                host_organization: Some("Springer".into()),
                source_type: Some("journal".into()),
            }),
        };
        let r = flatten_location(Some(&loc));
        assert_eq!(r.0, Some(true));
        assert_eq!(r.1.as_deref(), Some("https://example.com"));
        assert_eq!(r.5.as_deref(), Some("S1"));
        assert_eq!(r.10.as_deref(), Some("journal"));
    }

    #[test]
    fn flatten_topic_none_returns_all_none() {
        let r = flatten_topic(None);
        assert!(r.0.is_none());
        assert!(r.2.is_none());
        assert!(r.8.is_none());
    }

    #[test]
    fn flatten_topic_with_hierarchy() {
        let t = Topic {
            id: Some("T1".into()),
            display_name: Some("ML".into()),
            score: Some(0.95),
            subfield: Some(TopicClassification {
                id: Some("SF1".into()),
                display_name: Some("AI".into()),
            }),
            field: Some(TopicClassification {
                id: Some("F1".into()),
                display_name: Some("CS".into()),
            }),
            domain: None,
        };
        let r = flatten_topic(Some(&t));
        assert_eq!(r.0.as_deref(), Some("T1"));
        assert_eq!(r.2, Some(0.95));
        assert_eq!(r.3.as_deref(), Some("SF1"));
        assert!(r.7.is_none()); // domain.id
    }

    #[test]
    fn opt_vec_to_opt_empty_returns_none() {
        assert!(opt_vec_to_opt(vec![]).is_none());
    }

    #[test]
    fn opt_vec_to_opt_nonempty_returns_some() {
        let v = opt_vec_to_opt(vec!["a".to_string()]);
        assert_eq!(v.unwrap(), vec!["a"]);
    }

    #[test]
    fn works_accumulator_push_and_take_batch() {
        let mut acc = WorksAccumulator::new();
        acc.push(minimal_work());
        assert_eq!(acc.len(), 1);

        let batch = acc.take_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
        // Check work_id has OA prefix stripped
        let wid = batch.column(0).as_string::<i32>();
        assert_eq!(wid.value(0), "W999");
        // Check doi_norm (normalized DOI)
        let doi_norm = batch.column(2).as_string::<i32>();
        assert_eq!(doi_norm.value(0), "10.1234/test");
    }

    #[test]
    fn works_accumulator_content_hash_deterministic() {
        let mut acc1 = WorksAccumulator::new();
        let mut acc2 = WorksAccumulator::new();
        acc1.push(work_with_abstract());
        acc2.push(work_with_abstract());

        let b1 = acc1.take_batch().unwrap();
        let b2 = acc2.take_batch().unwrap();
        let h1 = b1.column(5).as_string::<i32>().value(0);
        let h2 = b2.column(5).as_string::<i32>().value(0);
        assert_eq!(h1, h2);
        assert!(!h1.is_empty());
    }

    #[test]
    fn works_accumulator_content_hash_differs_on_title_change() {
        let w1: WorkRow = sonic_rs::from_str(r#"{"title": "A", "id": "W1"}"#).unwrap();
        let w2: WorkRow = sonic_rs::from_str(r#"{"title": "B", "id": "W1"}"#).unwrap();
        let mut a1 = WorksAccumulator::new();
        let mut a2 = WorksAccumulator::new();
        a1.push(w1);
        a2.push(w2);
        let h1 = a1
            .take_batch()
            .unwrap()
            .column(5)
            .as_string::<i32>()
            .value(0)
            .to_string();
        let h2 = a2
            .take_batch()
            .unwrap()
            .column(5)
            .as_string::<i32>()
            .value(0)
            .to_string();
        assert_ne!(h1, h2);
    }

    #[test]
    fn works_accumulator_abstract_decoded_and_stripped() {
        let mut acc = WorksAccumulator::new();
        acc.push(work_with_abstract());
        let batch = acc.take_batch().unwrap();
        let abs = batch.column(4).as_string::<i32>().value(0);
        assert_eq!(abs, "hello world");
    }

    #[test]
    fn works_accumulator_shard_metadata() {
        let mut acc = WorksAccumulator::with_shard_metadata(Some("2024-01-01".into()), 42);
        acc.push(minimal_work());
        let batch = acc.take_batch().unwrap();
        let ncols = batch.num_columns();
        // shard_updated_date is second-to-last column
        let sud = batch.column(ncols - 2).as_string::<i32>().value(0);
        assert_eq!(sud, "2024-01-01");
        // oa_shard_idx is last column
        let idx = batch
            .column(ncols - 1)
            .as_any()
            .downcast_ref::<arrow::array::UInt16Array>()
            .unwrap();
        assert_eq!(idx.value(0), 42);
    }

    #[test]
    fn works_accumulator_schema_column_count() {
        let mut acc = WorksAccumulator::new();
        acc.push(minimal_work());
        let batch = acc.take_batch().unwrap();
        // Schema has 84 columns (82 + shard_updated_date + oa_shard_idx)
        assert_eq!(batch.num_columns(), crate::schema::works().fields().len());
    }
}
