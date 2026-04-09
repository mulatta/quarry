//! PubMed XML parser using hybrid SAX + serde deserialization.
//!
//! Streams `<PubmedArticle>` elements: SAX detects record boundaries,
//! `capture_element()` grabs raw XML, then serde deserializes structurally.
//! Mixed-content fields (ArticleTitle, AbstractText) are extracted via SAX
//! from the captured raw XML to preserve inline HTML text (e.g. `<i>`, `<sup>`).

use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;
use std::io::BufRead;

use super::capture_element;

// ── Output structs (unchanged — consumed by parquet_writer) ──

#[derive(Debug, Default)]
pub struct Paper {
    pub pmid: i32,
    pub doi: Option<String>,
    pub pmc_id: Option<String>,
    pub title: Option<String>,
    pub r#abstract: Option<String>,
    pub pub_year: Option<i16>,
    pub pub_date: Option<String>, // ISO "YYYY-MM-DD"
    pub journal_title: Option<String>,
    pub journal_issn: Option<String>,
    pub journal_abbr: Option<String>,
    pub volume: Option<String>,
    pub issue: Option<String>,
    pub pages: Option<String>,
    pub language: Option<String>,
    pub pub_type: Vec<String>,
    pub country: Option<String>,
    pub medline_status: Option<String>,
    pub created_date: Option<String>,
    pub revised_date: Option<String>,
    pub indexed_date: Option<String>,
}

#[derive(Debug, Default)]
pub struct Author {
    pub pmid: i32,
    pub author_position: i16,
    pub last_name: Option<String>,
    pub fore_name: Option<String>,
    pub initials: Option<String>,
    pub orcid: Option<String>,
    pub affiliation: Option<String>,
    pub is_collective: bool,
}

#[derive(Debug, Default)]
pub struct MeshHeading {
    pub pmid: i32,
    pub descriptor_ui: String,
    pub descriptor_name: String,
    pub qualifier_ui: Option<String>,
    pub qualifier_name: Option<String>,
    pub is_major_topic: bool,
}

#[derive(Debug, Default)]
pub struct Grant {
    pub pmid: i32,
    pub grant_id: Option<String>,
    pub acronym: Option<String>,
    pub agency: Option<String>,
    pub country: Option<String>,
}

#[derive(Debug, Default)]
pub struct Chemical {
    pub pmid: i32,
    pub registry_number: Option<String>,
    pub substance_ui: Option<String>,
    pub substance_name: Option<String>,
}

#[derive(Debug, Default)]
pub struct ParseResult {
    pub papers: Vec<Paper>,
    pub authors: Vec<Author>,
    pub mesh_headings: Vec<MeshHeading>,
    pub grants: Vec<Grant>,
    pub chemicals: Vec<Chemical>,
    pub delete_pmids: Vec<i32>,
}

impl ParseResult {
    pub fn extend(&mut self, other: ParseResult) {
        self.papers.extend(other.papers);
        self.authors.extend(other.authors);
        self.mesh_headings.extend(other.mesh_headings);
        self.grants.extend(other.grants);
        self.chemicals.extend(other.chemicals);
        self.delete_pmids.extend(other.delete_pmids);
    }
}

// ── Serde deserialization structs ──
//
// Structural fields are deserialized via serde. Mixed-content fields
// (ArticleTitle, AbstractText) are handled separately via SAX extraction
// from the raw XML to preserve inline HTML text.

#[derive(Deserialize, Debug)]
struct XmlPubmedArticle {
    #[serde(rename = "MedlineCitation")]
    medline_citation: XmlMedlineCitation,
    #[serde(rename = "PubmedData", default)]
    pubmed_data: Option<XmlPubmedData>,
}

#[derive(Deserialize, Debug)]
struct XmlMedlineCitation {
    #[serde(rename = "@Status", default)]
    status: Option<String>,
    #[serde(rename = "PMID")]
    pmid: XmlPmid,
    #[serde(rename = "DateRevised", default)]
    date_revised: Option<XmlDate>,
    #[serde(rename = "DateCompleted", default)]
    date_completed: Option<XmlDate>,
    #[serde(rename = "Article")]
    article: XmlArticle,
    #[serde(rename = "MedlineJournalInfo", default)]
    medline_journal_info: Option<XmlMedlineJournalInfo>,
    #[serde(rename = "ChemicalList", default)]
    chemical_list: Option<XmlChemicalList>,
    #[serde(rename = "MeshHeadingList", default)]
    mesh_heading_list: Option<XmlMeshHeadingList>,
    // CommentsCorrectionsList, KeywordList, etc. are silently ignored by serde.
}

#[derive(Deserialize, Debug)]
struct XmlPmid {
    #[serde(rename = "$text")]
    value: String,
}

#[derive(Deserialize, Debug, Default)]
struct XmlDate {
    #[serde(rename = "Year", default)]
    year: Option<String>,
    #[serde(rename = "Month", default)]
    month: Option<String>,
    #[serde(rename = "Day", default)]
    day: Option<String>,
    #[serde(rename = "MedlineDate", default)]
    medline_date: Option<String>,
}

#[derive(Deserialize, Debug)]
struct XmlArticle {
    #[serde(rename = "Journal")]
    journal: XmlJournal,
    // ArticleTitle: extracted via SAX (mixed content), not serde
    #[serde(rename = "Language", default)]
    language: Option<String>,
    #[serde(rename = "AuthorList", default)]
    author_list: Option<XmlAuthorList>,
    #[serde(rename = "GrantList", default)]
    grant_list: Option<XmlGrantList>,
    #[serde(rename = "Pagination", default)]
    pagination: Option<XmlPagination>,
    #[serde(rename = "PublicationTypeList", default)]
    publication_type_list: Option<XmlPublicationTypeList>,
    // Abstract: extracted via SAX (mixed content), not serde
}

#[derive(Deserialize, Debug)]
struct XmlJournal {
    #[serde(rename = "ISSN", default)]
    issn: Option<XmlIssn>,
    #[serde(rename = "JournalIssue", default)]
    journal_issue: Option<XmlJournalIssue>,
    #[serde(rename = "Title", default)]
    title: Option<String>,
    #[serde(rename = "ISOAbbreviation", default)]
    iso_abbreviation: Option<String>,
}

#[derive(Deserialize, Debug)]
struct XmlIssn {
    #[serde(rename = "$text")]
    value: String,
}

#[derive(Deserialize, Debug)]
struct XmlJournalIssue {
    #[serde(rename = "Volume", default)]
    volume: Option<String>,
    #[serde(rename = "Issue", default)]
    issue: Option<String>,
    #[serde(rename = "PubDate", default)]
    pub_date: Option<XmlDate>,
}

#[derive(Deserialize, Debug)]
struct XmlPagination {
    #[serde(rename = "MedlinePgn", default)]
    medline_pgn: Option<String>,
}

#[derive(Deserialize, Debug)]
struct XmlPublicationTypeList {
    #[serde(rename = "PublicationType", default)]
    types: Vec<XmlPublicationType>,
}

#[derive(Deserialize, Debug)]
struct XmlPublicationType {
    #[serde(rename = "$text")]
    name: String,
}

#[derive(Deserialize, Debug)]
struct XmlAuthorList {
    #[serde(rename = "Author", default)]
    authors: Vec<XmlAuthor>,
}

#[derive(Deserialize, Debug)]
struct XmlAuthor {
    #[serde(rename = "LastName", default)]
    last_name: Option<String>,
    #[serde(rename = "ForeName", default)]
    fore_name: Option<String>,
    #[serde(rename = "Initials", default)]
    initials: Option<String>,
    #[serde(rename = "CollectiveName", default)]
    collective_name: Option<String>,
    #[serde(rename = "Identifier", default)]
    identifiers: Vec<XmlIdentifier>,
    #[serde(rename = "AffiliationInfo", default)]
    affiliation_info: Vec<XmlAffiliationInfo>,
}

#[derive(Deserialize, Debug)]
struct XmlIdentifier {
    #[serde(rename = "@Source", default)]
    source: Option<String>,
    #[serde(rename = "$text")]
    value: String,
}

#[derive(Deserialize, Debug)]
struct XmlAffiliationInfo {
    #[serde(rename = "Affiliation", default)]
    affiliation: Option<String>,
}

#[derive(Deserialize, Debug)]
struct XmlGrantList {
    #[serde(rename = "Grant", default)]
    grants: Vec<XmlGrant>,
}

#[derive(Deserialize, Debug)]
struct XmlGrant {
    #[serde(rename = "GrantID", default)]
    grant_id: Option<String>,
    #[serde(rename = "Acronym", default)]
    acronym: Option<String>,
    #[serde(rename = "Agency", default)]
    agency: Option<String>,
    #[serde(rename = "Country", default)]
    country: Option<String>,
}

#[derive(Deserialize, Debug)]
struct XmlMedlineJournalInfo {
    #[serde(rename = "Country", default)]
    country: Option<String>,
}

#[derive(Deserialize, Debug)]
struct XmlChemicalList {
    #[serde(rename = "Chemical", default)]
    chemicals: Vec<XmlChemical>,
}

#[derive(Deserialize, Debug)]
struct XmlChemical {
    #[serde(rename = "RegistryNumber", default)]
    registry_number: Option<String>,
    #[serde(rename = "NameOfSubstance", default)]
    name_of_substance: Option<XmlNameOfSubstance>,
}

#[derive(Deserialize, Debug)]
struct XmlNameOfSubstance {
    #[serde(rename = "@UI", default)]
    ui: Option<String>,
    #[serde(rename = "$text")]
    name: String,
}

#[derive(Deserialize, Debug)]
struct XmlMeshHeadingList {
    #[serde(rename = "MeshHeading", default)]
    headings: Vec<XmlMeshHeading>,
}

#[derive(Deserialize, Debug)]
struct XmlMeshHeading {
    #[serde(rename = "DescriptorName")]
    descriptor_name: XmlDescriptorName,
    #[serde(rename = "QualifierName", default)]
    qualifiers: Vec<XmlQualifierName>,
}

#[derive(Deserialize, Debug)]
struct XmlDescriptorName {
    #[serde(rename = "@UI")]
    ui: String,
    #[serde(rename = "@MajorTopicYN", default)]
    major_topic_yn: Option<String>,
    #[serde(rename = "$text")]
    name: String,
}

#[derive(Deserialize, Debug)]
struct XmlQualifierName {
    #[serde(rename = "@UI")]
    ui: String,
    #[serde(rename = "@MajorTopicYN", default)]
    major_topic_yn: Option<String>,
    #[serde(rename = "$text")]
    name: String,
}

#[derive(Deserialize, Debug)]
struct XmlPubmedData {
    #[serde(rename = "History", default)]
    history: Option<XmlHistory>,
    #[serde(rename = "ArticleIdList", default)]
    article_id_list: Option<XmlArticleIdList>,
}

#[derive(Deserialize, Debug)]
struct XmlHistory {
    #[serde(rename = "PubMedPubDate", default)]
    dates: Vec<XmlPubMedPubDate>,
}

#[derive(Deserialize, Debug)]
struct XmlPubMedPubDate {
    #[serde(rename = "@PubStatus")]
    pub_status: String,
    #[serde(rename = "Year", default)]
    year: Option<String>,
    #[serde(rename = "Month", default)]
    month: Option<String>,
    #[serde(rename = "Day", default)]
    day: Option<String>,
}

#[derive(Deserialize, Debug)]
struct XmlArticleIdList {
    #[serde(rename = "ArticleId", default)]
    ids: Vec<XmlArticleId>,
}

#[derive(Deserialize, Debug)]
struct XmlArticleId {
    #[serde(rename = "@IdType")]
    id_type: String,
    #[serde(rename = "$text")]
    value: String,
}

#[derive(Deserialize, Debug)]
struct XmlDeleteCitation {
    #[serde(rename = "PMID", default)]
    pmids: Vec<XmlPmid>,
}

// ── Helper functions ──

fn month_to_num(s: &str) -> u32 {
    match s.get(..3).map(|m| m.to_ascii_lowercase()).as_deref() {
        Some("jan") => 1,
        Some("feb") => 2,
        Some("mar") => 3,
        Some("apr") => 4,
        Some("may") => 5,
        Some("jun") => 6,
        Some("jul") => 7,
        Some("aug") => 8,
        Some("sep") => 9,
        Some("oct") => 10,
        Some("nov") => 11,
        Some("dec") => 12,
        _ => s.parse().unwrap_or(1),
    }
}

fn format_date(y: u32, m: u32, d: u32) -> String {
    format!("{y:04}-{m:02}-{d:02}")
}

fn non_empty(s: String) -> Option<String> {
    let trimmed = s.trim().to_string();
    if trimmed.is_empty() { None } else { Some(trimmed) }
}

fn non_empty_ref(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
}

fn normalize_orcid(raw: &str) -> Option<String> {
    let val = raw.rsplit('/').next().unwrap_or(raw).trim();
    if val.is_empty() {
        return None;
    }
    if val.len() == 16 && val.chars().all(|c| c.is_ascii_digit()) {
        Some(format!("{}-{}-{}-{}", &val[..4], &val[4..8], &val[8..12], &val[12..]))
    } else {
        Some(val.to_string())
    }
}

/// Read text content until the matching end tag.
/// Handles mixed content like `<ArticleTitle>Foo <i>bar</i> baz</ArticleTitle>`
/// by concatenating all text nodes at all depths.
fn read_text_content<R: BufRead>(reader: &mut Reader<R>, buf: &mut Vec<u8>) -> String {
    let mut text = String::new();
    let mut depth: u32 = 1;
    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Text(e)) => {
                if let Ok(s) = e.unescape() {
                    text.push_str(&s);
                }
            }
            Ok(Event::CData(e)) => {
                text.push_str(&String::from_utf8_lossy(e.as_ref()));
            }
            Ok(Event::Start(_)) => {
                depth += 1;
            }
            Ok(Event::End(_)) => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }
    text.trim().to_string()
}

// ── Mixed content extraction from raw XML ──

/// Extract title and abstract from raw PubmedArticle XML via SAX.
/// These fields contain mixed content (`<i>`, `<sup>`, etc.) that serde
/// cannot handle without losing text from inline elements.
fn extract_mixed_content(raw_xml: &str) -> (Option<String>, Option<String>) {
    let mut reader = Reader::from_str(raw_xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::with_capacity(1024);

    let mut title: Option<String> = None;
    let mut r#abstract: Option<String> = None;
    let mut in_article = false;
    let mut in_abstract = false;

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"Article" => in_article = true,
                b"ArticleTitle" if in_article && title.is_none() => {
                    title = non_empty(read_text_content(&mut reader, &mut buf));
                }
                b"Abstract" if in_article => in_abstract = true,
                b"AbstractText" if in_abstract => {
                    let mut label: Option<String> = None;
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"Label" {
                            label = String::from_utf8(attr.value.to_vec()).ok();
                        }
                    }
                    let text = read_text_content(&mut reader, &mut buf);
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        let part = if let Some(l) = label {
                            format!("{l}: {text}")
                        } else {
                            text
                        };
                        if let Some(ref mut abs) = r#abstract {
                            abs.push(' ');
                            abs.push_str(&part);
                        } else {
                            r#abstract = Some(part);
                        }
                    }
                }
                _ => {}
            },
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"Abstract" => in_abstract = false,
                b"Article" => break, // no need to parse beyond </Article>
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    (title, r#abstract)
}

// ── Date resolution helpers ──

fn resolve_date(d: &XmlDate) -> Option<String> {
    let year: u32 = d.year.as_ref()?.parse().ok()?;
    let month = d.month.as_deref().map(month_to_num).unwrap_or(1);
    let day: u32 = d.day.as_deref().and_then(|s| s.parse().ok()).unwrap_or(1);
    Some(format_date(year, month, day))
}

fn resolve_pub_date(d: &XmlDate) -> (Option<i16>, Option<String>) {
    if let Some(ref y_str) = d.year {
        if let Ok(y) = y_str.parse::<u32>() {
            let month = d.month.as_deref().map(month_to_num).unwrap_or(1);
            let day: u32 = d.day.as_deref().and_then(|s| s.parse().ok()).unwrap_or(1);
            return (i16::try_from(y).ok(), Some(format_date(y, month, day)));
        }
    }
    if let Some(ref ml) = d.medline_date {
        let first4: String = ml.chars().take(4).collect();
        if let Ok(y) = first4.parse::<u32>() {
            return (i16::try_from(y).ok(), Some(format_date(y, 1, 1)));
        }
    }
    (None, None)
}

// ── Conversion: serde structs → output structs ──

fn convert_article(xml: XmlPubmedArticle, title: Option<String>, r#abstract: Option<String>) -> ParseResult {
    let mut result = ParseResult::default();
    let mc = xml.medline_citation;

    let pmid: i32 = mc.pmid.value.trim().parse().unwrap_or(0);
    if pmid == 0 {
        return result;
    }

    let mut paper = Paper::default();
    paper.pmid = pmid;
    paper.medline_status = mc.status;
    paper.title = title;
    paper.r#abstract = r#abstract;

    // Dates
    paper.revised_date = mc.date_revised.as_ref().and_then(resolve_date);
    paper.indexed_date = mc.date_completed.as_ref().and_then(resolve_date);

    // Journal
    let j = &mc.article.journal;
    paper.journal_issn = j.issn.as_ref().and_then(|i| non_empty_ref(&i.value));
    paper.journal_title = j.title.as_deref().and_then(non_empty_ref);
    paper.journal_abbr = j.iso_abbreviation.as_deref().and_then(non_empty_ref);

    if let Some(ji) = &j.journal_issue {
        paper.volume = ji.volume.as_deref().and_then(non_empty_ref);
        paper.issue = ji.issue.as_deref().and_then(non_empty_ref);
        if let Some(pd) = &ji.pub_date {
            let (pub_year, pub_date) = resolve_pub_date(pd);
            paper.pub_year = pub_year;
            paper.pub_date = pub_date;
        }
    }

    // Language, pages, pub types
    paper.language = mc.article.language.as_deref().and_then(non_empty_ref);
    if let Some(pag) = &mc.article.pagination {
        paper.pages = pag.medline_pgn.as_deref().and_then(non_empty_ref);
    }
    if let Some(ptl) = &mc.article.publication_type_list {
        paper.pub_type = ptl
            .types
            .iter()
            .map(|pt| pt.name.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    // MedlineJournalInfo -> Country
    paper.country = mc
        .medline_journal_info
        .as_ref()
        .and_then(|mji| mji.country.as_deref())
        .and_then(non_empty_ref);

    // PubmedData
    if let Some(pd) = &xml.pubmed_data {
        if let Some(hist) = &pd.history {
            for d in &hist.dates {
                if d.pub_status == "entrez" {
                    let year: u32 = match d.year.as_ref().and_then(|y| y.parse().ok()) {
                        Some(y) => y,
                        None => continue,
                    };
                    let month = d.month.as_deref().map(month_to_num).unwrap_or(1);
                    let day: u32 = d.day.as_deref().and_then(|s| s.parse().ok()).unwrap_or(1);
                    paper.created_date = Some(format_date(year, month, day));
                }
            }
        }
        if let Some(ail) = &pd.article_id_list {
            for aid in &ail.ids {
                let val = non_empty_ref(&aid.value);
                match aid.id_type.as_str() {
                    "doi" => paper.doi = val,
                    "pmc" => paper.pmc_id = val,
                    _ => {}
                }
            }
        }
    }

    result.papers.push(paper);

    // Authors
    if let Some(al) = &mc.article.author_list {
        for (i, xa) in al.authors.iter().enumerate() {
            let mut auth = Author::default();
            auth.pmid = pmid;
            auth.author_position = i16::try_from(i + 1).unwrap_or(i16::MAX);
            if let Some(cn) = &xa.collective_name {
                auth.is_collective = true;
                auth.last_name = non_empty_ref(cn);
            } else {
                auth.last_name = xa.last_name.as_deref().and_then(non_empty_ref);
                auth.fore_name = xa.fore_name.as_deref().and_then(non_empty_ref);
            }
            auth.initials = xa.initials.as_deref().and_then(non_empty_ref);
            for id in &xa.identifiers {
                if id.source.as_deref() == Some("ORCID") {
                    auth.orcid = normalize_orcid(&id.value);
                }
            }
            if let Some(ai) = xa.affiliation_info.first() {
                auth.affiliation = ai.affiliation.as_deref().and_then(non_empty_ref);
            }
            result.authors.push(auth);
        }
    }

    // MeSH headings
    if let Some(mhl) = &mc.mesh_heading_list {
        for mh in &mhl.headings {
            let desc_ui = mh.descriptor_name.ui.clone();
            let desc_name = mh.descriptor_name.name.trim().to_string();
            let desc_major = mh.descriptor_name.major_topic_yn.as_deref() == Some("Y");

            if mh.qualifiers.is_empty() {
                result.mesh_headings.push(MeshHeading {
                    pmid,
                    descriptor_ui: desc_ui,
                    descriptor_name: desc_name,
                    qualifier_ui: None,
                    qualifier_name: None,
                    is_major_topic: desc_major,
                });
            } else {
                for q in &mh.qualifiers {
                    result.mesh_headings.push(MeshHeading {
                        pmid,
                        descriptor_ui: desc_ui.clone(),
                        descriptor_name: desc_name.clone(),
                        qualifier_ui: Some(q.ui.clone()),
                        qualifier_name: Some(q.name.trim().to_string()),
                        is_major_topic: q.major_topic_yn.as_deref() == Some("Y"),
                    });
                }
            }
        }
    }

    // Grants
    if let Some(gl) = &mc.article.grant_list {
        for xg in &gl.grants {
            result.grants.push(Grant {
                pmid,
                grant_id: xg.grant_id.as_deref().and_then(non_empty_ref),
                acronym: xg.acronym.as_deref().and_then(non_empty_ref),
                agency: xg.agency.as_deref().and_then(non_empty_ref),
                country: xg.country.as_deref().and_then(non_empty_ref),
            });
        }
    }

    // Chemicals
    if let Some(cl) = &mc.chemical_list {
        for xc in &cl.chemicals {
            result.chemicals.push(Chemical {
                pmid,
                registry_number: xc.registry_number.as_deref().and_then(non_empty_ref),
                substance_ui: xc
                    .name_of_substance
                    .as_ref()
                    .and_then(|n| n.ui.as_deref())
                    .and_then(non_empty_ref),
                substance_name: xc
                    .name_of_substance
                    .as_ref()
                    .map(|n| n.name.as_str())
                    .and_then(non_empty_ref),
            });
        }
    }

    result
}

// ── File-level streaming parser ──

/// Parse a full PubMed XML file (possibly gzipped).
pub fn parse_file(path: &std::path::Path) -> std::io::Result<ParseResult> {
    let file = std::fs::File::open(path)?;
    let is_gz = path.extension().is_some_and(|ext| ext == "gz");

    let reader: Box<dyn BufRead> = if is_gz {
        Box::new(std::io::BufReader::new(flate2::read::GzDecoder::new(file)))
    } else {
        Box::new(std::io::BufReader::new(file))
    };

    parse_from_reader(reader)
}

/// Parse PubMed XML from a reader. Scans for `<PubmedArticle>` and
/// `<DeleteCitation>` boundaries via SAX, then uses serde for each record.
pub fn parse_from_reader<R: BufRead>(reader: R) -> std::io::Result<ParseResult> {
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.config_mut().trim_text(true);
    let mut buf = Vec::with_capacity(4096);
    let mut result = ParseResult::default();

    loop {
        buf.clear();
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"PubmedArticle" => {
                    let owned = e.to_owned();
                    let raw = match capture_element(&mut xml_reader, &owned, &mut buf) {
                        Ok(r) => r,
                        Err(err) => {
                            eprintln!("pubmed: WARN: capture failed: {err}");
                            continue;
                        }
                    };

                    // Extract mixed-content fields via SAX
                    let (title, r#abstract) = extract_mixed_content(&raw);

                    // Deserialize structural fields via serde
                    match quick_xml::de::from_str::<XmlPubmedArticle>(&raw) {
                        Ok(article) => {
                            let article_result = convert_article(article, title, r#abstract);
                            result.extend(article_result);
                        }
                        Err(err) => {
                            eprintln!("pubmed: WARN: serde failed: {err}");
                        }
                    }
                }
                b"DeleteCitation" => {
                    let owned = e.to_owned();
                    let raw = match capture_element(&mut xml_reader, &owned, &mut buf) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    if let Ok(dc) = quick_xml::de::from_str::<XmlDeleteCitation>(&raw) {
                        for pmid_elem in dc.pmids {
                            if let Ok(pmid) = pmid_elem.value.trim().parse::<i32>() {
                                result.delete_pmids.push(pmid);
                            }
                        }
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    Ok(result)
}
