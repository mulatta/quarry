//! PubMed XML pull-parser using quick-xml.
//!
//! Streams `<PubmedArticle>` elements and extracts fields matching
//! the Python `quarry.etl.parse` module 1:1.

use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::BufRead;

// ── Data structs ──

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

// ── Month name → number ──

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

// ── XML text extraction helpers ──

/// Read text content until the matching end tag.
/// Handles mixed content like `<ArticleTitle>Foo <i>bar</i> baz</ArticleTitle>`
/// by concatenating all text.
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


/// Parse <Year>, <Month>, <Day> children of a date element.
/// Returns ISO date string or None.
fn parse_date_children<R: BufRead>(reader: &mut Reader<R>, buf: &mut Vec<u8>) -> Option<String> {
    let mut year: Option<u32> = None;
    let mut month: u32 = 1;
    let mut day: u32 = 1;
    let mut depth: u32 = 1;

    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                match e.name().as_ref() {
                    b"Year" => {
                        let t = read_text_content(reader, buf);
                        year = t.parse().ok();
                        depth -= 1;
                    }
                    b"Month" => {
                        let t = read_text_content(reader, buf);
                        month = month_to_num(&t);
                        depth -= 1;
                    }
                    b"Day" => {
                        let t = read_text_content(reader, buf);
                        day = t.parse().unwrap_or(1);
                        depth -= 1;
                    }
                    _ => {}
                }
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

    year.map(|y| format_date(y, month, day))
}

// ── ORCID normalization ──

fn normalize_orcid(raw: &str) -> Option<String> {
    let val = raw.rsplit('/').next().unwrap_or(raw).trim();
    if val.is_empty() {
        return None;
    }
    // Insert hyphens if bare 16-digit string
    if val.len() == 16 && val.chars().all(|c| c.is_ascii_digit()) {
        Some(format!("{}-{}-{}-{}", &val[..4], &val[4..8], &val[8..12], &val[12..]))
    } else {
        Some(val.to_string())
    }
}

// ── Top-level article parser ──

/// Parse a `<PubmedArticle>` element from an already-positioned reader.
/// The reader should be positioned right after the `<PubmedArticle>` Start event.
/// Consumes events until `</PubmedArticle>` End event.
fn parse_pubmed_article_from_reader<R: BufRead>(
    reader: &mut Reader<R>,
    buf: &mut Vec<u8>,
) -> ParseResult {
    let mut result = ParseResult::default();

    let mut pmid: i32 = 0;
    let mut paper = Paper::default();
    let mut authors: Vec<Author> = Vec::new();
    let mut mesh_headings: Vec<MeshHeading> = Vec::new();
    let mut grants: Vec<Grant> = Vec::new();
    let mut chemicals: Vec<Chemical> = Vec::new();

    // Track which section we're in
    #[derive(PartialEq)]
    enum Section {
        None,
        MedlineCitation,
        Article,
        Journal,
        JournalIssue,
        PubDate,
        Abstract,
        AuthorList,
        MeshHeadingList,
        GrantList,
        ChemicalList,
        PubmedData,
        History,
        ArticleIdList,
        MedlineJournalInfo,
    }
    let mut section = Section::None;
    let mut status_attr: Option<String> = None;

    // PubDate state
    let mut pd_year: Option<u32> = None;
    let mut pd_month: u32 = 1;
    let mut pd_day: u32 = 1;
    let mut pd_medline_date: Option<String> = None;

    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"MedlineCitation" => {
                        section = Section::MedlineCitation;
                        // Extract Status attribute
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"Status" {
                                status_attr = String::from_utf8(attr.value.to_vec()).ok();
                            }
                        }
                    }
                    b"PMID" if section == Section::MedlineCitation && pmid == 0 => {
                        // Only read the first PMID (MedlineCitation direct child).
                        // Later PMIDs inside CommentsCorrections etc. must be skipped.
                        let text = read_text_content(reader, buf);
                        pmid = text.parse().unwrap_or(0);
                    }
                    b"DateRevised" if section == Section::MedlineCitation => {
                        paper.revised_date = parse_date_children(reader, buf);
                    }
                    b"DateCompleted" if section == Section::MedlineCitation => {
                        paper.indexed_date = parse_date_children(reader, buf);
                    }
                    b"Article" if section == Section::MedlineCitation => {
                        section = Section::Article;
                    }
                    b"Journal" if section == Section::Article => {
                        section = Section::Journal;
                    }
                    b"ISSN" if section == Section::Journal => {
                        paper.journal_issn = non_empty(read_text_content(reader, buf));
                    }
                    b"JournalIssue" if section == Section::Journal => {
                        section = Section::JournalIssue;
                    }
                    b"Volume" if section == Section::JournalIssue => {
                        paper.volume = non_empty(read_text_content(reader, buf));
                    }
                    b"Issue" if section == Section::JournalIssue => {
                        paper.issue = non_empty(read_text_content(reader, buf));
                    }
                    b"PubDate" if section == Section::JournalIssue => {
                        section = Section::PubDate;
                        pd_year = None;
                        pd_month = 1;
                        pd_day = 1;
                        pd_medline_date = None;
                    }
                    b"Year" if section == Section::PubDate => {
                        let t = read_text_content(reader, buf);
                        pd_year = t.parse().ok();
                    }
                    b"Month" if section == Section::PubDate => {
                        let t = read_text_content(reader, buf);
                        pd_month = month_to_num(&t);
                    }
                    b"Day" if section == Section::PubDate => {
                        let t = read_text_content(reader, buf);
                        pd_day = t.parse().unwrap_or(1);
                    }
                    b"MedlineDate" if section == Section::PubDate => {
                        pd_medline_date =
                            non_empty(read_text_content(reader, buf));
                    }
                    b"Title" if section == Section::Journal => {
                        paper.journal_title = non_empty(read_text_content(reader, buf));
                    }
                    b"ISOAbbreviation" if section == Section::Journal => {
                        paper.journal_abbr = non_empty(read_text_content(reader, buf));
                    }
                    b"ArticleTitle" if section == Section::Article => {
                        paper.title = non_empty(read_text_content(reader, buf));
                    }
                    b"Abstract" if section == Section::Article => {
                        section = Section::Abstract;
                    }
                    b"AbstractText" if section == Section::Abstract => {
                        let mut label: Option<String> = None;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"Label" {
                                label = String::from_utf8(attr.value.to_vec()).ok();
                            }
                        }
                        let text = read_text_content(reader, buf);
                        let text = text.trim().to_string();
                        if !text.is_empty() {
                            let part = if let Some(l) = label {
                                format!("{l}: {text}")
                            } else {
                                text
                            };
                            if let Some(ref mut abs) = paper.r#abstract {
                                abs.push(' ');
                                abs.push_str(&part);
                            } else {
                                paper.r#abstract = Some(part);
                            }
                        }
                    }
                    b"Language" if section == Section::Article => {
                        paper.language = non_empty(read_text_content(reader, buf));
                    }
                    b"Pagination" if section == Section::Article => {
                        // Read MedlinePgn inside
                        let mut depth: u32 = 1;
                        loop {
                            buf.clear();
                            match reader.read_event_into(buf) {
                                Ok(Event::Start(ref e2)) => {
                                    depth += 1;
                                    if e2.name().as_ref() == b"MedlinePgn" {
                                        paper.pages = non_empty(
                                            read_text_content(reader, buf),
                                        );
                                        depth -= 1;
                                    }
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
                    }
                    b"PublicationTypeList" if section == Section::Article => {
                        // Inline: read all PublicationType children
                        let mut depth: u32 = 1;
                        loop {
                            buf.clear();
                            match reader.read_event_into(buf) {
                                Ok(Event::Start(ref e2)) => {
                                    depth += 1;
                                    if e2.name().as_ref() == b"PublicationType" {
                                        let t = read_text_content(reader, buf);
                                        let t = t.trim().to_string();
                                        if !t.is_empty() {
                                            paper.pub_type.push(t);
                                        }
                                        depth -= 1;
                                    }
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
                    }
                    b"AuthorList" if section == Section::Article => {
                        section = Section::AuthorList;
                    }
                    b"Author" if section == Section::AuthorList => {
                        let auth = parse_author(reader, buf);
                        authors.push(auth);
                    }
                    b"GrantList" if section == Section::Article => {
                        section = Section::GrantList;
                    }
                    b"Grant" if section == Section::GrantList => {
                        let g = parse_grant(reader, buf);
                        grants.push(g);
                    }
                    b"MedlineJournalInfo" if section == Section::MedlineCitation => {
                        section = Section::MedlineJournalInfo;
                    }
                    b"Country" if section == Section::MedlineJournalInfo => {
                        paper.country = non_empty(read_text_content(reader, buf));
                    }
                    b"ChemicalList" if section == Section::MedlineCitation => {
                        section = Section::ChemicalList;
                    }
                    b"Chemical" if section == Section::ChemicalList => {
                        let c = parse_chemical(reader, buf);
                        chemicals.push(c);
                    }
                    b"MeshHeadingList" if section == Section::MedlineCitation => {
                        section = Section::MeshHeadingList;
                    }
                    b"MeshHeading" if section == Section::MeshHeadingList => {
                        let mh = parse_mesh_heading(reader, buf);
                        mesh_headings.extend(mh);
                    }
                    b"PubmedData" => {
                        section = Section::PubmedData;
                    }
                    b"History" if section == Section::PubmedData => {
                        section = Section::History;
                    }
                    b"PubMedPubDate" if section == Section::History => {
                        let mut pub_status: Option<String> = None;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"PubStatus" {
                                pub_status =
                                    String::from_utf8(attr.value.to_vec()).ok();
                            }
                        }
                        let d = parse_date_children(reader, buf);
                        if pub_status.as_deref() == Some("entrez") {
                            paper.created_date = d;
                        }
                    }
                    b"ArticleIdList" if section == Section::PubmedData => {
                        section = Section::ArticleIdList;
                    }
                    b"ArticleId" if section == Section::ArticleIdList => {
                        let mut id_type: Option<String> = None;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"IdType" {
                                id_type =
                                    String::from_utf8(attr.value.to_vec()).ok();
                            }
                        }
                        let val = non_empty(read_text_content(reader, buf));
                        match id_type.as_deref() {
                            Some("doi") => paper.doi = val,
                            Some("pmc") => paper.pmc_id = val,
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                match e.name().as_ref() {
                    b"PubDate" if section == Section::PubDate => {
                        // Resolve pub_year and pub_date
                        if let Some(y) = pd_year {
                            paper.pub_year = i16::try_from(y).ok();
                            paper.pub_date = Some(format_date(y, pd_month, pd_day));
                        } else if let Some(ref ml) = pd_medline_date {
                            // MedlineDate: extract first 4-digit year
                            let first4: String = ml.chars().take(4).collect();
                            if let Ok(y) = first4.parse::<u32>() {
                                paper.pub_year = i16::try_from(y).ok();
                                paper.pub_date = Some(format_date(y, 1, 1));
                            }
                        }
                        section = Section::JournalIssue;
                    }
                    b"JournalIssue" if section == Section::JournalIssue => {
                        section = Section::Journal;
                    }
                    b"Journal" if section == Section::Journal => {
                        section = Section::Article;
                    }
                    b"Abstract" if section == Section::Abstract => {
                        section = Section::Article;
                    }
                    b"AuthorList" if section == Section::AuthorList => {
                        section = Section::Article;
                    }
                    b"GrantList" if section == Section::GrantList => {
                        section = Section::Article;
                    }
                    b"Article" if section == Section::Article => {
                        section = Section::MedlineCitation;
                    }
                    b"MedlineJournalInfo" if section == Section::MedlineJournalInfo => {
                        section = Section::MedlineCitation;
                    }
                    b"ChemicalList" if section == Section::ChemicalList => {
                        section = Section::MedlineCitation;
                    }
                    b"MeshHeadingList" if section == Section::MeshHeadingList => {
                        section = Section::MedlineCitation;
                    }
                    b"MedlineCitation" if section == Section::MedlineCitation => {
                        section = Section::None;
                    }
                    b"History" if section == Section::History => {
                        section = Section::PubmedData;
                    }
                    b"ArticleIdList" if section == Section::ArticleIdList => {
                        section = Section::PubmedData;
                    }
                    b"PubmedData" if section == Section::PubmedData => {
                        section = Section::None;
                    }
                    b"PubmedArticle" => break,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    if pmid == 0 {
        return result;
    }

    paper.pmid = pmid;
    paper.medline_status = status_attr;

    // Set author PMIDs and positions
    for (i, a) in authors.iter_mut().enumerate() {
        a.pmid = pmid;
        a.author_position = i16::try_from(i + 1).unwrap_or(i16::MAX);
    }
    for mh in &mut mesh_headings {
        mh.pmid = pmid;
    }
    for g in &mut grants {
        g.pmid = pmid;
    }
    for c in &mut chemicals {
        c.pmid = pmid;
    }

    result.papers.push(paper);
    result.authors = authors;
    result.mesh_headings = mesh_headings;
    result.grants = grants;
    result.chemicals = chemicals;
    result
}

fn parse_author<R: BufRead>(reader: &mut Reader<R>, buf: &mut Vec<u8>) -> Author {
    let mut auth = Author::default();
    let mut depth: u32 = 1;

    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                match e.name().as_ref() {
                    b"LastName" => {
                        auth.last_name = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"ForeName" => {
                        auth.fore_name = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"Initials" => {
                        auth.initials = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"CollectiveName" => {
                        auth.is_collective = true;
                        auth.last_name = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"Identifier" => {
                        let mut is_orcid = false;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"Source" && attr.value.as_ref() == b"ORCID" {
                                is_orcid = true;
                            }
                        }
                        let val = read_text_content(reader, buf);
                        depth -= 1;
                        if is_orcid {
                            auth.orcid = normalize_orcid(&val);
                        }
                    }
                    b"AffiliationInfo" => {
                        // Read first Affiliation child
                        let mut d2: u32 = 1;
                        loop {
                            buf.clear();
                            match reader.read_event_into(buf) {
                                Ok(Event::Start(ref e2)) => {
                                    d2 += 1;
                                    if e2.name().as_ref() == b"Affiliation"
                                        && auth.affiliation.is_none()
                                    {
                                        auth.affiliation =
                                            non_empty(read_text_content(reader, buf));
                                        d2 -= 1;
                                    }
                                }
                                Ok(Event::End(_)) => {
                                    d2 -= 1;
                                    if d2 == 0 {
                                        break;
                                    }
                                }
                                Ok(Event::Eof) => break,
                                Err(_) => break,
                                _ => {}
                            }
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
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
    auth
}

fn parse_mesh_heading<R: BufRead>(
    reader: &mut Reader<R>,
    buf: &mut Vec<u8>,
) -> Vec<MeshHeading> {
    let mut desc_ui = String::new();
    let mut desc_name = String::new();
    let mut desc_major = false;
    let mut qualifiers: Vec<(String, String, bool)> = Vec::new();
    let mut depth: u32 = 1;

    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                match e.name().as_ref() {
                    b"DescriptorName" => {
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"UI" => {
                                    desc_ui =
                                        String::from_utf8(attr.value.to_vec()).unwrap_or_default();
                                }
                                b"MajorTopicYN" => {
                                    desc_major = attr.value.as_ref() == b"Y";
                                }
                                _ => {}
                            }
                        }
                        desc_name = read_text_content(reader, buf).trim().to_string();
                        depth -= 1;
                    }
                    b"QualifierName" => {
                        let mut q_ui = String::new();
                        let mut q_major = false;
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"UI" => {
                                    q_ui = String::from_utf8(attr.value.to_vec())
                                        .unwrap_or_default();
                                }
                                b"MajorTopicYN" => {
                                    q_major = attr.value.as_ref() == b"Y";
                                }
                                _ => {}
                            }
                        }
                        let q_name = read_text_content(reader, buf).trim().to_string();
                        depth -= 1;
                        qualifiers.push((q_ui, q_name, q_major));
                    }
                    _ => {}
                }
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

    if qualifiers.is_empty() {
        vec![MeshHeading {
            pmid: 0,
            descriptor_ui: desc_ui,
            descriptor_name: desc_name,
            qualifier_ui: None,
            qualifier_name: None,
            is_major_topic: desc_major,
        }]
    } else {
        qualifiers
            .into_iter()
            .map(|(q_ui, q_name, q_major)| MeshHeading {
                pmid: 0,
                descriptor_ui: desc_ui.clone(),
                descriptor_name: desc_name.clone(),
                qualifier_ui: Some(q_ui),
                qualifier_name: Some(q_name),
                is_major_topic: q_major,
            })
            .collect()
    }
}

fn parse_grant<R: BufRead>(reader: &mut Reader<R>, buf: &mut Vec<u8>) -> Grant {
    let mut g = Grant::default();
    let mut depth: u32 = 1;

    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                match e.name().as_ref() {
                    b"GrantID" => {
                        g.grant_id = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"Acronym" => {
                        g.acronym = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"Agency" => {
                        g.agency = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"Country" => {
                        g.country = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    _ => {}
                }
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
    g
}

fn parse_chemical<R: BufRead>(reader: &mut Reader<R>, buf: &mut Vec<u8>) -> Chemical {
    let mut c = Chemical::default();
    let mut depth: u32 = 1;

    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                match e.name().as_ref() {
                    b"RegistryNumber" => {
                        c.registry_number = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    b"NameOfSubstance" => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"UI" {
                                c.substance_ui = non_empty(
                                    String::from_utf8(attr.value.to_vec()).unwrap_or_default(),
                                );
                            }
                        }
                        c.substance_name = non_empty(read_text_content(reader, buf));
                        depth -= 1;
                    }
                    _ => {}
                }
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
    c
}

// ── File-level streaming parser ──

/// Parse a full PubMed XML file (possibly gzipped) and return accumulated ParseResult.
/// Splits on `<PubmedArticle>` boundaries using a two-pass approach:
/// 1. Stream gzip/plain → scan for `<PubmedArticle>` ... `</PubmedArticle>` boundaries
/// 2. Parse each article block independently
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

pub fn parse_from_reader<R: BufRead>(reader: R) -> std::io::Result<ParseResult> {
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.config_mut().trim_text(true);
    let mut buf = Vec::with_capacity(4096);
    let mut result = ParseResult::default();

    loop {
        buf.clear();
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"PubmedArticle" => {
                        // Directly parse from the stream — no capture/reparse
                        let article_result =
                            parse_pubmed_article_from_reader(&mut xml_reader, &mut buf);
                        result.extend(article_result);
                    }
                    b"DeleteCitation" => {
                        // Read all PMID children
                        let mut depth: u32 = 1;
                        loop {
                            buf.clear();
                            match xml_reader.read_event_into(&mut buf) {
                                Ok(Event::Start(ref e2)) => {
                                    depth += 1;
                                    if e2.name().as_ref() == b"PMID" {
                                        let text = read_text_content(&mut xml_reader, &mut buf);
                                        if let Ok(pmid) = text.trim().parse::<i32>() {
                                            result.delete_pmids.push(pmid);
                                        }
                                        depth -= 1;
                                    }
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
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    Ok(result)
}
