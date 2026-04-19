//! Unit tests for PubMed XML parser.
//!
//! Tests parse_from_reader with inline XML strings covering edge cases
//! that the integration test fixtures don't reach.

#[cfg(test)]
mod tests {
    use crate::parse::xml::*;

    /// Helper: wrap XML content in a minimal PubmedArticleSet.
    fn parse_xml(xml: &str) -> ParseResult {
        let wrapped = format!(
            r#"<?xml version="1.0"?>
<!DOCTYPE PubmedArticleSet>
<PubmedArticleSet>
{xml}
</PubmedArticleSet>"#
        );
        parse_from_reader(std::io::Cursor::new(wrapped)).unwrap()
    }

    /// Helper: build a minimal <PubmedArticle> with PMID and optional body.
    fn article(pmid: i32, body: &str) -> String {
        format!(
            r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID Version="1">{pmid}</PMID>
    <Article>
      <Journal>
        <JournalIssue>
          <PubDate><Year>2024</Year></PubDate>
        </JournalIssue>
      </Journal>
      <ArticleTitle>Test Title</ArticleTitle>
      {body}
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList/>
  </PubmedData>
</PubmedArticle>"#
        )
    }

    // ── Basic parsing ──

    #[test]
    fn test_minimal_article() {
        let result = parse_xml(&article(12345, ""));
        assert_eq!(result.papers.len(), 1);
        let p = &result.papers[0];
        assert_eq!(p.pmid, 12345);
        assert_eq!(p.title.as_deref(), Some("Test Title"));
        assert_eq!(p.pub_year, Some(2024));
        assert_eq!(p.medline_status.as_deref(), Some("MEDLINE"));
    }

    #[test]
    fn test_zero_pmid_skipped() {
        // PMID=0 should be skipped (treated as invalid)
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID Version="1">0</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>Bad</ArticleTitle>
    </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        assert!(result.papers.is_empty());
    }

    // ── Month parsing ──

    #[test]
    fn test_month_name_parsing() {
        let body = "";
        let xml = format!(
            r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>1</PMID>
    <Article>
      <Journal><JournalIssue>
        <PubDate><Year>2023</Year><Month>Mar</Month><Day>15</Day></PubDate>
      </JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
      {body}
    </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#
        );
        let result = parse_xml(&xml);
        let p = &result.papers[0];
        assert_eq!(p.pub_date.as_deref(), Some("2023-03-15"));
    }

    #[test]
    fn test_month_numeric_string() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>2</PMID>
    <Article>
      <Journal><JournalIssue>
        <PubDate><Year>2023</Year><Month>11</Month></PubDate>
      </JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        assert_eq!(result.papers[0].pub_date.as_deref(), Some("2023-11-01"));
    }

    // ── MedlineDate ──

    #[test]
    fn test_medline_date_fallback() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>3</PMID>
    <Article>
      <Journal><JournalIssue>
        <PubDate><MedlineDate>2022 Jan-Feb</MedlineDate></PubDate>
      </JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        let p = &result.papers[0];
        assert_eq!(p.pub_year, Some(2022));
        assert_eq!(p.pub_date.as_deref(), Some("2022-01-01"));
    }

    // ── Mixed content (inline HTML in title) ──

    #[test]
    fn test_mixed_content_title() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>4</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>Effect of <i>Helicobacter pylori</i> on gastric <sup>pH</sup></ArticleTitle>
    </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        let title = result.papers[0].title.as_deref().unwrap();
        assert!(title.contains("Helicobacter pylori"));
        assert!(title.contains("pH"));
    }

    // ── Structured abstract with labels ──

    #[test]
    fn test_structured_abstract() {
        let body = r#"<Abstract>
        <AbstractText Label="BACKGROUND">Background text.</AbstractText>
        <AbstractText Label="METHODS">Methods text.</AbstractText>
      </Abstract>"#;
        let result = parse_xml(&article(5, body));
        let abs = result.papers[0].r#abstract.as_deref().unwrap();
        assert!(abs.contains("BACKGROUND: Background text."));
        assert!(abs.contains("METHODS: Methods text."));
    }

    #[test]
    fn test_unstructured_abstract() {
        let body = r#"<Abstract>
        <AbstractText>A single paragraph abstract.</AbstractText>
      </Abstract>"#;
        let result = parse_xml(&article(6, body));
        let abs = result.papers[0].r#abstract.as_deref().unwrap();
        assert_eq!(abs, "A single paragraph abstract.");
    }

    #[test]
    fn test_empty_abstract() {
        let body = r#"<Abstract>
        <AbstractText></AbstractText>
      </Abstract>"#;
        let result = parse_xml(&article(7, body));
        assert!(result.papers[0].r#abstract.is_none());
    }

    // ── Authors ──

    #[test]
    fn test_collective_name_author() {
        let body = r#"<AuthorList>
        <Author>
          <CollectiveName>WHO Working Group</CollectiveName>
        </Author>
      </AuthorList>"#;
        let result = parse_xml(&article(8, body));
        assert_eq!(result.authors.len(), 1);
        let a = &result.authors[0];
        assert!(a.is_collective);
        assert_eq!(a.last_name.as_deref(), Some("WHO Working Group"));
        assert_eq!(a.pmid, 8);
        assert_eq!(a.author_position, 1);
    }

    #[test]
    fn test_author_orcid_url() {
        let body = r#"<AuthorList>
        <Author>
          <LastName>Smith</LastName>
          <ForeName>John</ForeName>
          <Identifier Source="ORCID">https://orcid.org/0000-0002-1234-5678</Identifier>
        </Author>
      </AuthorList>"#;
        let result = parse_xml(&article(9, body));
        assert_eq!(
            result.authors[0].orcid.as_deref(),
            Some("0000-0002-1234-5678")
        );
    }

    #[test]
    fn test_author_orcid_bare_digits() {
        let body = r#"<AuthorList>
        <Author>
          <LastName>Kim</LastName>
          <Identifier Source="ORCID">0000000212345678</Identifier>
        </Author>
      </AuthorList>"#;
        let result = parse_xml(&article(10, body));
        assert_eq!(
            result.authors[0].orcid.as_deref(),
            Some("0000-0002-1234-5678")
        );
    }

    #[test]
    fn test_multiple_authors_position() {
        let body = r#"<AuthorList>
        <Author><LastName>A</LastName></Author>
        <Author><LastName>B</LastName></Author>
        <Author><LastName>C</LastName></Author>
      </AuthorList>"#;
        let result = parse_xml(&article(11, body));
        assert_eq!(result.authors.len(), 3);
        assert_eq!(result.authors[0].author_position, 1);
        assert_eq!(result.authors[1].author_position, 2);
        assert_eq!(result.authors[2].author_position, 3);
    }

    // ── MeSH headings ──

    #[test]
    fn test_mesh_descriptor_only() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>20</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
    <MeshHeadingList>
      <MeshHeading>
        <DescriptorName UI="D000900" MajorTopicYN="N">Anti-Bacterial Agents</DescriptorName>
      </MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        assert_eq!(result.mesh_headings.len(), 1);
        let mh = &result.mesh_headings[0];
        assert_eq!(mh.pmid, 20);
        assert_eq!(mh.descriptor_ui, "D000900");
        assert_eq!(mh.descriptor_name, "Anti-Bacterial Agents");
        assert!(mh.qualifier_ui.is_none());
        assert!(!mh.is_major_topic);
    }

    #[test]
    fn test_mesh_with_qualifiers() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>21</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
    <MeshHeadingList>
      <MeshHeading>
        <DescriptorName UI="D006321" MajorTopicYN="N">Heart</DescriptorName>
        <QualifierName UI="Q000033" MajorTopicYN="N">anatomy &amp; histology</QualifierName>
        <QualifierName UI="Q000502" MajorTopicYN="Y">physiology</QualifierName>
      </MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        // 2 qualifiers → 2 MeshHeading rows (one per qualifier)
        assert_eq!(result.mesh_headings.len(), 2);
        assert_eq!(result.mesh_headings[0].qualifier_ui.as_deref(), Some("Q000033"));
        assert!(!result.mesh_headings[0].is_major_topic);
        assert_eq!(result.mesh_headings[1].qualifier_ui.as_deref(), Some("Q000502"));
        assert!(result.mesh_headings[1].is_major_topic); // MajorTopicYN="Y"
    }

    // ── Grants and Chemicals ──

    #[test]
    fn test_grant_parsing() {
        let body = r#"<GrantList>
        <Grant>
          <GrantID>R01 CA123456</GrantID>
          <Acronym>CA</Acronym>
          <Agency>NCI NIH HHS</Agency>
          <Country>United States</Country>
        </Grant>
      </GrantList>"#;
        let result = parse_xml(&article(30, body));
        assert_eq!(result.grants.len(), 1);
        let g = &result.grants[0];
        assert_eq!(g.pmid, 30);
        assert_eq!(g.grant_id.as_deref(), Some("R01 CA123456"));
        assert_eq!(g.acronym.as_deref(), Some("CA"));
        assert_eq!(g.agency.as_deref(), Some("NCI NIH HHS"));
        assert_eq!(g.country.as_deref(), Some("United States"));
    }

    #[test]
    fn test_chemical_parsing() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>31</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
    <ChemicalList>
      <Chemical>
        <RegistryNumber>50-78-2</RegistryNumber>
        <NameOfSubstance UI="D001241">Aspirin</NameOfSubstance>
      </Chemical>
    </ChemicalList>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        assert_eq!(result.chemicals.len(), 1);
        let c = &result.chemicals[0];
        assert_eq!(c.pmid, 31);
        assert_eq!(c.registry_number.as_deref(), Some("50-78-2"));
        assert_eq!(c.substance_ui.as_deref(), Some("D001241"));
        assert_eq!(c.substance_name.as_deref(), Some("Aspirin"));
    }

    // ── DOI and PMC ID ──

    #[test]
    fn test_article_ids() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>40</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="doi">10.1234/test.2024</ArticleId>
      <ArticleId IdType="pmc">PMC12345</ArticleId>
      <ArticleId IdType="pubmed">40</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        let p = &result.papers[0];
        assert_eq!(p.doi.as_deref(), Some("10.1234/test.2024"));
        assert_eq!(p.pmc_id.as_deref(), Some("PMC12345"));
    }

    // ── Date fields ──

    #[test]
    fn test_date_fields() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>50</PMID>
    <DateRevised><Year>2024</Year><Month>03</Month><Day>15</Day></DateRevised>
    <DateCompleted><Year>2024</Year><Month>01</Month><Day>10</Day></DateCompleted>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <History>
      <PubMedPubDate PubStatus="entrez">
        <Year>2023</Year><Month>06</Month><Day>01</Day>
      </PubMedPubDate>
    </History>
    <ArticleIdList/>
  </PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        let p = &result.papers[0];
        assert_eq!(p.revised_date.as_deref(), Some("2024-03-15"));
        assert_eq!(p.indexed_date.as_deref(), Some("2024-01-10"));
        assert_eq!(p.created_date.as_deref(), Some("2023-06-01"));
    }

    // ── DeleteCitation ──

    #[test]
    fn test_delete_citation() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>100</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle>
    </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>
<DeleteCitation>
  <PMID Version="1">999</PMID>
  <PMID Version="1">1000</PMID>
</DeleteCitation>"#;
        let result = parse_xml(xml);
        assert_eq!(result.papers.len(), 1);
        assert_eq!(result.delete_pmids, vec![999, 1000]);
    }

    // ── Publication types and pagination ──

    #[test]
    fn test_pub_types_and_pages() {
        let body = r#"<Pagination><MedlinePgn>123-130</MedlinePgn></Pagination>
      <PublicationTypeList>
        <PublicationType>Journal Article</PublicationType>
        <PublicationType>Review</PublicationType>
      </PublicationTypeList>"#;
        let result = parse_xml(&article(60, body));
        let p = &result.papers[0];
        assert_eq!(p.pages.as_deref(), Some("123-130"));
        assert_eq!(p.pub_type, vec!["Journal Article", "Review"]);
    }

    // ── Journal info ──

    #[test]
    fn test_journal_fields() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID>70</PMID>
    <Article>
      <Journal>
        <ISSN IssnType="Print">0028-0836</ISSN>
        <JournalIssue>
          <Volume>612</Volume>
          <Issue>7938</Issue>
          <PubDate><Year>2022</Year><Month>Dec</Month></PubDate>
        </JournalIssue>
        <Title>Nature</Title>
        <ISOAbbreviation>Nature</ISOAbbreviation>
      </Journal>
      <ArticleTitle>T</ArticleTitle>
      <Language>eng</Language>
    </Article>
    <MedlineJournalInfo>
      <Country>England</Country>
    </MedlineJournalInfo>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        let p = &result.papers[0];
        assert_eq!(p.journal_issn.as_deref(), Some("0028-0836"));
        assert_eq!(p.journal_title.as_deref(), Some("Nature"));
        assert_eq!(p.journal_abbr.as_deref(), Some("Nature"));
        assert_eq!(p.volume.as_deref(), Some("612"));
        assert_eq!(p.issue.as_deref(), Some("7938"));
        assert_eq!(p.language.as_deref(), Some("eng"));
        assert_eq!(p.country.as_deref(), Some("England"));
    }

    // ── Multiple articles ──

    #[test]
    fn test_multiple_languages_joined() {
        let xml = r#"<PubmedArticle>
  <MedlineCitation Status="MEDLINE" Owner="NLM">
    <PMID Version="1">99</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>Bilingual</ArticleTitle>
      <Language>eng</Language>
      <Language>fre</Language>
    </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList/></PubmedData>
</PubmedArticle>"#;
        let result = parse_xml(xml);
        assert_eq!(result.papers.len(), 1);
        assert_eq!(result.papers[0].language.as_deref(), Some("eng;fre"));
    }

    // ── Multiple articles ──

    #[test]
    fn test_multiple_articles() {
        let xml = format!(
            "{}{}",
            article(101, ""),
            article(102, ""),
        );
        let result = parse_xml(&xml);
        assert_eq!(result.papers.len(), 2);
        assert_eq!(result.papers[0].pmid, 101);
        assert_eq!(result.papers[1].pmid, 102);
    }
}
