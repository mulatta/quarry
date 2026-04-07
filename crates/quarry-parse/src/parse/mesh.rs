//! MeSH descriptor XML → structured entries.
//!
//! Streaming parser using quick-xml. Same pattern as existing PubMed XML parser.
//! Replaces Python `etl/mesh.py::parse_mesh_descriptors()`.

use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// A single MeSH descriptor → tree_number entry.
pub struct MeshEntry {
    pub descriptor_ui: String,
    pub descriptor_name: String,
    pub tree_number: String,
}

/// A single MeSH descriptor → term entry (synonym / entry term).
///
/// NLM MeSH XML stores synonyms as Term elements within ConceptList.
/// Each descriptor has one preferred term (= descriptor_name) and zero
/// or more non-preferred entry terms (alternative names, abbreviations).
/// Parsing these enables searching by synonym rather than requiring
/// users to know the exact MeSH descriptor name.
pub struct MeshTerm {
    pub descriptor_ui: String,
    pub descriptor_name: String,
    pub term: String,
    pub is_preferred: bool,
}

/// Parse result containing both tree entries and term entries.
pub struct MeshParseResult {
    pub tree_entries: Vec<MeshEntry>,
    pub term_entries: Vec<MeshTerm>,
}

/// Parse MeSH descriptor XML → tree entries + term entries.
///
/// Tree entries: each DescriptorRecord with N TreeNumbers produces N MeshEntry rows.
/// Term entries: each Term/String in ConceptList produces one MeshTerm row.
pub fn parse_mesh_xml(path: &Path) -> Result<MeshParseResult, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(256 * 1024, file);
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut tree_entries: Vec<MeshEntry> = Vec::with_capacity(400_000);
    let mut term_entries: Vec<MeshTerm> = Vec::with_capacity(200_000);
    let mut buf = Vec::with_capacity(4096);

    // State for current DescriptorRecord
    let mut in_descriptor = false;
    let mut current_ui = String::new();
    let mut current_name = String::new();
    let mut current_trees: Vec<String> = Vec::new();
    let mut current_terms: Vec<(String, bool)> = Vec::new(); // (term, is_preferred)

    // Track nesting for text capture.
    // MeSH XML nests String elements under both DescriptorName and Term.
    // We distinguish them by tracking which parent we're inside.
    #[derive(Clone, Copy, PartialEq)]
    enum Capture {
        None,
        DescriptorUI,
        DescriptorNameString,
        TreeNumber,
        TermString,
    }
    let mut capture = Capture::None;
    let mut in_descriptor_name = false;
    let mut in_term = false;
    let mut current_term_is_preferred = false;

    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"DescriptorRecord" => {
                    in_descriptor = true;
                    current_ui.clear();
                    current_name.clear();
                    current_trees.clear();
                    current_terms.clear();
                }
                b"DescriptorUI" if in_descriptor => capture = Capture::DescriptorUI,
                b"DescriptorName" if in_descriptor => in_descriptor_name = true,
                b"String" if in_descriptor_name => capture = Capture::DescriptorNameString,
                b"TreeNumber" if in_descriptor => capture = Capture::TreeNumber,
                b"Term" if in_descriptor => {
                    in_term = true;
                    // RecordPreferredTermYN="Y" marks the descriptor's primary term
                    current_term_is_preferred = e.attributes().flatten().any(|a| {
                        a.key.as_ref() == b"RecordPreferredTermYN"
                            && a.value.as_ref() == b"Y"
                    });
                }
                b"String" if in_term => capture = Capture::TermString,
                _ => {}
            },
            Ok(Event::Text(ref e)) => match capture {
                Capture::DescriptorUI => {
                    current_ui = e.unescape().unwrap_or_default().trim().to_string();
                }
                Capture::DescriptorNameString => {
                    current_name = e.unescape().unwrap_or_default().trim().to_string();
                }
                Capture::TreeNumber => {
                    let tn = e.unescape().unwrap_or_default().trim().to_string();
                    if !tn.is_empty() {
                        current_trees.push(tn);
                    }
                }
                Capture::TermString => {
                    let term = e.unescape().unwrap_or_default().trim().to_string();
                    if !term.is_empty() {
                        current_terms.push((term, current_term_is_preferred));
                    }
                }
                Capture::None => {}
            },
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"DescriptorUI" | b"TreeNumber" => capture = Capture::None,
                b"String"
                    if capture == Capture::DescriptorNameString
                        || capture == Capture::TermString =>
                {
                    capture = Capture::None;
                }
                b"DescriptorName" => in_descriptor_name = false,
                b"Term" => in_term = false,
                b"DescriptorRecord" => {
                    if !current_ui.is_empty() && !current_name.is_empty() {
                        for tn in &current_trees {
                            tree_entries.push(MeshEntry {
                                descriptor_ui: current_ui.clone(),
                                descriptor_name: current_name.clone(),
                                tree_number: tn.clone(),
                            });
                        }
                        for (term, is_pref) in &current_terms {
                            term_entries.push(MeshTerm {
                                descriptor_ui: current_ui.clone(),
                                descriptor_name: current_name.clone(),
                                term: term.clone(),
                                is_preferred: *is_pref,
                            });
                        }
                    }
                    in_descriptor = false;
                    in_descriptor_name = false;
                    in_term = false;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {e}").into()),
            _ => {}
        }
        buf.clear();
    }

    Ok(MeshParseResult {
        tree_entries,
        term_entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_test_xml() -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"<?xml version="1.0"?>
<!DOCTYPE DescriptorRecordSet>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D000001</DescriptorUI>
    <DescriptorName><String>Calcimycin</String></DescriptorName>
    <TreeNumberList>
      <TreeNumber>D03.633.100.221.173</TreeNumber>
    </TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList>
          <Term RecordPreferredTermYN="Y"><String>Calcimycin</String></Term>
          <Term RecordPreferredTermYN="N"><String>A-23187</String></Term>
          <Term RecordPreferredTermYN="N"><String>A23187</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord>
    <DescriptorUI>D000002</DescriptorUI>
    <DescriptorName><String>Temefos</String></DescriptorName>
    <TreeNumberList>
      <TreeNumber>D02.705.400.625.800</TreeNumber>
      <TreeNumber>D02.886.300.692.800</TreeNumber>
    </TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList>
          <Term RecordPreferredTermYN="Y"><String>Temefos</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>"#
        )
        .unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_mesh_tree_entries() {
        let f = write_test_xml();
        let result = parse_mesh_xml(f.path()).unwrap();
        let entries = &result.tree_entries;
        assert_eq!(entries.len(), 3); // 1 + 2 tree numbers
        assert_eq!(entries[0].descriptor_ui, "D000001");
        assert_eq!(entries[1].descriptor_ui, "D000002");
        assert_eq!(entries[2].descriptor_ui, "D000002");
        assert_eq!(entries[0].tree_number, "D03.633.100.221.173");
        assert_eq!(entries[1].tree_number, "D02.705.400.625.800");
        assert_eq!(entries[2].tree_number, "D02.886.300.692.800");
    }

    #[test]
    fn test_parse_mesh_term_entries() {
        let f = write_test_xml();
        let result = parse_mesh_xml(f.path()).unwrap();
        let terms = &result.term_entries;
        // D000001: 3 terms (Calcimycin, A-23187, A23187)
        // D000002: 1 term (Temefos)
        assert_eq!(terms.len(), 4);
        assert_eq!(terms[0].term, "Calcimycin");
        assert!(terms[0].is_preferred);
        assert_eq!(terms[1].term, "A-23187");
        assert!(!terms[1].is_preferred);
        assert_eq!(terms[2].term, "A23187");
        assert!(!terms[2].is_preferred);
        assert_eq!(terms[3].term, "Temefos");
        assert!(terms[3].is_preferred);
    }

    #[test]
    fn test_empty_descriptor_ui_skipped() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"<?xml version="1.0"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI></DescriptorUI>
    <DescriptorName><String>Empty UI</String></DescriptorName>
    <TreeNumberList><TreeNumber>A01</TreeNumber></TreeNumberList>
  </DescriptorRecord>
</DescriptorRecordSet>"#
        )
        .unwrap();
        f.flush().unwrap();

        let result = parse_mesh_xml(f.path()).unwrap();
        assert_eq!(result.tree_entries.len(), 0);
        assert_eq!(result.term_entries.len(), 0);
    }

    #[test]
    fn test_no_tree_numbers_still_has_terms() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"<?xml version="1.0"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D999999</DescriptorUI>
    <DescriptorName><String>No Trees</String></DescriptorName>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList>
          <Term RecordPreferredTermYN="Y"><String>No Trees</String></Term>
          <Term RecordPreferredTermYN="N"><String>Treeless</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>"#
        )
        .unwrap();
        f.flush().unwrap();

        let result = parse_mesh_xml(f.path()).unwrap();
        assert_eq!(result.tree_entries.len(), 0);
        // Terms are still extracted even without tree numbers
        assert_eq!(result.term_entries.len(), 2);
        assert_eq!(result.term_entries[0].descriptor_ui, "D999999");
    }

    #[test]
    fn test_xml_entity_unescape() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"<?xml version="1.0"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D000003</DescriptorUI>
    <DescriptorName><String>Anatomy &amp; Histology</String></DescriptorName>
    <TreeNumberList><TreeNumber>A01.001</TreeNumber></TreeNumberList>
  </DescriptorRecord>
</DescriptorRecordSet>"#
        )
        .unwrap();
        f.flush().unwrap();

        let result = parse_mesh_xml(f.path()).unwrap();
        assert_eq!(result.tree_entries.len(), 1);
        assert_eq!(result.tree_entries[0].descriptor_name, "Anatomy & Histology");
    }

    #[test]
    fn test_multi_concept_descriptor() {
        // Real MeSH descriptors can have multiple Concepts, each with its own TermList.
        // All terms from all concepts should be extracted.
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"<?xml version="1.0"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D000100</DescriptorUI>
    <DescriptorName><String>Acetaminophen</String></DescriptorName>
    <TreeNumberList><TreeNumber>D02.065.199.092</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList>
          <Term RecordPreferredTermYN="Y"><String>Acetaminophen</String></Term>
          <Term RecordPreferredTermYN="N"><String>Tylenol</String></Term>
        </TermList>
      </Concept>
      <Concept PreferredConceptYN="N">
        <TermList>
          <Term RecordPreferredTermYN="N"><String>Paracetamol</String></Term>
          <Term RecordPreferredTermYN="N"><String>APAP</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>"#
        )
        .unwrap();
        f.flush().unwrap();

        let result = parse_mesh_xml(f.path()).unwrap();
        assert_eq!(result.tree_entries.len(), 1);
        // All 4 terms from both concepts should be captured
        assert_eq!(result.term_entries.len(), 4);
        let term_strs: Vec<&str> = result.term_entries.iter().map(|t| t.term.as_str()).collect();
        assert!(term_strs.contains(&"Acetaminophen"));
        assert!(term_strs.contains(&"Tylenol"));
        assert!(term_strs.contains(&"Paracetamol"));
        assert!(term_strs.contains(&"APAP"));
        // Only RecordPreferredTermYN="Y" is marked preferred
        assert_eq!(result.term_entries.iter().filter(|t| t.is_preferred).count(), 1);
    }
}
