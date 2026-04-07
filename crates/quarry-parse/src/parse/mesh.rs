//! MeSH descriptor XML → structured entries.
//!
//! Hybrid parser: SAX boundary detection + serde deserialization.
//! SAX scans for `<DescriptorRecord>` boundaries, `capture_element()` grabs
//! the raw XML, then `quick_xml::de::from_str()` deserializes structurally.
//!
//! This approach eliminates the nested-element overwrite bug that affected the
//! previous SAX-only parser: serde structs scope fields to their parent element,
//! so `<DescriptorUI>` inside `<PharmacologicalAction>/<DescriptorReferredTo>`
//! is never confused with the top-level `<DescriptorUI>`.

use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use super::capture_element;

// ── Output structs (unchanged — consumed by parquet_writer) ──

/// A single MeSH descriptor → tree_number entry.
pub struct MeshEntry {
    pub descriptor_ui: String,
    pub descriptor_name: String,
    pub tree_number: String,
}

/// A single MeSH descriptor → term entry (synonym / entry term).
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

// ── Serde deserialization structs ──
//
// Only the fields we need are declared. quick-xml serde silently ignores
// undeclared elements (PharmacologicalAction, SeeRelatedList, AllowableQualifiersList,
// etc.), which is exactly what prevents the nested DescriptorUI bug.

#[derive(Deserialize)]
struct XmlDescriptorRecord {
    #[serde(rename = "DescriptorUI")]
    descriptor_ui: String,
    #[serde(rename = "DescriptorName")]
    descriptor_name: XmlDescriptorName,
    #[serde(rename = "TreeNumberList", default)]
    tree_number_list: Option<XmlTreeNumberList>,
    #[serde(rename = "ConceptList", default)]
    concept_list: Option<XmlConceptList>,
}

#[derive(Deserialize)]
struct XmlDescriptorName {
    /// MeSH uses `<String>` as element name for text content.
    #[serde(rename = "String")]
    value: String,
}

#[derive(Deserialize)]
struct XmlTreeNumberList {
    #[serde(rename = "TreeNumber", default)]
    tree_numbers: Vec<String>,
}

#[derive(Deserialize)]
struct XmlConceptList {
    #[serde(rename = "Concept", default)]
    concepts: Vec<XmlConcept>,
}

#[derive(Deserialize)]
struct XmlConcept {
    #[serde(rename = "TermList", default)]
    term_list: Option<XmlTermList>,
}

#[derive(Deserialize)]
struct XmlTermList {
    #[serde(rename = "Term", default)]
    terms: Vec<XmlTerm>,
}

#[derive(Deserialize)]
struct XmlTerm {
    #[serde(rename = "@RecordPreferredTermYN", default)]
    record_preferred_term_yn: Option<String>,
    #[serde(rename = "String")]
    value: String,
}

// ── Conversion ──

fn convert_record(rec: XmlDescriptorRecord) -> (Vec<MeshEntry>, Vec<MeshTerm>) {
    let ui = &rec.descriptor_ui;
    let name = rec.descriptor_name.value.trim();

    if ui.is_empty() || name.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Tree entries
    let tree_entries: Vec<MeshEntry> = rec
        .tree_number_list
        .into_iter()
        .flat_map(|tnl| tnl.tree_numbers)
        .filter(|tn| !tn.is_empty())
        .map(|tn| MeshEntry {
            descriptor_ui: ui.clone(),
            descriptor_name: name.to_string(),
            tree_number: tn,
        })
        .collect();

    // Term entries
    let term_entries: Vec<MeshTerm> = rec
        .concept_list
        .into_iter()
        .flat_map(|cl| cl.concepts)
        .filter_map(|c| c.term_list)
        .flat_map(|tl| tl.terms)
        .filter(|t| !t.value.trim().is_empty())
        .map(|t| MeshTerm {
            descriptor_ui: ui.clone(),
            descriptor_name: name.to_string(),
            term: t.value.trim().to_string(),
            is_preferred: t.record_preferred_term_yn.as_deref() == Some("Y"),
        })
        .collect();

    (tree_entries, term_entries)
}

// ── Main parser ──

/// Parse MeSH descriptor XML → tree entries + term entries.
///
/// Uses SAX to find `<DescriptorRecord>` boundaries, then serde for each record.
pub fn parse_mesh_xml(path: &Path) -> Result<MeshParseResult, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(256 * 1024, file);
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut tree_entries: Vec<MeshEntry> = Vec::with_capacity(400_000);
    let mut term_entries: Vec<MeshTerm> = Vec::with_capacity(200_000);
    let mut buf = Vec::with_capacity(4096);

    loop {
        buf.clear();
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"DescriptorRecord" => {
                let owned = e.to_owned();
                let raw = capture_element(&mut xml, &owned, &mut buf)?;
                match quick_xml::de::from_str::<XmlDescriptorRecord>(&raw) {
                    Ok(rec) => {
                        let (trees, terms) = convert_record(rec);
                        tree_entries.extend(trees);
                        term_entries.extend(terms);
                    }
                    Err(err) => {
                        eprintln!("mesh: WARN: failed to parse DescriptorRecord: {err}");
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {e}").into()),
            _ => {}
        }
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

    /// Regression test for the nested DescriptorUI bug.
    ///
    /// MeSH XML contains `<DescriptorUI>` at multiple nesting levels:
    /// - Direct child of `<DescriptorRecord>` (the real one)
    /// - Inside `<PharmacologicalAction>/<DescriptorReferredTo>` (cross-reference)
    /// - Inside `<SeeRelatedList>/<SeeRelatedDescriptor>/<DescriptorReferredTo>`
    ///
    /// The old SAX parser overwrote `current_ui` with nested instances,
    /// causing 34% of descriptors to have corrupted term mappings.
    /// With serde, undeclared elements are silently ignored, so the
    /// nested DescriptorUI never affects the top-level field.
    #[test]
    fn test_nested_descriptor_ui_not_captured() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"<?xml version="1.0"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D000001</DescriptorUI>
    <DescriptorName><String>Calcimycin</String></DescriptorName>
    <TreeNumberList>
      <TreeNumber>D03.633.100.221.173</TreeNumber>
    </TreeNumberList>
    <PharmacologicalAction>
      <DescriptorReferredTo>
        <DescriptorUI>D000900</DescriptorUI>
        <DescriptorName><String>Anti-Infective Agents</String></DescriptorName>
      </DescriptorReferredTo>
    </PharmacologicalAction>
    <PharmacologicalAction>
      <DescriptorReferredTo>
        <DescriptorUI>D061207</DescriptorUI>
        <DescriptorName><String>Calcium Ionophores</String></DescriptorName>
      </DescriptorReferredTo>
    </PharmacologicalAction>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList>
          <Term RecordPreferredTermYN="Y"><String>Calcimycin</String></Term>
          <Term RecordPreferredTermYN="N"><String>A-23187</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord>
    <DescriptorUI>D015202</DescriptorUI>
    <DescriptorName><String>Protein Engineering</String></DescriptorName>
    <SeeRelatedList>
      <SeeRelatedDescriptor>
        <DescriptorReferredTo>
          <DescriptorUI>D016297</DescriptorUI>
          <DescriptorName><String>Mutagenesis, Site-Directed</String></DescriptorName>
        </DescriptorReferredTo>
      </SeeRelatedDescriptor>
      <SeeRelatedDescriptor>
        <DescriptorReferredTo>
          <DescriptorUI>D019020</DescriptorUI>
          <DescriptorName><String>Directed Molecular Evolution</String></DescriptorName>
        </DescriptorReferredTo>
      </SeeRelatedDescriptor>
    </SeeRelatedList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList>
          <Term RecordPreferredTermYN="Y"><String>Protein Engineering</String></Term>
          <Term RecordPreferredTermYN="N"><String>Engineering, Protein</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>"#
        )
        .unwrap();
        f.flush().unwrap();

        let result = parse_mesh_xml(f.path()).unwrap();

        // D000001: terms must be under D000001, NOT D061207
        let d1_terms: Vec<&MeshTerm> = result
            .term_entries
            .iter()
            .filter(|t| t.descriptor_ui == "D000001")
            .collect();
        assert_eq!(d1_terms.len(), 2);
        assert_eq!(d1_terms[0].term, "Calcimycin");
        assert_eq!(d1_terms[0].descriptor_name, "Calcimycin");

        // D015202: terms must be under D015202, NOT D019020
        let d15202_terms: Vec<&MeshTerm> = result
            .term_entries
            .iter()
            .filter(|t| t.descriptor_ui == "D015202")
            .collect();
        assert_eq!(d15202_terms.len(), 2);
        assert_eq!(d15202_terms[0].term, "Protein Engineering");
        assert_eq!(d15202_terms[0].descriptor_name, "Protein Engineering");
        assert!(d15202_terms[0].is_preferred);

        // No terms should exist for the nested reference UIs
        assert!(result.term_entries.iter().all(|t| t.descriptor_ui != "D061207"));
        assert!(result.term_entries.iter().all(|t| t.descriptor_ui != "D019020"));
        assert!(result.term_entries.iter().all(|t| t.descriptor_ui != "D000900"));
        assert!(result.term_entries.iter().all(|t| t.descriptor_ui != "D016297"));

        // Tree entries: D000001 only (D015202 has no TreeNumberList)
        assert_eq!(result.tree_entries.len(), 1);
        assert_eq!(result.tree_entries[0].descriptor_ui, "D000001");
    }
}
