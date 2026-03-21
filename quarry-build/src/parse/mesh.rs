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

/// Parse MeSH descriptor XML → list of (ui, name, tree_number) entries.
///
/// Each DescriptorRecord with N TreeNumbers produces N entries.
pub fn parse_mesh_xml(path: &Path) -> Result<Vec<MeshEntry>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(256 * 1024, file);
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut entries: Vec<MeshEntry> = Vec::with_capacity(400_000);
    let mut buf = Vec::with_capacity(4096);

    // State for current DescriptorRecord
    let mut in_descriptor = false;
    let mut current_ui = String::new();
    let mut current_name = String::new();
    let mut current_trees: Vec<String> = Vec::new();

    // Track nesting for text capture
    #[derive(Clone, Copy, PartialEq)]
    enum Capture {
        None,
        DescriptorUI,
        DescriptorNameString,
        TreeNumber,
    }
    let mut capture = Capture::None;
    let mut in_descriptor_name = false;

    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"DescriptorRecord" => {
                    in_descriptor = true;
                    current_ui.clear();
                    current_name.clear();
                    current_trees.clear();
                }
                b"DescriptorUI" if in_descriptor => capture = Capture::DescriptorUI,
                b"DescriptorName" if in_descriptor => in_descriptor_name = true,
                b"String" if in_descriptor_name => capture = Capture::DescriptorNameString,
                b"TreeNumber" if in_descriptor => capture = Capture::TreeNumber,
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
                Capture::None => {}
            },
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"DescriptorUI" | b"TreeNumber" => capture = Capture::None,
                b"String" if capture == Capture::DescriptorNameString => {
                    capture = Capture::None;
                }
                b"DescriptorName" => in_descriptor_name = false,
                b"DescriptorRecord" => {
                    if !current_ui.is_empty() && !current_name.is_empty() {
                        for tn in &current_trees {
                            entries.push(MeshEntry {
                                descriptor_ui: current_ui.clone(),
                                descriptor_name: current_name.clone(),
                                tree_number: tn.clone(),
                            });
                        }
                    }
                    in_descriptor = false;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {e}").into()),
            _ => {}
        }
        buf.clear();
    }

    Ok(entries)
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
  </DescriptorRecord>
  <DescriptorRecord>
    <DescriptorUI>D000002</DescriptorUI>
    <DescriptorName><String>Temefos</String></DescriptorName>
    <TreeNumberList>
      <TreeNumber>D02.705.400.625.800</TreeNumber>
      <TreeNumber>D02.886.300.692.800</TreeNumber>
    </TreeNumberList>
  </DescriptorRecord>
</DescriptorRecordSet>"#
        )
        .unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_mesh_xml() {
        let f = write_test_xml();
        let entries = parse_mesh_xml(f.path()).unwrap();
        assert_eq!(entries.len(), 3); // 1 + 2 tree numbers
        assert_eq!(entries[0].descriptor_ui, "D000001");
        assert_eq!(entries[1].descriptor_ui, "D000002");
        assert_eq!(entries[2].descriptor_ui, "D000002");
        assert_eq!(entries[0].tree_number, "D03.633.100.221.173");
        assert_eq!(entries[1].tree_number, "D02.705.400.625.800");
        assert_eq!(entries[2].tree_number, "D02.886.300.692.800");
    }
}
