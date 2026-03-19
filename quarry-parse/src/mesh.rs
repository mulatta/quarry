//! MeSH descriptor XML → Arrow RecordBatch.
//!
//! Streaming parser using quick-xml. Same pattern as existing PubMed XML parser.
//! Replaces Python `etl/mesh.py::parse_mesh_descriptors()`.

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

struct MeshEntry {
    descriptor_ui: String,
    descriptor_name: String,
    tree_number: String,
}

/// Parse MeSH descriptor XML → list of (ui, name, tree_number) entries.
///
/// Each DescriptorRecord with N TreeNumbers produces N entries.
pub fn parse_mesh_xml(path: &Path) -> Result<RecordBatch, Box<dyn std::error::Error>> {
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

    // Build Arrow RecordBatch
    let schema = Arc::new(Schema::new(vec![
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("tree_number", DataType::Utf8, false),
    ]));

    let ui_arr: StringArray = entries.iter().map(|e| Some(e.descriptor_ui.as_str())).collect();
    let name_arr: StringArray = entries
        .iter()
        .map(|e| Some(e.descriptor_name.as_str()))
        .collect();
    let tree_arr: StringArray = entries
        .iter()
        .map(|e| Some(e.tree_number.as_str()))
        .collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(ui_arr), Arc::new(name_arr), Arc::new(tree_arr)],
    )?;

    Ok(batch)
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
        let batch = parse_mesh_xml(f.path()).unwrap();
        assert_eq!(batch.num_rows(), 3); // 1 + 2 tree numbers

        let ui = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ui.value(0), "D000001");
        assert_eq!(ui.value(1), "D000002");
        assert_eq!(ui.value(2), "D000002");

        let tree = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(tree.value(0), "D03.633.100.221.173");
        assert_eq!(tree.value(1), "D02.705.400.625.800");
        assert_eq!(tree.value(2), "D02.886.300.692.800");
    }
}
