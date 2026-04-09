pub mod mesh;
pub mod xml;

#[cfg(test)]
mod xml_tests;

use quick_xml::Reader;
use std::io::BufRead;

/// Capture the raw XML of the current element (already opened) into a String.
///
/// The caller has just consumed an `Event::Start(e)` for the element.
/// This function reads all events until the matching `Event::End`, tracking
/// depth to handle nested elements with the same name. Returns the complete
/// XML fragment including the opening and closing tags.
///
/// Used by both MeSH and PubMed parsers for hybrid SAX-boundary + serde
/// deserialization: SAX detects record boundaries, this function captures the
/// raw XML, then `quick_xml::de::from_str()` deserializes it structurally.
pub fn capture_element<R: BufRead>(
    reader: &mut Reader<R>,
    start_event: &quick_xml::events::BytesStart<'_>,
    buf: &mut Vec<u8>,
) -> Result<String, Box<dyn std::error::Error>> {
    use quick_xml::events::Event;

    let mut writer = quick_xml::Writer::new(Vec::with_capacity(4096));

    // Write the opening tag (already consumed by caller).
    writer.write_event(Event::Start(start_event.to_owned()))?;

    let mut depth: u32 = 1;
    loop {
        buf.clear();
        match reader.read_event_into(buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                writer.write_event(Event::Start(e.to_owned()))?;
            }
            Ok(Event::End(ref e)) => {
                depth -= 1;
                writer.write_event(Event::End(e.to_owned()))?;
                if depth == 0 {
                    break;
                }
            }
            Ok(Event::Text(ref e)) => {
                writer.write_event(Event::Text(e.to_owned()))?;
            }
            Ok(Event::CData(ref e)) => {
                writer.write_event(Event::CData(e.to_owned()))?;
            }
            Ok(Event::Empty(ref e)) => {
                writer.write_event(Event::Empty(e.to_owned()))?;
            }
            Ok(Event::Eof) => {
                return Err("unexpected EOF while capturing element".into());
            }
            Err(e) => return Err(e.into()),
            _ => {} // PI, Comment, Decl — skip
        }
    }

    Ok(String::from_utf8(writer.into_inner())?)
}
