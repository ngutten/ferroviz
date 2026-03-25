//! JSON serialization for VisGraph.

use crate::model::VisGraph;

/// Render a VisGraph as a pretty-printed JSON string.
pub fn render_json(vis: &VisGraph) -> String {
    serde_json::to_string_pretty(vis).expect("VisGraph serialization should not fail")
}

/// Render a VisGraph as a compact JSON string.
pub fn render_json_compact(vis: &VisGraph) -> String {
    serde_json::to_string(vis).expect("VisGraph serialization should not fail")
}

/// Parse a VisGraph from a JSON string.
pub fn parse_json(json: &str) -> Result<VisGraph, serde_json::Error> {
    serde_json::from_str(json)
}
