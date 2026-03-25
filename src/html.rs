//! Self-contained HTML renderer for VisGraph.
//!
//! Produces a single HTML file with embedded JS, CSS, and the graph data as JSON.
//! Uses dagre-d3 (loaded from CDN) for DAG layout and rendering.

use crate::model::VisGraph;

const VIEWER_JS: &str = include_str!("assets/viewer.js");
const VIEWER_CSS: &str = include_str!("assets/viewer.css");

/// Render a VisGraph as a self-contained HTML string.
pub fn render_html(vis: &VisGraph) -> String {
    let json_data = serde_json::to_string(vis).expect("VisGraph serialization should not fail");

    let node_count = vis.nodes.len();
    let edge_count = vis.edges.len();
    let fusion_count = vis.fusion_groups.len();
    let has_runtime = vis.runtime.is_some();

    let stats = format!(
        "{} nodes, {} edges, {} fusion groups{}",
        node_count,
        edge_count,
        fusion_count,
        if has_runtime { ", runtime profiled" } else { "" }
    );

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ferroviz - Compute Graph</title>
<style>
{css}
</style>
</head>
<body>
<header>
  <h1>Ferroviz</h1>
  <span class="stats">{stats}</span>
  <div class="legend">
    <span class="legend-item"><span class="legend-swatch" style="background:#42a5f5;"></span>Elementwise</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#ab47bc;"></span>Reduction</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#26a69a;"></span>MatMul</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#66bb6a;"></span>Linear</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#ffa726;"></span>Activation</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#7e57c2;"></span>Fused</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#e53935;"></span>CPU round-trip</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#ff9800;"></span>Device transition</span>
  </div>
</header>
<div class="main">
  <div id="graph-container">
    <svg id="graph-svg"></svg>
  </div>
  <div id="detail-panel"></div>
</div>
<div id="timeline-container"></div>

<!-- d3 v5 (required by dagre-d3 0.6.x) and dagre-d3 from CDN -->
<script src="https://d3js.org/d3.v5.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre-d3@0.6.4/dist/dagre-d3.min.js"></script>

<script>
window.__FERROVIZ_DATA__ = {json_data};
</script>
<script>
{js}
</script>
</body>
</html>"#,
        css = VIEWER_CSS,
        stats = stats,
        json_data = json_data,
        js = VIEWER_JS,
    )
}
