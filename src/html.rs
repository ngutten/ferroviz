//! Self-contained HTML renderer for VisGraph.
//!
//! Produces a single HTML file with embedded JS, CSS, and the graph data as JSON.
//! Uses dagre-d3 (loaded from CDN) for DAG layout and rendering.

use crate::model::{TrainingStepVis, VisGraph};

const VIEWER_JS: &str = include_str!("assets/viewer.js");
const VIEWER_CSS: &str = include_str!("assets/viewer.css");

/// Render a VisGraph as a self-contained HTML string.
pub fn render_html(vis: &VisGraph) -> String {
    let json_data = serde_json::to_string(vis).expect("VisGraph serialization should not fail");
    render_html_with_data(&json_data, "null", vis_stats(vis).as_str())
}

/// Render a TrainingStepVis (forward + backward) as a self-contained HTML string.
///
/// The viewer shows both graphs with a tab switcher.
pub fn render_html_training_step(vis: &TrainingStepVis) -> String {
    let forward_json = serde_json::to_string(&vis.forward)
        .expect("VisGraph serialization should not fail");
    let backward_json = serde_json::to_string(&vis.backward)
        .expect("VisGraph serialization should not fail");

    let stats = format!(
        "Forward: {} | Backward: {} nodes, {} edges",
        vis_stats(&vis.forward),
        vis.backward.nodes.len(),
        vis.backward.edges.len(),
    );

    render_html_with_data(&forward_json, &backward_json, &stats)
}

fn vis_stats(vis: &VisGraph) -> String {
    let node_count = vis.nodes.len();
    let edge_count = vis.edges.len();
    let fusion_count = vis.fusion_groups.len();
    let has_runtime = vis.runtime.is_some();

    format!(
        "{} nodes, {} edges, {} fusion groups{}",
        node_count,
        edge_count,
        fusion_count,
        if has_runtime { ", runtime profiled" } else { "" }
    )
}

fn render_html_with_data(forward_json: &str, backward_json: &str, stats: &str) -> String {
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
    <span class="legend-item"><span class="legend-swatch swatch-gpu" style="background:#42a5f5;"></span>Elementwise</span>
    <span class="legend-item"><span class="legend-swatch swatch-gpu" style="background:#ab47bc;"></span>Reduction</span>
    <span class="legend-item"><span class="legend-swatch swatch-gpu" style="background:#26a69a;"></span>MatMul</span>
    <span class="legend-item"><span class="legend-swatch swatch-gpu" style="background:#66bb6a;"></span>Linear</span>
    <span class="legend-item"><span class="legend-swatch swatch-gpu" style="background:#ffa726;"></span>Activation</span>
    <span class="legend-item"><span class="legend-swatch swatch-gpu" style="background:#7e57c2;"></span>Fused</span>
    <span class="legend-item"><span class="legend-swatch swatch-gpu" style="background:#f06292;"></span>Backward</span>
    <span class="legend-item"><span class="legend-swatch" style="background:transparent; border: 2px solid #42a5f5;"></span>CPU</span>
    <span class="legend-item"><span class="legend-swatch swatch-gpu swatch-fallback" style="background:#ff9800;"></span>GPU fallback</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#e53935;"></span>CPU round-trip</span>
  </div>
</header>
<div class="tab-bar" id="tab-bar" style="display:none;">
  <button class="tab active" data-tab="forward">Forward</button>
  <button class="tab" data-tab="backward">Backward</button>
</div>
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
window.__FERROVIZ_DATA__ = {forward_json};
window.__FERROVIZ_BACKWARD__ = {backward_json};
</script>
<script>
{js}
</script>
</body>
</html>"#,
        css = VIEWER_CSS,
        stats = stats,
        forward_json = forward_json,
        backward_json = backward_json,
        js = VIEWER_JS,
    )
}
