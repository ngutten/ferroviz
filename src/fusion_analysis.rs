//! Wraps ferrotorch-jit's `find_fusion_groups()` and converts results
//! into `FusionCluster` data for the visualization graph.

use ferrotorch_jit::graph::IrGraph;
use ferrotorch_jit::FusionGroupKind;

use crate::model::FusionCluster;

/// Analyze fusion groups in the graph and return clusters with assigned IDs.
pub fn analyze_fusion_groups(graph: &IrGraph) -> Vec<FusionCluster> {
    let groups = ferrotorch_jit::dag_fusion::find_fusion_groups(graph);

    groups
        .into_iter()
        .enumerate()
        .map(|(id, group)| FusionCluster {
            id,
            kind: format_fusion_kind(&group.kind),
            node_ids: group.node_ids.iter().map(|nid| nid.0).collect(),
        })
        .collect()
}

/// Build a map from node ID to cluster ID for fast lookup.
pub fn build_cluster_map(clusters: &[FusionCluster]) -> std::collections::HashMap<usize, usize> {
    let mut map = std::collections::HashMap::new();
    for cluster in clusters {
        for &node_id in &cluster.node_ids {
            map.insert(node_id, cluster.id);
        }
    }
    map
}

fn format_fusion_kind(kind: &FusionGroupKind) -> String {
    match kind {
        FusionGroupKind::Elementwise => "Elementwise".to_string(),
        FusionGroupKind::Reduction => "Reduction".to_string(),
        FusionGroupKind::MatMul => "MatMul".to_string(),
        FusionGroupKind::Linear => "Linear".to_string(),
        FusionGroupKind::Opaque => "Opaque".to_string(),
    }
}
