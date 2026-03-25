//! The `capture()` and `analyze()` APIs that orchestrate tracing, optimization,
//! instrumented interpretation, and fusion analysis into a VisGraph.

use std::collections::HashMap;

use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use ferrotorch_jit::graph::IrGraph;
use ferrotorch_jit::OptimizationConfig;

use crate::fusion_analysis::{analyze_fusion_groups, build_cluster_map};
use crate::instrument::instrumented_interpret;
use crate::model::*;

/// Trace a function, optimize the graph, run the instrumented interpreter,
/// analyze fusion groups, and return the full VisGraph with runtime data.
pub fn capture<T, F>(
    f: F,
    example_inputs: &[Tensor<T>],
) -> FerrotorchResult<VisGraph>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    // 1. Trace to get the IR graph
    let graph = ferrotorch_jit::trace(&f, example_inputs)?;

    // 2. Clone and optimize
    let mut opt_graph = graph.clone();
    let config = OptimizationConfig {
        constant_folding: true,
        dead_code_elimination: true,
        operator_fusion: true,
        memory_planning: false,
    };
    ferrotorch_jit::optimize(&mut opt_graph, &config);

    // 3. Fusion analysis on the optimized graph
    let fusion_clusters = analyze_fusion_groups(&opt_graph);
    let cluster_map = build_cluster_map(&fusion_clusters);

    // 4. Run instrumented interpreter on the optimized graph
    let (_output, profile) = instrumented_interpret(&opt_graph, example_inputs)?;

    // 5. Build runtime event lookup by node_id
    let event_map: HashMap<usize, &OpEvent> = profile
        .op_events
        .iter()
        .map(|e| (e.node_id, e))
        .collect();

    // 6. Build VisGraph
    let vis = build_vis_graph(&opt_graph, &fusion_clusters, &cluster_map, Some(&profile), &event_map);

    Ok(vis)
}

/// Static analysis only — no runtime execution. Builds a VisGraph from an IrGraph
/// with fusion analysis but no timing or device observation data.
pub fn analyze(graph: &IrGraph) -> VisGraph {
    let fusion_clusters = analyze_fusion_groups(graph);
    let cluster_map = build_cluster_map(&fusion_clusters);
    let event_map: HashMap<usize, &OpEvent> = HashMap::new();

    build_vis_graph(graph, &fusion_clusters, &cluster_map, None, &event_map)
}

fn build_vis_graph(
    graph: &IrGraph,
    fusion_clusters: &[FusionCluster],
    cluster_map: &HashMap<usize, usize>,
    runtime: Option<&RuntimeProfile>,
    event_map: &HashMap<usize, &OpEvent>,
) -> VisGraph {
    // Build value→producer node mapping
    let mut value_producer: HashMap<usize, usize> = HashMap::new();
    for node in &graph.nodes {
        for out in &node.outputs {
            value_producer.insert(out.0, node.id.0);
        }
    }

    // Build nodes
    let nodes: Vec<VisNode> = graph
        .nodes
        .iter()
        .map(|node| {
            let nid = node.id.0;
            let event = event_map.get(&nid);

            let output_shapes: Vec<Vec<usize>> = node
                .outputs
                .iter()
                .filter_map(|vid| graph.values.iter().find(|v| v.id == *vid))
                .map(|v| v.shape.clone())
                .collect();

            VisNode {
                id: nid,
                op_label: op_label(&node.op),
                category: classify_op(&node.op),
                cluster_id: cluster_map.get(&nid).copied(),
                output_shapes,
                observed_input_devices: event.map(|e| e.input_devices.clone()),
                observed_output_device: event.map(|e| e.output_device),
                observed_duration_us: event.map(|e| e.duration_us),
                requires_grad: event.map(|e| e.requires_grad),
                cuda_kernels: vec![],
                cuda_memcpy: vec![],
            }
        })
        .collect();

    // Build edges from node inputs
    let mut edges: Vec<VisEdge> = Vec::new();
    for node in &graph.nodes {
        for input_vid in &node.inputs {
            if let Some(&from_node) = value_producer.get(&input_vid.0) {
                let value = graph.values.iter().find(|v| v.id == *input_vid);
                let shape = value.map(|v| v.shape.clone()).unwrap_or_default();
                let size_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();

                // Detect device transitions from runtime data
                let from_event = event_map.get(&from_node);
                let to_event = event_map.get(&node.id.0);
                let transition = match (from_event, to_event) {
                    (Some(fe), Some(te)) => {
                        detect_transition(&fe.output_device, &te.output_device)
                    }
                    _ => None,
                };

                edges.push(VisEdge {
                    from_node,
                    to_node: node.id.0,
                    shape,
                    size_bytes,
                    observed_device_transition: transition,
                });
            }
        }
    }

    VisGraph {
        nodes,
        edges,
        fusion_groups: fusion_clusters.to_vec(),
        runtime: runtime.cloned(),
        #[cfg(feature = "cuda-trace")]
        cuda_trace: None,
    }
}
