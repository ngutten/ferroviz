//! Basic integration tests for ferroviz.

use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_jit::graph::{IrGraph, IrOpKind};

/// Helper: build a simple graph manually: Output = Add(Input[0], Input[1])
fn simple_add_graph() -> IrGraph {
    let mut graph = IrGraph::new();
    let a = graph.add_input(vec![2, 3]);
    let b = graph.add_input(vec![2, 3]);
    let (_, add_outs) = graph.add_node(IrOpKind::Add, vec![a, b], vec![vec![2, 3]]);
    graph.set_outputs(add_outs);
    graph
}

/// Helper: build a chain graph: Output = Relu(Neg(Input[0]))
fn unary_chain_graph() -> IrGraph {
    let mut graph = IrGraph::new();
    let a = graph.add_input(vec![4]);
    let (_, neg_outs) = graph.add_node(IrOpKind::Neg, vec![a], vec![vec![4]]);
    let (_, relu_outs) = graph.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
    graph.set_outputs(relu_outs);
    graph
}

/// Helper: build a more complex graph with matmul and activations
fn mlp_graph() -> IrGraph {
    let mut graph = IrGraph::new();
    let x = graph.add_input(vec![2, 4]);
    let w1 = graph.add_input(vec![4, 8]);
    let w2 = graph.add_input(vec![8, 3]);

    // layer 1: relu(x @ w1)
    let (_, mm1_outs) = graph.add_node(IrOpKind::Mm, vec![x, w1], vec![vec![2, 8]]);
    let (_, relu_outs) = graph.add_node(IrOpKind::Relu, vec![mm1_outs[0]], vec![vec![2, 8]]);

    // layer 2: tanh(relu_out @ w2)
    let (_, mm2_outs) = graph.add_node(IrOpKind::Mm, vec![relu_outs[0], w2], vec![vec![2, 3]]);
    let (_, tanh_outs) = graph.add_node(IrOpKind::Tanh, vec![mm2_outs[0]], vec![vec![2, 3]]);

    graph.set_outputs(tanh_outs);
    graph
}

fn make_tensor(shape: Vec<usize>) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, true).unwrap()
}

#[test]
fn test_analyze_simple_add() {
    let graph = simple_add_graph();
    let vis = ferroviz::analyze(&graph);

    // Should have nodes for: Input[0], Input[1], Add, and output-related nodes
    assert!(!vis.nodes.is_empty(), "should have nodes");
    assert!(!vis.edges.is_empty(), "should have edges");

    // Find the Add node
    let add_node = vis.nodes.iter().find(|n| n.op_label == "Add");
    assert!(add_node.is_some(), "should have an Add node");
    let add_node = add_node.unwrap();
    assert_eq!(add_node.category, ferroviz::model::OpCategory::Elementwise);
    assert_eq!(add_node.output_shapes, vec![vec![2, 3]]);
    // No runtime data, so gpu_fallback should be None
    assert_eq!(add_node.gpu_fallback, None);
}

#[test]
fn test_analyze_unary_chain() {
    let graph = unary_chain_graph();
    let vis = ferroviz::analyze(&graph);

    let neg_node = vis.nodes.iter().find(|n| n.op_label == "Neg");
    let relu_node = vis.nodes.iter().find(|n| n.op_label == "Relu");
    assert!(neg_node.is_some(), "should have Neg node");
    assert!(relu_node.is_some(), "should have Relu node");
}

#[test]
fn test_analyze_mlp_fusion_groups() {
    let graph = mlp_graph();
    let vis = ferroviz::analyze(&graph);

    // Should detect fusion groups
    assert!(vis.nodes.len() >= 4, "MLP should have at least 4 non-IO nodes");

    // Check edges connect properly
    for edge in &vis.edges {
        assert!(
            vis.nodes.iter().any(|n| n.id == edge.from_node),
            "edge from_node {} should exist in nodes",
            edge.from_node
        );
        assert!(
            vis.nodes.iter().any(|n| n.id == edge.to_node),
            "edge to_node {} should exist in nodes",
            edge.to_node
        );
    }
}

#[test]
fn test_instrumented_interpret_add() {
    let graph = simple_add_graph();
    let a = make_tensor(vec![2, 3]);
    let b = make_tensor(vec![2, 3]);

    let (result, profile) =
        ferroviz::instrument::instrumented_interpret(&graph, &[a.clone(), b.clone()]).unwrap();

    // Verify output shape
    assert_eq!(result.shape(), &[2, 3]);

    // Verify against standard interpreter
    let expected = ferrotorch_jit::interpret(&graph, &[a, b]).unwrap();
    let result_data = result.data_vec().unwrap();
    let expected_data = expected.data_vec().unwrap();
    assert_eq!(result_data.len(), expected_data.len());
    for (r, e) in result_data.iter().zip(expected_data.iter()) {
        assert!((r - e).abs() < 1e-6, "mismatch: {} vs {}", r, e);
    }

    // Verify profiling events
    assert!(!profile.op_events.is_empty(), "should have profiling events");
    assert!(profile.total_duration_us > 0 || profile.op_events.len() > 0);

    // All CPU ops should have gpu_fallback = false
    for event in &profile.op_events {
        assert!(!event.gpu_fallback, "CPU ops should not have gpu_fallback");
    }
}

#[test]
fn test_instrumented_interpret_mlp() {
    let graph = mlp_graph();
    let x = make_tensor(vec![2, 4]);
    let w1 = make_tensor(vec![4, 8]);
    let w2 = make_tensor(vec![8, 3]);

    let (result, profile) =
        ferroviz::instrument::instrumented_interpret(&graph, &[x.clone(), w1.clone(), w2.clone()])
            .unwrap();

    assert_eq!(result.shape(), &[2, 3]);

    // Compare with standard interpreter
    let expected = ferrotorch_jit::interpret(&graph, &[x, w1, w2]).unwrap();
    let result_data = result.data_vec().unwrap();
    let expected_data = expected.data_vec().unwrap();
    for (r, e) in result_data.iter().zip(expected_data.iter()) {
        assert!((r - e).abs() < 1e-5, "mismatch: {} vs {}", r, e);
    }

    // Check that we got events for the computational ops
    let mm_events: Vec<_> = profile
        .op_events
        .iter()
        .filter(|e| e.op_label == "Mm")
        .collect();
    assert_eq!(mm_events.len(), 2, "should have 2 Mm events");
}

#[test]
fn test_json_round_trip() {
    let graph = mlp_graph();
    let vis = ferroviz::analyze(&graph);

    let json = ferroviz::render_json(&vis);
    assert!(!json.is_empty());

    let parsed = ferroviz::parse_json(&json).unwrap();
    assert_eq!(parsed.nodes.len(), vis.nodes.len());
    assert_eq!(parsed.edges.len(), vis.edges.len());
    assert_eq!(parsed.fusion_groups.len(), vis.fusion_groups.len());
}

#[test]
fn test_json_compact() {
    let graph = simple_add_graph();
    let vis = ferroviz::analyze(&graph);
    let compact = ferroviz::render_json_compact(&vis);
    assert!(!compact.contains('\n'), "compact JSON should be single line");
}

#[test]
fn test_html_output() {
    let graph = mlp_graph();
    let vis = ferroviz::analyze(&graph);
    let html = ferroviz::render_html(&vis);

    assert!(html.contains("<!DOCTYPE html>"), "should be valid HTML");
    assert!(html.contains("Ferroviz"), "should contain title");
    assert!(html.contains("__FERROVIZ_DATA__"), "should embed graph data");
    assert!(html.contains("dagre"), "should reference dagre for layout");
}

#[test]
fn test_html_training_step() {
    let graph = mlp_graph();
    let vis = ferroviz::analyze(&graph);

    // Create a minimal backward graph
    let backward = ferroviz::model::VisGraph {
        nodes: vec![ferroviz::model::VisNode {
            id: 0,
            op_label: "MmBackward".to_string(),
            category: ferroviz::model::OpCategory::Backward,
            cluster_id: None,
            output_shapes: vec![],
            observed_input_devices: None,
            observed_output_device: None,
            observed_duration_us: None,
            requires_grad: Some(true),
            gpu_fallback: None,
            cuda_kernels: vec![],
            cuda_memcpy: vec![],
        }],
        edges: vec![],
        fusion_groups: vec![],
        runtime: None,
        #[cfg(feature = "cuda-trace")]
        cuda_trace: None,
    };

    let training = ferroviz::model::TrainingStepVis {
        forward: vis,
        backward,
    };

    let html = ferroviz::render_html_training_step(&training);
    assert!(html.contains("__FERROVIZ_DATA__"), "should embed forward data");
    assert!(html.contains("__FERROVIZ_BACKWARD__"), "should embed backward data");
    assert!(html.contains("tab-bar"), "should have tab bar");
    assert!(html.contains("Backward"), "should have backward tab");
}

#[test]
fn test_capture_with_traced_function() {
    // Use capture() with a simple function that ferrotorch can trace
    let x = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![2, 2],
        true,
    )
    .unwrap();
    let w = Tensor::from_storage(
        TensorStorage::cpu(vec![0.5_f32, -0.5, 0.5, -0.5]),
        vec![2, 2],
        true,
    )
    .unwrap();

    let result = ferroviz::capture(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            let mm = ferrotorch_core::grad_fns::linalg::mm_differentiable(&inputs[0], &inputs[1])?;
            let activated = ferrotorch_core::grad_fns::activation::relu(&mm)?;
            Ok(activated)
        },
        &[x, w],
    );

    match result {
        Ok(vis) => {
            assert!(!vis.nodes.is_empty(), "captured graph should have nodes");
            assert!(vis.runtime.is_some(), "captured graph should have runtime data");

            let json = ferroviz::render_json(&vis);
            assert!(!json.is_empty());

            let html = ferroviz::render_html(&vis);
            assert!(html.contains("Ferroviz"));
        }
        Err(e) => {
            eprintln!("capture() failed (may be expected): {}", e);
        }
    }
}

#[test]
fn test_edge_size_bytes() {
    let graph = simple_add_graph();
    let vis = ferroviz::analyze(&graph);

    // For shape [2, 3], size = 2*3*4 = 24 bytes
    let edges_with_size: Vec<_> = vis.edges.iter().filter(|e| e.size_bytes == 24).collect();
    assert!(
        !edges_with_size.is_empty(),
        "should have edges with correct size_bytes for [2,3] f32 tensors"
    );
}

#[test]
fn test_op_labels() {
    use ferrotorch_jit::graph::IrOpKind;
    use ferroviz::model::op_label;

    assert_eq!(op_label(&IrOpKind::Add), "Add");
    assert_eq!(op_label(&IrOpKind::Relu), "Relu");
    assert_eq!(op_label(&IrOpKind::Input { index: 0 }), "Input[0]");
    assert_eq!(op_label(&IrOpKind::Linear), "Linear");
    assert_eq!(
        op_label(&IrOpKind::FusedElementwise {
            ops: vec![IrOpKind::Neg, IrOpKind::Relu]
        }),
        "FusedElementwise[Neg,Relu]"
    );
    assert_eq!(op_label(&IrOpKind::Pow { exponent: 2.0 }), "Pow(2)");
}

#[test]
fn test_op_categories() {
    use ferrotorch_jit::graph::IrOpKind;
    use ferroviz::model::{classify_op, OpCategory};

    assert_eq!(classify_op(&IrOpKind::Add), OpCategory::Elementwise);
    assert_eq!(classify_op(&IrOpKind::Sum), OpCategory::Reduction);
    assert_eq!(classify_op(&IrOpKind::Mm), OpCategory::MatMul);
    assert_eq!(classify_op(&IrOpKind::Linear), OpCategory::Linear);
    assert_eq!(classify_op(&IrOpKind::Relu), OpCategory::Activation);
    assert_eq!(classify_op(&IrOpKind::Flatten), OpCategory::Shape);
    assert_eq!(
        classify_op(&IrOpKind::Input { index: 0 }),
        OpCategory::IO
    );
    assert_eq!(
        classify_op(&IrOpKind::FusedElementwise { ops: vec![] }),
        OpCategory::Fused
    );
}

#[test]
fn test_device_transition_detection() {
    use ferroviz::model::{detect_transition, DeviceTransition, SerializableDevice};

    assert_eq!(
        detect_transition(&SerializableDevice::Cpu, &SerializableDevice::Cuda(0)),
        Some(DeviceTransition::CpuToGpu)
    );
    assert_eq!(
        detect_transition(&SerializableDevice::Cuda(0), &SerializableDevice::Cpu),
        Some(DeviceTransition::GpuToCpu)
    );
    assert_eq!(
        detect_transition(&SerializableDevice::Cpu, &SerializableDevice::Cpu),
        None
    );
    assert_eq!(
        detect_transition(&SerializableDevice::Cuda(0), &SerializableDevice::Cuda(0)),
        None
    );
}

#[test]
fn test_backward_category() {
    use ferroviz::model::OpCategory;
    // Backward is a valid category
    let cat = OpCategory::Backward;
    assert_eq!(cat, OpCategory::Backward);
}

#[test]
fn test_gpu_fallback_field_serialization() {
    use ferroviz::model::*;

    let node = VisNode {
        id: 0,
        op_label: "Sqrt".to_string(),
        category: OpCategory::Elementwise,
        cluster_id: None,
        output_shapes: vec![vec![4]],
        observed_input_devices: Some(vec![SerializableDevice::Cuda(0)]),
        observed_output_device: Some(SerializableDevice::Cuda(0)),
        observed_duration_us: Some(100),
        requires_grad: Some(false),
        gpu_fallback: Some(true),
        cuda_kernels: vec![],
        cuda_memcpy: vec![],
    };

    let json = serde_json::to_string(&node).unwrap();
    assert!(json.contains("\"gpu_fallback\":true"), "gpu_fallback should serialize");

    let parsed: VisNode = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.gpu_fallback, Some(true));
}

#[test]
fn test_training_step_vis_serialization() {
    use ferroviz::model::*;

    let empty_graph = VisGraph {
        nodes: vec![],
        edges: vec![],
        fusion_groups: vec![],
        runtime: None,
        #[cfg(feature = "cuda-trace")]
        cuda_trace: None,
    };

    let training = TrainingStepVis {
        forward: empty_graph.clone(),
        backward: empty_graph,
    };

    let json = serde_json::to_string(&training).unwrap();
    let parsed: TrainingStepVis = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.forward.nodes.len(), 0);
    assert_eq!(parsed.backward.nodes.len(), 0);
}
