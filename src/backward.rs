//! Backward pass visualization via autograd graph walking.
//!
//! After a forward pass produces a loss tensor and `.backward()` is called,
//! we walk the `grad_fn` chain to build a structural backward graph showing
//! which backward ops exist and how they connect.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use ferrotorch_core::{Float, Tensor};

use crate::model::*;

/// Walk the autograd graph from a loss tensor's `grad_fn` and build a VisGraph
/// representing the backward pass structure.
///
/// Each `GradFn` node becomes a `VisNode` with `category = Backward`.
/// Edges follow the `GradFn::inputs()` tensor references — each input tensor
/// that has its own `grad_fn` provides the next node in the backward chain.
///
/// No execution or timing is performed — this is a structural snapshot.
pub fn capture_backward_graph<T: Float>(loss: &Tensor<T>) -> VisGraph {
    let mut nodes: Vec<VisNode> = Vec::new();
    let mut edges: Vec<VisEdge> = Vec::new();

    // BFS through the grad_fn chain. We use the Arc pointer address as
    // a unique identifier and assign sequential IDs for the VisGraph.
    let mut visited: HashSet<usize> = HashSet::new();
    // Queue holds (grad_fn Arc pointer as usize, the Arc itself)
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut ptr_to_id: HashMap<usize, usize> = HashMap::new();
    // Store grad_fn Arcs so we can re-access them by pointer
    let mut ptr_to_arc: HashMap<usize, Arc<dyn ferrotorch_core::GradFn<T>>> = HashMap::new();
    let mut next_id: usize = 0;

    // Seed from the loss tensor's grad_fn
    if let Some(root_gf) = loss.grad_fn() {
        let root_ptr = Arc::as_ptr(root_gf) as *const () as usize;
        visited.insert(root_ptr);
        let root_id = next_id;
        ptr_to_id.insert(root_ptr, root_id);
        ptr_to_arc.insert(root_ptr, root_gf.clone());
        next_id += 1;

        nodes.push(VisNode {
            id: root_id,
            op_label: root_gf.name().to_string(),
            category: OpCategory::Backward,
            cluster_id: None,
            output_shapes: vec![],
            observed_input_devices: None,
            observed_output_device: None,
            observed_duration_us: None,
            requires_grad: Some(true),
            gpu_fallback: None,
            cuda_kernels: vec![],
            cuda_memcpy: vec![],
        });

        queue.push_back(root_ptr);

        while let Some(current_ptr) = queue.pop_front() {
            let current_id = ptr_to_id[&current_ptr];
            let gf = ptr_to_arc[&current_ptr].clone();

            // Each input is a tensor; if it has a grad_fn, that's the next
            // backward node in the chain.
            for input_tensor in gf.inputs() {
                if let Some(input_gf) = input_tensor.grad_fn() {
                    let input_ptr = Arc::as_ptr(input_gf) as *const () as usize;

                    let input_id = if let Some(&existing_id) = ptr_to_id.get(&input_ptr) {
                        existing_id
                    } else {
                        let new_id = next_id;
                        next_id += 1;
                        ptr_to_id.insert(input_ptr, new_id);
                        ptr_to_arc.insert(input_ptr, input_gf.clone());

                        nodes.push(VisNode {
                            id: new_id,
                            op_label: input_gf.name().to_string(),
                            category: OpCategory::Backward,
                            cluster_id: None,
                            output_shapes: vec![],
                            observed_input_devices: None,
                            observed_output_device: None,
                            observed_duration_us: None,
                            requires_grad: Some(true),
                            gpu_fallback: None,
                            cuda_kernels: vec![],
                            cuda_memcpy: vec![],
                        });

                        if visited.insert(input_ptr) {
                            queue.push_back(input_ptr);
                        }

                        new_id
                    };

                    // Edge from current backward op to its input backward op
                    // (gradient flows from loss toward leaves)
                    edges.push(VisEdge {
                        from_node: current_id,
                        to_node: input_id,
                        shape: vec![],
                        size_bytes: 0,
                        observed_device_transition: None,
                    });
                } else if input_tensor.is_leaf() && input_tensor.requires_grad() {
                    // Leaf parameter — add as a terminal node
                    let leaf_id = next_id;
                    next_id += 1;

                    let dev = Some(SerializableDevice::from(input_tensor.device()));
                    nodes.push(VisNode {
                        id: leaf_id,
                        op_label: format!("Param{:?}", input_tensor.shape()),
                        category: OpCategory::IO,
                        cluster_id: None,
                        output_shapes: vec![input_tensor.shape().to_vec()],
                        observed_input_devices: None,
                        observed_output_device: dev,
                        observed_duration_us: None,
                        requires_grad: Some(true),
                        gpu_fallback: None,
                        cuda_kernels: vec![],
                        cuda_memcpy: vec![],
                    });

                    edges.push(VisEdge {
                        from_node: current_id,
                        to_node: leaf_id,
                        shape: input_tensor.shape().to_vec(),
                        size_bytes: input_tensor.shape().iter().product::<usize>()
                            * std::mem::size_of::<f32>(),
                        observed_device_transition: None,
                    });
                }
            }
        }
    }

    VisGraph {
        nodes,
        edges,
        fusion_groups: vec![],
        runtime: None,
        #[cfg(feature = "cuda-trace")]
        cuda_trace: None,
    }
}
