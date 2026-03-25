//! Backward pass visualization via autograd graph walking.
//!
//! After a forward pass produces a loss tensor, we walk the `grad_fn` chain
//! to build a structural backward graph. Optionally, we register gradient
//! hooks on intermediate tensors, run `.backward()`, and record per-op
//! timing from the hook timestamps.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ferrotorch_core::{Float, FerrotorchResult, Tensor};

use crate::model::*;

/// A timestamp event recorded by a gradient hook during backward.
#[derive(Debug, Clone)]
struct HookEvent {
    /// The backward node (grad_fn) that produced this gradient.
    parent_node_id: usize,
    /// Wall-clock timestamp when the hook fired.
    timestamp: Instant,
    /// Device of the gradient tensor that triggered this hook.
    device: SerializableDevice,
}

/// Walk the autograd graph from a loss tensor's `grad_fn` and build a VisGraph
/// representing the backward pass structure.
///
/// Each `GradFn` node becomes a `VisNode` with `category = Backward`.
/// Edges follow the `GradFn::inputs()` tensor references — each input tensor
/// that has its own `grad_fn` provides the next node in the backward chain.
///
/// No execution or timing is performed — this is a structural snapshot.
pub fn capture_backward_graph<T: Float>(loss: &Tensor<T>) -> VisGraph {
    build_backward_graph(loss, false).0
}

/// Walk the autograd graph, register gradient hooks, run `.backward()`, and
/// return a VisGraph with per-op timing from the hook timestamps.
///
/// Each backward node's `observed_duration_us` is computed from the wall-clock
/// gap between consecutive hook firings. This is CPU-side timing — on GPU the
/// actual kernel execution may overlap. For precise GPU timing, combine with
/// nsys + `cuda-trace`.
///
/// The loss tensor must NOT have had `.backward()` called yet.
pub fn capture_backward_graph_timed<T: Float>(
    loss: &Tensor<T>,
) -> FerrotorchResult<VisGraph> {
    let (mut vis, hook_log) = build_backward_graph(loss, true);

    // Run backward — this fires the hooks we registered
    loss.backward()?;

    // Build timing from hook events
    let events = hook_log
        .expect("hook_log should exist when timed=true")
        .lock()
        .unwrap()
        .clone();

    if events.is_empty() {
        return Ok(vis);
    }

    // Sort by timestamp (should already be in order, but be safe)
    let mut sorted = events.clone();
    sorted.sort_by_key(|e| e.timestamp);

    // Group events by parent_node_id and take the LAST timestamp and device
    // per node (the node finishes when its last output gradient hook fires).
    let mut node_last_ts: HashMap<usize, Instant> = HashMap::new();
    let mut node_device: HashMap<usize, SerializableDevice> = HashMap::new();
    for ev in &sorted {
        node_last_ts.insert(ev.parent_node_id, ev.timestamp);
        node_device.insert(ev.parent_node_id, ev.device);
    }

    // Build a sorted list of (last_timestamp, node_id)
    let mut node_completion: Vec<(Instant, usize)> = node_last_ts.into_iter()
        .map(|(id, ts)| (ts, id))
        .collect();
    node_completion.sort_by_key(|(ts, _)| *ts);

    // Compute durations as gaps between consecutive node completions.
    // The first node's duration = time from backward start to its completion.
    let backward_start = sorted.first().map(|e| e.timestamp).unwrap();
    let mut op_events: Vec<OpEvent> = Vec::new();
    let mut prev_ts = backward_start;

    // Build a lookup from node_id to label
    let node_labels: HashMap<usize, String> = vis.nodes.iter()
        .map(|n| (n.id, n.op_label.clone()))
        .collect();

    for (ts, node_id) in &node_completion {
        let duration_us = ts.duration_since(prev_ts).as_micros() as u64;
        let label = node_labels.get(node_id).cloned().unwrap_or_default();

        let device = node_device.get(node_id).copied()
            .unwrap_or(SerializableDevice::Cpu);
        op_events.push(OpEvent {
            node_id: *node_id,
            op_label: label,
            input_devices: vec![],
            output_device: device,
            duration_us,
            output_shape: vec![],
            requires_grad: true,
            gpu_fallback: false,
        });

        prev_ts = *ts;
    }

    let total_duration_us = if let Some(last) = sorted.last() {
        last.timestamp.duration_since(backward_start).as_micros() as u64
    } else {
        0
    };

    // Attach timing to nodes
    let event_map: HashMap<usize, &OpEvent> = op_events.iter()
        .map(|e| (e.node_id, e))
        .collect();
    for node in &mut vis.nodes {
        if let Some(ev) = event_map.get(&node.id) {
            node.observed_duration_us = Some(ev.duration_us);
            node.observed_output_device = Some(ev.output_device);
        }
    }

    vis.runtime = Some(RuntimeProfile {
        total_duration_us,
        op_events,
    });

    Ok(vis)
}

/// Core graph-building logic shared by both timed and untimed variants.
///
/// When `register_hooks` is true, registers gradient hooks on intermediate
/// tensors and returns a shared log that will be populated during `.backward()`.
fn build_backward_graph<T: Float>(
    loss: &Tensor<T>,
    register_hooks: bool,
) -> (VisGraph, Option<Arc<Mutex<Vec<HookEvent>>>>) {
    let hook_log: Arc<Mutex<Vec<HookEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let mut hook_handles = Vec::new();

    let mut nodes: Vec<VisNode> = Vec::new();
    let mut edges: Vec<VisEdge> = Vec::new();

    let mut visited: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut ptr_to_id: HashMap<usize, usize> = HashMap::new();
    let mut ptr_to_arc: HashMap<usize, Arc<dyn ferrotorch_core::GradFn<T>>> = HashMap::new();
    let mut next_id: usize = 0;

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

                    edges.push(VisEdge {
                        from_node: current_id,
                        to_node: input_id,
                        shape: vec![],
                        size_bytes: 0,
                        observed_device_transition: None,
                    });

                    // Register a gradient hook on this intermediate tensor.
                    // The hook fires when current_id's grad_fn finishes
                    // computing this tensor's gradient.
                    if register_hooks {
                        let log = hook_log.clone();
                        let parent_id = current_id;
                        if let Ok(handle) = input_tensor.register_hook(move |grad| {
                            log.lock().unwrap().push(HookEvent {
                                parent_node_id: parent_id,
                                timestamp: Instant::now(),
                                device: SerializableDevice::from(grad.device()),
                            });
                            None // don't modify the gradient
                        }) {
                            hook_handles.push(handle);
                        }
                    }
                } else if input_tensor.is_leaf() && input_tensor.requires_grad() {
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

                    // Register hook on leaf parameters too
                    if register_hooks {
                        let log = hook_log.clone();
                        let parent_id = current_id;
                        if let Ok(handle) = input_tensor.register_hook(move |grad| {
                            log.lock().unwrap().push(HookEvent {
                                parent_node_id: parent_id,
                                timestamp: Instant::now(),
                                device: SerializableDevice::from(grad.device()),
                            });
                            None
                        }) {
                            hook_handles.push(handle);
                        }
                    }
                }
            }
        }
    }

    // Keep hook handles alive — they'll be dropped after backward() is called
    // by the caller. We leak them into a Box so they survive this function.
    if register_hooks {
        std::mem::forget(hook_handles);
    }

    let vis = VisGraph {
        nodes,
        edges,
        fusion_groups: vec![],
        runtime: None,
        #[cfg(feature = "cuda-trace")]
        cuda_trace: None,
    };

    let log = if register_hooks { Some(hook_log) } else { None };
    (vis, log)
}
