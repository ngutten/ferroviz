//! Example: Vision-Attention architecture with full training step.
//!
//! Architecture:
//!   Conv2d -> permute+reshape to (B,S,F) -> MultiheadAttention -> permute+reshape
//!   back to (B,C,H,W) -> Linear -> GELU -> Softmax -> MSELoss -> backward -> AdamW step
//!
//! Demonstrates ferroviz capture on a realistic mixed conv/attention pipeline
//! with GPU execution, backward pass visualization, and GPU fallback detection.

use ferrotorch_core::{Device, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_nn::{
    Conv2d, Linear, MultiheadAttention, GELU, LayerNorm, Module, Reduction,
    MSELoss, Softmax,
};
use ferrotorch_optim::{AdamW, AdamWConfig, Optimizer};

/// A small vision-attention block.
///
/// Flow:
///   input (B, C_in, H, W)
///   -> Conv2d -> (B, C, H', W')
///   -> permute(0,2,3,1) + reshape -> (B, S, C)  where S = H'*W'
///   -> LayerNorm
///   -> MultiheadAttention over the sequence
///   -> permute + reshape back -> (B, C, H', W')
///   -> flatten to (B, C*H'*W') -> Linear -> (B, num_classes)
///   -> GELU
///   -> Softmax
struct VisionAttentionBlock {
    conv: Conv2d<f32>,
    ln: LayerNorm<f32>,
    attn: MultiheadAttention<f32>,
    head: Linear<f32>,
    gelu: GELU,
    softmax: Softmax,
    embed_dim: usize,
    spatial_h: usize,
    spatial_w: usize,
}

impl VisionAttentionBlock {
    fn new(
        in_channels: usize,
        conv_channels: usize,
        kernel_size: (usize, usize),
        num_heads: usize,
        num_classes: usize,
        input_h: usize,
        input_w: usize,
    ) -> FerrotorchResult<Self> {
        let conv = Conv2d::new(in_channels, conv_channels, kernel_size, (1, 1), (0, 0), true)?;
        let spatial_h = input_h - kernel_size.0 + 1;
        let spatial_w = input_w - kernel_size.1 + 1;
        let embed_dim = conv_channels;
        let seq_len = spatial_h * spatial_w;

        let ln = LayerNorm::new(vec![embed_dim], 1e-5, true)?;
        let attn = MultiheadAttention::new(embed_dim, num_heads, true)?;
        let head = Linear::new(seq_len * embed_dim, num_classes, true)?;
        let gelu = GELU::new();
        let softmax = Softmax::new(-1);

        Ok(VisionAttentionBlock {
            conv, ln, attn, head, gelu, softmax,
            embed_dim, spatial_h, spatial_w,
        })
    }

    fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let batch_size = input.shape()[0];

        // 1) Conv2d: (B, C_in, H, W) -> (B, C, H', W')
        let conv_out = self.conv.forward(input)?;

        // 2) Permute to (B, H', W', C) then reshape to (B, S, C)
        let permuted = ferrotorch_core::permute_t(&conv_out, &[0, 2, 3, 1])?;
        let seq_len = self.spatial_h * self.spatial_w;
        let reshaped = ferrotorch_core::view_t(
            &permuted,
            &[batch_size as i64, seq_len as i64, self.embed_dim as i64],
        )?;

        // 3) LayerNorm + MultiheadAttention over the sequence
        let normed = self.ln.forward(&reshaped)?;
        let attn_out = self.attn.forward(&normed)?;

        // 4) Permute + reshape back to (B, C, H', W')
        let spatial = ferrotorch_core::view_t(
            &attn_out,
            &[
                batch_size as i64,
                self.spatial_h as i64,
                self.spatial_w as i64,
                self.embed_dim as i64,
            ],
        )?;
        let back_to_conv = ferrotorch_core::permute_t(&spatial, &[0, 3, 1, 2])?;

        // 5) Flatten to (B, C*H'*W') -> Linear
        let flat = ferrotorch_core::view_t(
            &back_to_conv,
            &[batch_size as i64, -1],
        )?;
        let projected = self.head.forward(&flat)?;

        // 6) GELU activation
        let activated = self.gelu.forward(&projected)?;

        // 7) Softmax
        let output = self.softmax.forward(&activated)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&ferrotorch_nn::Parameter<f32>> {
        let mut params = Vec::new();
        params.extend(Module::parameters(&self.conv));
        params.extend(Module::parameters(&self.ln));
        params.extend(Module::parameters(&self.attn));
        params.extend(Module::parameters(&self.head));
        params
    }

    /// Move all parameters to the given device.
    fn to_device(&mut self, device: Device) -> FerrotorchResult<()> {
        self.conv.to_device(device)?;
        self.ln.to_device(device)?;
        self.attn.to_device(device)?;
        self.head.to_device(device)?;
        Ok(())
    }
}

fn main() -> FerrotorchResult<()> {
    // ── Hyperparameters ──
    let batch_size = 4;
    let in_channels = 3;
    let input_h = 8;
    let input_w = 8;
    let conv_channels = 16;
    let kernel_size = (3, 3);
    let num_heads = 4;
    let num_classes = 10;

    // ── Determine device ──
    let device = if ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        println!("GPU backend available — running on CUDA:0");
        Device::Cuda(0)
    } else {
        println!("No GPU backend — running on CPU");
        Device::Cpu
    };

    // ── Build model and move to device ──
    let mut model = VisionAttentionBlock::new(
        in_channels, conv_channels, kernel_size,
        num_heads, num_classes,
        input_h, input_w,
    )?;
    model.to_device(device)?;

    // ── Random input and target on device ──
    let input = random_tensor(vec![batch_size, in_channels, input_h, input_w]).to(device)?;
    let target = random_tensor(vec![batch_size, num_classes]).to(device)?;

    // ── Optimizer ──
    let params: Vec<ferrotorch_nn::Parameter<f32>> =
        model.parameters().into_iter().cloned().collect();
    let mut optimizer = AdamW::new(
        params,
        AdamWConfig {
            lr: 1e-3,
            weight_decay: 0.01,
            ..AdamWConfig::default()
        },
    );

    // ── Loss function ──
    let loss_fn = MSELoss::new(Reduction::Mean);

    println!("=== Vision-Attention Training Step ===");
    println!("Device: {:?}", device);
    println!("Input shape:  {:?}", input.shape());
    println!("Target shape: {:?}", target.shape());
    println!();

    // ── Forward pass ──
    let output = model.forward(&input)?;
    println!("Output shape: {:?}", output.shape());

    // ── Compute loss ──
    let loss = loss_fn.forward(&output, &target)?;
    let loss_val = loss.to(Device::Cpu)?.data_vec()?[0];
    println!("Loss: {:.6}", loss_val);

    // ── Backward pass ──
    loss.backward()?;
    println!("Backward pass complete.");

    // ── Capture backward graph from the loss tensor ──
    println!();
    println!("=== Backward Graph ===");
    let backward_vis = ferroviz::capture_backward_graph(&loss);
    println!("Backward graph: {} nodes, {} edges",
        backward_vis.nodes.len(), backward_vis.edges.len());
    for node in &backward_vis.nodes {
        println!("  {} (id={})", node.op_label, node.id);
    }

    // ── AdamW step ──
    optimizer.step()?;
    optimizer.zero_grad()?;
    println!("\nAdamW step complete.");

    // ══════════════════════════════════════════════════════
    //  Build the full architecture as an IR graph and
    //  visualize it with ferroviz (static + instrumented)
    //  using GPU tensors to exercise GPU fallback
    // ══════════════════════════════════════════════════════
    println!();
    println!("=== Ferroviz Graph Visualization ===");

    let vis = build_full_architecture_graph(device);

    println!("Graph: {} nodes, {} edges, {} fusion groups",
        vis.nodes.len(), vis.edges.len(), vis.fusion_groups.len());

    if let Some(ref rt) = vis.runtime {
        println!("Total runtime: {} µs", rt.total_duration_us);
        println!();
        println!("Per-op breakdown:");
        for event in &rt.op_events {
            if event.duration_us > 0 || event.gpu_fallback {
                let fallback_tag = if event.gpu_fallback { " [GPU FALLBACK]" } else { "" };
                println!("  {:30} {:>6} µs  device={:?}  shape={:?}{}",
                    event.op_label, event.duration_us,
                    event.output_device, event.output_shape, fallback_tag);
            }
        }
    }

    // Write outputs to examples/output/
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("output");
    std::fs::create_dir_all(&out_dir).expect("failed to create output dir");

    // ── Forward-only graph ──
    let json = ferroviz::render_json(&vis);
    let json_path = out_dir.join("vision_attention_graph.json");
    std::fs::write(&json_path, &json).expect("failed to write JSON");
    println!("\nWrote {} ({} bytes)", json_path.display(), json.len());

    let html = ferroviz::render_html(&vis);
    let html_path = out_dir.join("vision_attention_graph.html");
    std::fs::write(&html_path, &html).expect("failed to write HTML");
    println!("Wrote {} ({} bytes)", html_path.display(), html.len());

    // ── Training step (forward + backward) ──
    let training_vis = ferroviz::model::TrainingStepVis {
        forward: vis,
        backward: backward_vis,
    };

    let training_html = ferroviz::render_html_training_step(&training_vis);
    let training_path = out_dir.join("vision_attention_training.html");
    std::fs::write(&training_path, &training_html).expect("failed to write training HTML");
    println!("Wrote {} ({} bytes)", training_path.display(), training_html.len());

    println!("\nDone.");
    Ok(())
}

/// Build the full vision-attention architecture as an IR graph and run
/// the instrumented interpreter to get per-op profiling data.
///
/// When `device` is a CUDA device, input tensors are placed on GPU so the
/// instrumented interpreter exercises GPU execution paths (and GPU fallback
/// for ops with missing kernels).
fn build_full_architecture_graph(device: Device) -> ferroviz::VisGraph {
    use ferrotorch_jit::graph::{IrGraph, IrOpKind};

    let mut g = IrGraph::new();

    // ── Graph inputs ──
    let x_patches = g.add_input(vec![144, 27]);
    let conv_w    = g.add_input(vec![27, 16]);
    let conv_b    = g.add_input(vec![16]);
    let attn_qw   = g.add_input(vec![16, 16]);
    let attn_kw   = g.add_input(vec![16, 16]);
    let attn_vw   = g.add_input(vec![16, 16]);
    let attn_ow   = g.add_input(vec![16, 16]);
    let head_w    = g.add_input(vec![10, 576]);
    let head_b    = g.add_input(vec![10]);
    let target    = g.add_input(vec![4, 10]);

    // ── Conv2d (im2col matmul): patches @ weight + bias ──
    let (_, conv_mm) = g.add_node(
        IrOpKind::Mm, vec![x_patches, conv_w], vec![vec![144, 16]],
    );
    let (_, conv_biased) = g.add_node(
        IrOpKind::Add, vec![conv_mm[0], conv_b], vec![vec![144, 16]],
    );

    // ── Reshape to (B, S, C) = (4, 36, 16) ──
    let (_, seq) = g.add_node(
        IrOpKind::Reshape { shape: vec![4, 36, 16] },
        vec![conv_biased[0]],
        vec![vec![4, 36, 16]],
    );

    // ── Multi-head attention ──
    let (_, flat_seq) = g.add_node(
        IrOpKind::Reshape { shape: vec![144, 16] },
        vec![seq[0]],
        vec![vec![144, 16]],
    );

    // Q, K, V projections
    let (_, q_flat) = g.add_node(IrOpKind::Mm, vec![flat_seq[0], attn_qw], vec![vec![144, 16]]);
    let (_, k_flat) = g.add_node(IrOpKind::Mm, vec![flat_seq[0], attn_kw], vec![vec![144, 16]]);
    let (_, v_flat) = g.add_node(IrOpKind::Mm, vec![flat_seq[0], attn_vw], vec![vec![144, 16]]);

    // Attention scores
    let (_, k_t) = g.add_node(
        IrOpKind::Transpose, vec![k_flat[0]], vec![vec![16, 144]],
    );
    let (_, scores) = g.add_node(
        IrOpKind::Mm, vec![q_flat[0], k_t[0]], vec![vec![144, 144]],
    );
    let (_, attn_w) = g.add_node(
        IrOpKind::Softmax, vec![scores[0]], vec![vec![144, 144]],
    );

    // Attention output
    let (_, attn_out) = g.add_node(
        IrOpKind::Mm, vec![attn_w[0], v_flat[0]], vec![vec![144, 16]],
    );

    // Output projection
    let (_, proj_flat) = g.add_node(
        IrOpKind::Mm, vec![attn_out[0], attn_ow], vec![vec![144, 16]],
    );

    // Reshape back
    let (_, proj) = g.add_node(
        IrOpKind::Reshape { shape: vec![4, 36, 16] },
        vec![proj_flat[0]],
        vec![vec![4, 36, 16]],
    );
    let (_, spatial) = g.add_node(
        IrOpKind::Reshape { shape: vec![4, 16, 6, 6] },
        vec![proj[0]],
        vec![vec![4, 16, 6, 6]],
    );

    // ── Flatten -> Linear -> GELU -> Softmax ──
    let (_, flat) = g.add_node(
        IrOpKind::Reshape { shape: vec![4, 576] },
        vec![spatial[0]],
        vec![vec![4, 576]],
    );
    let (_, linear_out) = g.add_node(
        IrOpKind::Linear, vec![flat[0], head_w, head_b], vec![vec![4, 10]],
    );
    let (_, gelu_out) = g.add_node(
        IrOpKind::Gelu, vec![linear_out[0]], vec![vec![4, 10]],
    );
    let (_, sm_out) = g.add_node(
        IrOpKind::Softmax, vec![gelu_out[0]], vec![vec![4, 10]],
    );

    // ── MSE Loss: mean((pred - target)^2) ──
    let (_, diff) = g.add_node(
        IrOpKind::Sub, vec![sm_out[0], target], vec![vec![4, 10]],
    );
    let (_, sq) = g.add_node(
        IrOpKind::Mul, vec![diff[0], diff[0]], vec![vec![4, 10]],
    );
    let (_, loss) = g.add_node(
        IrOpKind::Mean, vec![sq[0]], vec![vec![1]],
    );

    g.set_outputs(loss);

    // ── Static analysis ──
    let mut vis = ferroviz::analyze(&g);

    // ── Instrumented interpreter with device-appropriate tensors ──
    let inputs: Vec<Tensor<f32>> = vec![
        random_tensor_on(vec![144, 27], device),
        random_tensor_on(vec![27, 16], device),
        random_tensor_on(vec![16], device),
        random_tensor_on(vec![16, 16], device),
        random_tensor_on(vec![16, 16], device),
        random_tensor_on(vec![16, 16], device),
        random_tensor_on(vec![16, 16], device),
        random_tensor_on(vec![10, 576], device),
        random_tensor_on(vec![10], device),
        random_tensor_on(vec![4, 10], device),
    ];

    match ferroviz::instrument::instrumented_interpret(&g, &inputs) {
        Ok((_output, profile)) => {
            let event_map: std::collections::HashMap<usize, &ferroviz::model::OpEvent> =
                profile.op_events.iter().map(|e| (e.node_id, e)).collect();

            for node in &mut vis.nodes {
                if let Some(event) = event_map.get(&node.id) {
                    node.observed_input_devices = Some(event.input_devices.clone());
                    node.observed_output_device = Some(event.output_device);
                    node.observed_duration_us = Some(event.duration_us);
                    node.requires_grad = Some(event.requires_grad);
                    node.gpu_fallback = Some(event.gpu_fallback);
                }
            }
            vis.runtime = Some(profile);
        }
        Err(e) => {
            eprintln!("Instrumented interpret warning: {}", e);
        }
    }

    vis
}

fn random_tensor(shape: Vec<usize>) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.7123 + 0.3).sin() * 0.5)
        .collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, true).unwrap()
}

fn random_tensor_on(shape: Vec<usize>, device: Device) -> Tensor<f32> {
    let t = random_tensor(shape);
    if device != Device::Cpu {
        t.to(device).unwrap()
    } else {
        t
    }
}
