//! Example: Vision-Attention architecture with full training step.
//!
//! Architecture:
//!   Conv2d → permute+reshape to (B,S,F) → MultiheadAttention → permute+reshape
//!   back to (B,C,H,W) → Linear → GELU → Softmax → MSELoss → backward → AdamW step
//!
//! Demonstrates ferroviz capture on a realistic mixed conv/attention pipeline
//! operating on random data.

use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_nn::{
    Conv2d, Linear, MultiheadAttention, GELU, LayerNorm, Module, Reduction,
    MSELoss, Softmax,
};
use ferrotorch_optim::{AdamW, AdamWConfig, Optimizer};

/// A small vision-attention block.
///
/// Flow:
///   input (B, C_in, H, W)
///   → Conv2d → (B, C, H', W')
///   → permute(0,2,3,1) + reshape → (B, S, C)  where S = H'*W'
///   → LayerNorm
///   → MultiheadAttention over the sequence
///   → permute + reshape back → (B, C, H', W')
///   → flatten to (B, C*H'*W') → Linear → (B, num_classes)
///   → GELU
///   → Softmax
struct VisionAttentionBlock {
    conv: Conv2d<f32>,
    ln: LayerNorm<f32>,
    attn: MultiheadAttention<f32>,
    head: Linear<f32>,
    gelu: GELU,
    softmax: Softmax,
    // Cached spatial dims for reshape (set after first forward)
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
        // Output spatial dims after conv with no padding, stride 1
        let spatial_h = input_h - kernel_size.0 + 1;
        let spatial_w = input_w - kernel_size.1 + 1;
        let embed_dim = conv_channels;
        let seq_len = spatial_h * spatial_w;

        let ln = LayerNorm::new(vec![embed_dim], 1e-5, true)?;
        let attn = MultiheadAttention::new(embed_dim, num_heads, true)?;

        // Linear head: from flattened (seq_len * embed_dim) to num_classes
        let head = Linear::new(seq_len * embed_dim, num_classes, true)?;

        let gelu = GELU::new();
        let softmax = Softmax::new(-1);

        Ok(VisionAttentionBlock {
            conv, ln, attn, head, gelu, softmax,
            embed_dim,
            spatial_h, spatial_w,
        })
    }

    fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let batch_size = input.shape()[0];

        // 1) Conv2d: (B, C_in, H, W) → (B, C, H', W')
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
        //    attn_out is (B, S, C) → reshape to (B, H', W', C) → permute to (B, C, H', W')
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

        // 5) Flatten to (B, C*H'*W') → Linear (in-place style: direct projection)
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
}

fn main() -> FerrotorchResult<()> {
    // ── Hyperparameters ──
    let batch_size = 4;
    let in_channels = 3;
    let input_h = 8;
    let input_w = 8;
    let conv_channels = 16; // must be divisible by num_heads
    let kernel_size = (3, 3);
    let num_heads = 4;
    let num_classes = 10;

    // ── Build model ──
    let model = VisionAttentionBlock::new(
        in_channels, conv_channels, kernel_size,
        num_heads, num_classes,
        input_h, input_w,
    )?;

    // ── Random input and target ──
    let input = random_tensor(vec![batch_size, in_channels, input_h, input_w]);
    let target = random_tensor(vec![batch_size, num_classes]);

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
    println!("Input shape:  {:?}", input.shape());
    println!("Target shape: {:?}", target.shape());
    println!();

    // ── Forward pass ──
    let output = model.forward(&input)?;
    println!("Output shape: {:?}", output.shape());

    // ── Compute loss ──
    let loss = loss_fn.forward(&output, &target)?;
    let loss_val = loss.data_vec()?[0];
    println!("Loss: {:.6}", loss_val);

    // ── Backward pass ──
    loss.backward()?;
    println!("Backward pass complete.");

    // ── AdamW step ──
    optimizer.step()?;
    optimizer.zero_grad()?;
    println!("AdamW step complete.");

    // ══════════════════════════════════════════════════════
    //  Build the full architecture as an IR graph and
    //  visualize it with ferroviz (static + instrumented)
    // ══════════════════════════════════════════════════════
    println!();
    println!("=== Ferroviz Graph Visualization ===");

    let vis = build_full_architecture_graph();

    println!("Graph: {} nodes, {} edges, {} fusion groups",
        vis.nodes.len(), vis.edges.len(), vis.fusion_groups.len());

    if let Some(ref rt) = vis.runtime {
        println!("Total runtime: {} µs", rt.total_duration_us);
        println!();
        println!("Per-op breakdown:");
        for event in &rt.op_events {
            if event.duration_us > 0 {
                println!("  {:30} {:>6} µs  device={:?}  shape={:?}",
                    event.op_label, event.duration_us,
                    event.output_device, event.output_shape);
            }
        }
    }

    // Write outputs
    let json = ferroviz::render_json(&vis);
    std::fs::write("vision_attention_graph.json", &json)
        .expect("failed to write JSON");
    println!("\nWrote vision_attention_graph.json ({} bytes)", json.len());

    let html = ferroviz::render_html(&vis);
    std::fs::write("vision_attention_graph.html", &html)
        .expect("failed to write HTML");
    println!("Wrote vision_attention_graph.html ({} bytes)", html.len());

    println!("\nDone.");
    Ok(())
}

/// Build the full vision-attention architecture as an IR graph and run
/// the instrumented interpreter to get per-op profiling data.
///
/// Architecture (matching the nn-based model above):
///   Conv2d (as Matmul + Add)
///   → Reshape to (B, S, C)
///   → Q/K/V projections (Mm)
///   → Transpose K
///   → Attention scores (Matmul + Softmax)
///   → Attention output (Matmul)
///   → Output projection (Mm)
///   → Reshape back to (B, C, H', W')
///   → Flatten
///   → Linear head
///   → GELU
///   → Softmax
///   → MSE Loss (Sub + Mul + Mean)
fn build_full_architecture_graph() -> ferroviz::VisGraph {
    use ferrotorch_jit::graph::{IrGraph, IrOpKind};

    let mut g = IrGraph::new();

    // ── Graph inputs (model weights + data) ──
    //
    // The conv is represented in im2col style: the input image patches are
    // pre-extracted into a matrix (B*S, C_in*kH*kW) so the conv becomes a
    // plain matmul. This matches how conv2d actually executes on most backends.
    //
    // B=4, C_in=3, H=8, W=8, C_out=16, kernel=(3,3) → H'=6, W'=6, S=36
    // im2col patches: (4*36, 3*3*3) = (144, 27)
    // conv weight reshaped: (27, 16) = (C_in*kH*kW, C_out)
    let x_patches = g.add_input(vec![144, 27]);          // 0: im2col image patches
    let conv_w    = g.add_input(vec![27, 16]);            // 1: conv weight (im2col layout)
    let conv_b    = g.add_input(vec![16]);                // 2: conv bias
    let attn_qw   = g.add_input(vec![16, 16]);            // 3: Q projection weight
    let attn_kw   = g.add_input(vec![16, 16]);            // 4: K projection weight
    let attn_vw   = g.add_input(vec![16, 16]);            // 5: V projection weight
    let attn_ow   = g.add_input(vec![16, 16]);            // 6: output projection weight
    let head_w    = g.add_input(vec![10, 576]);            // 7: linear head weight
    let head_b    = g.add_input(vec![10]);                 // 8: linear head bias
    let target    = g.add_input(vec![4, 10]);              // 9: MSE target

    // ── Conv2d (im2col matmul): patches @ weight + bias ──
    // (144, 27) @ (27, 16) → (144, 16)
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
    // Flatten (B, S, C) → (B*S, C) = (144, 16) for Mm projections
    let (_, flat_seq) = g.add_node(
        IrOpKind::Reshape { shape: vec![144, 16] },
        vec![seq[0]],
        vec![vec![144, 16]],
    );

    // Q, K, V projections: (144, 16) @ (16, 16) → (144, 16)
    let (_, q_flat) = g.add_node(IrOpKind::Mm, vec![flat_seq[0], attn_qw], vec![vec![144, 16]]);
    let (_, k_flat) = g.add_node(IrOpKind::Mm, vec![flat_seq[0], attn_kw], vec![vec![144, 16]]);
    let (_, v_flat) = g.add_node(IrOpKind::Mm, vec![flat_seq[0], attn_vw], vec![vec![144, 16]]);

    // ── Attention scores ──
    // In 2D: Q (144, 16) @ K^T (16, 144) → scores (144, 144)
    // This is a simplified single-batch attention — in practice, each batch is separate,
    // but the IR represents the fused computation.
    let (_, k_t) = g.add_node(
        IrOpKind::Transpose, vec![k_flat[0]], vec![vec![16, 144]],
    );
    let (_, scores) = g.add_node(
        IrOpKind::Mm, vec![q_flat[0], k_t[0]], vec![vec![144, 144]],
    );
    let (_, attn_w) = g.add_node(
        IrOpKind::Softmax, vec![scores[0]], vec![vec![144, 144]],
    );

    // Attention output: (144, 144) @ V (144, 16) → (144, 16)
    let (_, attn_out) = g.add_node(
        IrOpKind::Mm, vec![attn_w[0], v_flat[0]], vec![vec![144, 16]],
    );

    // Output projection: (144, 16) @ (16, 16) → (144, 16)
    let (_, proj_flat) = g.add_node(
        IrOpKind::Mm, vec![attn_out[0], attn_ow], vec![vec![144, 16]],
    );

    // Reshape back to (B, S, C) = (4, 36, 16)
    let (_, proj) = g.add_node(
        IrOpKind::Reshape { shape: vec![4, 36, 16] },
        vec![proj_flat[0]],
        vec![vec![4, 36, 16]],
    );

    // ── Reshape back to spatial (B, C, H', W') ──
    let (_, spatial) = g.add_node(
        IrOpKind::Reshape { shape: vec![4, 16, 6, 6] },
        vec![proj[0]],
        vec![vec![4, 16, 6, 6]],
    );

    // ── Flatten spatial dims → (B, C*H'*W') = (4, 576) ──
    let (_, flat) = g.add_node(
        IrOpKind::Reshape { shape: vec![4, 576] },
        vec![spatial[0]],
        vec![vec![4, 576]],
    );
    let (_, linear_out) = g.add_node(
        IrOpKind::Linear, vec![flat[0], head_w, head_b], vec![vec![4, 10]],
    );

    // ── GELU activation ──
    let (_, gelu_out) = g.add_node(
        IrOpKind::Gelu, vec![linear_out[0]], vec![vec![4, 10]],
    );

    // ── Softmax ──
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

    // ── Static analysis: fusion groups etc. ──
    let mut vis = ferroviz::analyze(&g);

    // ── Instrumented interpreter: get per-op runtime data ──
    let inputs: Vec<Tensor<f32>> = vec![
        random_tensor(vec![144, 27]),           // x_patches (im2col)
        random_tensor(vec![27, 16]),            // conv_w
        random_tensor(vec![16]),                // conv_b
        random_tensor(vec![16, 16]),            // attn_qw
        random_tensor(vec![16, 16]),            // attn_kw
        random_tensor(vec![16, 16]),            // attn_vw
        random_tensor(vec![16, 16]),            // attn_ow
        random_tensor(vec![10, 576]),            // head_w
        random_tensor(vec![10]),                 // head_b
        random_tensor(vec![4, 10]),              // target
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
    // Simple deterministic pseudo-random data
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.7123 + 0.3).sin() * 0.5)
        .collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, true).unwrap()
}
