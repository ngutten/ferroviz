# Ferroviz

Compute graph visualizer for [ferrotorch](https://crates.io/crates/ferrotorch-core). Ferroviz instruments graph execution to show which ops actually run on GPU vs CPU, where data crosses device boundaries, which ops get fused, and how long each op takes.

## Why

When debugging ferrotorch model performance, there's no built-in way to see what's really happening at runtime. Some ops (like `sum_f32`) do GPU->CPU->GPU round-trips where the output tensor reports `device=Cuda` despite executing on CPU. Simply checking `tensor.device()` won't catch this. Ferroviz fills this gap by instrumenting execution and (optionally) correlating with CUDA-level profiling data from nsys.

## Installation

```toml
[dependencies]
ferroviz = "0.1.0"

# Optional: enable nsys SQLite trace parsing
# ferroviz = { version = "0.1.0", features = ["cuda-trace"] }
```

## Usage

### Capture and visualize a model

```rust
use ferrotorch_core::{Tensor, FerrotorchResult};

// Trace, optimize, and profile a function in one call
let vis = ferroviz::capture(|inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
    // your forward pass here
    Ok(result)
}, &example_inputs)?;

// Interactive HTML graph viewer
let html = ferroviz::render_html(&vis);
std::fs::write("graph.html", html)?;

// Machine-readable JSON (e.g. for Claude Code to analyze)
let json = ferroviz::render_json(&vis);
std::fs::write("graph.json", json)?;
```

### Static analysis only (no execution)

```rust
use ferrotorch_jit::graph::IrGraph;

let graph: IrGraph = /* load or build your graph */;
let vis = ferroviz::analyze(&graph);
```

### CUDA trace correlation (requires `cuda-trace` feature)

```rust
// 1. Run your model under nsys:
//    nsys profile -o trace.sqlite ./my_model

// 2. Capture with ferroviz to get per-op time windows
let mut vis = ferroviz::capture(f, &inputs)?;

// 3. Correlate nsys data to find hidden GPU<->CPU transfers
ferroviz::correlate_nsys(&mut vis, Path::new("trace.sqlite"))?;

// Nodes now have cuda_kernels and cuda_memcpy populated.
// Ops with memcpy events but no kernel launches are doing hidden round-trips.
```

## API

| Function | Purpose |
|----------|---------|
| `capture(f, inputs)` | Trace + optimize + instrument + analyze in one call |
| `analyze(graph)` | Static analysis only (fusion groups, categories), no runtime |
| `render_html(vis)` | Self-contained HTML with interactive DAG viewer |
| `render_json(vis)` | Pretty-printed JSON for programmatic use |
| `render_json_compact(vis)` | Single-line JSON |
| `parse_json(s)` | Deserialize a VisGraph from JSON |
| `correlate_nsys(vis, path)` | Correlate nsys SQLite trace *(cuda-trace feature)* |

## HTML viewer features

- DAG layout with nodes in topological order (dagre-d3)
- Fusion groups shown as dashed bounding boxes
- Click any node for details: op type, shapes, device, timing, CUDA kernels
- Edge labels show tensor shape and size in bytes
- Color coding: red = hidden CPU round-trip, orange = device transition, purple = fused
- Timeline bar chart of per-op execution time
- Scroll and zoom for large graphs

## Example: vision-attention benchmark

`examples/vision_attention.rs` implements a mixed conv/attention architecture and runs a full training step, then visualizes the compute graph with ferroviz.

### Architecture

```
Input (B=4, C=3, H=8, W=8)
  -> Conv2d(3, 16, kernel=3x3)             -- im2col matmul + bias
  -> Permute + Reshape to (B, S=36, C=16)  -- spatial to sequence
  -> Q/K/V projections (Mm)                -- 3 parallel linear projections
  -> Transpose K, Q @ K^T                  -- attention scores
  -> Softmax                               -- attention weights
  -> Attn weights @ V                      -- weighted values
  -> Output projection (Mm)
  -> Reshape back to (B, C=16, H'=6, W'=6) -- sequence to spatial
  -> Reshape to (B, 576) -> Linear(576, 10) -- classification head
  -> GELU
  -> Softmax
  -> MSE Loss (Sub, Mul, Mean)
```

The example also runs the real `ferrotorch-nn` model with backpropagation and an AdamW optimizer step to demonstrate a complete training iteration.

### Running it

```sh
cargo run --example vision_attention
```

### Output

Pre-generated outputs are in [`examples/output/`](examples/output/):

- [`vision_attention_graph.json`](examples/output/vision_attention_graph.json) -- full VisGraph with 31 nodes, 33 edges, 20 fusion groups, and per-op runtime profiling
- [`vision_attention_graph.html`](examples/output/vision_attention_graph.html) -- interactive viewer (open in a browser)

### Per-op runtime breakdown (CPU, typical run)

```
Op                                 Time     Shape
Mm (conv im2col)                   37 us    [144, 16]
Add (conv bias)                   192 us    [144, 16]
Reshape (to sequence)               2 us    [4, 36, 16]
Reshape (flatten for proj)          1 us    [144, 16]
Mm (Q projection)                   6 us    [144, 16]
Mm (K projection)                   6 us    [144, 16]
Mm (V projection)                   5 us    [144, 16]
Transpose (K^T)                   698 us    [16, 144]
Mm (attention scores)              22 us    [144, 144]
Softmax (attention weights)      2021 us    [144, 144]
Mm (attn @ V)                      23 us    [144, 16]
Mm (output projection)              6 us    [144, 16]
Reshape (to B,S,C)                   3 us    [4, 36, 16]
Reshape (to spatial)                 1 us    [4, 16, 6, 6]
Reshape (flatten)                    1 us    [4, 576]
Linear (head)                      15 us    [4, 10]
Gelu                               11 us    [4, 10]
Softmax (output)                    6 us    [4, 10]
Sub (loss)                           6 us    [4, 10]
Mul (loss squared)                  10 us    [4, 10]
Mean (loss reduction)                2 us    []
```

Softmax on the 144x144 attention matrix dominates at ~2ms. The transpose is also expensive (~700us) because it materializes a full copy. On a real model these would be the first targets for optimization.

## Project structure

```
ferroviz/
  Cargo.toml
  src/
    lib.rs                -- public API re-exports
    model.rs              -- VisGraph, VisNode, VisEdge, OpCategory, etc.
    capture.rs            -- capture() and analyze() orchestration
    instrument.rs         -- instrumented interpreter (mirrors ferrotorch-jit)
    fusion_analysis.rs    -- wraps find_fusion_groups() -> FusionCluster
    cuda_trace.rs         -- nsys SQLite parsing (cuda-trace feature)
    html.rs               -- self-contained HTML renderer
    json.rs               -- JSON serialization
    assets/
      viewer.js           -- interactive graph viewer (dagre-d3)
      viewer.css          -- dark-theme styles
  examples/
    vision_attention.rs   -- benchmark example
    output/               -- pre-generated outputs
  tests/
    test_basic.rs         -- 13 integration tests
```

## License

[CC0 1.0](LICENSE) -- public domain.
