# Ferroviz: GPU execution and backward pass support

## Context

We are using ferroviz to visualize the execution graph of a Pythia-160M
transformer trainer built on ferrotorch 0.1.8. The goal is to profile a
full training step (forward + backward + optimizer) on GPU and identify
optimization opportunities.

Two gaps in the current implementation block this.

---

## Issue 1: Instrumented interpreter cannot run on GPU

### What happens

`instrumented_interpret()` dispatches ops through `ferrotorch_core::grad_fns::*`,
which are device-transparent — they will run on GPU if the input tensors are on
GPU. However, the **published** ferrotorch 0.1.8 crate is missing GPU kernels
for several elementwise ops that appear in real model graphs:

```
invalid argument: sqrt_f32 GPU op not yet implemented
```

This surfaces immediately when the graph contains a decomposed LayerNorm
(Mean → Sub → Mul → Mean → **Sqrt** → Div → Mul → Add), which every
transformer block does.

### Upstream status (ferrotorch `main` branch)

The missing GPU kernels have been implemented on ferrotorch's `main` branch
but are **not yet published** to crates.io. Key commits after the v0.1.8
release tag (`a346e28`):

| Commit   | Date       | Description |
|----------|------------|-------------|
| `dc42866`| 2026-03-24 | `feat: add GPU dispatch for div, exp, log, sqrt, pow, abs, sigmoid, tanh [CL-218]` |
| `3fa9739`| 2026-03-24 | `fix: GPU-native gradient accumulation — eliminate CPU round-trip in backward [CL-259]` |
| `40bc247`| 2026-03-25 | `feat: GPU parallel reduction kernel — eliminate CPU round-trip for sum/mean/sum_dim [CL-338]` |
| `5f17ccc`| 2026-03-25 | `feat: zero-copy stride-based views for permute, transpose, narrow [CL-261]` |
| `91f06f8`| 2026-03-25 | `feat: GPU-resident Adam optimizer state with fused CUDA kernel [CL-260]` |

Once these land in a published release, ferroviz's instrumented interpreter
should work on GPU without changes — it already dispatches through the
device-transparent `grad_fns::*` layer.

### Current workaround

Fall back to CPU for the instrumented interpreter. This gives timing data
but it reflects CPU execution, not GPU — defeating the purpose of the tool
for GPU profiling.

### What we need

**Short-term (ferroviz-side):** Until the next ferrotorch release, add a
fallback mode to the instrumented interpreter that catches per-op GPU
failures and transparently round-trips through CPU for unsupported ops,
while still recording the device the op *attempted* to run on and flagging
the fallback. This would at least show which ops lack GPU kernels — which
is itself valuable diagnostic information.

**Upstream (ferrotorch-side):** Publish a new crate version containing the
commits listed above. Once `dc42866` (GPU dispatch for sqrt/div/exp/log/
pow/abs/sigmoid/tanh) and `40bc247` (GPU reduction kernels for sum/mean)
are on crates.io, ferroviz GPU execution should work out of the box.

---

## Issue 2: No backward pass visualization

### What happens

`capture()` traces the forward pass via `ferrotorch_jit::trace()`, which
records the forward computation graph only. The instrumented interpreter
then re-executes that forward graph with timing. There is no mechanism to
trace, record, or visualize the backward pass.

For training workloads the backward pass is typically 2-3x the cost of the
forward pass, and in our case it is where the worst ferrotorch 0.1.8
performance bugs manifest (batch_transpose CPU round-trip in BMM backward,
GELU erf backward CPU fallback, LinearFusedBackward gradient count
mismatch). A forward-only view misses all of this.

### What we need

A `capture_training_step()` or similar API that captures both forward and
backward execution. Possible approaches:

**Option A: Autograd graph walk (structural)**

After the forward pass produces a loss tensor, walk the `grad_fn` chain
from the loss backward through the autograd graph. Each `GradFn` node has
a `name()` and `inputs()`, which gives the backward graph structure. This
could be rendered as a second DAG (or overlaid on the forward graph with
backward edges) without needing to execute backward.

Sketch:
```rust
pub fn capture_backward_graph<T: Float>(loss: &Tensor<T>) -> VisGraph {
    // Walk loss.grad_fn() recursively via inputs()
    // Build VisNodes with category=Backward, op_label=grad_fn.name()
    // Edges follow the inputs() references
}
```

**Option B: Instrumented backward execution (timed)**

Actually execute `.backward()` on the loss tensor, with per-grad-fn timing.
This is harder because ferrotorch's autograd doesn't have instrumentation
hooks — you'd need to wrap each `GradFn::backward()` call with timing, or
use a monkey-patching approach.

Sketch:
```rust
pub fn capture_training_step<T, F>(
    f: F,
    inputs: &[Tensor<T>],
) -> FerrotorchResult<TrainingStepVis>
where
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    // 1. Forward: trace + instrumented interpret (existing)
    // 2. Backward: call output.backward(), measure per-grad-fn timing
    // 3. Return combined visualization
}

pub struct TrainingStepVis {
    pub forward: VisGraph,
    pub backward: VisGraph,  // grad_fn nodes + timing
}
```

**Option C: Defer to nsys correlation (pragmatic)**

The existing `correlate_nsys()` with the `cuda-trace` feature already
correlates CUDA kernel launches and memcpy events to forward ops. If the
nsys trace covers the full training step, the kernel/memcpy data will
include backward kernels. The gap is that these backward kernels can't be
attributed to specific grad_fn nodes without the autograd graph structure
from Option A.

A combined approach — Option A for structure + Option C for GPU timing —
would give the most complete picture.

### Upstream status

ferrotorch `main` has commits that fix many backward GPU issues:
- `3fa9739`: GPU-native gradient accumulation — eliminate CPU round-trip in backward
- The CHANGELOG for 0.1.3 lists "Perf Phase 5B: Wire backward GPU kernels —
  eliminate all CPU roundtrips in backward passes (#255)" and "Fix batched
  matmul and broadcast matmul backward crash on GPU (#228)"

These fixes address the backend-level backward problems (batch_transpose
CPU round-trip, GELU backward fallback, etc.) but do **not** add backward
*visualization* to ferroviz. Even with a perfect backward pass on GPU,
ferroviz still can't show what happens during it.

### Impact

Without backward visualization:
- Cannot see the 2-3x backward cost at all
- Cannot identify which backward ops trigger CPU round-trips on GPU
  (batch_transpose, GELU erf, etc.)
- Cannot assess whether JIT fusion of forward ops also benefits backward
  (fused forward ops may have simpler backward graphs)
- The `profile` command in hcc-train partially fills this gap with manual
  per-component timing, but it's hand-written and doesn't produce a graph

---

## Issue 3 (minor): `ferrotorch_jit::trace()` panics on common model ops

### What happens

Attempting to JIT-trace a real Pythia model forward pass:

```rust
let vis = ferroviz::capture(|inputs| {
    model.forward(&inputs[0], None, None, 0)
}, &[input])?;
```

Panics with:
```
BUG: tensor TensorId(452) not found in tensor_to_ir map during IR construction
```

The tracer cannot handle:
- **Embedding lookups** (index/gather ops)
- **`Tensor::from_storage()` inside the traced function** (RoPE builds
  cos/sin tensors from CPU data)
- **`.split()` and `.contiguous()`** operations

This forces users to construct IR graphs manually (as in the
`vision_attention` example), which is tedious and can't represent the
full model faithfully.

### What we need

This is primarily a ferrotorch-jit issue, but ferroviz could help by:
- Documenting which ops are traceable vs. not
- Providing a `capture_eager()` API that instruments the eager execution
  directly (wrapping tensor ops at the ferrotorch_core level) rather than
  going through JIT tracing, avoiding the tracer limitations entirely

---

---

## Upgrade path: ferrotorch `main` branch

Upgrading ferrotorch from the published 0.1.8 crate to a git dependency
on `main` would unblock GPU execution (Issue 1) and likely fix many of the
backward GPU bugs documented in ISSUES_v0.1.8.md. Relevant unreleased
commits beyond what's listed above:

- `5f17ccc`: zero-copy stride-based views for permute, transpose, narrow
  (fixes the `permute_t` CPU round-trip that dominates our GPU training)
- `d0a682d`: fp16/bf16 Tensor Core kernels
- `91f06f8`: GPU-resident Adam optimizer state with fused CUDA kernel
- `30d8d2b`: tensor-tensor in-place ops with GPU dispatch
- `3c7eacd`: vectorizable exp/log polynomial kernels

The CHANGELOG also documents fixes in 0.1.3 for: FlashAttention/RoPE/
KVCache crashes on GPU (#229), permute/split/chunk on GPU (#225), JIT
fusion engine on GPU (#237), and the PTX register name collision that
caused all elementwise kernels to silently fall back to CPU.

To test: change `Cargo.toml` deps from `"0.1.8"` to
`{ git = "https://github.com/dollspace-gay/ferrotorch", branch = "main" }`
for all ferrotorch crates, rebuild, and re-run `vizgraph` with `--seq-len 16`.

---

## Priority

For our training visualization use case:

1. **Issue 2 (backward)** — highest impact; forward-only misses the majority
   of training cost and all of the worst GPU bottlenecks
2. **Issue 1 (GPU)** — likely unblocked by upgrading to ferrotorch `main`;
   ferroviz itself needs no changes, just the upstream kernels
3. **Issue 3 (trace)** — quality-of-life; manual graph construction works
   but is fragile
