//! Ferroviz: Compute graph visualizer for ferrotorch.
//!
//! Ferroviz provides runtime observation of ferrotorch compute graphs,
//! showing which ops run on GPU vs CPU, where data crosses device boundaries,
//! which ops get fused, and per-op execution time.
//!
//! # Quick start
//!
//! ```ignore
//! use ferrotorch_core::Tensor;
//!
//! let inputs = vec![Tensor::from_storage(/* ... */)];
//! let vis = ferroviz::capture(|inputs| {
//!     // your model forward pass
//!     Ok(inputs[0].clone())
//! }, &inputs).unwrap();
//!
//! // JSON output for programmatic consumption
//! let json = ferroviz::render_json(&vis);
//!
//! // HTML output for interactive visualization
//! let html = ferroviz::render_html(&vis);
//! ```

pub mod model;
pub mod instrument;
pub mod capture;
pub mod fusion_analysis;
pub mod json;
pub mod html;

#[cfg(feature = "cuda-trace")]
pub mod cuda_trace;

// Re-export the public API at crate root
pub use capture::{capture, analyze};
pub use json::{render_json, render_json_compact, parse_json};
pub use html::render_html;
pub use model::VisGraph;

#[cfg(feature = "cuda-trace")]
pub use cuda_trace::correlate_nsys;
