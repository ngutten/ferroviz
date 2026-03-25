use ferrotorch_core::Device;
use serde::{Deserialize, Serialize};

/// Complete analysis result for a compute graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisGraph {
    pub nodes: Vec<VisNode>,
    pub edges: Vec<VisEdge>,
    pub fusion_groups: Vec<FusionCluster>,
    pub runtime: Option<RuntimeProfile>,
    #[cfg(feature = "cuda-trace")]
    pub cuda_trace: Option<CudaTrace>,
}

/// A node in the visualization graph, corresponding to one IR operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisNode {
    /// IrNodeId.0
    pub id: usize,
    /// Human-readable label, e.g. "Matmul", "FusedElementwise[Neg,Relu]"
    pub op_label: String,
    pub category: OpCategory,
    /// Fusion group this node belongs to, if any.
    pub cluster_id: Option<usize>,
    pub output_shapes: Vec<Vec<usize>>,
    // Runtime-observed fields (None if no runtime data)
    pub observed_input_devices: Option<Vec<SerializableDevice>>,
    pub observed_output_device: Option<SerializableDevice>,
    pub observed_duration_us: Option<u64>,
    pub requires_grad: Option<bool>,
    // CUDA trace fields (empty if no nsys data)
    pub cuda_kernels: Vec<CudaKernelLaunch>,
    pub cuda_memcpy: Vec<CudaMemcpy>,
}

/// An edge in the visualization graph, representing a tensor flowing between ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisEdge {
    pub from_node: usize,
    pub to_node: usize,
    pub shape: Vec<usize>,
    /// shape.product() * sizeof(f32)
    pub size_bytes: usize,
    pub observed_device_transition: Option<DeviceTransition>,
}

/// A group of fused operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCluster {
    pub id: usize,
    pub kind: String,
    pub node_ids: Vec<usize>,
}

/// Runtime profiling data from the instrumented interpreter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProfile {
    pub total_duration_us: u64,
    pub op_events: Vec<OpEvent>,
}

/// A single op execution event recorded by the instrumented interpreter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpEvent {
    pub node_id: usize,
    pub op_label: String,
    pub input_devices: Vec<SerializableDevice>,
    pub output_device: SerializableDevice,
    pub duration_us: u64,
    pub output_shape: Vec<usize>,
    pub requires_grad: bool,
}

/// CUDA trace data from nsys.
#[cfg(feature = "cuda-trace")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaTrace {
    pub total_kernel_time_ns: u64,
    pub total_memcpy_time_ns: u64,
    pub kernel_count: usize,
    pub memcpy_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaKernelLaunch {
    pub kernel_name: String,
    pub duration_ns: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaMemcpy {
    pub direction: MemcpyDirection,
    pub bytes: usize,
    pub duration_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpCategory {
    IO,
    Elementwise,
    Reduction,
    MatMul,
    Linear,
    Activation,
    Shape,
    Control,
    Fused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceTransition {
    CpuToGpu,
    GpuToCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemcpyDirection {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
}

/// A serializable wrapper around `Device` since `Device` doesn't impl Serialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializableDevice {
    Cpu,
    Cuda(usize),
}

impl From<Device> for SerializableDevice {
    fn from(d: Device) -> Self {
        match d {
            Device::Cpu => SerializableDevice::Cpu,
            Device::Cuda(n) => SerializableDevice::Cuda(n),
        }
    }
}

impl From<SerializableDevice> for Device {
    fn from(d: SerializableDevice) -> Self {
        match d {
            SerializableDevice::Cpu => Device::Cpu,
            SerializableDevice::Cuda(n) => Device::Cuda(n),
        }
    }
}

/// Detect a device transition between two devices.
pub fn detect_transition(from: &SerializableDevice, to: &SerializableDevice) -> Option<DeviceTransition> {
    match (from, to) {
        (SerializableDevice::Cpu, SerializableDevice::Cuda(_)) => Some(DeviceTransition::CpuToGpu),
        (SerializableDevice::Cuda(_), SerializableDevice::Cpu) => Some(DeviceTransition::GpuToCpu),
        _ => None,
    }
}

/// Classify an IrOpKind into an OpCategory.
pub fn classify_op(op: &ferrotorch_jit::graph::IrOpKind) -> OpCategory {
    use ferrotorch_jit::graph::IrOpKind;
    match op {
        IrOpKind::Input { .. } | IrOpKind::Output | IrOpKind::Constant { .. } => OpCategory::IO,
        IrOpKind::Add | IrOpKind::Sub | IrOpKind::Mul | IrOpKind::Div
        | IrOpKind::Neg | IrOpKind::Sqrt | IrOpKind::Abs
        | IrOpKind::Exp | IrOpKind::Log | IrOpKind::Pow { .. } => OpCategory::Elementwise,
        IrOpKind::Sum | IrOpKind::Mean | IrOpKind::Prod => OpCategory::Reduction,
        IrOpKind::Matmul | IrOpKind::Mm | IrOpKind::Mv | IrOpKind::Dot
        | IrOpKind::Transpose => OpCategory::MatMul,
        IrOpKind::Linear => OpCategory::Linear,
        IrOpKind::Relu | IrOpKind::Sigmoid | IrOpKind::Tanh
        | IrOpKind::Gelu | IrOpKind::Silu | IrOpKind::Softmax
        | IrOpKind::LogSoftmax => OpCategory::Activation,
        IrOpKind::Reshape { .. } | IrOpKind::Flatten | IrOpKind::Squeeze { .. }
        | IrOpKind::Unsqueeze { .. } | IrOpKind::Cat { .. } => OpCategory::Shape,
        IrOpKind::Cond | IrOpKind::Scan => OpCategory::Control,
        IrOpKind::FusedElementwise { .. } => OpCategory::Fused,
    }
}

/// Format an IrOpKind into a human-readable label.
pub fn op_label(op: &ferrotorch_jit::graph::IrOpKind) -> String {
    use ferrotorch_jit::graph::IrOpKind;
    match op {
        IrOpKind::Input { index } => format!("Input[{}]", index),
        IrOpKind::Output => "Output".to_string(),
        IrOpKind::Constant { shape, .. } => format!("Constant{:?}", shape),
        IrOpKind::Add => "Add".to_string(),
        IrOpKind::Sub => "Sub".to_string(),
        IrOpKind::Mul => "Mul".to_string(),
        IrOpKind::Div => "Div".to_string(),
        IrOpKind::Neg => "Neg".to_string(),
        IrOpKind::Pow { exponent } => format!("Pow({})", exponent),
        IrOpKind::Sqrt => "Sqrt".to_string(),
        IrOpKind::Abs => "Abs".to_string(),
        IrOpKind::Exp => "Exp".to_string(),
        IrOpKind::Log => "Log".to_string(),
        IrOpKind::Sum => "Sum".to_string(),
        IrOpKind::Mean => "Mean".to_string(),
        IrOpKind::Prod => "Prod".to_string(),
        IrOpKind::Matmul => "Matmul".to_string(),
        IrOpKind::Mm => "Mm".to_string(),
        IrOpKind::Mv => "Mv".to_string(),
        IrOpKind::Dot => "Dot".to_string(),
        IrOpKind::Transpose => "Transpose".to_string(),
        IrOpKind::Linear => "Linear".to_string(),
        IrOpKind::Relu => "Relu".to_string(),
        IrOpKind::Sigmoid => "Sigmoid".to_string(),
        IrOpKind::Tanh => "Tanh".to_string(),
        IrOpKind::Gelu => "Gelu".to_string(),
        IrOpKind::Silu => "Silu".to_string(),
        IrOpKind::Softmax => "Softmax".to_string(),
        IrOpKind::LogSoftmax => "LogSoftmax".to_string(),
        IrOpKind::Reshape { shape } => format!("Reshape{:?}", shape),
        IrOpKind::Flatten => "Flatten".to_string(),
        IrOpKind::Squeeze { axis } => format!("Squeeze({})", axis),
        IrOpKind::Unsqueeze { axis } => format!("Unsqueeze({})", axis),
        IrOpKind::Cat { axis } => format!("Cat(axis={})", axis),
        IrOpKind::Cond => "Cond".to_string(),
        IrOpKind::Scan => "Scan".to_string(),
        IrOpKind::FusedElementwise { ops } => {
            let names: Vec<String> = ops.iter().map(|o| op_label(o)).collect();
            format!("FusedElementwise[{}]", names.join(","))
        }
    }
}
