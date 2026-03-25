//! Instrumented interpreter that mirrors ferrotorch-jit's interpreter
//! but records per-op timing, device information, and tensor shapes.

use std::collections::HashMap;
use std::time::Instant;

use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use ferrotorch_jit::graph::{IrGraph, IrNodeId, IrOpKind, IrValueId};

use crate::model::{OpEvent, RuntimeProfile, SerializableDevice};

/// Run the graph through an instrumented interpreter, recording profiling events.
pub fn instrumented_interpret<T: Float>(
    graph: &IrGraph,
    inputs: &[Tensor<T>],
) -> FerrotorchResult<(Tensor<T>, RuntimeProfile)> {
    assert_eq!(
        inputs.len(),
        graph.input_values.len(),
        "input count mismatch: got {}, expected {}",
        inputs.len(),
        graph.input_values.len()
    );
    assert_eq!(
        graph.output_values.len(),
        1,
        "instrumented interpreter expects exactly one graph output"
    );

    let mut values: HashMap<IrValueId, Tensor<T>> = HashMap::new();
    let mut events: Vec<OpEvent> = Vec::new();
    let total_start = Instant::now();

    let topo_order = graph.topological_order();

    for &node_id in &topo_order {
        let node = graph.nodes.iter().find(|n| n.id == node_id).unwrap();
        let op_label_str = crate::model::op_label(&node.op);

        // Collect input devices
        let input_devices: Vec<SerializableDevice> = node
            .inputs
            .iter()
            .filter_map(|id| values.get(id).map(|t| SerializableDevice::from(t.device())))
            .collect();

        let start = Instant::now();
        let result = dispatch_op(node_id, &node.op, &node.inputs, &mut values, inputs)?;
        let elapsed = start.elapsed();

        // Store outputs
        let output_device: SerializableDevice;
        let output_shape: Vec<usize>;
        let requires_grad: bool;

        match &result {
            Some(tensor) => {
                output_device = SerializableDevice::from(tensor.device());
                output_shape = tensor.shape().to_vec();
                requires_grad = tensor.requires_grad();
                for &out_id in &node.outputs {
                    values.insert(out_id, tensor.clone());
                }
            }
            None => {
                // Output node — just forwards input
                output_device = input_devices.first().copied().unwrap_or(SerializableDevice::Cpu);
                output_shape = vec![];
                requires_grad = false;
            }
        }

        events.push(OpEvent {
            node_id: node_id.0,
            op_label: op_label_str,
            input_devices,
            output_device,
            duration_us: elapsed.as_micros() as u64,
            output_shape,
            requires_grad,
        });
    }

    let total_elapsed = total_start.elapsed();
    let output_id = graph.output_values[0];
    let output = values
        .remove(&output_id)
        .expect("output value not computed");

    let profile = RuntimeProfile {
        total_duration_us: total_elapsed.as_micros() as u64,
        op_events: events,
    };

    Ok((output, profile))
}

/// Dispatch a single op, returning the result tensor (None for Output nodes).
fn dispatch_op<T: Float>(
    _node_id: IrNodeId,
    op: &IrOpKind,
    inputs: &[IrValueId],
    values: &mut HashMap<IrValueId, Tensor<T>>,
    graph_inputs: &[Tensor<T>],
) -> FerrotorchResult<Option<Tensor<T>>> {
    use ferrotorch_core::grad_fns::{
        activation, arithmetic, linalg, reduction, shape, transcendental,
    };

    match op {
        IrOpKind::Input { index } => {
            Ok(Some(graph_inputs[*index].clone()))
        }

        IrOpKind::Constant { data, shape: s } => {
            let converted: Vec<T> = data.iter().map(|&v| T::from(v).unwrap()).collect();
            let tensor = Tensor::from_storage(
                ferrotorch_core::TensorStorage::cpu(converted),
                s.clone(),
                false,
            )?;
            Ok(Some(tensor))
        }

        IrOpKind::Output => {
            // Forward the input value to the output slot
            let input = get_value(values, inputs[0])?;
            // Output node just forwards input
            Ok(Some(input.clone()))
        }

        // Binary arithmetic
        IrOpKind::Add => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(arithmetic::add(&a, &b)?))
        }
        IrOpKind::Sub => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(arithmetic::sub(&a, &b)?))
        }
        IrOpKind::Mul => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(arithmetic::mul(&a, &b)?))
        }
        IrOpKind::Div => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(arithmetic::div(&a, &b)?))
        }

        // Unary arithmetic
        IrOpKind::Neg => {
            let a = get_unary(values, inputs)?;
            Ok(Some(arithmetic::neg(&a)?))
        }
        IrOpKind::Sqrt => {
            let a = get_unary(values, inputs)?;
            Ok(Some(arithmetic::sqrt(&a)?))
        }
        IrOpKind::Abs => {
            let a = get_unary(values, inputs)?;
            Ok(Some(arithmetic::abs(&a)?))
        }
        IrOpKind::Pow { exponent } => {
            let a = get_unary(values, inputs)?;
            Ok(Some(arithmetic::pow(&a, *exponent)?))
        }
        IrOpKind::Exp => {
            let a = get_unary(values, inputs)?;
            Ok(Some(transcendental::exp(&a)?))
        }
        IrOpKind::Log => {
            let a = get_unary(values, inputs)?;
            Ok(Some(transcendental::log(&a)?))
        }

        // Reductions
        IrOpKind::Sum => {
            let a = get_unary(values, inputs)?;
            Ok(Some(reduction::sum(&a)?))
        }
        IrOpKind::Mean => {
            let a = get_unary(values, inputs)?;
            Ok(Some(reduction::mean(&a)?))
        }
        IrOpKind::Prod => {
            let a = get_unary(values, inputs)?;
            Ok(Some(reduction::prod(&a)?))
        }

        // Activations
        IrOpKind::Relu => {
            let a = get_unary(values, inputs)?;
            Ok(Some(activation::relu(&a)?))
        }
        IrOpKind::Sigmoid => {
            let a = get_unary(values, inputs)?;
            Ok(Some(activation::sigmoid(&a)?))
        }
        IrOpKind::Tanh => {
            let a = get_unary(values, inputs)?;
            Ok(Some(activation::tanh(&a)?))
        }
        IrOpKind::Gelu => {
            let a = get_unary(values, inputs)?;
            Ok(Some(activation::gelu(&a)?))
        }
        IrOpKind::Silu => {
            let a = get_unary(values, inputs)?;
            Ok(Some(activation::silu(&a)?))
        }
        IrOpKind::Softmax => {
            let a = get_unary(values, inputs)?;
            Ok(Some(activation::softmax(&a)?))
        }
        IrOpKind::LogSoftmax => {
            let a = get_unary(values, inputs)?;
            Ok(Some(activation::log_softmax(&a)?))
        }

        // Linalg
        IrOpKind::Mm => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(linalg::mm_differentiable(&a, &b)?))
        }
        IrOpKind::Matmul => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(linalg::matmul_differentiable(&a, &b)?))
        }
        IrOpKind::Mv => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(linalg::mv_differentiable(&a, &b)?))
        }
        IrOpKind::Dot => {
            let (a, b) = get_binary(values, inputs)?;
            Ok(Some(linalg::dot_differentiable(&a, &b)?))
        }
        IrOpKind::Transpose => {
            let a = get_unary(values, inputs)?;
            Ok(Some(shape::transpose_2d(&a)?))
        }
        IrOpKind::Linear => {
            let input = get_value(values, inputs[0])?;
            let weight = get_value(values, inputs[1])?;
            let bias = if inputs.len() >= 3 {
                Some(get_value(values, inputs[2])?)
            } else {
                None
            };
            Ok(Some(linalg::linear_fused(&input, &weight, bias)?))
        }

        // Shape ops
        IrOpKind::Reshape { shape: new_shape } => {
            let a = get_unary(values, inputs)?;
            Ok(Some(shape::reshape(&a, &new_shape)?))
        }
        IrOpKind::Flatten => {
            let a = get_unary(values, inputs)?;
            Ok(Some(shape::flatten(&a)?))
        }
        IrOpKind::Squeeze { axis } => {
            let a = get_unary(values, inputs)?;
            Ok(Some(shape::squeeze(&a, *axis as isize)?))
        }
        IrOpKind::Unsqueeze { axis } => {
            let a = get_unary(values, inputs)?;
            Ok(Some(shape::unsqueeze(&a, *axis as isize)?))
        }
        IrOpKind::Cat { axis } => {
            let tensors: Vec<Tensor<T>> = inputs
                .iter()
                .map(|id| get_value(values, *id).cloned())
                .collect::<FerrotorchResult<Vec<_>>>()?;
            Ok(Some(shape::cat(&tensors, *axis as isize)?))
        }

        // Fused elementwise
        IrOpKind::FusedElementwise { ops } => {
            let mut current = get_unary(values, inputs)?.clone();
            for sub_op in ops {
                current = apply_elementwise_op(&current, sub_op)?;
            }
            Ok(Some(current))
        }

        IrOpKind::Cond | IrOpKind::Scan => {
            Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!("{:?} must be lowered before interpretation", op),
            })
        }
    }
}

/// Apply a single elementwise op (for FusedElementwise chains).
fn apply_elementwise_op<T: Float>(
    input: &Tensor<T>,
    op: &IrOpKind,
) -> FerrotorchResult<Tensor<T>> {
    use ferrotorch_core::grad_fns::{activation, arithmetic, transcendental};

    match op {
        IrOpKind::Neg => arithmetic::neg(input),
        IrOpKind::Sqrt => arithmetic::sqrt(input),
        IrOpKind::Abs => arithmetic::abs(input),
        IrOpKind::Pow { exponent } => arithmetic::pow(input, *exponent),
        IrOpKind::Exp => transcendental::exp(input),
        IrOpKind::Log => transcendental::log(input),
        IrOpKind::Relu => activation::relu(input),
        IrOpKind::Sigmoid => activation::sigmoid(input),
        IrOpKind::Tanh => activation::tanh(input),
        IrOpKind::Gelu => activation::gelu(input),
        IrOpKind::Silu => activation::silu(input),
        _ => Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("{:?} is not a supported elementwise op in fusion", op),
        }),
    }
}

fn get_value<T: Float>(
    values: &HashMap<IrValueId, Tensor<T>>,
    id: IrValueId,
) -> FerrotorchResult<&Tensor<T>> {
    values.get(&id).ok_or_else(|| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("value {:?} not found during interpretation", id),
        }
    })
}

fn get_unary<'a, T: Float>(
    values: &'a HashMap<IrValueId, Tensor<T>>,
    inputs: &[IrValueId],
) -> FerrotorchResult<&'a Tensor<T>> {
    assert!(!inputs.is_empty(), "unary op requires at least one input");
    get_value(values, inputs[0])
}

fn get_binary<'a, T: Float>(
    values: &'a HashMap<IrValueId, Tensor<T>>,
    inputs: &[IrValueId],
) -> FerrotorchResult<(&'a Tensor<T>, &'a Tensor<T>)> {
    assert!(inputs.len() >= 2, "binary op requires at least two inputs");
    Ok((get_value(values, inputs[0])?, get_value(values, inputs[1])?))
}
