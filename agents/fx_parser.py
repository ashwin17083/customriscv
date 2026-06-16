"""
FX Parser Agent — Converts a PyTorch FX graph into the Custom IR.

This agent requires NO LLM. It performs deterministic analysis of the
torch.fx.GraphModule and builds an IRGraph with all layer info, shapes,
and weight metadata.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ir import IRGraph, IRNode, IROpType
from state import AgentState
from tools.export_weights import export_weights_binary

logger = logging.getLogger(__name__)

# ── Mapping from torch.nn module types to IR ops ────────────────
MODULE_TYPE_MAP: dict[type, str] = {
    nn.Conv2d: IROpType.CONV2D,
    nn.Linear: IROpType.LINEAR,
    nn.ReLU: IROpType.RELU,
    nn.BatchNorm2d: IROpType.BATCHNORM,
    nn.MaxPool2d: IROpType.MAXPOOL2D,
    nn.AdaptiveAvgPool2d: IROpType.AVGPOOL2D,
    nn.AvgPool2d: IROpType.AVGPOOL2D,
    nn.Dropout: IROpType.DROPOUT,
    nn.Flatten: IROpType.FLATTEN,
    nn.Softmax: IROpType.SOFTMAX,
    nn.Embedding: IROpType.EMBEDDING,
    nn.LayerNorm: IROpType.LAYERNORM,
}

# ── Mapping from torch functions to IR ops ──────────────────────
FUNCTION_MAP: dict[Any, str] = {}
METHOD_MAP: dict[str, str] = {
    "relu": IROpType.RELU,
    "flatten": IROpType.FLATTEN,
    "view": IROpType.RESHAPE,
    "reshape": IROpType.RESHAPE,
    "transpose": IROpType.TRANSPOSE,
    "permute": IROpType.TRANSPOSE,
    "contiguous": None,  # no-op, skip
    "softmax": IROpType.SOFTMAX,
    "chunk": IROpType.SPLIT,
    "split": IROpType.SPLIT,
    "mean": None,  # handled inline
    "size": None,   # metadata, skip
}

# Populate FUNCTION_MAP after import (torch.nn.functional may not
# be available at module-level in all environments)
def _init_function_map():
    import torch.nn.functional as F
    import operator

    FUNCTION_MAP.update({
        F.relu: IROpType.RELU,
        F.silu: IROpType.SILU,
        F.softmax: IROpType.SOFTMAX,
        F.linear: IROpType.LINEAR,
        F.conv2d: IROpType.CONV2D,
        F.embedding: IROpType.EMBEDDING,
        F.layer_norm: IROpType.LAYERNORM,
        F.dropout: IROpType.DROPOUT,
        torch.matmul: IROpType.MATMUL,
        torch.bmm: IROpType.MATMUL,
        torch.cat: IROpType.CONCAT,
        torch.split: IROpType.SPLIT,
        operator.add: IROpType.ADD,
        operator.mul: IROpType.MUL,
        torch.add: IROpType.ADD,
        torch.mul: IROpType.MUL,
        torch.flatten: IROpType.FLATTEN,
        torch.reshape: IROpType.RESHAPE,
        torch.transpose: IROpType.TRANSPOSE,
    })

_init_function_map()


def _get_module_for_target(model: nn.Module, target: str) -> nn.Module | None:
    """Resolve a dotted target path to the actual nn.Module."""
    parts = target.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current if isinstance(current, nn.Module) else None


def _extract_conv2d_params(module: nn.Conv2d) -> dict:
    """Extract Conv2D parameters for the IR node."""
    return {
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": list(module.kernel_size),
        "stride": list(module.stride),
        "padding": list(module.padding),
        "groups": module.groups,
        "bias": module.bias is not None,
    }


def _extract_linear_params(module: nn.Linear) -> dict:
    """Extract Linear layer parameters for the IR node."""
    return {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
    }


def _extract_embedding_params(module: nn.Embedding) -> dict:
    """Extract Embedding parameters for the IR node."""
    return {
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
    }


def _extract_pool_params(module: nn.Module) -> dict:
    """Extract pooling parameters."""
    if isinstance(module, nn.MaxPool2d):
        ks = module.kernel_size
        if isinstance(ks, int):
            ks = [ks, ks]
        st = module.stride
        if isinstance(st, int):
            st = [st, st]
        pd = module.padding
        if isinstance(pd, int):
            pd = [pd, pd]
        return {"kernel_size": list(ks), "stride": list(st), "padding": list(pd)}
    elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
        if isinstance(module, nn.AdaptiveAvgPool2d):
            os = module.output_size
            if isinstance(os, int):
                os = [os, os]
            return {"output_size": list(os)}
        return {}
    return {}


def _extract_norm_params(module: nn.Module) -> dict:
    """Extract normalization parameters."""
    if isinstance(module, nn.LayerNorm):
        ns = module.normalized_shape
        return {
            "normalized_shape": list(ns),
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
        }
    return {}


def _infer_shape(node, known_shapes: dict[str, tuple]) -> tuple:
    """
    Try to infer output shape from the node's metadata.
    Falls back to shape propagation heuristics.
    """
    # Try torch.fx metadata first
    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
        meta = node.meta['tensor_meta']
        if hasattr(meta, 'shape'):
            return tuple(meta.shape)
    if hasattr(node, 'meta') and 'val' in node.meta:
        val = node.meta['val']
        if hasattr(val, 'shape'):
            return tuple(val.shape)

    # Fallback: check if any input shapes are known
    for arg in node.args:
        if hasattr(arg, 'name') and arg.name in known_shapes:
            return known_shapes[arg.name]

    return ()


def _node_inputs(node) -> list[str]:
    """Extract input node names from FX node args."""
    inputs = []
    for arg in node.args:
        if hasattr(arg, 'name'):
            inputs.append(arg.name)
        elif isinstance(arg, (list, tuple)):
            for a in arg:
                if hasattr(a, 'name'):
                    inputs.append(a.name)
    return inputs


def parse_fx_graph(state: AgentState) -> dict:
    """
    LangGraph node function: Parse FX graph → Custom IR.

    Reads: state["fx_graph"], state["model"]
    Writes: state["ir_graph"], state["ir_summary"], state["weights_metadata"],
            state["weights_path"], state["total_params"], state["model_memory_bytes"]
    """
    fx_module = state["fx_graph"]
    model = state["model"]
    model_name = state.get("model_name", "unnamed_model")

    logger.info(f"Parsing FX graph for model: {model_name}")

    ir_graph = IRGraph(model_name=model_name)
    known_shapes: dict[str, tuple] = {}
    weight_metadata: dict[str, dict] = {}

    # ── Collect all weight tensors ──────────────────────────────
    state_dict = model.state_dict()
    for param_name, param_tensor in state_dict.items():
        weight_metadata[param_name] = {
            "shape": list(param_tensor.shape),
            "dtype": str(param_tensor.dtype).replace("torch.", ""),
            "numel": param_tensor.numel(),
        }

    ir_graph.weight_metadata = weight_metadata

    # ── Run shape propagation if possible ───────────────────────
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        # Create dummy input for shape propagation
        input_shapes = state.get("input_shapes", {})
        if not input_shapes:
            # Try to infer from the first placeholder
            for node in fx_module.graph.nodes:
                if node.op == "placeholder":
                    # Default: assume batch=1
                    # For LLM models: [1, seq_len] integer input
                    # For CNN models: [1, C, H, W] float input
                    break

        # Attempt shape propagation with a sample input
        sample_input = state.get("sample_input", None)
        if sample_input is not None:
            ShapeProp(fx_module).propagate(sample_input)
    except Exception as e:
        logger.warning(f"Shape propagation failed: {e}. Using heuristic shapes.")

    # ── Iterate FX graph nodes ──────────────────────────────────
    for node in fx_module.graph.nodes:

        if node.op == "placeholder":
            # Input tensor
            shape = _infer_shape(node, known_shapes)
            ir_node = IRNode(
                id=node.name,
                op=IROpType.TENSOR_INPUT,
                inputs=[],
                shape=shape,
            )
            ir_graph.input_shapes[node.name] = shape
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "call_module":
            # nn.Module call (e.g., self.conv1)
            target_module = _get_module_for_target(model, node.target)
            if target_module is None:
                logger.warning(f"Could not resolve module: {node.target}")
                continue

            module_type = type(target_module)
            ir_op = MODULE_TYPE_MAP.get(module_type)

            if ir_op is None:
                # Try to detect custom modules (RMSNorm, SwiGLU, etc.)
                class_name = module_type.__name__.lower()
                if "rmsnorm" in class_name or "rms_norm" in class_name:
                    ir_op = IROpType.RMSNORM
                elif "swiglu" in class_name:
                    ir_op = IROpType.SWIGLU
                elif "rotary" in class_name or "rope" in class_name:
                    ir_op = IROpType.ROTARY_EMBEDDING
                elif "attention" in class_name:
                    ir_op = IROpType.ATTENTION
                else:
                    logger.warning(
                        f"Unsupported module type: {module_type.__name__} "
                        f"at {node.target}. Skipping."
                    )
                    continue

            # Extract op-specific parameters
            params = {}
            weight_key = ""
            bias_key = ""

            if ir_op == IROpType.CONV2D:
                params = _extract_conv2d_params(target_module)
                weight_key = f"{node.target}.weight"
                if params.get("bias"):
                    bias_key = f"{node.target}.bias"

            elif ir_op == IROpType.LINEAR:
                params = _extract_linear_params(target_module)
                weight_key = f"{node.target}.weight"
                if params.get("bias"):
                    bias_key = f"{node.target}.bias"

            elif ir_op == IROpType.EMBEDDING:
                params = _extract_embedding_params(target_module)
                weight_key = f"{node.target}.weight"

            elif ir_op in (IROpType.MAXPOOL2D, IROpType.AVGPOOL2D):
                params = _extract_pool_params(target_module)

            elif ir_op in (IROpType.LAYERNORM, IROpType.RMSNORM):
                params = _extract_norm_params(target_module)
                if hasattr(target_module, 'weight') and target_module.weight is not None:
                    weight_key = f"{node.target}.weight"
                if hasattr(target_module, 'bias') and target_module.bias is not None:
                    bias_key = f"{node.target}.bias"

            elif ir_op == IROpType.ATTENTION:
                # Try to extract attention params from submodules
                if hasattr(target_module, 'num_heads'):
                    params["num_heads"] = target_module.num_heads
                if hasattr(target_module, 'head_dim'):
                    params["head_dim"] = target_module.head_dim
                # Collect weight keys for Q, K, V, O projections
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'wq', 'wk', 'wv', 'wo']:
                    full_key = f"{node.target}.{proj}.weight"
                    if full_key in weight_metadata:
                        params[f"{proj}_weight"] = full_key

            shape = _infer_shape(node, known_shapes)
            inputs = _node_inputs(node)

            ir_node = IRNode(
                id=node.name,
                op=ir_op,
                inputs=inputs,
                params=params,
                shape=shape,
                weight_key=weight_key,
                bias_key=bias_key,
            )
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "call_function":
            # Functional call (e.g., torch.relu, operator.add)
            ir_op = FUNCTION_MAP.get(node.target)

            if ir_op is None:
                target_name = getattr(node.target, '__name__', str(node.target))
                # Try matching by name
                name_lower = target_name.lower()
                if "silu" in name_lower:
                    ir_op = IROpType.SILU
                elif "gelu" in name_lower:
                    ir_op = IROpType.RELU  # Approximate
                elif "embedding" in name_lower:
                    ir_op = IROpType.EMBEDDING
                elif "add" in name_lower:
                    ir_op = IROpType.ADD
                elif "sub" in name_lower:
                    ir_op = IROpType.SUB
                elif "mul" in name_lower:
                    ir_op = IROpType.MUL
                elif "div" in name_lower:
                    ir_op = IROpType.DIV
                elif "sqrt" in name_lower:
                    ir_op = IROpType.SQRT
                elif "mean" in name_lower:
                    ir_op = IROpType.MEAN
                elif "softmax" in name_lower:
                    ir_op = IROpType.SOFTMAX
                elif "permute" in name_lower or "transpose" in name_lower:
                    ir_op = IROpType.TRANSPOSE
                elif "view" in name_lower or "reshape" in name_lower or "unsqueeze" in name_lower or "slice" in name_lower:
                    ir_op = IROpType.RESHAPE
                elif "mm" in name_lower or "matmul" in name_lower or "bmm" in name_lower:
                    ir_op = IROpType.MATMUL
                elif "cat" in name_lower:
                    ir_op = IROpType.CONCAT
                elif "split" in name_lower or "chunk" in name_lower:
                    ir_op = IROpType.SPLIT
                else:
                    logger.warning(f"Unsupported function: {target_name}")
                    continue

            shape = _infer_shape(node, known_shapes)
            inputs = _node_inputs(node)

            params = {}
            # Extract function-specific params from kwargs
            if ir_op == IROpType.SOFTMAX:
                params["dim"] = node.kwargs.get("dim", -1)
            elif ir_op == IROpType.FLATTEN:
                if len(node.args) >= 2:
                    params["start_dim"] = node.args[1] if not hasattr(node.args[1], 'name') else 1
                if len(node.args) >= 3:
                    params["end_dim"] = node.args[2] if not hasattr(node.args[2], 'name') else -1

            ir_node = IRNode(
                id=node.name,
                op=ir_op,
                inputs=inputs,
                params=params,
                shape=shape,
            )
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "call_method":
            # Method call (e.g., x.view(), x.flatten())
            method_name = node.target
            ir_op = METHOD_MAP.get(method_name)

            if ir_op is None:
                # Skip no-op methods
                if method_name in ("contiguous", "size", "dim", "float", "half"):
                    continue
                logger.warning(f"Unsupported method: {method_name}")
                continue

            shape = _infer_shape(node, known_shapes)
            inputs = _node_inputs(node)

            params = {}
            if ir_op == IROpType.RESHAPE:
                # Try to extract target shape from args
                shape_args = []
                for arg in node.args[1:]:
                    if isinstance(arg, int):
                        shape_args.append(arg)
                    elif hasattr(arg, 'name'):
                        shape_args.append(-1)  # dynamic dim
                if shape_args:
                    params["target_shape"] = shape_args

            elif ir_op == IROpType.TRANSPOSE:
                if len(node.args) >= 3:
                    dim_args = node.args[1:3]
                    params["dim0"] = dim_args[0] if isinstance(dim_args[0], int) else 0
                    params["dim1"] = dim_args[1] if isinstance(dim_args[1], int) else 1

            ir_node = IRNode(
                id=node.name,
                op=ir_op,
                inputs=inputs,
                params=params,
                shape=shape,
            )
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "output":
            inputs = _node_inputs(node)
            ir_node = IRNode(
                id="output",
                op=IROpType.TENSOR_OUTPUT,
                inputs=inputs,
                shape=(),
            )
            ir_graph.add_node(ir_node)

        elif node.op == "get_attr":
            # Constant attribute access — skip (handled via weight_key)
            continue

    # ── Validate the IR graph ───────────────────────────────────
    issues = ir_graph.validate()
    if issues:
        logger.warning(f"IR validation issues:\n" + "\n".join(issues))

    # ── Save weights to .npz file ───────────────────────────────
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "weights.npz")

    weight_arrays = {}
    for name, tensor in state_dict.items():
        weight_arrays[name] = tensor.detach().cpu().numpy()
    np.savez(weights_path, **weight_arrays)
    logger.info(f"Weights saved to: {weights_path}")

    # ── Export weights to binary + manifest ──────────────────────
    precision = state.get("weight_precision", "f32")
    bin_path, manifest = export_weights_binary(
        weights_path, output_dir, precision
    )
    logger.info(f"Binary weights exported to: {bin_path}")

    # ── Build summary ───────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    model_memory = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )

    return {
        "ir_graph": ir_graph.to_dict(),
        "ir_summary": ir_graph.layer_summary(),
        "weights_metadata": weight_metadata,
        "weights_path": weights_path,
        "weights_bin_path": bin_path,
        "weights_manifest": manifest,
        "weight_precision": precision,
        "weight_mode": state.get("weight_mode", "embedded"),
        "total_params": total_params,
        "model_memory_bytes": model_memory,
        "fx_graph_str": str(fx_module.graph),
    }
