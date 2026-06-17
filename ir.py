"""
Custom Hardware-Friendly Intermediate Representation (IR).

Bridges PyTorch FX graph semantics to C code generation.
Supports both CNN operations (Conv2D, Pool, etc.) and
LLM operations (Attention, RMSNorm, RoPE, SiLU) for TinyLlama.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import math


class IROpType(str, Enum):
    """All supported IR operation types."""

    # ── Tensor I/O ──────────────────────────────────────────────
    TENSOR_INPUT = "TENSOR_INPUT"
    TENSOR_OUTPUT = "TENSOR_OUTPUT"

    # ── CNN Operations ──────────────────────────────────────────
    CONV2D = "CONV2D"
    RELU = "RELU"
    BATCHNORM = "BATCHNORM"
    MAXPOOL2D = "MAXPOOL2D"
    AVGPOOL2D = "AVGPOOL2D"
    FLATTEN = "FLATTEN"

    # ── Common Operations ───────────────────────────────────────
    LINEAR = "LINEAR"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    SQRT = "SQRT"
    MEAN = "MEAN"
    SOFTMAX = "SOFTMAX"
    DROPOUT = "DROPOUT"          # No-op in inference

    # ── LLM Operations (for TinyLlama / LLaMA-style models) ────
    EMBEDDING = "EMBEDDING"           # Token embedding lookup
    RMSNORM = "RMSNORM"               # Root Mean Square Layer Norm
    LAYERNORM = "LAYERNORM"           # Standard Layer Norm
    ATTENTION = "ATTENTION"           # Multi-head self-attention block
    MATMUL = "MATMUL"                 # Raw matrix multiplication
    SILU = "SILU"                     # SiLU activation (x * sigmoid(x))
    SWIGLU = "SWIGLU"                 # SwiGLU gating: SiLU(xW1) * (xW2)
    ROTARY_EMBEDDING = "ROTARY_EMBEDDING"  # Rotary Positional Encoding
    RESHAPE = "RESHAPE"               # Tensor reshape
    TRANSPOSE = "TRANSPOSE"           # Tensor transpose/permute
    SPLIT = "SPLIT"                   # Tensor split (for Q/K/V)
    CONCAT = "CONCAT"                 # Tensor concatenation

    # ── Quantization ────────────────────────────────────────────
    QUANTIZE = "QUANTIZE"             # Float → Int8/Int4
    DEQUANTIZE = "DEQUANTIZE"         # Int8/Int4 → Float


# ── Default parameters for each op type ─────────────────────────
OP_DEFAULT_PARAMS: dict[str, dict] = {
    IROpType.CONV2D: {
        "kernel_size": [3, 3], "stride": [1, 1],
        "padding": [0, 0], "groups": 1, "bias": True
    },
    IROpType.LINEAR: {"bias": True},
    IROpType.MAXPOOL2D: {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
    IROpType.AVGPOOL2D: {"output_size": [1, 1]},
    IROpType.ATTENTION: {
        "num_heads": 4, "head_dim": 64, "causal": True
    },
    IROpType.EMBEDDING: {"num_embeddings": 32000, "embedding_dim": 2048},
    IROpType.RMSNORM: {"eps": 1e-5},
    IROpType.LAYERNORM: {"eps": 1e-5},
    IROpType.ROTARY_EMBEDDING: {"max_seq_len": 2048, "base": 10000.0},
    IROpType.SOFTMAX: {"dim": -1},
    IROpType.SUB: {},
    IROpType.DIV: {},
    IROpType.SQRT: {},
    IROpType.MEAN: {"dim": -1, "keepdim": True},
    IROpType.FLATTEN: {"start_dim": 1, "end_dim": -1},
    IROpType.RESHAPE: {"target_shape": []},
    IROpType.TRANSPOSE: {"dim0": 0, "dim1": 1},
    IROpType.SPLIT: {"num_splits": 3, "dim": -1},
    IROpType.CONCAT: {"dim": -1},
    IROpType.QUANTIZE: {"bits": 8, "scheme": "symmetric"},
    IROpType.DEQUANTIZE: {"bits": 8, "scheme": "symmetric"},
}

# ── C helper prototypes for generated model artifacts ─────────────────
# Keep the helper signatures next to the IR operation definitions so codegen
# and verification derive their requirements from the IR itself instead of a
# separate code-generation contract module.
HELPER_SIGNATURES_BY_OP: dict[str, tuple[str, ...]] = {
    IROpType.CONV2D: (
        "void conv2d(const float* in, const float* weight, const float* bias, float* out, int IC, int OC, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.LINEAR: (
        "void linear(const float* in, const float* weight, const float* bias, float* out, int in_features, int out_features);",
    ),
    IROpType.RELU: ("void relu(float* data, int size);",),
    IROpType.SILU: ("void silu(float* data, int size);",),
    IROpType.RMSNORM: (
        "void rmsnorm(const float* in, const float* weight, float* out, int size, float eps);",
    ),
    IROpType.LAYERNORM: (
        "void layernorm(const float* in, const float* weight, const float* bias, float* out, int size, float eps);",
    ),
    IROpType.SOFTMAX: ("void softmax(float* data, int size);",),
    IROpType.MATMUL: (
        "void matmul(const float* A, const float* B, float* C, int M, int K, int N);",
    ),
    IROpType.EMBEDDING: (
        "void embedding(const float* table, int token_id, float* out, int dim);",
    ),
    IROpType.ROTARY_EMBEDDING: (
        "void rope(float* q, float* k, int head_dim, int pos, int num_heads);",
    ),
    IROpType.ATTENTION: (
        "void attention(const float* x, const float* wq, const float* wk, const float* wv, const float* wo, float* out, int seq_len, int dim, int num_heads, int head_dim);",
    ),
    IROpType.BATCHNORM: (
        "void batchnorm2d(const float* in, const float* weight, const float* bias, float* out, int C, int H, int W, float eps);",
    ),
    IROpType.MAXPOOL2D: (
        "void maxpool2d(const float* in, float* out, int C, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.AVGPOOL2D: (
        "void avgpool2d(const float* in, float* out, int C, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.FLATTEN: (
        "void flatten(const float* in, float* out, int size);",
    ),
    IROpType.ADD: (
        "void add_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.SUB: (
        "void sub_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.MUL: (
        "void mul_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.DIV: (
        "void div_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.SQRT: (
        "void sqrt_tensor(const float* in, float* out, int size);",
    ),
    IROpType.MEAN: (
        "void mean_tensor(const float* in, float* out, int outer, int reduce, int inner);",
    ),
    IROpType.RESHAPE: (
        "void reshape_copy(const float* in, float* out, int size);",
    ),
    IROpType.TRANSPOSE: (
        "void transpose2d(const float* in, float* out, int rows, int cols);",
    ),
    IROpType.SPLIT: (
        "void split_tensor(const float* in, float* out0, float* out1, float* out2, int part_size);",
    ),
    IROpType.CONCAT: (
        "void concat_tensors(const float* a, const float* b, float* out, int a_size, int b_size);",
    ),
    IROpType.QUANTIZE: (
        "void quantize_f32_to_i8(const float* in, signed char* out, int size, float scale);",
    ),
    IROpType.DEQUANTIZE: (
        "void dequantize_i8_to_f32(const signed char* in, float* out, int size, float scale);",
    ),
}


def helper_name_from_signature(signature: str) -> str:
    """Return the C function name from a simple helper prototype."""
    before_paren = signature.split("(", 1)[0].strip()
    return before_paren.split()[-1]


def required_helper_signatures(ir_dict: dict) -> list[str]:
    """Return required helper prototypes for all non-I/O operations in IR order."""
    ir_graph = IRGraph.from_dict(ir_dict)
    seen: set[str] = set()
    signatures: list[str] = []
    for node in ir_graph.topological_order():
        for signature in HELPER_SIGNATURES_BY_OP.get(node.op, ()):
            if signature not in seen:
                seen.add(signature)
                signatures.append(signature)
    return signatures



@dataclass
class IRNode:
    """A single operation node in the IR graph."""

    id: str                                   # Unique node identifier
    op: str                                   # IROpType value
    inputs: list[str] = field(default_factory=list)  # Input node IDs
    params: dict[str, Any] = field(default_factory=dict)  # Op-specific parameters
    shape: tuple = ()                         # Output tensor shape
    dtype: str = "float32"                    # Output data type
    weight_key: str = ""                      # Key into weights dict (if any)
    bias_key: str = ""                        # Key into weights dict for bias

    def num_elements(self) -> int:
        """Number of elements in the output tensor."""
        if not self.shape:
            return 0
        result = 1
        for s in self.shape:
            result *= s
        return result

    def memory_bytes(self) -> int:
        """Memory footprint of the output tensor in bytes."""
        dtype_sizes = {"float32": 4, "float16": 2, "int8": 1, "int4": 1}
        return self.num_elements() * dtype_sizes.get(self.dtype, 4)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage in LangGraph state."""
        return {
            "id": self.id,
            "op": self.op,
            "inputs": self.inputs,
            "params": self.params,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "weight_key": self.weight_key,
            "bias_key": self.bias_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IRNode:
        """Deserialize from dictionary."""
        return cls(
            id=d["id"],
            op=d["op"],
            inputs=d.get("inputs", []),
            params=d.get("params", {}),
            shape=tuple(d.get("shape", ())),
            dtype=d.get("dtype", "float32"),
            weight_key=d.get("weight_key", ""),
            bias_key=d.get("bias_key", ""),
        )


@dataclass
class IRGraph:
    """
    The complete IR graph — a DAG of IRNodes.

    Designed to be serializable (for LangGraph state) and
    pretty-printable (for LLM consumption).
    """

    nodes: list[IRNode] = field(default_factory=list)
    input_shapes: dict[str, tuple] = field(default_factory=dict)
    weight_metadata: dict[str, dict] = field(default_factory=dict)
    model_name: str = ""

    # ── Graph Operations ────────────────────────────────────────

    def add_node(self, node: IRNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def get_node(self, node_id: str) -> IRNode | None:
        """Look up a node by ID."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def topological_order(self) -> list[IRNode]:
        """Return nodes in topological order (inputs before consumers)."""
        id_to_node = {n.id: n for n in self.nodes}
        visited: set[str] = set()
        order: list[IRNode] = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = id_to_node.get(node_id)
            if node is None:
                return
            for inp in node.inputs:
                visit(inp)
            order.append(node)

        for n in self.nodes:
            visit(n.id)
        return order

    def validate(self) -> list[str]:
        """Validate the graph structure. Returns list of issues."""
        issues = []
        node_ids = {n.id for n in self.nodes}

        for node in self.nodes:
            # Check inputs exist
            for inp in node.inputs:
                if inp not in node_ids:
                    issues.append(
                        f"Node '{node.id}' references unknown input '{inp}'"
                    )

            # Check op type is valid
            try:
                IROpType(node.op)
            except ValueError:
                issues.append(
                    f"Node '{node.id}' has unknown op type '{node.op}'"
                )

            # Check shapes are non-empty (except for output nodes)
            if node.op != IROpType.TENSOR_OUTPUT and not node.shape:
                issues.append(
                    f"Node '{node.id}' ({node.op}) has empty shape"
                )

        return issues

    # ── Memory Estimation ───────────────────────────────────────

    def total_activation_memory(self) -> int:
        """Total activation memory (all intermediate tensors)."""
        return sum(n.memory_bytes() for n in self.nodes)

    def total_weight_memory(self) -> int:
        """Total weight memory from metadata."""
        total = 0
        for meta in self.weight_metadata.values():
            numel = 1
            for s in meta.get("shape", []):
                numel *= s
            dtype_size = {"float32": 4, "float16": 2, "int8": 1}.get(
                meta.get("dtype", "float32"), 4
            )
            total += numel * dtype_size
        return total

    def total_params(self) -> int:
        """Total number of parameters."""
        total = 0
        for meta in self.weight_metadata.values():
            numel = 1
            for s in meta.get("shape", []):
                numel *= s
            total += numel
        return total

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the entire graph to a dictionary."""
        return {
            "model_name": self.model_name,
            "nodes": [n.to_dict() for n in self.nodes],
            "input_shapes": {
                k: list(v) for k, v in self.input_shapes.items()
            },
            "weight_metadata": self.weight_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IRGraph:
        """Deserialize from dictionary."""
        return cls(
            model_name=d.get("model_name", ""),
            nodes=[IRNode.from_dict(n) for n in d.get("nodes", [])],
            input_shapes={
                k: tuple(v) for k, v in d.get("input_shapes", {}).items()
            },
            weight_metadata=d.get("weight_metadata", {}),
        )

    # ── Pretty Printing ────────────────────────────────────────

    def pretty_print(self) -> str:
        """
        Format the IR graph for LLM consumption.

        Produces a clear, structured text representation
        that the code generation LLM can parse.
        """
        lines = []
        lines.append(f"# IR Graph: {self.model_name}")
        lines.append(f"# Total Parameters: {self.total_params():,}")
        lines.append(
            f"# Weight Memory: {self.total_weight_memory() / 1024:.1f} KB"
        )
        lines.append(
            f"# Activation Memory: "
            f"{self.total_activation_memory() / 1024:.1f} KB"
        )
        lines.append("")

        # Input shapes
        lines.append("## Inputs:")
        for name, shape in self.input_shapes.items():
            lines.append(f"  {name}: shape={shape}")
        lines.append("")

        # Node list
        lines.append("## Operations (topological order):")
        for node in self.topological_order():
            inputs_str = ", ".join(node.inputs) if node.inputs else "none"
            line = (
                f"  [{node.id}] {node.op}"
                f"  inputs=({inputs_str})"
                f"  shape={node.shape}"
            )
            if node.weight_key:
                wmeta = self.weight_metadata.get(node.weight_key, {})
                wshape = wmeta.get("shape", "?")
                line += f"  weight={node.weight_key}{wshape}"
            if node.bias_key:
                line += f"  bias={node.bias_key}"
            if node.params:
                # Show non-default params only
                param_str = ", ".join(
                    f"{k}={v}" for k, v in node.params.items()
                )
                line += f"  params={{{param_str}}}"
            lines.append(line)

        lines.append("")

        # Weight summary table
        lines.append("## Weight Tensors:")
        for name, meta in self.weight_metadata.items():
            shape = meta.get("shape", "?")
            dtype = meta.get("dtype", "float32")
            numel = meta.get("numel", "?")
            lines.append(f"  {name}: shape={shape} dtype={dtype} numel={numel}")

        return "\n".join(lines)

    def layer_summary(self) -> str:
        """Concise layer summary for human review."""
        lines = [f"Model: {self.model_name}", ""]

        op_counts: dict[str, int] = {}
        for node in self.nodes:
            op_counts[node.op] = op_counts.get(node.op, 0) + 1

        lines.append("Layer counts:")
        for op, count in sorted(op_counts.items()):
            lines.append(f"  {op}: {count}")

        lines.append(f"\nTotal parameters: {self.total_params():,}")
        lines.append(
            f"Weight memory: {self.total_weight_memory() / (1024*1024):.2f} MB"
        )
        lines.append(
            f"Activation memory: "
            f"{self.total_activation_memory() / 1024:.1f} KB"
        )

        return "\n".join(lines)


# ── C Code Pattern Hints ────────────────────────────────────────
# These help the code generator understand what each op maps to in C.

C_PATTERNS: dict[str, str] = {
    IROpType.TENSOR_INPUT: "float* {id}  /* function parameter */",
    IROpType.TENSOR_OUTPUT: "return {id};",
    IROpType.CONV2D: (
        "for (oc) for (oh) for (ow) for (ic) for (kh) for (kw)\n"
        "  out[oc][oh][ow] += in[ic][ih+kh][iw+kw] * weight[oc][ic][kh][kw];"
    ),
    IROpType.LINEAR: (
        "for (i) for (j) out[i] += in[j] * weight[i][j];\n"
        "out[i] += bias[i];"
    ),
    IROpType.RELU: "out[i] = (in[i] > 0) ? in[i] : 0;",
    IROpType.SILU: "out[i] = in[i] * (1.0f / (1.0f + expf(-in[i])));",
    IROpType.RMSNORM: (
        "float rms = sqrt(mean(x[i]*x[i]) + eps);\n"
        "out[i] = (x[i] / rms) * weight[i];"
    ),
    IROpType.EMBEDDING: "for (i) out[i] = embed_table[token_id][i];",
    IROpType.ATTENTION: (
        "Q = X @ Wq; K = X @ Wk; V = X @ Wv;\n"
        "scores = (Q @ K^T) / sqrt(head_dim);\n"
        "if (causal) apply_causal_mask(scores);\n"
        "attn = softmax(scores);\n"
        "out = attn @ V;"
    ),
    IROpType.MATMUL: "for (i) for (j) for (k) C[i][j] += A[i][k] * B[k][j];",
    IROpType.SOFTMAX: (
        "float max_val = max(x);\n"
        "float sum = 0;\n"
        "for (i) { x[i] = expf(x[i] - max_val); sum += x[i]; }\n"
        "for (i) x[i] /= sum;"
    ),
    IROpType.ROTARY_EMBEDDING: (
        "for (i in 0..head_dim/2):\n"
        "  cos_θ = cos(pos * freq[i]);\n"
        "  sin_θ = sin(pos * freq[i]);\n"
        "  out[2i]   = x[2i]*cos_θ - x[2i+1]*sin_θ;\n"
        "  out[2i+1] = x[2i]*sin_θ + x[2i+1]*cos_θ;"
    ),
    IROpType.SUB: "out[i] = in0[i] - in1[i];",
    IROpType.DIV: "out[i] = in0[i] / in1[i];",
    IROpType.SQRT: "out[i] = sqrtf(in[i]);",
    IROpType.MEAN: "out[0] = mean(in);",

}
