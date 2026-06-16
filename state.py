"""
Shared LangGraph state definition for the Agentic RISC-V Compiler pipeline.

All agents read from and write to this shared TypedDict state.
LangGraph merges partial updates automatically.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class VerificationResult(TypedDict, total=False):
    """Structured result from the verification agent."""
    passed: bool
    errors: list[str]
    warnings: list[str]
    compiler_output: str


class SimulationResult(TypedDict, total=False):
    """Structured result from the Hazard3 simulation agent."""
    success: bool
    cycles: int
    execution_trace: str
    output_values: list[float]
    output_match: bool
    raw_log: str


class SynthesisResult(TypedDict, total=False):
    """Structured result from the OpenROAD synthesis agent."""
    success: bool
    power_watts: float
    area_mm2: float
    frequency_mhz: float
    cell_count: int
    detailed_report: str


class AgentState(TypedDict, total=False):
    """
    Shared state for the LangGraph pipeline.

    All agents receive the full state and return partial updates.
    LangGraph handles merging.
    """

    # ── Input ───────────────────────────────────────────────────
    model_name: str                          # Human-readable model name
    model: Any                               # The nn.Module instance (not serialized)
    fx_graph: Any                            # torch.fx.GraphModule
    fx_graph_str: str                        # String representation of FX graph
    sample_input: Any                        # Sample input tensor for shape propagation

    # ── IR (FX Parser → Code Generator) ────────────────────────
    ir_graph: dict                           # Serialized IRGraph (dict form)
    ir_summary: str                          # Human-readable layer summary

    # ── Weights ─────────────────────────────────────────────────
    weights_metadata: dict[str, dict]        # param_name → {shape, dtype, numel}
    weights_path: str                        # Path to saved .npz weights file
    total_params: int                        # Total parameter count
    model_memory_bytes: int                  # Estimated memory footprint
    weights_bin_path: str                    # Path to saved weights.bin binary file
    weights_manifest: dict[str, dict]        # param_name → {c_name, offset, size_bytes, numel, shape, c_type, precision}
    weight_precision: str                    # Precision mode: 'f32', 'f16', 'bf16', 'mxfp8'
    weight_mode: str                         # 'embedded' (bare-metal) or 'binary' (hosted)

    # ── Generated Code (Code Generator → Verifier) ─────────────
    generated_code: str                      # model.c content
    generated_header: str                    # weights.h content
    generated_model_header: str              # model.h content
    code_path: str                           # Path to saved model.c
    header_path: str                         # Path to saved weights.h
    model_header_path: str                   # Path to saved model.h

    # ── Verification ────────────────────────────────────────────
    verification_result: VerificationResult
    verification_attempts: int               # Counter (max 5)
    verification_feedback: str               # Formatted feedback for LLM
    verification_exhausted: bool             # True if max verification attempts reached

    # ── Human-in-the-loop ───────────────────────────────────────
    human_approved: bool                     # True if user approved
    human_feedback: str                      # Optional user comments
    human_action: str                        # 'approve', 'retry', 'verify', 'quit'

    # ── Simulation (Hazard3) ────────────────────────────────────
    simulation_result: SimulationResult
    reference_outputs: list[float]           # PyTorch reference output values

    # ── Synthesis (OpenROAD) ────────────────────────────────────
    synthesis_result: SynthesisResult

    # ── Optimization Loop ───────────────────────────────────────
    optimization_suggestions: list[str]      # List of optimization suggestions
    optimization_iteration: int              # Current iteration (max 3)
    enable_optimization: bool                # Toggle optimization loop

    # ── Final Report ────────────────────────────────────────────
    final_report: str                        # Markdown report

    # ── Error Handling ──────────────────────────────────────────
    error: Optional[str]                     # Fatal error message
