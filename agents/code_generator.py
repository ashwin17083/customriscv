"""
Code Generator Agent — Converts Custom IR → RISC-V C code.

Uses Qwen2.5-Coder-32B via local vLLM server (OpenAI-compatible API)
to generate model.h/model.c from the IR graph.

weights.h is generated deterministically (not by the LLM) using
the export_weights utility. The LLM then produces a model.h contract
before implementing model.c against that contract.

On verification retries, the previously generated header and code are
fed back to the LLM in "repair mode" (Codex-inspired) so it can make
targeted fixes rather than regenerating from scratch.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ir import IRGraph
from state import AgentState
from tools.export_weights import (
    export_weights_binary,
    generate_weights_header,
    generate_weights_loader,
)

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")  # vLLM doesn't need real key
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "200000"))

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "codegen.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
LLM_MAX_TOKENS = 250_000


def _read_generated_artifact_from_output(
    state: AgentState,
    path_key: str,
    state_key: str,
    default_filename: str,
) -> str:
    """Read a generated artifact from disk, falling back to state content."""
    code_path = state.get(path_key, "")
    if code_path:
        path = Path(code_path)
        if path.exists():
            return path.read_text(encoding="utf-8")

    default_path = OUTPUT_DIR / default_filename
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")

    return state.get(state_key, "")


def _read_generated_code_from_output(state: AgentState) -> str:
    """Read the generated model.c from disk, falling back to state content."""
    return _read_generated_artifact_from_output(
        state, "code_path", "generated_code", "model.c"
    )


def _read_generated_model_header_from_output(state: AgentState) -> str:
    """Read generated model.h from disk, falling back to state content."""
    return _read_generated_artifact_from_output(
        state, "model_header_path", "generated_model_header", "model.h"
    )


def _load_system_prompt() -> str:
    """Load the code generation system prompt."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def extract_weight_variable_names(state: AgentState) -> list[str]:
    """
    Return the C variable names exported by weights.h for LLM prompt context.

    This helper centralizes weight-name extraction for both normal models and
    LLM-style checkpoints with dotted parameter names such as
    ``model.layers.0.self_attn.q_proj.weight``.
    """
    manifest = state.get("weights_manifest", {})
    if manifest:
        return [
            str(info.get("c_name", name.replace(".", "_")))
            for name, info in sorted(manifest.items())
        ]

    weight_metadata = state.get("weights_metadata", {})
    if not weight_metadata:
        weight_metadata = state.get("ir_graph", {}).get("weight_metadata", {})

    return [name.replace(".", "_") for name in sorted(weight_metadata.keys())]


def _build_user_prompt(state: AgentState) -> str:
    """
    Build the user prompt containing IR graph, weight metadata,
    detailed implementation guidance, any previous feedback, and
    optimization suggestions.

    On retries (when verification_feedback is set), reads the
    previously generated code from the output file and includes it
    in a REPAIR MODE section so the LLM
    can make targeted fixes instead of regenerating from scratch.
    """
    ir_dict = state.get("ir_graph", {})
    ir_graph = IRGraph.from_dict(ir_dict)
    is_retry = bool(state.get("verification_feedback", ""))

    sections = []

    # ── IR Graph ────────────────────────────────────────────────
    sections.append("=" * 60)
    sections.append("IR GRAPH")
    sections.append("=" * 60)
    sections.append(ir_graph.pretty_print())

    # ── Weight Summary (C names for #include "weights.h") ───────
    sections.append("")
    sections.append("=" * 60)
    sections.append("WEIGHT TENSORS (available in weights.h — use these C names)")
    sections.append("=" * 60)

    # Prefer manifest (has c_name), fall back to weight_metadata
    manifest = state.get("weights_manifest", {})
    if manifest:
        for name in sorted(manifest.keys()):
            info = manifest[name]
            c_name = info["c_name"]
            numel = info["numel"]
            shape = info["shape"]
            c_type = info.get("c_type", "float")
            sections.append(
                f"  {c_type} {c_name}[{numel}];  "
                f"// shape={shape}, originally: {name}"
            )
    else:
        for name, meta in ir_graph.weight_metadata.items():
            shape = meta.get("shape", [])
            numel = meta.get("numel", 0)
            c_name = name.replace(".", "_")
            sections.append(
                f"  float {c_name}[{numel}];  "
                f"// shape={shape}, originally: {name}"
            )

    # ── FUNCTION HEADER ─────────────────────────────────────────
    functions_h = state.get("generated_functions_header", "")
    if functions_h:
        sections.append("")
        sections.append("=" * 60)
        sections.append("FUNCTION PROTOTYPES (from model_functions.h)")
        sections.append("=" * 60)
        sections.append(functions_h)

    # ── REPAIR MODE: Current Code + Errors ──────────────────────
    if is_retry:
        current_code = _read_generated_code_from_output(state)
        feedback = state.get("verification_feedback", "")

        if current_code:
            sections.append("")
            sections.append("=" * 60)
            sections.append(
                "⚠️  CURRENT CODE (contains errors — you are in REPAIR MODE)"
            )
            sections.append("=" * 60)
            sections.append(current_code)

        sections.append("")
        sections.append("=" * 60)
        sections.append("⚠️  VERIFICATION ERRORS — FIX THESE")
        sections.append("=" * 60)
        sections.append(feedback)
        sections.append("")
        sections.append(
            "You are in REPAIR MODE. Fix ALL the errors listed above "
            "in the current code. Output the COMPLETE fixed model.c file. "
            "Keep the overall structure intact. Mark fixes with "
            "// FIX: <description> comments."
        )
    else:
        # ── First attempt: generate from scratch ────────────────
        # (No verification feedback yet)
        pass

    # ── Optimization Suggestions (if in optimization loop) ──────
    suggestions = state.get("optimization_suggestions", [])
    if suggestions:
        sections.append("")
        sections.append("=" * 60)
        sections.append("🔧 OPTIMIZATION SUGGESTIONS — APPLY THESE")
        sections.append("=" * 60)
        for i, s in enumerate(suggestions, 1):
            sections.append(f"  {i}. {s}")
        sections.append("")
        sections.append(
            "Apply the above optimizations to the generated code. "
            "Mark optimized sections with '// OPTIMIZED: <description>'."
        )

    # ── Detailed generation guidance ────────────────────────────
    sections.append("")
    sections.append("=" * 60)
    sections.append("IMPLEMENTATION DETAILING REQUIREMENTS")
    sections.append("=" * 60)
    sections.append(
        "Before emitting code, internally map every IR node to exact "
        "buffer names, tensor extents, helper calls, loop bounds, and "
        "weight arrays. The final answer must still contain only the "
        "single requested model.c code block, but the implementation "
        "should reflect this detailed plan with clear constants, explicit "
        "shape comments, and operation-by-operation comments."
    )
    sections.append(
        "Use the full available context to preserve all generated code "
        "during repairs and optimizations; do not omit helper functions "
        "or unrelated model_inference steps while fixing localized issues."
    )

    # ── Task Instruction ────────────────────────────────────────
    sections.append("")
    sections.append("=" * 60)
    sections.append("TASK")
    sections.append("=" * 60)
    if is_retry:
        sections.append(
            "Fix the errors in the current code and output the COMPLETE "
            "fixed model.c file. Do NOT rewrite from scratch — make "
            "targeted fixes. Output exactly ONE ```c model.c code block."
        )
    else:
        sections.append(
            "Generate the complete model.c file for this neural network "
            "model targeting RISC-V rv32imac. "
            "Do NOT generate weights.h or model.h in this response — both "
            "are already available. Just #include \"model.h\" and use the "
            "weight array names listed above through the model.h -> weights.h "
            "include chain. "
            "Output exactly ONE ```c model.c code block."
        )

    return "\n".join(sections)

def _build_header_prompt(state: AgentState) -> str:
    """Prompt for generating the model_functions.h header."""
    ir_dict = state.get("ir_graph", {})
    ir_graph = IRGraph.from_dict(ir_dict)
    
    sections = []
    sections.append("=" * 60)
    sections.append("IR GRAPH")
    sections.append("=" * 60)
    sections.append(ir_graph.pretty_print())
    sections.append("")
    sections.append("=" * 60)
    sections.append("TASK")
    sections.append("=" * 60)
    sections.append(
        "Generate a C header file named `model_functions.h` containing ONLY the function "
        "prototypes (declarations) needed to implement this neural network on bare-metal RISC-V. "
        "Include `void model_inference(const float* input, float* output);` "
        "Do NOT implement the functions. Output exactly ONE ```c model_functions.h code block."
    )
    return "\n".join(sections)


def _build_weight_context(state: AgentState) -> str:
    """Build deterministic weight context shared by model.h/model.c prompts."""
    lines = [
        "=" * 60,
        "WEIGHT TENSORS (available in weights.h — use these C names)",
        "=" * 60,
    ]
    manifest = state.get("weights_manifest", {})
    if manifest:
        for name in sorted(manifest.keys()):
            info = manifest[name]
            lines.append(
                f"  {info.get('c_type', 'float')} {info['c_name']}[{info['numel']}]; "
                f"// shape={info['shape']}, originally: {name}"
            )
    else:
        weight_metadata = state.get("weights_metadata", {})
        if not weight_metadata:
            weight_metadata = state.get("ir_graph", {}).get("weight_metadata", {})
        for name, meta in sorted(weight_metadata.items()):
            lines.append(
                f"  float {name.replace('.', '_')}[{meta.get('numel', 0)}]; "
                f"// shape={meta.get('shape', [])}, originally: {name}"
            )
    lines.append("")
    lines.append("Weight variable names only:")
    lines.append(", ".join(extract_weight_variable_names(state)) or "(none)")
    return "\n".join(lines)


def _build_model_header_prompt(state: AgentState) -> str:
    """
    Build step-2 prompt: generate model.h with includes, interface contracts,
    dependencies, tensor block declarations, and function stubs only.
    """
    ir_graph = IRGraph.from_dict(state.get("ir_graph", {}))
    is_retry = bool(state.get("verification_feedback", ""))
    sections = [
        "=" * 60,
        "IR GRAPH",
        "=" * 60,
        ir_graph.pretty_print(),
        "",
        _build_weight_context(state),
        "",
        "=" * 60,
        "STEP 2 TASK: CREATE model.h",
        "=" * 60,
        "Create a C99 header file named model.h. It must include weights.h, "
        "define input/output tensor contracts, document dependencies, declare "
        "static-size tensor block macros/constants, and provide prototypes for "
        "all helper functions and model_inference(). Do not implement function "
        "bodies and do not define storage in model.h.",
        "Output exactly ONE ```c model.h code block.",
    ]

    if is_retry:
        current_header = _read_generated_model_header_from_output(state)
        feedback = state.get("verification_feedback", "")
        if current_header:
            sections.extend([
                "",
                "=" * 60,
                "CURRENT model.h (contains errors — REPAIR MODE)",
                "=" * 60,
                current_header,
            ])
        sections.extend([
            "",
            "=" * 60,
            "VERIFICATION ERRORS — FIX HEADER-RELEVANT ISSUES",
            "=" * 60,
            feedback,
            "Make targeted fixes and output the COMPLETE fixed model.h.",
        ])

    return "\n".join(sections)


def _build_model_c_prompt(state: AgentState, model_h: str) -> str:
    """Build step-3 prompt: implement model.c against the generated model.h."""
    base = _build_user_prompt(state)
    sections = [
        base,
        "",
        "=" * 60,
        "STEP 2 model.h CONTRACT TO IMPLEMENT",
        "=" * 60,
        model_h,
        "",
        "=" * 60,
        "STEP 3 TASK: IMPLEMENT model.c",
        "=" * 60,
        "Implement every prototype and tensor block declared in model.h. "
        "Include \"model.h\" (which includes weights.h) and keep model.c "
        "consistent with the inputs, outputs, dependencies, and stubs above. "
        "Output exactly ONE ```c model.c code block.",
    ]
    return "\n".join(sections)


def _extract_c_artifact(response: str, filename: Literal["model.h", "model.c"]) -> str:
    """Extract a named C artifact from an LLM response."""
    escaped = re.escape(filename)
    blocks = re.findall(
        rf"```(?:c|C)?\s*(?:{escaped})?\s*\n(.*?)```",
        response,
        re.DOTALL,
    )
    if blocks:
        return max(blocks, key=len).strip()
    return response.strip()

def _extract_header_c(response: str) -> str:
    blocks = re.findall(
        r'```(?:c|C)?\s*(?:model_functions\.h)?\s*\n(.*?)```',
        response,
        re.DOTALL,
    )
    if blocks:
        return max(blocks, key=len).strip()
    return response.strip()


def _generate_deterministic_header(state: AgentState) -> str:
    """
    Generate weights.h deterministically from the weight data.

    This replaces the old approach of having the LLM generate weights.h.
    The header contains actual weight values (not placeholders) and is
    guaranteed to match the model's state_dict.

    Works for any model type (LLMs, CNNs, etc.) since it operates
    on the raw numpy arrays from model.state_dict().
    """
    npz_path = state.get("weights_path", "")
    manifest = state.get("weights_manifest", {})
    precision = state.get("weight_precision", "f32")
    mode = state.get("weight_mode", "embedded")

    if not npz_path or not os.path.exists(npz_path):
        logger.error(f"Weights file not found: {npz_path}")
        # Fall back to generating from weight_metadata
        return _generate_fallback_header(state)

    if not manifest:
        # Generate manifest from npz if not already in state
        logger.warning("No manifest in state — generating from .npz")
        output_dir = os.path.dirname(npz_path)
        _, manifest = export_weights_binary(
            npz_path, output_dir, precision
        )

    return generate_weights_header(
        manifest=manifest,
        npz_path=npz_path,
        precision=precision,
        mode=mode,
    )


def _generate_deterministic_loader(state: AgentState) -> str:
    """
    Generate weights_loader.c for binary mode.
    Only needed when weight_mode is 'binary'.
    """
    manifest = state.get("weights_manifest", {})
    precision = state.get("weight_precision", "f32")

    return generate_weights_loader(
        manifest=manifest,
        precision=precision,
    )


def _generate_fallback_header(state: AgentState) -> str:
    """
    Generate a minimal weights.h if no .npz file is available.
    Uses zero-initialized arrays (legacy fallback).
    """
    weight_metadata = state.get("weights_metadata", {})
    ir_dict = state.get("ir_graph", {})
    if not weight_metadata:
        weight_metadata = ir_dict.get("weight_metadata", {})

    lines = [
        "#pragma once",
        "/* Auto-generated weight declarations */",
        "/* WARNING: Fallback mode — weights are zero-initialized */",
        "",
    ]
    for name, meta in sorted(weight_metadata.items()):
        c_name = name.replace(".", "_")
        numel = meta.get("numel", 1)
        shape = meta.get("shape", [])
        lines.append(f"// {name}: shape={shape}")
        lines.append(f"static const float {c_name}[{numel}] = {{0}};")
        lines.append("")
    return "\n".join(lines)


def generate_code(state: AgentState) -> dict:
    """
    LangGraph node function: Generate C code from IR.

    Reads: state["ir_graph"], state["verification_feedback"],
           state["optimization_suggestions"], state["generated_code"],
           state["code_path"],
           state["weights_path"], state["weights_manifest"],
           state["weight_precision"], state["weight_mode"]
    Writes: state["generated_code"], state["generated_header"],
            state["code_path"], state["header_path"]
    """
    logger.info("Generating C code from IR graph...")

    attempt = state.get("verification_attempts", 0)
    opt_iter = state.get("optimization_iteration", 0)
    is_retry = bool(state.get("verification_feedback", ""))
    logger.info(
        f"  Verification attempt: {attempt}, "
        f"Optimization iteration: {opt_iter}, "
        f"Repair mode: {is_retry}"
    )

    # ── Generate deterministic weights.h ────────────────────────
    weights_h = _generate_deterministic_header(state)
    mode = state.get("weight_mode", "embedded")

    # For binary mode, also generate the loader
    loader_c = ""
    if mode == "binary":
        loader_c = _generate_deterministic_loader(state)

    # ── Build LLM messages ──────────────────────────────────────
    system_prompt = _load_system_prompt()

    # ── Call vLLM ───────────────────────────────────────────────
    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        model=VLLM_MODEL,
        temperature=0.2 if not is_retry else 0.1,  # Lower temp on retries
        max_tokens=LLM_MAX_TOKENS,
    )

    # Step 2: ask the LLM for model.h, which defines includes, tensor
    # contracts, dependencies, and all function stubs to be implemented.
    header_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_model_header_prompt(state)),
    ]

    logger.info(f"Calling LLM ({VLLM_MODEL}) for step 2 model.h ...")
    if is_retry:
        logger.info("  → REPAIR MODE: feeding back current code + errors")
    header_response = llm.invoke(header_messages)
    raw_header_response = header_response.content
    logger.info(f"LLM model.h response length: {len(raw_header_response)} chars")

    # ── Extract model.h ─────────────────────────────────────────
    model_h = _extract_c_artifact(raw_header_response, "model.h")

    # Step 3: ask the LLM to implement model.c against the model.h contract.
    c_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_model_c_prompt(state, model_h)),
    ]

    logger.info(f"Calling LLM ({VLLM_MODEL}) for step 3 model.c ...")
    c_response = llm.invoke(c_messages)
    raw_c_response = c_response.content
    logger.info(f"LLM model.c response length: {len(raw_c_response)} chars")

    # ── Extract model.c ─────────────────────────────────────────
    model_c = _extract_c_artifact(raw_c_response, "model.c")

    # ── Write files ─────────────────────────────────────────────
    code_path = OUTPUT_DIR / "model.c"
    model_header_path = OUTPUT_DIR / "model.h"
    header_path = OUTPUT_DIR / "weights.h"
    funcs_path = OUTPUT_DIR / "model_functions.h"

    code_path.write_text(model_c, encoding="utf-8")
    model_header_path.write_text(model_h, encoding="utf-8")
    header_path.write_text(weights_h, encoding="utf-8")
    funcs_path.write_text(functions_h, encoding="utf-8")
    
    # Save to persistent storage for retries
    (OUTPUT_DIR / "_latest_model.c").write_text(model_c, encoding="utf-8")

    logger.info(f"Generated code written to: {code_path}")
    logger.info(f"Generated model header written to: {model_header_path}")
    logger.info(f"Generated header written to: {header_path}")

    # Write loader if in binary mode
    if loader_c:
        loader_path = OUTPUT_DIR / "weights_loader.c"
        loader_path.write_text(loader_c, encoding="utf-8")
        logger.info(f"Generated loader written to: {loader_path}")

    return {
        "generated_code": model_c,
        "generated_header": weights_h,
        "generated_model_header": model_h,
        "code_path": str(code_path),
        "model_header_path": str(model_header_path),
        "header_path": str(header_path),
        "functions_header_path": str(funcs_path),
    }
