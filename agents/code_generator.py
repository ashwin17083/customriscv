"""
Code Generator Agent — Converts Custom IR → RISC-V C code.

Uses Qwen2.5-Coder-32B via local vLLM server (OpenAI-compatible API)
to generate model.c from the IR graph.

weights.h is generated deterministically (not by the LLM) using
the export_weights utility. The LLM only produces model.c.

On verification retries, the previously generated code is fed back
to the LLM in "repair mode" (Codex-inspired) so it can make targeted
fixes rather than regenerating from scratch.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

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

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "codegen.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _load_system_prompt() -> str:
    """Load the code generation system prompt."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_user_prompt(state: AgentState) -> str:
    """
    Build the user prompt containing IR graph, weight metadata,
    and any previous feedback or optimization suggestions.

    On retries (when verification_feedback is set), includes the
    previously generated code in a REPAIR MODE section so the LLM
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

    # ── REPAIR MODE: Current Code + Errors ──────────────────────
    if is_retry:
        current_code = state.get("generated_code", "")
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
            "Do NOT generate weights.h — it is auto-generated. "
            "Just #include \"weights.h\" and use the weight array names "
            "listed above. "
            "Output exactly ONE ```c model.c code block."
        )

    return "\n".join(sections)


def _extract_model_c(response: str) -> str:
    """
    Extract model.c from the LLM response.

    Looks for a fenced code block. Since we now only ask for model.c,
    this is simpler than the old two-file extraction.
    """
    # Strategy 1: Named code block (```c model.c ... ```)
    blocks = re.findall(
        r'```(?:c|C)?\s*(?:model\.c)?\s*\n(.*?)```',
        response,
        re.DOTALL,
    )

    if blocks:
        # Return the largest code block (most likely the full model.c)
        return max(blocks, key=len).strip()

    # Strategy 2: Any code block
    all_blocks = re.findall(r'```(?:\w*)\s*\n(.*?)```', response, re.DOTALL)
    for block in all_blocks:
        block = block.strip()
        if "model_inference" in block or '#include "weights.h"' in block:
            return block

    # Strategy 3: Use entire response
    logger.warning("Could not extract code blocks — using raw response")
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
    user_prompt = _build_user_prompt(state)

    # ── Call vLLM ───────────────────────────────────────────────
    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        model=VLLM_MODEL,
        temperature=0.2 if not is_retry else 0.1,  # Lower temp on retries
        max_tokens=8192,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info(f"Calling LLM ({VLLM_MODEL}) ...")
    if is_retry:
        logger.info("  → REPAIR MODE: feeding back current code + errors")
    response = llm.invoke(messages)
    raw_response = response.content
    logger.info(f"LLM response length: {len(raw_response)} chars")

    # ── Extract model.c ─────────────────────────────────────────
    model_c = _extract_model_c(raw_response)

    # ── Write files ─────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    code_path = OUTPUT_DIR / "model.c"
    header_path = OUTPUT_DIR / "weights.h"

    code_path.write_text(model_c, encoding="utf-8")
    header_path.write_text(weights_h, encoding="utf-8")

    logger.info(f"Generated code written to: {code_path}")
    logger.info(f"Generated header written to: {header_path}")

    # Write loader if in binary mode
    if loader_c:
        loader_path = OUTPUT_DIR / "weights_loader.c"
        loader_path.write_text(loader_c, encoding="utf-8")
        logger.info(f"Generated loader written to: {loader_path}")

    return {
        "generated_code": model_c,
        "generated_header": weights_h,
        "code_path": str(code_path),
        "header_path": str(header_path),
    }
