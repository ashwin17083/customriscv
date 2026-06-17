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
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ir import required_helper_signatures
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
LLM_MAX_TOKENS = 200_000


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


def _is_repair_mode(state: AgentState) -> bool:
    """Return True when generation should receive previous artifacts for repair."""
    if state.get("verification_feedback", ""):
        return True
    verification_result = state.get("verification_result", {})
    if verification_result and not verification_result.get("passed", False):
        return True
    return state.get("verification_attempts", 0) > 0


def _build_repair_context_sections(state: AgentState, target_artifact: str) -> list[str]:
    """Build repair-mode context with original generated artifacts and errors."""
    feedback = state.get("verification_feedback", "")
    current_header = _read_generated_model_header_from_output(state)
    current_code = _read_generated_code_from_output(state)

    sections: list[str] = [
        "",
        "=" * 60,
        f"⚠️  REPAIR MODE CONTEXT FOR {target_artifact}",
        "=" * 60,
        "Use the original generated artifacts below as the starting point. "
        "Make targeted fixes only; do not drop unrelated declarations, helper "
        "functions, buffers, or model_inference steps.",
    ]

    if current_header:
        sections.extend([
            "",
            "=" * 60,
            "ORIGINAL model.h FROM PREVIOUS GENERATION",
            "=" * 60,
            current_header,
        ])

    if current_code:
        sections.extend([
            "",
            "=" * 60,
            "ORIGINAL model.c FROM PREVIOUS GENERATION",
            "=" * 60,
            current_code,
        ])

    sections.extend([
        "",
        "=" * 60,
        "VERIFICATION ERRORS — FIX THESE",
        "=" * 60,
        feedback or "(No verifier feedback was provided.)",
    ])
    return sections


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
    is_retry = _is_repair_mode(state)

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

    # ── REPAIR MODE: Original Artifacts + Errors ────────────────
    if is_retry:
        sections.extend(_build_repair_context_sections(state, "model.c"))
        sections.append("")
        sections.append(
            "You are in REPAIR MODE. Fix ALL the errors listed above "
            "using the original model.h and model.c as context. Output the "
            "COMPLETE fixed model.c file. Keep the overall structure intact. "
            "Mark fixes with // FIX: <description> comments."
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
            "Context for model generation follows."
            "The actual generation instructions are provided later."
            "Do not generate code based only on this section."
            "model targeting RISC-V rv32imac. "
            "Do NOT generate weights.h or model.h in this response — both "
            "are already available. Just #include \"model.h\" and use the "
            "weight array names listed above through the model.h -> weights.h "
            "include chain. "
            "Output exactly ONE ```c model.c code block."
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
    required_helpers = required_helper_signatures(state.get("ir_graph", {}))
    is_retry = _is_repair_mode(state)
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
        "",
        "REQUIRED HELPER PROTOTYPES:",
        *(f"  {signature}" for signature in required_helpers),
        "If the implementation needs any additional non-static helper function, "
        "declare its prototype in model.h as well.",
        "Output exactly ONE ```c model.h code block.",
    ]

    if is_retry:
        sections.extend(_build_repair_context_sections(state, "model.h"))
        sections.append(
            "Make targeted header fixes and output the COMPLETE fixed model.h. "
            "Preserve any declarations that are still needed by the original model.c."
        )

    return "\n".join(sections)


def _build_model_c_prompt(state: AgentState, model_h: str) -> str:
    """Build step-3 prompt: implement model.c against the generated model.h."""
    base = _build_user_prompt(state)
    required_helpers = required_helper_signatures(state.get("ir_graph", {}))
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
        "The following IR-required helpers must have matching non-static "
        "definitions in model.c:",
        *(f"  {signature}" for signature in required_helpers),
        "Output exactly ONE ```c model.c code block.",
    ]
    return "\n".join(sections)


def _raw_c_artifact_start(text: str, filename: str) -> int | None:
    """Find where an unfenced LLM response appears to start the C artifact."""
    markers = (
        ("#pragma once", "#ifndef", "#include", "void model_inference")
        if filename.endswith(".h")
        else ('#include "model.h"', "#include <", "void model_inference")
    )
    positions = [text.find(marker) for marker in markers if text.find(marker) >= 0]
    if not positions:
        return None
    return min(positions)


def _extract_raw_c_artifact_text(text: str, filename: str) -> str | None:
    """Extract plausible raw C from an unfenced response, trimming leading prose."""
    stripped = text.strip()
    if not stripped:
        return None

    start = _raw_c_artifact_start(stripped, filename)
    if start is None:
        return None

    artifact = stripped[start:].strip()
    if filename.endswith(".c") and "void model_inference" not in artifact:
        return None
    return artifact


def _looks_like_c_artifact(text: str, filename: str) -> bool:
    """Return True when unfenced LLM text appears to contain the requested C artifact."""
    return _extract_raw_c_artifact_text(text, filename) is not None


def _extract_c_artifact(response, filename):
    """
    Extract a generated C artifact from an LLM response.

    Prefer a fenced block labelled with the requested filename. If the LLM omits
    the filename label (or omits fences entirely but returns plausible C), recover
    the artifact instead of failing the pipeline immediately. This makes step-2
    model.h generation tolerant of common local-model formatting drift while
    still raising a clear error for responses that do not contain usable C.
    """
    named_pattern = rf"```(?:c|C)?\s*{re.escape(filename)}\s*\n(.*?)```"
    named_blocks = re.findall(named_pattern, response, re.DOTALL)
    if named_blocks:
        if len(named_blocks) > 1:
            logger.warning(
                "Multiple %s code blocks detected (%d). Using largest.",
                filename,
                len(named_blocks),
            )
        return max(named_blocks, key=len).strip()

    generic_blocks = re.findall(
        r"```(?:c|C)?\s*\n(.*?)```",
        response,
        re.DOTALL,
    )
    if generic_blocks:
        logger.warning(
            "No fenced code block named %s found. Using largest generic C block.",
            filename,
        )
        return max(generic_blocks, key=len).strip()

    raw_artifact = _extract_raw_c_artifact_text(response, filename)
    if raw_artifact is not None:
        logger.warning(
            "No fenced code block named %s found. Extracting raw C artifact text.",
            filename,
        )
        return raw_artifact

    raise ValueError(
        f"Could not extract {filename}: expected a fenced code block named "
        f"{filename}, a generic C code block, or raw C artifact text."
    )


def _write_llm_call_log(
    *,
    step_number: int,
    attempt: int,
    messages,
    raw_response: str,
) -> Path:
    """Persist the exact LLM input prompt and raw response for debugging."""
    log_dir = OUTPUT_DIR / "llm_call"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = log_dir / f"llm_step_{step_number}_{timestamp}_attempt{attempt}.txt"

    lines = [
        f"LLM step: {step_number}",
        f"Attempt: {attempt}",
        f"UTC timestamp: {timestamp}",
        "",
        "=" * 80,
        "INPUT PROMPT",
        "=" * 80,
    ]
    for index, message in enumerate(messages, 1):
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", str(message))
        lines.extend([
            "",
            f"--- Message {index}: {role} ---",
            str(content),
        ])

    lines.extend([
        "",
        "=" * 80,
        "RAW LLM RESPONSE",
        "=" * 80,
        raw_response,
        "",
    ])
    log_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LLM step %d call log written to: %s", step_number, log_path)
    return log_path


def _invoke_llm_and_extract_artifact(
    llm, messages, filename: str, step_name: str, step_number: int
) -> str:
    """Invoke the LLM and retry once if artifact extraction fails."""
    last_error: ValueError | None = None
    for attempt in range(1, 3):
        response = llm.invoke(messages)
        raw_response = response.content
        _write_llm_call_log(
            step_number=step_number,
            attempt=attempt,
            messages=messages,
            raw_response=raw_response,
        )
        logger.info(
            "LLM %s response length: %d chars",
            filename,
            len(raw_response),
        )
        try:
            return _extract_c_artifact(raw_response, filename)
        except ValueError as exc:
            last_error = exc
            if attempt == 1:
                logger.warning(
                    "%s extraction failed: %s. Retrying generation once.",
                    step_name,
                    exc,
                )
            else:
                logger.error("%s extraction failed after retry: %s", step_name, exc)
    raise last_error or ValueError(f"Could not extract {filename}.")

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


def _write_generated_artifacts(
    model_c: str,
    model_h: str,
    weights_h: str,
    loader_c: str = "",
) -> dict[str, str]:
    """
    Write generated code artifacts and return their state path fields.

    Keeping all file writes in one helper avoids legacy loose variables and
    makes the generation write path easier to test end-to-end.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    code_path = OUTPUT_DIR / "model.c"
    model_header_path = OUTPUT_DIR / "model.h"
    header_path = OUTPUT_DIR / "weights.h"

    code_path.write_text(model_c, encoding="utf-8")
    model_header_path.write_text(model_h, encoding="utf-8")
    header_path.write_text(weights_h, encoding="utf-8")

    logger.info(f"Generated code written to: {code_path}")
    logger.info(f"Generated model header written to: {model_header_path}")
    logger.info(f"Generated header written to: {header_path}")

    paths = {
        "code_path": str(code_path),
        "model_header_path": str(model_header_path),
        "header_path": str(header_path),
    }

    if loader_c:
        loader_path = OUTPUT_DIR / "weights_loader.c"
        loader_path.write_text(loader_c, encoding="utf-8")
        logger.info(f"Generated loader written to: {loader_path}")
        paths["loader_path"] = str(loader_path)

    return paths


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
    is_retry = _is_repair_mode(state)
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
        temperature=0.2 if not is_retry else 0.0,  # Lower temp on retries
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
    # ── Extract model.h ─────────────────────────────────────────
    model_h = _invoke_llm_and_extract_artifact(
        llm, header_messages, "model.h", "step 2 model.h", 2
    )

    # Step 3: ask the LLM to implement model.c against the model.h contract.
    c_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_model_c_prompt(state, model_h)),
    ]

    logger.info(f"Calling LLM ({VLLM_MODEL}) for step 3 model.c ...")
    # ── Extract model.c ─────────────────────────────────────────
    model_c = _invoke_llm_and_extract_artifact(
        llm, c_messages, "model.c", "step 3 model.c", 3
    )

    # ── Write files ─────────────────────────────────────────────
    artifact_paths = _write_generated_artifacts(
        model_c=model_c,
        model_h=model_h,
        weights_h=weights_h,
        loader_c=loader_c,
    )

    return {
        "generated_code": model_c,
        "generated_header": weights_h,
        "generated_model_header": model_h,
        **artifact_paths,
    }
