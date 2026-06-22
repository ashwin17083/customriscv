"""
Hardware-Aware Optimizer Agent — Rewrites verified C code for custom HW accelerators.

Reads hw_config.yaml (project root) to learn about available:
  - Custom instructions (systolic arrays, VPUs, etc.)
  - Multi-core / multi-accelerator distribution strategy
  - Vector extensions

Produces model_optimized.h + model_optimized.c that use these custom
instructions instead of standard C loops, while preserving the model's
mathematical semantics.

Uses the same two-phase LLM generation pattern as the code generator:
  Phase 1 → model_optimized.h  (header + custom instruction wrappers)
  Phase 2 → model_optimized.c  (full rewrite using the custom instructions)

On verification retries (opt_verification_feedback set), runs in REPAIR MODE.

Supports the same LLM_BACKEND env var as code_generator.py:
  LLM_BACKEND=vllm   → Qwen2.5-Coder-32B via vLLM (default)
  LLM_BACKEND=ollama → deepseek-coder-v2:16b-lite-instruct-q4_K_M via Ollama
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

# Re-use helpers from code_generator — avoids code duplication
from agents.code_generator import (
    _build_llm,
    _extract_c_artifact,
    LLM_MAX_TOKENS,
    OUTPUT_DIR,
)
from ir import IRGraph
from state import AgentState, LLMCallStats

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "hw_optimizer.txt"
HW_CONFIG_PATH = Path(__file__).parent.parent / "hw_config.yaml"

MAX_OPT_VERIFICATION_ATTEMPTS = 3


# ── hw_config helpers ──────────────────────────────────────────

def _load_hw_config() -> dict:
    """Load hw_config.yaml from the project root. Returns empty dict if missing."""
    if not HW_CONFIG_PATH.exists():
        logger.warning(
            f"hw_config.yaml not found at {HW_CONFIG_PATH}. "
            "HW-aware optimization will proceed with no custom instructions."
        )
        return {}
    try:
        with open(HW_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded hw_config.yaml: target={config.get('target', {})}")
        return config
    except Exception as e:
        logger.error(f"Failed to parse hw_config.yaml: {e}")
        return {}


def _hw_config_summary(config: dict) -> str:
    """Format hw_config as a readable prompt section."""
    if not config:
        return "(No hardware configuration provided — generating standard C)"

    lines = []

    target = config.get("target", {})
    if target:
        lines.append(f"Target ISA : {target.get('isa', 'rv32imac')}")
        lines.append(f"Description: {target.get('description', '')}")
        lines.append("")

    custom_instrs = config.get("custom_instructions", [])
    enabled_instrs = [ci for ci in custom_instrs if ci.get("enabled", False)]
    if enabled_instrs:
        lines.append("CUSTOM INSTRUCTIONS (enabled):")
        for ci in enabled_instrs:
            lines.append(f"  [{ci['name']}] {ci.get('description', '')}")
            lines.append(f"    applies_to : {ci.get('applies_to', [])}")
            if ci.get("tile_size"):
                lines.append(f"    tile_size  : {ci['tile_size']}")
            lines.append(f"    example    : {ci.get('usage_example', '')}")
            lines.append(f"    definition :\n{ci.get('header_definition', '')}")
            lines.append("")
    else:
        lines.append("CUSTOM INSTRUCTIONS: none enabled")

    mc = config.get("multi_core", {})
    if mc.get("enabled"):
        lines.append(f"MULTI-CORE: {mc.get('num_cores')} cores, "
                     f"{mc.get('num_systolic_arrays')} systolic arrays, "
                     f"strategy={mc.get('distribution_strategy')}")

    vec = config.get("vector_extension", {})
    if vec.get("enabled"):
        lines.append(f"VECTOR EXT : {vec.get('extension')} (vlen={vec.get('vlen')} bits)")

    notes = config.get("notes", "")
    if notes:
        lines.append(f"\nNotes:\n{notes}")

    return "\n".join(lines)


# ── Disk helpers ───────────────────────────────────────────────

def _read_file_or_state(path_key: str, state_key: str, state: AgentState,
                        default_filename: str = "") -> str:
    """Read a file from path in state, falling back to state field, then disk."""
    path_value = state.get(path_key, "")
    if path_value:
        p = Path(path_value)
        if p.exists():
            return p.read_text(encoding="utf-8")

    if default_filename:
        default_path = OUTPUT_DIR / default_filename
        if default_path.exists():
            return default_path.read_text(encoding="utf-8")

    return state.get(state_key, "")


def _read_original_model_c(state: AgentState) -> str:
    return _read_file_or_state("code_path", "generated_code", state, "model.c")


def _read_original_model_h(state: AgentState) -> str:
    return _read_file_or_state("model_header_path", "generated_model_header", state, "model.h")


def _read_optimized_model_c(state: AgentState) -> str:
    return _read_file_or_state("optimized_code_path", "optimized_code", state, "model_optimized.c")


def _read_optimized_model_h(state: AgentState) -> str:
    return _read_file_or_state("optimized_header_path", "optimized_header", state, "model_optimized.h")


# ── Prompt builders ────────────────────────────────────────────

def _build_hw_header_prompt(
    state: AgentState,
    hw_config: dict,
    is_retry: bool,
) -> str:
    """Build the Phase-1 prompt: generate model_optimized.h."""
    sections: list[str] = []

    sections += [
        "=" * 60,
        "HARDWARE CONFIGURATION (hw_config.yaml)",
        "=" * 60,
        _hw_config_summary(hw_config),
    ]

    ir_dict = state.get("ir_graph", {})
    if ir_dict:
        ir_graph = IRGraph.from_dict(ir_dict)
        sections += [
            "",
            "=" * 60,
            "IR GRAPH SUMMARY",
            "=" * 60,
            ir_graph.layer_summary(),
        ]

    original_h = _read_original_model_h(state)
    sections += [
        "",
        "=" * 60,
        "ORIGINAL model.h (verified — use as base)",
        "=" * 60,
        original_h,
    ]

    if is_retry:
        current_h = _read_optimized_model_h(state)
        feedback = state.get("opt_verification_feedback", "")
        if current_h:
            sections += [
                "",
                "=" * 60,
                "⚠️  CURRENT model_optimized.h (REPAIR MODE — contains errors)",
                "=" * 60,
                current_h,
            ]
        sections += [
            "",
            "=" * 60,
            "⚠️  VERIFICATION ERRORS — FIX THESE IN model_optimized.h",
            "=" * 60,
            feedback,
        ]

    sections += [
        "",
        "=" * 60,
        "TASK: GENERATE model_optimized.h",
        "=" * 60,
        "Create model_optimized.h that:",
        "  1. Starts with #pragma once",
        "  2. #include \"model.h\"  (which pulls in weights.h)",
        "  3. Copies VERBATIM every enabled 'header_definition' block from hw_config",
        "  4. Adds any extra prototypes needed by model_optimized.c",
        "Output exactly ONE ```c model_optimized.h code block.",
    ]

    return "\n".join(sections)


def _build_hw_c_prompt(
    state: AgentState,
    hw_config: dict,
    optimized_h: str,
    is_retry: bool,
) -> str:
    """Build the Phase-2 prompt: generate model_optimized.c."""
    sections: list[str] = []

    sections += [
        "=" * 60,
        "HARDWARE CONFIGURATION (hw_config.yaml)",
        "=" * 60,
        _hw_config_summary(hw_config),
    ]

    original_c = _read_original_model_c(state)
    sections += [
        "",
        "=" * 60,
        "ORIGINAL model.c (verified — rewrite this using custom HW instructions)",
        "=" * 60,
        original_c,
    ]

    sections += [
        "",
        "=" * 60,
        "model_optimized.h CONTRACT (implement against this)",
        "=" * 60,
        optimized_h,
    ]

    if is_retry:
        current_c = _read_optimized_model_c(state)
        feedback = state.get("opt_verification_feedback", "")
        if current_c:
            sections += [
                "",
                "=" * 60,
                "⚠️  CURRENT model_optimized.c (REPAIR MODE — contains errors)",
                "=" * 60,
                current_c,
            ]
        sections += [
            "",
            "=" * 60,
            "⚠️  VERIFICATION ERRORS — FIX THESE IN model_optimized.c",
            "=" * 60,
            feedback,
            "",
            "You are in REPAIR MODE. Fix ALL the errors listed above. "
            "Keep all correct custom instruction calls intact. "
            "Mark fixes with // FIX: <description> comments.",
        ]

    sections += [
        "",
        "=" * 60,
        "TASK: GENERATE model_optimized.c",
        "=" * 60,
        "Implement model_optimized.c that:",
        "  1. #include \"model_optimized.h\"",
        "  2. Replaces eligible operations with custom instructions per hw_config",
        "  3. Preserves all function signatures from model.c",
        "  4. Keeps void model_inference(const float* input, float* output) as entry point",
        "  5. Does NOT use malloc/calloc/free or any OS calls",
        "Output exactly ONE ```c model_optimized.c code block.",
    ]

    return "\n".join(sections)


# ── File writing ───────────────────────────────────────────────

def _write_optimized_artifacts(optimized_c: str, optimized_h: str) -> dict[str, str]:
    """Write model_optimized.h and model_optimized.c to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    c_path = OUTPUT_DIR / "model_optimized.c"
    h_path = OUTPUT_DIR / "model_optimized.h"
    c_path.write_text(optimized_c, encoding="utf-8")
    h_path.write_text(optimized_h, encoding="utf-8")
    logger.info(f"Written: {c_path}")
    logger.info(f"Written: {h_path}")
    return {
        "optimized_code_path": str(c_path),
        "optimized_header_path": str(h_path),
    }


# ── LLM call helper ────────────────────────────────────────────

def _llm_call(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    label: str,
) -> tuple[str, int, int, float]:
    """Invoke LLM and return (content, input_tokens, output_tokens, latency_s)."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    latency = time.perf_counter() - t0

    content = response.content
    usage = getattr(response, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        in_tok  = usage.get("input_tokens",  0)
        out_tok = usage.get("output_tokens", 0)
    else:
        in_tok  = getattr(usage, "input_tokens",  0)
        out_tok = getattr(usage, "output_tokens", 0)

    logger.info(
        f"  {label} — input_tokens={in_tok}, output_tokens={out_tok}, "
        f"latency={latency:.2f}s, response_len={len(content)} chars"
    )
    return content, in_tok, out_tok, latency


# ── Main agent function ────────────────────────────────────────

def hw_optimize(state: AgentState) -> dict:
    """
    LangGraph node function: Hardware-aware code rewriting agent.

    Reads:  state["generated_code"]         (original verified model.c)
            state["generated_model_header"]  (original verified model.h)
            state["ir_graph"]
            state["opt_verification_feedback"]  (on retry)
            hw_config.yaml from project root
    Writes: state["optimized_code"]
            state["optimized_header"]
            state["optimized_code_path"]
            state["optimized_header_path"]
            state["hw_config"]
            Telemetry fields
    """
    attempt = state.get("opt_verification_attempts", 0)
    is_retry = bool(state.get("opt_verification_feedback", ""))

    logger.info("=" * 60)
    logger.info("HW-AWARE OPTIMIZER")
    logger.info("=" * 60)
    logger.info(
        f"  Opt verification attempt: {attempt}/{MAX_OPT_VERIFICATION_ATTEMPTS}, "
        f"Repair mode: {is_retry}"
    )

    # ── Load hardware config ─────────────────────────────────────
    hw_config = _load_hw_config()

    # ── Load system prompt ───────────────────────────────────────
    if not PROMPT_PATH.exists():
        logger.error(f"HW optimizer prompt not found: {PROMPT_PATH}")
        return {"error": f"Missing prompt file: {PROMPT_PATH}"}
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    # ── Build LLM ───────────────────────────────────────────────
    llm = _build_llm(temperature=0.1 if not is_retry else 0.0)

    # ── Telemetry accumulators ───────────────────────────────────
    call_stats: list[LLMCallStats] = []
    total_input = 0
    total_output = 0
    total_latency = 0.0

    # ══ Phase 1: Generate model_optimized.h ══════════════════════
    logger.info("Phase 1: Generating model_optimized.h ...")
    h_prompt = _build_hw_header_prompt(state, hw_config, is_retry)
    h_content, h_in, h_out, h_lat = _llm_call(llm, system_prompt, h_prompt, "model_optimized.h")
    optimized_h = _extract_c_artifact(h_content, "model_optimized.h")

    call_stats.append(LLMCallStats(
        agent="hw_optimizer",
        call_label="model_optimized.h",
        input_tokens=h_in,
        output_tokens=h_out,
        total_tokens=h_in + h_out,
        latency_s=h_lat,
    ))
    total_input   += h_in
    total_output  += h_out
    total_latency += h_lat

    # ══ Phase 2: Generate model_optimized.c ══════════════════════
    logger.info("Phase 2: Generating model_optimized.c ...")
    c_prompt = _build_hw_c_prompt(state, hw_config, optimized_h, is_retry)
    c_content, c_in, c_out, c_lat = _llm_call(llm, system_prompt, c_prompt, "model_optimized.c")
    optimized_c = _extract_c_artifact(c_content, "model_optimized.c")

    call_stats.append(LLMCallStats(
        agent="hw_optimizer",
        call_label="model_optimized.c",
        input_tokens=c_in,
        output_tokens=c_out,
        total_tokens=c_in + c_out,
        latency_s=c_lat,
    ))
    total_input   += c_in
    total_output  += c_out
    total_latency += c_lat

    # ── Write files ──────────────────────────────────────────────
    artifact_paths = _write_optimized_artifacts(optimized_c, optimized_h)

    # ── Merge telemetry ──────────────────────────────────────────
    existing_stats: list[LLMCallStats] = list(state.get("llm_call_stats") or [])
    existing_stats.extend(call_stats)
    prev_input   = state.get("total_input_tokens",  0) or 0
    prev_output  = state.get("total_output_tokens", 0) or 0
    prev_latency = state.get("total_llm_latency_s", 0.0) or 0.0
    prev_al: dict = dict(state.get("agent_latencies") or {})
    prev_al["hw_optimizer"] = prev_al.get("hw_optimizer", 0.0) + total_latency

    return {
        "optimized_code":   optimized_c,
        "optimized_header": optimized_h,
        **artifact_paths,
        "hw_config": hw_config,
        # Clear feedback so next verifier run starts fresh
        "opt_verification_feedback": "",
        # Telemetry
        "llm_call_stats":     existing_stats,
        "total_input_tokens":  prev_input  + total_input,
        "total_output_tokens": prev_output + total_output,
        "total_llm_latency_s": prev_latency + total_latency,
        "agent_latencies": prev_al,
    }
