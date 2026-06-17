"""
Optimization Agent — Hardware-aware hints pass.

Injects knowledge of the target micro-architecture (e.g. systolic arrays,
VPUs, SIMD units) into a new code-generator run so the output C code is
structured to exploit available hardware resources.

Runs ONCE, after human approval, BEFORE simulation.  It does not loop.
Uses Qwen2.5-Coder-32B via local vLLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ir import IRGraph
from state import AgentState, LLMCallStats

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "optimizer.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
LLM_MAX_TOKENS = 200_000

MAX_OPTIMIZATION_ITERATIONS = 3


def _read_generated_code_from_output(state: AgentState) -> str:
    """Read model.c from disk so optimization uses verified output content."""
    code_path = state.get("code_path", "")
    if code_path:
        path = Path(code_path)
        if path.exists():
            return path.read_text(encoding="utf-8")

    default_path = OUTPUT_DIR / "model.c"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")

    return state.get("generated_code", "")


def _load_system_prompt() -> str:
    """Load the optimization system prompt."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_optimization_prompt(state: AgentState) -> str:
    """Build the prompt with current metrics and code."""
    sections = []

    # ── Current Code ────────────────────────────────────────────
    code = _read_generated_code_from_output(state)
    sections.append("=" * 60)
    sections.append("CURRENT GENERATED C CODE")
    sections.append("=" * 60)
    sections.append(code)

    # ── IR Graph ────────────────────────────────────────────────
    ir_dict = state.get("ir_graph", {})
    if ir_dict:
        ir_graph = IRGraph.from_dict(ir_dict)
        sections.append("")
        sections.append("=" * 60)
        sections.append("IR GRAPH SUMMARY")
        sections.append("=" * 60)
        sections.append(ir_graph.layer_summary())

    # ── Simulation Results ──────────────────────────────────────
    sim = state.get("simulation_result", {})
    if sim:
        sections.append("")
        sections.append("=" * 60)
        sections.append("SIMULATION RESULTS (Hazard3 RISC-V)")
        sections.append("=" * 60)
        sections.append(f"  Clock Cycles: {sim.get('cycles', 'N/A'):,}")
        sections.append(f"  Output Match: {sim.get('output_match', 'N/A')}")

    # ── Synthesis Results ───────────────────────────────────────
    synth = state.get("synthesis_result", {})
    if synth:
        sections.append("")
        sections.append("=" * 60)
        sections.append("SYNTHESIS RESULTS (OpenROAD)")
        sections.append("=" * 60)
        sections.append(f"  Power:     {synth.get('power_watts', 'N/A')} W")
        sections.append(f"  Area:      {synth.get('area_mm2', 'N/A')} mm²")
        sections.append(f"  Frequency: {synth.get('frequency_mhz', 'N/A')} MHz")
        sections.append(f"  Cells:     {synth.get('cell_count', 'N/A')}")

    # ── Previous Optimizations ──────────────────────────────────
    prev_suggestions = state.get("optimization_suggestions", [])
    if prev_suggestions:
        sections.append("")
        sections.append("=" * 60)
        sections.append("PREVIOUSLY APPLIED OPTIMIZATIONS")
        sections.append("=" * 60)
        for s in prev_suggestions:
            sections.append(f"  • {s}")
        sections.append(
            "\nSuggest NEW optimizations that are different from the above."
        )

    sections.append("")
    sections.append("=" * 60)
    sections.append("TASK")
    sections.append("=" * 60)
    sections.append(
        "Analyze the simulation and synthesis results above in detail. "
        "Reference concrete functions, buffers, loops, weight arrays, and "
        "IR nodes from the current output file wherever possible. "
        "Suggest up to 5 concrete code optimizations to improve "
        "performance (reduce cycles), power, and/or area. "
        "Return your suggestions as a JSON array."
    )

    return "\n".join(sections)


def _parse_suggestions(response: str) -> list[str]:
    """Extract optimization suggestions from LLM response."""
    suggestions = []

    # Try to parse JSON array from response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            for item in items:
                if isinstance(item, dict):
                    suggestion = item.get("suggestion", "")
                    category = item.get("category", "")
                    target = item.get("target", "")
                    if suggestion:
                        s = f"[{category}] {target}: {suggestion}"
                        suggestions.append(s)
                elif isinstance(item, str):
                    suggestions.append(item)
        except json.JSONDecodeError:
            pass

    # Fallback: extract bullet points
    if not suggestions:
        for line in response.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•")
                         or line.startswith("*")):
                suggestions.append(line.lstrip("-•* "))
            elif re.match(r'^\d+[\.\)]\s', line):
                suggestions.append(re.sub(r'^\d+[\.\)]\s*', '', line))

    return suggestions[:5]  # Max 5 suggestions


def optimize(state: AgentState) -> dict:
    """
    LangGraph node function: Hardware-aware optimization hints pass.

    Reads: state["generated_code"], state["ir_graph"]
           (simulation_result / synthesis_result not yet available at this
           point in the new flow — optimizer works from IR + code only).
    Writes: state["optimization_suggestions"], state["optimization_iteration"]
    """
    iteration = state.get("optimization_iteration", 0) + 1
    logger.info(f"Optimization iteration {iteration}/{MAX_OPTIMIZATION_ITERATIONS}")

    # ── Build and send LLM prompt ───────────────────────────────
    system_prompt = _load_system_prompt()
    user_prompt = _build_optimization_prompt(state)

    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        model=VLLM_MODEL,
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info(f"Calling LLM for optimization suggestions...")
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    llm_latency = time.perf_counter() - t0
    raw_response = response.content

    # ── Parse suggestions ────────────────────────────────
    suggestions = _parse_suggestions(raw_response)

    logger.info(f"Generated {len(suggestions)} optimization suggestions:")
    for i, s in enumerate(suggestions, 1):
        logger.info(f"  {i}. {s}")

    # ── Token stats ────────────────────────────────────────
    usage = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  0) if isinstance(usage, dict) else getattr(usage, "input_tokens",  0)
    out_tok = usage.get("output_tokens", 0) if isinstance(usage, dict) else getattr(usage, "output_tokens", 0)
    logger.info(
        f"  optimizer — input_tokens={in_tok}, output_tokens={out_tok}, "
        f"latency={llm_latency:.2f}s"
    )

    new_stat = LLMCallStats(
        agent="optimizer",
        call_label=f"optimize_iter_{iteration}",
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        latency_s=llm_latency,
    )

    existing_stats: list[LLMCallStats] = list(state.get("llm_call_stats") or [])
    existing_stats.append(new_stat)
    prev_input   = state.get("total_input_tokens",  0) or 0
    prev_output  = state.get("total_output_tokens", 0) or 0
    prev_latency = state.get("total_llm_latency_s", 0.0) or 0.0
    prev_agent_latencies: dict = dict(state.get("agent_latencies") or {})
    prev_agent_latencies["optimizer"] = (
        prev_agent_latencies.get("optimizer", 0.0) + llm_latency
    )

    return {
        "optimization_suggestions": suggestions,
        "optimization_iteration": iteration,
        # Telemetry
        "llm_call_stats": existing_stats,
        "total_input_tokens":  prev_input  + in_tok,
        "total_output_tokens": prev_output + out_tok,
        "total_llm_latency_s": prev_latency + llm_latency,
        "agent_latencies": prev_agent_latencies,
    }
