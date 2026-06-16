"""
Optimization Agent — Analyzes simulation/synthesis results and suggests
code optimizations to improve performance, power, and area.

Uses Qwen2.5-Coder-32B via local vLLM to generate optimization suggestions
based on hardware metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ir import IRGraph
from state import AgentState

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "optimizer.txt"

MAX_OPTIMIZATION_ITERATIONS = 3


def _load_system_prompt() -> str:
    """Load the optimization system prompt."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_optimization_prompt(state: AgentState) -> str:
    """Build the prompt with current metrics and code."""
    sections = []

    # ── Current Code ────────────────────────────────────────────
    code = state.get("generated_code", "")
    sections.append("=" * 60)
    sections.append("CURRENT GENERATED C CODE")
    sections.append("=" * 60)
    # Truncate if very long to fit context
    if len(code) > 6000:
        sections.append(code[:3000])
        sections.append("\n... [truncated] ...\n")
        sections.append(code[-3000:])
    else:
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
        "Analyze the simulation and synthesis results above. "
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
    LangGraph node function: Generate optimization suggestions.

    Reads: state["generated_code"], state["simulation_result"],
           state["synthesis_result"], state["ir_graph"]
    Writes: state["optimization_suggestions"], state["optimization_iteration"],
            state["verification_attempts"] (reset for re-verification)
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
        max_tokens=4096,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info(f"Calling LLM for optimization suggestions...")
    response = llm.invoke(messages)
    raw_response = response.content

    # ── Parse suggestions ───────────────────────────────────────
    suggestions = _parse_suggestions(raw_response)

    logger.info(f"Generated {len(suggestions)} optimization suggestions:")
    for i, s in enumerate(suggestions, 1):
        logger.info(f"  {i}. {s}")

    return {
        "optimization_suggestions": suggestions,
        "optimization_iteration": iteration,
        # Reset verification counter for re-verification after optimization
        "verification_attempts": 0,
        "verification_feedback": "",
    }
