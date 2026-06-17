"""
Report Agent — Generates the final summary report.

Aggregates all pipeline results into a formatted Markdown report
including model info, code stats, simulation, synthesis, and
optimization history.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from ir import IRGraph
from state import AgentState

logger = logging.getLogger(__name__)


def generate_report(state: AgentState) -> dict:
    """
    LangGraph node function: Generate final report.

    Reads: all state fields
    Writes: state["final_report"]
    """
    logger.info("=" * 60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 60)

    model_name = state.get("model_name", "Unknown Model")
    ir_dict = state.get("ir_graph", {})
    ir_graph = IRGraph.from_dict(ir_dict) if ir_dict else None
    sim = state.get("simulation_result", {})
    synth = state.get("synthesis_result", {})
    verification = state.get("verification_result", {})
    opt_suggestions = state.get("optimization_suggestions", [])
    opt_iteration = state.get("optimization_iteration", 0)

    lines = []

    # ── Header ──────────────────────────────────────────────────
    lines.append("# 🚀 Agentic RISC-V Compiler — Final Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Target ISA:** rv32imac (RISC-V)")
    lines.append(f"**Processor:** Hazard3")
    lines.append("")

    # ── Model Summary ───────────────────────────────────────────
    lines.append("## 📊 Model Summary")
    lines.append("")
    if ir_graph:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(
            f"| Total Parameters | {state.get('total_params', 0):,} |"
        )
        lines.append(
            f"| Weight Memory | "
            f"{state.get('model_memory_bytes', 0) / (1024*1024):.2f} MB |"
        )
        lines.append(
            f"| Activation Memory | "
            f"{ir_graph.total_activation_memory() / 1024:.1f} KB |"
        )
        lines.append(f"| IR Nodes | {len(ir_graph.nodes)} |")
        lines.append("")

        # Op breakdown
        op_counts: dict[str, int] = {}
        for node in ir_graph.nodes:
            op_counts[node.op] = op_counts.get(node.op, 0) + 1

        lines.append("### Layer Breakdown")
        lines.append("")
        lines.append("| Operation | Count |")
        lines.append("|-----------|-------|")
        for op, count in sorted(op_counts.items()):
            lines.append(f"| {op} | {count} |")
        lines.append("")

    # ── Verification ────────────────────────────────────────────
    lines.append("## ✅ Verification")
    lines.append("")
    v_attempts = state.get("verification_attempts", 0)
    v_passed = verification.get("passed", False)
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Status | {'PASSED ✅' if v_passed else 'FAILED ❌'} |")
    lines.append(f"| Attempts | {v_attempts} |")
    lines.append(f"| Errors | {len(verification.get('errors', []))} |")
    lines.append(f"| Warnings | {len(verification.get('warnings', []))} |")
    lines.append("")

    if verification.get("warnings"):
        lines.append("### Warnings")
        for w in verification["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    # ── Generated Code Stats ────────────────────────────────────
    lines.append("## 💻 Generated Code")
    lines.append("")
    code = state.get("generated_code", "")
    header = state.get("generated_header", "")
    lines.append(f"| File | Lines | Size |")
    lines.append(f"|------|-------|------|")
    lines.append(
        f"| model.c | {len(code.splitlines())} | "
        f"{len(code.encode('utf-8')) / 1024:.1f} KB |"
    )
    lines.append(
        f"| weights.h | {len(header.splitlines())} | "
        f"{len(header.encode('utf-8')) / 1024:.1f} KB |"
    )
    lines.append("")

    # ── Simulation Results ──────────────────────────────────────
    lines.append("## ⚡ Simulation Results (Hazard3)")
    lines.append("")
    if sim and sim.get("success"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Clock Cycles | {sim.get('cycles', 'N/A'):,} |")
        lines.append(
            f"| Output Correctness | "
            f"{'Match ✅' if sim.get('output_match') else 'Mismatch ⚠️'} |"
        )
        lines.append("")
    else:
        error = sim.get("raw_log", "Simulation not run or failed")
        lines.append(f"⚠️ Simulation did not complete: {error}")
        lines.append("")

    # ── Synthesis Results ───────────────────────────────────────
    lines.append("## 🔧 Synthesis Results (OpenROAD)")
    lines.append("")
    if synth and synth.get("success"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Power | {synth.get('power_watts', 0):.4f} W |")
        lines.append(f"| Area | {synth.get('area_mm2', 0):.4f} mm² |")
        lines.append(f"| Max Frequency | {synth.get('frequency_mhz', 0):.1f} MHz |")
        lines.append(f"| Cell Count | {synth.get('cell_count', 0):,} |")
        lines.append("")

        # Derived metrics
        cycles = sim.get("cycles", 0)
        freq = synth.get("frequency_mhz", 0)
        if cycles > 0 and freq > 0:
            exec_time_ms = (cycles / (freq * 1e6)) * 1000
            lines.append("### Derived Metrics")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Estimated Execution Time | {exec_time_ms:.2f} ms |")
            lines.append(
                f"| Energy per Inference | "
                f"{synth['power_watts'] * exec_time_ms / 1000:.6f} J |"
            )
            lines.append(
                f"| Throughput | "
                f"{1000 / exec_time_ms:.1f} inferences/sec |"
            )
            lines.append("")
    else:
        lines.append("⚠️ Synthesis did not complete or was not run.")
        lines.append("")

    # ── Optimization History ────────────────────────────────────
    if opt_iteration > 0:
        lines.append("## 🔄 Optimization History")
        lines.append("")
        lines.append(f"**Iterations completed:** {opt_iteration}")
        lines.append("")
        if opt_suggestions:
            lines.append("### Applied Optimizations")
            for i, s in enumerate(opt_suggestions, 1):
                lines.append(f"{i}. {s}")
            lines.append("")

    # ── Telemetry ────────────────────────────────────────────────
    lines.append("## 📈 Telemetry")
    lines.append("")
    total_in  = state.get("total_input_tokens",  0) or 0
    total_out = state.get("total_output_tokens", 0) or 0
    total_llm = state.get("total_llm_latency_s", 0.0) or 0.0
    lines.append("### Token Usage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Input Tokens  | {total_in:,} |")
    lines.append(f"| Total Output Tokens | {total_out:,} |")
    lines.append(f"| Total Tokens        | {total_in + total_out:,} |")
    lines.append(f"| Total LLM Latency   | {total_llm:.2f}s |")
    lines.append("")

    agent_latencies: dict = state.get("agent_latencies") or {}
    if agent_latencies:
        lines.append("### Per-Agent Wall-Clock Latency")
        lines.append("")
        lines.append("| Agent | Latency (s) |")
        lines.append("|-------|-------------|")
        for agent, lat in sorted(agent_latencies.items()):
            lines.append(f"| {agent} | {lat:.2f} |")
        lines.append("")

    call_stats: list = state.get("llm_call_stats") or []
    if call_stats:
        lines.append("### LLM Call Breakdown")
        lines.append("")
        lines.append("| Agent | Call | Input Tokens | Output Tokens | Total | Latency (s) |")
        lines.append("|-------|------|-------------|--------------|-------|-------------|")
        for s in call_stats:
            if isinstance(s, dict):
                a, lbl = s.get("agent","?"), s.get("call_label","?")
                inp, out = s.get("input_tokens",0), s.get("output_tokens",0)
                tot, lat = s.get("total_tokens",0), s.get("latency_s",0.0)
            else:
                a, lbl = getattr(s,"agent","?"), getattr(s,"call_label","?")
                inp, out = getattr(s,"input_tokens",0), getattr(s,"output_tokens",0)
                tot, lat = getattr(s,"total_tokens",0), getattr(s,"latency_s",0.0)
            lines.append(f"| {a} | {lbl} | {inp:,} | {out:,} | {tot:,} | {lat:.2f} |")
        lines.append("")

    # ── Pipeline Summary ────────────────────────────────────────

    lines.append("## 📋 Pipeline Summary")
    lines.append("")
    lines.append("```")
    lines.append("PyTorch Model")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("FX Graph Trace ────── ✅")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("Custom IR ─────────── ✅")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"C Code Generation ─── ✅ ({v_attempts} attempt(s))")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"Verification ──────── {'✅' if v_passed else '❌'}")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"Human Approval ────── {'✅' if state.get('human_approved') else '⏳'}")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"Hazard3 Simulation ── {'✅' if sim.get('success') else '⏳'}")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"OpenROAD Synthesis ── {'✅' if synth.get('success') else '⏳'}")
    if opt_iteration > 0:
        lines.append("    │")
        lines.append("    ▼")
        lines.append(f"Optimization ──────── ✅ ({opt_iteration} iteration(s))")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("Final Report ──────── ✅")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by Agentic RISC-V Compiler*")

    report = "\n".join(lines)

    # Save report to file
    output_dir = Path(os.getcwd()) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {report_path}")

    return {"final_report": report}
