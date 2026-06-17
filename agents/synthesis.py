"""
Synthesis Agent — Runs OpenROAD to synthesize the Hazard3 SoC
and measure power, area, and frequency.

Wraps the OpenROAD-flow-scripts (ORFS) pipeline:
1. Yosys logic synthesis
2. OpenROAD physical design (floorplan → place → CTS → route)
3. Parse timing/power/area reports
"""

from __future__ import annotations

import logging
import os
import time

from state import AgentState
from tools.openroad import run_openroad_flow

logger = logging.getLogger(__name__)


def synthesize(state: AgentState) -> dict:
    """
    LangGraph node function: Run OpenROAD synthesis.

    Reads: state["simulation_result"]
    Writes: state["synthesis_result"]
    """
    logger.info("=" * 60)
    logger.info("OPENROAD SYNTHESIS")
    logger.info("=" * 60)

    t_start = time.perf_counter()
    sim_result = state.get("simulation_result", {})

    # ── Run the OpenROAD flow ───────────────────────────────────
    logger.info("Running OpenROAD synthesis flow...")
    logger.info("  Steps: Yosys → Floorplan → Place → CTS → Route → Report")

    synthesis_result = run_openroad_flow(
        design_name=state.get("model_name", "model"),
        sim_cycles=sim_result.get("cycles", 0),
    )

    if synthesis_result["success"]:
        logger.info("✅ Synthesis completed successfully")
        logger.info(f"  Power:     {synthesis_result['power_watts']:.4f} W")
        logger.info(f"  Area:      {synthesis_result['area_mm2']:.4f} mm²")
        logger.info(f"  Frequency: {synthesis_result['frequency_mhz']:.1f} MHz")
        logger.info(f"  Cells:     {synthesis_result['cell_count']:,}")
    else:
        logger.error(
            f"Synthesis failed: "
            f"{synthesis_result.get('detailed_report', 'Unknown error')}"
        )

    synth_elapsed = time.perf_counter() - t_start
    logger.info(f"  Synthesis wall-clock: {synth_elapsed:.2f}s")
    prev_al: dict = dict(state.get("agent_latencies") or {})
    prev_al["synthesizer"] = prev_al.get("synthesizer", 0.0) + synth_elapsed

    return {
        "synthesis_result": synthesis_result,
        "agent_latencies": prev_al,
    }
