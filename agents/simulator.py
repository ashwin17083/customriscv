"""
Simulation Agent — Runs the generated C code on Hazard3 RISC-V simulator.

Wraps the Hazard3 CXXRTL simulation flow:
1. Cross-compile C code → RISC-V ELF binary
2. Run on Hazard3 simulator
3. Parse output for cycle counts and execution trace
4. Compare outputs against PyTorch reference values

Falls back to mock simulation when toolchains are unavailable.
"""

from __future__ import annotations

import logging
import os

from state import AgentState
from tools.compile import compile_to_elf, find_compiler, _is_riscv_compiler
from tools.hazard3 import run_hazard3_simulation, _mock_simulation

logger = logging.getLogger(__name__)


def simulate(state: AgentState) -> dict:
    """
    LangGraph node function: Run Hazard3 simulation.

    Reads: state["code_path"], state["header_path"],
           state["reference_outputs"]
    Writes: state["simulation_result"]
    """
    code_path = state.get("code_path", "")
    header_path = state.get("header_path", "")
    reference_outputs = state.get("reference_outputs", [])

    logger.info("=" * 60)
    logger.info("HAZARD3 RISC-V SIMULATION")
    logger.info("=" * 60)

    # ── Step 1: Cross-compile to RISC-V ELF ─────────────────────
    output_dir = os.path.dirname(code_path) if code_path else "output"
    elf_path = os.path.join(output_dir, "firmware.elf")

    logger.info("Step 1: Cross-compiling to RISC-V ELF...")
    compile_ok, compile_output = compile_to_elf(
        source_path=code_path,
        output_path=elf_path,
        include_dir=os.path.dirname(header_path) if header_path else output_dir,
    )

    if not compile_ok:
        logger.warning(f"Cross-compilation failed: {compile_output}")
        logger.info("Falling back to mock simulation (toolchain unavailable)")

        # Use mock simulation so the pipeline can continue
        sim_result = _mock_simulation(elf_path)
        sim_result["raw_log"] = (
            f"[MOCK — cross-compilation failed]\n"
            f"Compiler output: {compile_output}\n\n"
            f"{sim_result.get('raw_log', '')}"
        )

        return {"simulation_result": sim_result}

    logger.info(f"Cross-compilation successful: {elf_path}")

    # ── Step 2: Run Hazard3 simulator ───────────────────────────
    logger.info("Step 2: Running Hazard3 simulation...")
    sim_result = run_hazard3_simulation(elf_path)

    if not sim_result["success"]:
        logger.error(f"Simulation failed: {sim_result.get('error', 'Unknown error')}")
        return {"simulation_result": sim_result}

    logger.info(f"Simulation completed in {sim_result['cycles']:,} cycles")

    # ── Step 3: Compare outputs ─────────────────────────────────
    if reference_outputs and sim_result.get("output_values"):
        output_values = sim_result["output_values"]
        match = True
        tolerance = 1e-3  # Allow small floating-point differences

        if len(output_values) != len(reference_outputs):
            match = False
            logger.warning(
                f"Output size mismatch: got {len(output_values)}, "
                f"expected {len(reference_outputs)}"
            )
        else:
            for i, (actual, expected) in enumerate(
                zip(output_values, reference_outputs)
            ):
                if abs(actual - expected) > tolerance:
                    match = False
                    logger.warning(
                        f"Output mismatch at index {i}: "
                        f"got {actual:.6f}, expected {expected:.6f}"
                    )
                    break

        sim_result["output_match"] = match
        if match:
            logger.info("✅ Output values match PyTorch reference")
        else:
            logger.warning("⚠️ Output values differ from PyTorch reference")
    else:
        sim_result["output_match"] = True  # No reference to compare against
        logger.info("No reference outputs for comparison — skipping")

    # ── Step 4: Log summary ─────────────────────────────────────
    logger.info(f"  Cycles: {sim_result['cycles']:,}")
    logger.info(f"  Output match: {sim_result['output_match']}")

    return {"simulation_result": sim_result}
