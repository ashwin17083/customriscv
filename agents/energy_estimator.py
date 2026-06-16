"""
Energy Estimation Agent — Runs after verification/human approval.

1. Cross-compile model.c to RISC-V ELF (model.elf)
2. Disassemble and count static instructions
3. Parse model.c loops for dynamic instruction estimate
4. Compute runtime and energy from environment assumptions
"""

from __future__ import annotations

import logging
import os

from state import AgentState
from tools.compile import compile_model_elf
from tools.estimate_energy import estimate_energy

logger = logging.getLogger(__name__)


def estimate_energy_node(state: AgentState) -> dict:
    """
    LangGraph node function: compile ELF and estimate inference energy.

    Reads: state["code_path"]
    Writes: state["energy_estimation_result"], state["elf_path"]
    """
    code_path = state.get("code_path", "")
    output_dir = os.path.dirname(code_path) if code_path else "output"
    elf_path = os.path.join(output_dir, "model.elf")
    report_path = os.path.join(output_dir, "energy_report.md")
    disasm_path = os.path.join(output_dir, "model.disasm")

    logger.info("=" * 60)
    logger.info("RISC-V ENERGY ESTIMATION")
    logger.info("=" * 60)

    if not code_path or not os.path.isfile(code_path):
        logger.error("No generated model.c available for energy estimation")
        return {
            "energy_estimation_result": {
                "success": False,
                "error": "model.c not found",
            }
        }

    logger.info("Step 1: Cross-compiling to RISC-V ELF...")
    compile_ok, compile_output, elf_path = compile_model_elf(
        model_c_path=code_path,
        output_dir=output_dir,
    )

    if not compile_ok:
        logger.error(f"ELF compilation failed: {compile_output}")
        return {
            "energy_estimation_result": {
                "success": False,
                "error": f"ELF compilation failed:\n{compile_output}",
                "elf_path": elf_path,
            }
        }

    logger.info(f"ELF compiled: {elf_path}")
    logger.info("Step 2: Analyzing instructions and estimating energy...")

    result = estimate_energy(
        elf_path=elf_path,
        source_path=code_path,
        report_path=report_path,
        disasm_path=disasm_path,
    )

    if result.success:
        logger.info(
            f"Static instructions: {result.static_instructions:,}, "
            f"dynamic estimate: {result.estimated_dynamic_instructions:,}"
        )
        logger.info(
            f"Runtime: {result.runtime_seconds:.6e} s, "
            f"Energy: {result.energy_joules:.6e} J"
        )
        logger.info(f"Report saved: {report_path}")
    else:
        logger.error(f"Energy estimation failed: {result.error}")

    return {
        "elf_path": elf_path,
        "energy_estimation_result": {
            "success": result.success,
            "error": result.error,
            "elf_path": elf_path,
            "source_path": code_path,
            "report_path": report_path,
            "disasm_path": disasm_path,
            "static_instructions": result.static_instructions,
            "estimated_dynamic_instructions": result.estimated_dynamic_instructions,
            "assumed_cpi": result.assumed_cpi,
            "estimated_cycles": result.estimated_cycles,
            "frequency_hz": result.frequency_hz,
            "runtime_seconds": result.runtime_seconds,
            "openroad_power_watts": result.openroad_power_watts,
            "energy_joules": result.energy_joules,
            "loop_count": len(result.loops),
        },
    }
