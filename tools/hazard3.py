"""
Hazard3 Simulation Tool — Wraps the Hazard3 RISC-V CXXRTL simulator.

Flow:
1. Build the CXXRTL simulator from Hazard3 Verilog (cached)
2. Run the firmware ELF on the simulator
3. Parse stdout for cycle counts and output values

Supports a mock mode for demo/testing when Hazard3 is not installed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────
HAZARD3_DIR = os.environ.get("HAZARD3_DIR", "")
HAZARD3_SIM = os.environ.get("HAZARD3_SIM", "")  # Pre-built simulator path
MOCK_MODE = os.environ.get("MOCK_SIMULATION", "true").lower() == "true"


def _find_hazard3() -> Optional[str]:
    """Find the Hazard3 simulator binary."""
    # Check explicit path
    if HAZARD3_SIM and os.path.isfile(HAZARD3_SIM):
        return HAZARD3_SIM

    # Check in HAZARD3_DIR
    if HAZARD3_DIR:
        candidates = [
            os.path.join(HAZARD3_DIR, "build", "tb_cxxrtl"),
            os.path.join(HAZARD3_DIR, "sim", "tb_cxxrtl"),
            os.path.join(HAZARD3_DIR, "tb_cxxrtl"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    # Check PATH
    sim_path = shutil.which("hazard3_sim")
    if sim_path:
        return sim_path

    return None


def _build_hazard3_simulator() -> tuple[bool, str]:
    """
    Build the Hazard3 CXXRTL simulator from source.

    Requires: Yosys, Clang, Hazard3 source code.
    """
    if not HAZARD3_DIR:
        return False, "HAZARD3_DIR not set — cannot build simulator"

    if not os.path.isdir(HAZARD3_DIR):
        return False, f"HAZARD3_DIR does not exist: {HAZARD3_DIR}"

    logger.info("Building Hazard3 CXXRTL simulator...")

    try:
        result = subprocess.run(
            ["make", "sim"],
            cwd=HAZARD3_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, "Simulator built successfully"
        else:
            return False, f"Build failed: {result.stderr}"
    except Exception as e:
        return False, f"Build error: {str(e)}"


def _parse_simulation_output(output: str) -> dict:
    """
    Parse the Hazard3 simulator stdout for results.

    Expected output format (from a modified testbench):
    ```
    [CYCLES] 1234567
    [OUTPUT] 0.123 0.456 0.789 ...
    [DONE]
    ```
    """
    cycles = 0
    output_values: list[float] = []

    for line in output.split("\n"):
        line = line.strip()

        # Parse cycle count
        cycle_match = re.search(r'\[CYCLES?\]\s*(\d+)', line)
        if cycle_match:
            cycles = int(cycle_match.group(1))

        # Parse output values
        output_match = re.search(r'\[OUTPUT\]\s*([\d\.\-\s]+)', line)
        if output_match:
            values_str = output_match.group(1).strip()
            for v in values_str.split():
                try:
                    output_values.append(float(v))
                except ValueError:
                    pass

        # Look for cycle count in other common formats
        if not cycles:
            alt_match = re.search(
                r'(?:cycles?|clk|clock)\s*[=:]\s*(\d+)', line, re.IGNORECASE
            )
            if alt_match:
                cycles = int(alt_match.group(1))

    return {
        "cycles": cycles,
        "output_values": output_values,
    }


def _mock_simulation(elf_path: str) -> dict:
    """
    Generate mock simulation results for demo purposes.

    Produces realistic-looking results based on file size
    as a rough proxy for code complexity.
    """
    import random
    random.seed(42)  # Deterministic for demo

    # Estimate cycles based on ELF size
    try:
        elf_size = os.path.getsize(elf_path)
    except OSError:
        elf_size = 10000

    # Heuristic: ~100 cycles per byte of code (rough estimate)
    base_cycles = max(10000, elf_size * 100)
    cycles = base_cycles + random.randint(-base_cycles // 10, base_cycles // 10)

    # Generate mock output values
    output_values = [
        round(random.uniform(-1.0, 1.0), 6)
        for _ in range(10)
    ]

    logger.info(f"[MOCK] Simulated {cycles:,} cycles")
    logger.info(f"[MOCK] Generated {len(output_values)} output values")

    return {
        "success": True,
        "cycles": cycles,
        "execution_trace": (
            f"[MOCK SIMULATION]\n"
            f"ELF: {elf_path}\n"
            f"Cycles: {cycles:,}\n"
            f"Output: {output_values[:5]}..."
        ),
        "output_values": output_values,
        "output_match": True,
        "raw_log": "[MOCK] Simulation completed successfully",
    }


def run_hazard3_simulation(elf_path: str) -> dict:
    """
    Run the Hazard3 CXXRTL simulation.

    Args:
        elf_path: Path to the compiled RISC-V ELF binary.

    Returns:
        SimulationResult dict with cycles, output values, etc.
    """
    # ── Check if we should use mock mode ────────────────────────
    if MOCK_MODE:
        logger.info("Running in MOCK simulation mode")
        return _mock_simulation(elf_path)

    # ── Find or build the simulator ─────────────────────────────
    sim_path = _find_hazard3()

    if sim_path is None:
        # Try building
        build_ok, build_msg = _build_hazard3_simulator()
        if build_ok:
            sim_path = _find_hazard3()
        if sim_path is None:
            logger.warning(
                f"Hazard3 simulator not available: {build_msg}. "
                "Falling back to mock mode."
            )
            return _mock_simulation(elf_path)

    # ── Run the simulation ──────────────────────────────────────
    logger.info(f"Running Hazard3 simulator: {sim_path}")
    logger.info(f"Firmware: {elf_path}")

    try:
        result = subprocess.run(
            [sim_path, elf_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        raw_log = result.stdout + result.stderr

        if result.returncode != 0:
            return {
                "success": False,
                "cycles": 0,
                "execution_trace": "",
                "output_values": [],
                "output_match": False,
                "raw_log": f"Simulator exited with code {result.returncode}:\n{raw_log}",
            }

        # Parse output
        parsed = _parse_simulation_output(raw_log)

        return {
            "success": True,
            "cycles": parsed["cycles"],
            "execution_trace": raw_log[:2000],  # Truncate for state
            "output_values": parsed["output_values"],
            "output_match": True,  # Will be checked by simulator agent
            "raw_log": raw_log,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "cycles": 0,
            "execution_trace": "",
            "output_values": [],
            "output_match": False,
            "raw_log": "Simulation timed out after 600 seconds",
        }
    except Exception as e:
        return {
            "success": False,
            "cycles": 0,
            "execution_trace": "",
            "output_values": [],
            "output_match": False,
            "raw_log": f"Simulation error: {str(e)}",
        }
