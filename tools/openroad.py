"""
OpenROAD Synthesis Tool — Wraps the OpenROAD-flow-scripts pipeline.

Flow:
1. Yosys logic synthesis
2. OpenROAD physical design (floorplan → place → CTS → route)
3. Parse timing/power/area reports

Supports mock mode for demo/testing when OpenROAD is not installed.
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
OPENROAD_FLOW_DIR = os.environ.get("OPENROAD_FLOW_DIR", "")
OPENROAD_BIN = os.environ.get("OPENROAD_BIN", "")
YOSYS_BIN = os.environ.get("YOSYS_BIN", "")
PDK = os.environ.get("OPENROAD_PDK", "sky130hd")
MOCK_MODE = os.environ.get("MOCK_SYNTHESIS", "true").lower() == "true"


def _find_openroad() -> Optional[str]:
    """Find the OpenROAD executable."""
    if OPENROAD_BIN and os.path.isfile(OPENROAD_BIN):
        return OPENROAD_BIN
    return shutil.which("openroad")


def _find_yosys() -> Optional[str]:
    """Find the Yosys executable."""
    if YOSYS_BIN and os.path.isfile(YOSYS_BIN):
        return YOSYS_BIN
    return shutil.which("yosys")


def _parse_power_report(report: str) -> float:
    """Extract total power from OpenROAD power report."""
    # Look for pattern: "Total ... <number> W" or "total_power: <number>"
    patterns = [
        r'[Tt]otal\s+(?:\w+\s+)*(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*[Ww]',
        r'total_power\s*[=:]\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Total\s+Power\s*[=:]\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, report)
        if match:
            return float(match.group(1))
    return 0.0


def _parse_area_report(report: str) -> tuple[float, int]:
    """Extract area and cell count from OpenROAD area report."""
    area = 0.0
    cells = 0

    # Area patterns
    area_patterns = [
        r'[Tt]otal\s+[Aa]rea\s*[=:]\s*(\d+\.?\d*)',
        r'[Dd]ie\s+[Aa]rea\s*[=:]\s*(\d+\.?\d*)',
        r'area\s*[=:]\s*(\d+\.?\d*)',
    ]
    for pattern in area_patterns:
        match = re.search(pattern, report)
        if match:
            area = float(match.group(1))
            break

    # Cell count patterns
    cell_patterns = [
        r'[Nn]umber\s+of\s+[Cc]ells?\s*[=:]\s*(\d+)',
        r'[Cc]ell\s+[Cc]ount\s*[=:]\s*(\d+)',
        r'(\d+)\s+cells?',
    ]
    for pattern in cell_patterns:
        match = re.search(pattern, report)
        if match:
            cells = int(match.group(1))
            break

    return area, cells


def _parse_timing_report(report: str) -> float:
    """Extract maximum frequency from OpenROAD timing report."""
    # Look for clock period or frequency
    period_match = re.search(
        r'[Cc]lock\s+[Pp]eriod\s*[=:]\s*(\d+\.?\d*)', report
    )
    if period_match:
        period_ns = float(period_match.group(1))
        if period_ns > 0:
            return 1000.0 / period_ns  # Convert ns to MHz

    freq_match = re.search(
        r'[Ff]requency\s*[=:]\s*(\d+\.?\d*)\s*[Mm][Hh]z', report
    )
    if freq_match:
        return float(freq_match.group(1))

    # Slack-based: f = 1 / (target_period - slack)
    slack_match = re.search(r'[Ss]lack\s*[=:]\s*(-?\d+\.?\d*)', report)
    if slack_match:
        slack = float(slack_match.group(1))
        # Assume 10ns target period (100 MHz)
        effective_period = 10.0 - slack
        if effective_period > 0:
            return 1000.0 / effective_period

    return 0.0


def _mock_synthesis(design_name: str, sim_cycles: int) -> dict:
    """
    Generate mock synthesis results for demo purposes.

    Produces realistic-looking results for a Hazard3 SoC
    on Sky130 process.
    """
    import random
    random.seed(hash(design_name) % 2**32)

    # Hazard3 baseline: ~15k cells, ~0.5mm² on Sky130
    # Application code adds some overhead
    base_cells = 15000
    app_cells = random.randint(2000, 8000)
    total_cells = base_cells + app_cells

    # Area scales with cell count (~33 um² per cell on Sky130)
    area_um2 = total_cells * 33.0
    area_mm2 = area_um2 / 1e6

    # Power: ~50-200 mW typical for Hazard3 at moderate frequency
    power_mw = 50 + random.uniform(30, 150)
    power_w = power_mw / 1000

    # Frequency: 50-150 MHz on Sky130
    freq_mhz = random.uniform(50, 150)

    detailed_report = (
        f"[MOCK SYNTHESIS REPORT]\n"
        f"Design: {design_name}\n"
        f"PDK: {PDK}\n"
        f"Target Clock: 10.0 ns (100 MHz)\n"
        f"\n"
        f"=== Synthesis (Yosys) ===\n"
        f"  Cells: {total_cells:,}\n"
        f"  Hazard3 Core: {base_cells:,} cells\n"
        f"  Application Logic: {app_cells:,} cells\n"
        f"\n"
        f"=== Physical Design (OpenROAD) ===\n"
        f"  Die Area: {area_mm2:.4f} mm²\n"
        f"  Utilization: {random.uniform(40, 75):.1f}%\n"
        f"\n"
        f"=== Timing ===\n"
        f"  Max Frequency: {freq_mhz:.1f} MHz\n"
        f"  WNS (Worst Negative Slack): {random.uniform(-0.5, 0.5):.3f} ns\n"
        f"\n"
        f"=== Power ===\n"
        f"  Total Power: {power_w:.4f} W ({power_mw:.1f} mW)\n"
        f"  Dynamic: {power_mw * 0.7:.1f} mW\n"
        f"  Leakage: {power_mw * 0.3:.1f} mW\n"
    )

    if sim_cycles > 0 and freq_mhz > 0:
        exec_time_ms = (sim_cycles / (freq_mhz * 1e6)) * 1000
        energy_j = power_w * exec_time_ms / 1000
        detailed_report += (
            f"\n=== Derived Metrics ===\n"
            f"  Execution Time: {exec_time_ms:.2f} ms "
            f"({sim_cycles:,} cycles @ {freq_mhz:.0f} MHz)\n"
            f"  Energy/Inference: {energy_j:.6f} J\n"
            f"  Throughput: {1000 / exec_time_ms:.1f} inf/s\n"
        )

    logger.info(f"[MOCK] Synthesis completed: {total_cells} cells, "
                f"{area_mm2:.4f} mm², {power_w:.4f} W, {freq_mhz:.1f} MHz")

    return {
        "success": True,
        "power_watts": round(power_w, 4),
        "area_mm2": round(area_mm2, 4),
        "frequency_mhz": round(freq_mhz, 1),
        "cell_count": total_cells,
        "detailed_report": detailed_report,
    }


def run_openroad_flow(
    design_name: str = "model",
    sim_cycles: int = 0,
    rtl_dir: str = "",
) -> dict:
    """
    Run the OpenROAD synthesis flow.

    Args:
        design_name: Name of the design.
        sim_cycles: Clock cycles from simulation (for derived metrics).
        rtl_dir: Directory containing RTL source files.

    Returns:
        SynthesisResult dict with power, area, frequency, etc.
    """
    # ── Check for mock mode ─────────────────────────────────────
    if MOCK_MODE:
        logger.info("Running in MOCK synthesis mode")
        return _mock_synthesis(design_name, sim_cycles)

    # ── Find tools ──────────────────────────────────────────────
    openroad = _find_openroad()
    yosys = _find_yosys()

    if not openroad or not yosys:
        logger.warning(
            f"OpenROAD or Yosys not found "
            f"(openroad={openroad}, yosys={yosys}). "
            "Falling back to mock mode."
        )
        return _mock_synthesis(design_name, sim_cycles)

    # ── Run ORFS Make flow ──────────────────────────────────────
    if OPENROAD_FLOW_DIR and os.path.isdir(OPENROAD_FLOW_DIR):
        logger.info("Running OpenROAD-flow-scripts...")

        try:
            # Run full flow
            result = subprocess.run(
                [
                    "make",
                    f"DESIGN_CONFIG={design_name}/config.mk",
                ],
                cwd=os.path.join(OPENROAD_FLOW_DIR, "flow"),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "power_watts": 0,
                    "area_mm2": 0,
                    "frequency_mhz": 0,
                    "cell_count": 0,
                    "detailed_report": (
                        f"ORFS failed:\n{result.stderr[:2000]}"
                    ),
                }

            # Parse reports
            flow_output = result.stdout + result.stderr
            power = _parse_power_report(flow_output)
            area, cells = _parse_area_report(flow_output)
            freq = _parse_timing_report(flow_output)

            return {
                "success": True,
                "power_watts": power,
                "area_mm2": area / 1e6,  # Convert from um² to mm²
                "frequency_mhz": freq,
                "cell_count": cells,
                "detailed_report": flow_output[:5000],
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "power_watts": 0,
                "area_mm2": 0,
                "frequency_mhz": 0,
                "cell_count": 0,
                "detailed_report": "Synthesis timed out after 30 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "power_watts": 0,
                "area_mm2": 0,
                "frequency_mhz": 0,
                "cell_count": 0,
                "detailed_report": f"Synthesis error: {str(e)}",
            }
    else:
        logger.warning("OPENROAD_FLOW_DIR not set. Falling back to mock mode.")
        return _mock_synthesis(design_name, sim_cycles)
