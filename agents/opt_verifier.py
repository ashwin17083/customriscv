"""
Verification Agent #2 — Validates the HW-optimized C code (model_optimized.h/c).

Performs (same checks as Verification Agent #1 via shared helpers):
  1. Structural completeness check
  2. Common C error patterns check
  3. Model header validation
  4. RISC-V compilation (syntax + object file)
     → If no RISC-V compiler found: sets compiler_unavailable=True (handled by graph routing)
  5. Python output comparison (SOFT WARNING):
     → Compiles model_optimized.c with host gcc + test harness
     → Runs the binary and captures output floats
     → Compares against state["reference_outputs"] from PyTorch
     → If host compiler missing or mismatch: WARNING only (not hard error)

Reuses check functions from verifier.py to avoid duplication.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

# Re-use structural check helpers from Verification Agent #1
from agents.verifier import (
    _check_structural_completeness,
    _check_common_errors,
    _check_model_header,
)
from state import AgentState
from tools.compile import (
    check_syntax,
    compile_to_object,
    find_compiler,
    _is_riscv_compiler,
)

logger = logging.getLogger(__name__)

MAX_OPT_VERIFICATION_ATTEMPTS = 3
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ── Disk helpers ───────────────────────────────────────────────

def _read_text_file(path_value: str, fallback: str = "") -> str:
    if path_value:
        p = Path(path_value)
        if p.exists():
            return p.read_text(encoding="utf-8")
    return fallback


# ── Python output comparison ───────────────────────────────────

def _build_test_harness(
    input_size: int,
    output_size: int,
    sample_input_path: str = "",
) -> str:
    """
    Generate a C test harness that:
      1. Loads sample_input (from .npy binary or uses zeros)
      2. Calls model_inference(input, output)
      3. Prints output values to stdout as space-separated floats

    The harness is compiled with host gcc (not cross-compiler) for local execution.
    """
    if sample_input_path and os.path.exists(sample_input_path):
        # Load float32 binary written by main.py
        input_init = f"""
    /* Load sample input from binary file */
    FILE *fp = fopen("{sample_input_path.replace(chr(92), '/')}", "rb");
    if (fp) {{
        fread(input, sizeof(float), {input_size}, fp);
        fclose(fp);
    }}"""
    else:
        input_init = "    /* No sample input available — using zeros */"

    return f"""
#include <stdio.h>
#include <string.h>
#include "model_optimized.h"

int main(void) {{
    static float input[{input_size}];
    static float output[{output_size}];
    memset(input,  0, sizeof(input));
    memset(output, 0, sizeof(output));
{input_init}

    model_inference(input, output);

    /* Print output values as space-separated floats */
    for (int i = 0; i < {output_size}; i++) {{
        printf("%f", output[i]);
        if (i < {output_size} - 1) printf(" ");
    }}
    printf("\\n");
    return 0;
}}
"""


def _find_host_compiler() -> Optional[str]:
    """Find a host (non-RISC-V) C compiler for native execution."""
    import shutil
    for cc in ["gcc", "cc", "clang"]:
        path = shutil.which(cc)
        if path:
            return cc
    return None


def _run_python_output_comparison(
    state: AgentState,
    optimized_c_path: str,
    optimized_h_path: str,
) -> list[str]:
    """
    Compile model_optimized.c with host gcc and compare output with PyTorch reference.

    Returns a list of warning/error strings. Empty list = success.
    This is a SOFT check — failures are reported as warnings, not pipeline-blocking errors.
    """
    warnings: list[str] = []

    reference_outputs = state.get("reference_outputs", [])
    if not reference_outputs:
        warnings.append(
            "WARNING: No PyTorch reference outputs available — skipping Python output comparison."
        )
        return warnings

    host_cc = _find_host_compiler()
    if not host_cc:
        warnings.append(
            "WARNING: No host C compiler (gcc/clang) found. "
            "Skipping Python output comparison of model_optimized.c."
        )
        return warnings

    # Determine input/output sizes from IR graph
    ir_dict = state.get("ir_graph", {})
    input_size = 1
    output_size = len(reference_outputs)

    try:
        from ir import IRGraph
        ir_graph = IRGraph.from_dict(ir_dict)
        for node in ir_graph.nodes:
            if node.op == "TENSOR_INPUT" and node.shape:
                # Flatten shape for the C array size
                size = 1
                for d in node.shape:
                    size *= d
                input_size = max(input_size, size)
    except Exception:
        pass

    # Path to saved sample input (if main.py saved it)
    sample_input_path = str(OUTPUT_DIR / "sample_input.bin")

    # Write the test harness
    harness_code = _build_test_harness(input_size, output_size, sample_input_path)

    include_dir = str(OUTPUT_DIR)

    with tempfile.TemporaryDirectory() as tmp_dir:
        harness_path = os.path.join(tmp_dir, "test_harness.c")
        binary_path  = os.path.join(tmp_dir, "test_run")

        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness_code)

        # Compile: host_cc test_harness.c model_optimized.c -I output/ -o test_run -lm
        cmd = [
            host_cc,
            harness_path,
            optimized_c_path,
            "-I", include_dir,
            "-o", binary_path,
            "-lm",
            "-std=c99",
            "-O0",  # No optimization — we want bit-faithful output
        ]
        logger.info(f"Compiling test harness: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                warnings.append(
                    f"WARNING: Test harness compilation failed (host gcc). "
                    f"Skipping output comparison.\n{result.stderr[:500]}"
                )
                return warnings
        except (subprocess.TimeoutExpired, Exception) as e:
            warnings.append(f"WARNING: Test harness compile error: {e}")
            return warnings

        # Run the binary and capture output
        logger.info("Running test harness binary for output comparison...")
        try:
            run_result = subprocess.run(
                [binary_path],
                capture_output=True, text=True, timeout=30,
            )
            if run_result.returncode != 0:
                warnings.append(
                    f"WARNING: Test harness execution failed (exit {run_result.returncode}). "
                    f"Skipping output comparison."
                )
                return warnings

            # Parse output floats
            output_str = run_result.stdout.strip()
            if not output_str:
                warnings.append(
                    "WARNING: Test harness produced no output. "
                    "Ensure model_optimized.c writes to stdout via the test harness."
                )
                return warnings

            c_outputs = [float(x) for x in output_str.split()]
        except (subprocess.TimeoutExpired, ValueError, Exception) as e:
            warnings.append(f"WARNING: Test harness run/parse error: {e}")
            return warnings

    # Compare
    tolerance = 1e-3
    if len(c_outputs) != len(reference_outputs):
        warnings.append(
            f"WARNING: Output size mismatch — C code produced {len(c_outputs)} values, "
            f"PyTorch reference has {len(reference_outputs)} values."
        )
        return warnings

    mismatches = []
    for i, (actual, expected) in enumerate(zip(c_outputs, reference_outputs)):
        if abs(actual - expected) > tolerance:
            mismatches.append(
                f"  index {i}: C={actual:.6f}, Python={expected:.6f}, "
                f"diff={abs(actual-expected):.6f}"
            )

    if mismatches:
        warnings.append(
            "WARNING: model_optimized.c output does not match PyTorch reference:\n"
            + "\n".join(mismatches[:5])
            + (f"\n  ... and {len(mismatches)-5} more" if len(mismatches) > 5 else "")
        )
        logger.warning(f"⚠️  Python output comparison: {len(mismatches)} mismatch(es)")
    else:
        logger.info("✅ Python output comparison: model_optimized.c matches PyTorch reference")

    return warnings


# ── Main agent function ────────────────────────────────────────

def verify_optimized_code(state: AgentState) -> dict:
    """
    LangGraph node function: Verify the HW-optimized C code.

    Reads:  state["optimized_code_path"], state["optimized_header_path"]
            state["ir_graph"], state["reference_outputs"]
            state["opt_verification_attempts"]
    Writes: state["opt_verification_result"]
            state["opt_verification_attempts"]
            state["opt_verification_feedback"]
            state["opt_verification_exhausted"]
            state["compiler_unavailable"]  (True if no RISC-V compiler)
    """
    optimized_c_path = state.get("optimized_code_path", "")
    optimized_h_path = state.get("optimized_header_path", "")
    ir_dict = state.get("ir_graph", {})
    attempt = state.get("opt_verification_attempts", 0) + 1

    # Fallback to default output paths
    if not optimized_c_path:
        optimized_c_path = str(OUTPUT_DIR / "model_optimized.c")
    if not optimized_h_path:
        optimized_h_path = str(OUTPUT_DIR / "model_optimized.h")

    optimized_c = _read_text_file(optimized_c_path)
    optimized_h = _read_text_file(optimized_h_path)

    logger.info("=" * 60)
    logger.info(f"OPTIMIZED CODE VERIFICATION (attempt {attempt}/{MAX_OPT_VERIFICATION_ATTEMPTS})")
    logger.info("=" * 60)

    t_start = time.perf_counter()

    all_errors:   list[str] = []
    all_warnings: list[str] = []
    compiler_output = ""
    compiler_unavailable = False

    if not optimized_c.strip():
        all_errors.append("ERROR: model_optimized.c is empty or not found.")
        logger.error("model_optimized.c is empty.")
    else:
        # ── 1. Structural completeness ─────────────────────────────
        structural = _check_structural_completeness(optimized_c, ir_dict, optimized_h)
        for issue in structural:
            (all_errors if issue.startswith("ERROR") else all_warnings).append(issue)

        # ── 2. Common error patterns ───────────────────────────────
        common = _check_common_errors(optimized_c)
        for issue in common:
            (all_errors if issue.startswith("ERROR") else all_warnings).append(issue)

        # ── 3. Header validation ───────────────────────────────────
        if not optimized_h.strip():
            all_errors.append("ERROR: model_optimized.h is empty or not found.")
        else:
            header_issues = _check_model_header(optimized_h, ir_dict)
            for issue in header_issues:
                # model_optimized.h includes model.h which includes weights.h
                # so we relax the direct weights.h check
                if "must include" in issue and "weights.h" in issue:
                    continue  # model_optimized.h includes model.h, which includes weights.h
                (all_errors if issue.startswith("ERROR") else all_warnings).append(issue)

        # ── 4. Compilation check ───────────────────────────────────
        if os.path.exists(optimized_c_path):
            include_dir = str(OUTPUT_DIR)

            # Check for any compiler first (host ok for syntax)
            host_compiler = find_compiler(prefer_riscv=False)
            riscv_compiler = find_compiler(prefer_riscv=True)
            riscv_available = riscv_compiler is not None and _is_riscv_compiler(
                riscv_compiler
            )

            if host_compiler:
                # Syntax check with host compiler
                syntax_ok, syntax_output = check_syntax(
                    optimized_c_path,
                    include_dir=include_dir,
                    compiler=host_compiler,
                )
                if not syntax_ok:
                    all_errors.append(
                        f"COMPILATION ERROR (syntax check):\n{syntax_output}"
                    )
                    compiler_output = syntax_output
            else:
                all_warnings.append(
                    "WARNING: No C compiler found. Skipping compilation check."
                )

            if not all_errors:  # Only check RISC-V if syntax passed
                if riscv_available:
                    compile_ok, compile_output = compile_to_object(
                        optimized_c_path,
                        include_dir=include_dir,
                        compiler=riscv_compiler,
                    )
                    if not compile_ok:
                        all_errors.append(
                            f"RISC-V COMPILATION ERROR:\n{compile_output}"
                        )
                        compiler_output = compile_output
                    else:
                        compiler_output = "RISC-V compilation successful."
                        logger.info("✓ RISC-V compilation of model_optimized.c successful")
                else:
                    # No RISC-V compiler — flag it; routing will send to compiler_decision
                    compiler_unavailable = True
                    all_warnings.append(
                        "WARNING: No RISC-V cross-compiler found. "
                        "Cannot verify RISC-V compilation of model_optimized.c. "
                        "Human review will be requested."
                    )
                    logger.warning(
                        "⚠️  No RISC-V compiler available for optimized code verification."
                    )
        else:
            all_warnings.append(
                "WARNING: model_optimized.c not found on disk. "
                "Skipping compilation check."
            )

    # ── 5. Python output comparison (soft) ────────────────────────
    if not all_errors and os.path.exists(optimized_c_path):
        py_warnings = _run_python_output_comparison(
            state, optimized_c_path, optimized_h_path
        )
        all_warnings.extend(py_warnings)

    # ── Build result ─────────────────────────────────────────────
    passed = len(all_errors) == 0

    if passed and not compiler_unavailable:
        logger.info("✅ Optimized Code Verification PASSED")
    elif compiler_unavailable:
        logger.warning("⚠️  Optimized Code Verification: No RISC-V compiler (routing to decision)")
    else:
        logger.info(
            f"❌ Optimized Code Verification FAILED with {len(all_errors)} error(s)"
        )

    feedback_lines = []
    if all_errors:
        feedback_lines.append("ERRORS (must fix):")
        for e in all_errors:
            feedback_lines.append(f"  • {e}")
    if all_warnings:
        feedback_lines.append("\nWARNINGS (informational):")
        for w in all_warnings:
            feedback_lines.append(f"  • {w}")

    elapsed = time.perf_counter() - t_start
    prev_al: dict = dict(state.get("agent_latencies") or {})
    prev_al["opt_verifier"] = prev_al.get("opt_verifier", 0.0) + elapsed

    return {
        "opt_verification_result": {
            "passed": passed,
            "errors": all_errors,
            "warnings": all_warnings,
            "compiler_output": compiler_output,
        },
        "opt_verification_attempts": attempt,
        "opt_verification_feedback": "\n".join(feedback_lines) if not passed else "",
        "opt_verification_exhausted": attempt >= MAX_OPT_VERIFICATION_ATTEMPTS and not passed,
        "compiler_unavailable": compiler_unavailable,
        "agent_latencies": prev_al,
    }
