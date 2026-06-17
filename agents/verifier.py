"""
Verification Agent — Validates the generated C code.

Performs:
1. Syntax check (compile with -fsyntax-only)
2. Full compilation to object file
3. Structural validation (all IR ops mapped to C)
4. Static analysis for common issues

No LLM required — fully deterministic.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from agents.codegen_contract import (
    helper_name_from_signature,
    required_helper_signatures,
)
from ir import IRGraph, IROpType
from state import AgentState
from tools.compile import (
    check_syntax,
    compile_to_object,
    find_compiler,
)

logger = logging.getLogger(__name__)

MAX_VERIFICATION_ATTEMPTS = 5


def _read_text_file(path_value: str, fallback: str) -> str:
    """Read generated artifact text from disk when available."""
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return fallback


def _check_structural_completeness(
    code: str, ir_dict: dict, model_header: str = ""
) -> list[str]:
    """
    Check that the generated C code covers all IR operations.
    Returns a list of error/warning messages.
    """
    issues = []
    ir_graph = IRGraph.from_dict(ir_dict)

    if "model_inference" not in code:
        issues.append(
            "ERROR: Missing 'model_inference' function. "
            "The entry point must be: "
            "void model_inference(const float* input, float* output);"
        )

    if '#include "weights.h"' not in code and '#include "model.h"' not in code:
        issues.append(
            'ERROR: Missing #include "model.h" or #include "weights.h". '
            "Weight arrays must be imported through the generated headers."
        )

    for node in ir_graph.nodes:
        if node.weight_key:
            c_name = node.weight_key.replace(".", "_")
            if c_name not in code:
                issues.append(
                    f"WARNING: Weight '{node.weight_key}' (C name: {c_name}) "
                    f"referenced by node '{node.id}' ({node.op}) "
                    f"not found in generated code."
                )

    op_patterns = {
        IROpType.CONV2D: ["conv2d", "convolution", "kernel"],
        IROpType.LINEAR: ["linear", "gemm", "matmul", "weight"],
        IROpType.RELU: ["relu", "> 0", "max("],
        IROpType.SILU: ["silu", "sigmoid", "expf"],
        IROpType.RMSNORM: ["rmsnorm", "rms", "sqrt"],
        IROpType.SOFTMAX: ["softmax", "expf", "sum"],
        IROpType.EMBEDDING: ["embed", "token"],
        IROpType.ATTENTION: ["attention", "query", "score"],
    }

    ops_in_graph = {node.op for node in ir_graph.nodes}
    code_lower = code.lower()

    for op in ops_in_graph:
        if op in (IROpType.TENSOR_INPUT, IROpType.TENSOR_OUTPUT, IROpType.DROPOUT):
            continue
        patterns = op_patterns.get(op, [])
        if patterns and not any(p in code_lower for p in patterns):
            issues.append(
                f"WARNING: IR operation '{op}' may not be implemented — "
                f"none of the expected patterns {patterns} found in code."
            )

    return issues


def _check_model_header(model_header: str, ir_dict: dict | None = None) -> list[str]:
    """Validate the LLM-generated model.h contract."""
    issues: list[str] = []
    if not model_header.strip():
        issues.append("ERROR: model.h is empty.")
        return issues

    if "#pragma once" not in model_header and "#ifndef" not in model_header:
        issues.append("WARNING: model.h has no include guard.")
    if '#include "weights.h"' not in model_header:
        issues.append('ERROR: model.h must include "weights.h".')
    if "model_inference" not in model_header:
        issues.append(
            "ERROR: model.h must declare model_inference(const float* input, float* output)."
        )
    if re.search(r"\bstatic\s+float\s+\w+\s*\[", model_header):
        issues.append(
            "ERROR: model.h must not define activation storage arrays; define storage in model.c."
        )
    if ir_dict is not None:
        for signature in required_helper_signatures(ir_dict):
            helper_name = helper_name_from_signature(signature)
            if not re.search(rf"\b{re.escape(helper_name)}\s*\(", model_header):
                issues.append(
                    f"ERROR: model.h is missing required helper prototype "
                    f"for IR operation helper '{helper_name}': {signature}"
                )
    return issues


def _check_required_helper_definitions(code: str, ir_dict: dict) -> list[str]:
    """Validate that model.c implements every IR-required helper function."""
    issues: list[str] = []
    for signature in required_helper_signatures(ir_dict):
        helper_name = helper_name_from_signature(signature)
        if not re.search(
            rf"(^|\n)\s*(?:static\s+)?(?:inline\s+)?\w[\w\s\*]*\b"
            rf"{re.escape(helper_name)}\s*\([^;]*\)\s*\{{",
            code,
            re.DOTALL,
        ):
            issues.append(
                f"ERROR: model.c is missing required helper implementation "
                f"for '{helper_name}' declared by the IR contract."
            )
    return issues


def _check_common_errors(code: str) -> list[str]:
    """Check for common C code errors."""
    issues = []

    if re.search(r"\b(malloc|calloc|realloc|free)\b", code):
        issues.append(
            "ERROR: Dynamic memory allocation detected (malloc/calloc/free). "
            "All arrays must be statically allocated for bare-metal RISC-V."
        )

    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.count('"') % 2 != 0 and "//" not in stripped.split('"')[0]:
            issues.append(
                f"WARNING: Possible unterminated string literal on line {i}: "
                f"{stripped[:60]}..."
            )

    cpp_patterns = [
        (r"\bclass\b", "C++ 'class' keyword"),
        (r"\btemplate\b", "C++ 'template' keyword"),
        (r"\bnew\b\s+\w+", "C++ 'new' operator"),
        (r"\bstd::", "C++ std:: namespace"),
        (r"\bcout\b", "C++ cout"),
        (r"\bvector\b", "C++ vector"),
    ]
    for pattern, description in cpp_patterns:
        if re.search(pattern, code):
            issues.append(
                f"ERROR: C++ feature detected: {description}. "
                "Code must be pure C99."
            )

    array_decls = re.findall(r"(?:static\s+)?(?:const\s+)?float\s+\w+\[(\d+)\]", code)
    for size_str in array_decls:
        size = int(size_str)
        mem_mb = (size * 4) / (1024 * 1024)
        if mem_mb > 100:
            issues.append(
                f"WARNING: Very large array ({size} floats = {mem_mb:.1f} MB). "
                "This may exceed RISC-V memory. Consider quantization."
            )

    return issues


def _check_header(header: str, weight_metadata: dict) -> list[str]:
    """
    Validate the weights.h header file.

    Since weights.h is now deterministically generated by the pipeline
    (not by the LLM), this check is simpler — it mainly validates
    that the header was generated correctly and contains all expected
    weight tensors with actual values (not zero placeholders).
    """
    issues = []

    if not header.strip():
        issues.append("ERROR: weights.h is empty.")
        return issues

    if "#pragma once" not in header and "#ifndef" not in header:
        issues.append(
            "WARNING: weights.h has no include guard. "
            "Add '#pragma once' or '#ifndef WEIGHTS_H'."
        )

    for name, meta in weight_metadata.items():
        c_name = name.replace(".", "_")
        if c_name not in header:
            issues.append(
                f"WARNING: Weight '{name}' (C name: {c_name}) "
                f"not declared in weights.h."
            )

    if "Auto-generated" in header and "Fallback mode" in header:
        issues.append(
            "WARNING: weights.h is using fallback zero-initialized values. "
            "The weight export may have failed. Check weights.npz exists."
        )

    return issues


def verify_code(state: AgentState) -> dict:
    """
    LangGraph node function: Verify the generated C code.

    Reads: state["generated_code"], state["generated_header"],
    state["ir_graph"], state["weights_metadata"], state["code_path"],
    state["header_path"]
    Writes: state["verification_result"], state["verification_attempts"],
    state["verification_feedback"]
    """
    code_path = state.get("code_path", "")
    header_path = state.get("header_path", "")
    model_header_path = state.get("model_header_path", "")
    code = _read_text_file(code_path, state.get("generated_code", ""))
    header = _read_text_file(header_path, state.get("generated_header", ""))
    model_header = _read_text_file(
        model_header_path, state.get("generated_model_header", "")
    )
    ir_dict = state.get("ir_graph", {})
    weight_metadata = state.get("weights_metadata", {})
    attempt = state.get("verification_attempts", 0) + 1

    logger.info(f"Verification attempt {attempt}/{MAX_VERIFICATION_ATTEMPTS}")

    all_errors: list[str] = []
    all_warnings: list[str] = []
    compiler_output = ""

    structural = _check_structural_completeness(code, ir_dict, model_header)
    for issue in structural:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    common = _check_common_errors(code)
    for issue in common:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    helper_definition_issues = _check_required_helper_definitions(code, ir_dict)
    for issue in helper_definition_issues:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    model_header_issues = _check_model_header(model_header, ir_dict)
    for issue in model_header_issues:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    header_issues = _check_header(header, weight_metadata)
    for issue in header_issues:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    if code_path and os.path.exists(code_path):
        compiler = find_compiler()
        if compiler:
            syntax_ok, syntax_output = check_syntax(
                code_path, include_dir=os.path.dirname(header_path)
            )
            if not syntax_ok:
                all_errors.append(f"COMPILATION ERROR (syntax check):\n{syntax_output}")
                compiler_output = syntax_output
            else:
                compile_ok, compile_output = compile_to_object(
                    code_path, include_dir=os.path.dirname(header_path)
                )
                if not compile_ok:
                    all_errors.append(f"COMPILATION ERROR:\n{compile_output}")
                    compiler_output = compile_output
                else:
                    compiler_output = "Compilation successful."
                    logger.info("✓ Compilation successful")
        else:
            all_warnings.append(
                "WARNING: No C compiler found. Skipping compilation check. "
                "Install riscv32-unknown-elf-gcc or gcc."
            )
    else:
        all_warnings.append("WARNING: Code file not found on disk. Skipping compilation check.")

    passed = len(all_errors) == 0

    if passed:
        logger.info("✅ Verification PASSED")
    else:
        logger.info(f"❌ Verification FAILED with {len(all_errors)} error(s)")

    feedback_lines = []
    if all_errors:
        feedback_lines.append("ERRORS (must fix):")
        for e in all_errors:
            feedback_lines.append(f" • {e}")
    if all_warnings:
        feedback_lines.append("\nWARNINGS (should fix):")
        for w in all_warnings:
            feedback_lines.append(f" • {w}")

    return {
        "verification_result": {
            "passed": passed,
            "errors": all_errors,
            "warnings": all_warnings,
            "compiler_output": compiler_output,
        },
        "verification_attempts": attempt,
        "verification_feedback": "\n".join(feedback_lines) if not passed else "",
    }
