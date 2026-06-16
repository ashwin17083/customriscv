"""
RISC-V ELF instruction count and energy estimation.

Builds on objdump disassembly for static instruction counts and parses
model.c for constant-bounded loops to estimate dynamic instruction count.
Combines with FREQUENCY_HZ, ASSUMED_CPI, and OPENROAD_POWER_WATTS env vars.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Defaults (override via environment)
DEFAULT_FREQUENCY_HZ = 100_000_000
DEFAULT_ASSUMED_CPI = 1.5
DEFAULT_OPENROAD_POWER_WATTS = 0.120

OBJDUMP_CANDIDATES = [
    "riscv32-unknown-elf-objdump",
    "riscv64-unknown-elf-objdump",
    "riscv-none-elf-objdump",
]

DISASM_LINE_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{2,}(?:\s+[0-9a-fA-F]{2,})*)\s+(\S+)"
)
DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", re.MULTILINE)
CONST_RE = re.compile(
    r"\b(?:static\s+)?(?:const\s+)?(?:unsigned\s+)?(?:int|long|size_t)\s+"
    r"([A-Za-z_]\w*)\s*=\s*([^;]+);"
)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
FOR_LOOP_RE = re.compile(r"\bfor\s*\(([^;]+);([^;]+);([^)]*)\)\s*\{")

AVG_INST_PER_STATEMENT = 3
LOOP_OVERHEAD_INSTRUCTIONS = 8


@dataclass
class LoopInfo:
    """Parsed loop with estimated trip count and body size."""

    variable: str
    trip_count: Optional[int]
    body_statements: int
    body_instruction_estimate: int
    line: int
    depth: int = 0
    nested: list["LoopInfo"] = field(default_factory=list)
    condition: str = ""
    increment: str = ""


@dataclass
class EnergyEstimate:
    """Full energy estimation result."""

    elf_path: str
    source_path: str
    static_instructions: int
    estimated_dynamic_instructions: int
    assumed_cpi: float
    estimated_cycles: float
    frequency_hz: float
    runtime_seconds: float
    openroad_power_watts: float
    energy_joules: float
    loops: list[LoopInfo] = field(default_factory=list)
    disasm_path: str = ""
    report_path: str = ""
    success: bool = True
    error: str = ""


class ConstEvaluator(ast.NodeVisitor):
    """Safely evaluate integer expressions made from constants and operators."""

    def __init__(self, constants: dict[str, int]):
        self.constants = constants

    def visit_Expression(self, node: ast.Expression) -> int:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int:
        if isinstance(node.value, int):
            return node.value
        raise ValueError("non-integer constant")

    def visit_Name(self, node: ast.Name) -> int:
        if node.id in self.constants:
            return self.constants[node.id]
        raise ValueError(f"unknown constant: {node.id}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> int:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if right == 0:
                raise ValueError("division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.LShift):
            return left << right
        if isinstance(node.op, ast.RShift):
            return left >> right
        raise ValueError("unsupported binary operator")

    def generic_visit(self, node: ast.AST) -> int:
        raise ValueError(f"unsupported expression: {type(node).__name__}")


def find_objdump(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve objdump binary from explicit arg, env, or PATH."""
    import shutil

    candidates = [explicit, os.environ.get("RISCV_OBJDUMP", ""), *OBJDUMP_CANDIDATES]
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate)
        if path:
            return candidate
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw}")
    return value


def run_objdump(elf_path: str, objdump: Optional[str] = None) -> str:
    """Disassemble ELF and return text output."""
    tool = find_objdump(objdump)
    if not tool:
        raise RuntimeError(
            "No RISC-V objdump found. Set RISCV_OBJDUMP or install "
            "riscv64-unknown-elf-objdump."
        )
    result = subprocess.run(
        [tool, "-d", elf_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"objdump failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def count_static_instructions(disasm: str) -> int:
    """Count disassembled instruction lines."""
    return sum(1 for line in disasm.splitlines() if DISASM_LINE_RE.match(line))


def _strip_comments_preserve_layout(source: str) -> str:
    """Remove C comments while preserving line/column layout for spans."""
    out: list[str] = []
    i = 0
    in_block = False
    in_string: str | None = None
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if in_block:
            if ch == "*" and nxt == "/":
                out.extend("  ")
                i += 2
                in_block = False
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if in_string:
            out.append(ch)
            if ch == "\\" and nxt:
                out.append(nxt)
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in {'"', "'"}:
            in_string = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "*":
            out.extend("  ")
            i += 2
            in_block = True
            continue
        if ch == "/" and nxt == "/":
            while i < len(source) and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _clean_int_expr(expr: str) -> str:
    cleaned = expr.strip()
    cleaned = cleaned.split("//", 1)[0].strip()
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned)
    cleaned = re.sub(r"\([A-Za-z_][\w\s\*]*\)", "", cleaned)
    cleaned = re.sub(r"\b([0-9]+)[uUlL]+\b", r"\1", cleaned)
    return cleaned.strip()


def _eval_expr(expr: str, constants: dict[str, int]) -> Optional[int]:
    """Evaluate a simple integer expression using known constants."""
    cleaned = _clean_int_expr(expr)
    if not cleaned:
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
        return ConstEvaluator(constants).visit(tree)
    except Exception:
        return None


def parse_constants(source: str) -> dict[str, int]:
    """Extract integer constants from #define and const declarations."""
    constants: dict[str, int] = {}
    clean_source = _strip_comments_preserve_layout(source)
    changed = True
    while changed:
        changed = False
        matches: list[tuple[str, str]] = []
        for match in DEFINE_RE.finditer(clean_source):
            name, expr = match.groups()
            if "(" not in name:
                matches.append((name, expr))
        matches.extend(CONST_RE.findall(clean_source))
        for name, expr in matches:
            if name in constants:
                continue
            value = _eval_expr(expr, constants)
            if value is not None:
                constants[name] = value
                changed = True
    return constants


def _parse_init(init: str, constants: dict[str, int]) -> tuple[str, int] | None:
    init = re.sub(
        r"^\s*(?:unsigned\s+)?(?:int|long|size_t|uint\d+_t|int\d+_t)\s+",
        "",
        init.strip(),
    )
    match = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", init)
    if not match:
        return None
    var, expr = match.groups()
    value = _eval_expr(expr, constants)
    if value is None:
        return None
    return var, value


def _parse_increment(increment: str, var: str, constants: dict[str, int]) -> Optional[int]:
    inc = increment.strip()
    if inc in {f"{var}++", f"++{var}"}:
        return 1
    if inc in {f"{var}--", f"--{var}"}:
        return -1

    match = re.fullmatch(rf"{re.escape(var)}\s*([+-])=\s*(.+)", inc)
    if match:
        sign, expr = match.groups()
        value = _eval_expr(expr, constants)
        if value is None:
            return None
        return value if sign == "+" else -value

    match = re.fullmatch(
        rf"{re.escape(var)}\s*=\s*{re.escape(var)}\s*([+-])\s*(.+)", inc
    )
    if match:
        sign, expr = match.groups()
        value = _eval_expr(expr, constants)
        if value is None:
            return None
        return value if sign == "+" else -value

    return None


def _normalize_condition(condition: str, var: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+?)\s*(<=|>=|<|>)\s*(.+)", condition.strip())
    if not match:
        return None
    left, op, right = [part.strip() for part in match.groups()]
    if left == var:
        return op, right
    if right == var:
        inverted = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]
        return inverted, left
    return None


def _compute_trip_count(start: int, op: str, bound: int, step: int) -> Optional[int]:
    if step == 0:
        return None
    if step > 0:
        if op == "<":
            return max(0, math.ceil((bound - start) / step))
        if op == "<=":
            return max(0, math.ceil((bound - start + 1) / step))
        return None

    step_abs = -step
    if op == ">":
        return max(0, math.ceil((start - bound) / step_abs))
    if op == ">=":
        return max(0, math.ceil((start - bound + 1) / step_abs))
    return None


def _trip_count_from_for_header(
    init: str,
    condition: str,
    increment: str,
    constants: dict[str, int],
) -> tuple[str, Optional[int]]:
    """Compute loop trip count for supported for-loop forms."""
    parsed_init = _parse_init(init, constants)
    if parsed_init is None:
        return "?", None
    var, start = parsed_init
    normalized = _normalize_condition(condition, var)
    step = _parse_increment(increment, var, constants)
    if normalized is None or step is None:
        return var, None
    op, bound_expr = normalized
    bound = _eval_expr(bound_expr, constants)
    if bound is None:
        return var, None
    return var, _compute_trip_count(start, op, bound, step)


def _count_body_statements(body: str) -> int:
    """Count non-empty, non-comment statement-ish lines in a loop body."""
    statements = 0
    clean_body = _strip_comments_preserve_layout(body)
    for line in clean_body.splitlines():
        stripped = line.strip()
        if not stripped or stripped in ("{", "}"):
            continue
        if stripped.startswith("for ") or stripped.startswith("for("):
            continue
        statements += max(1, stripped.count(";"))
    return max(1, statements)


def _extract_loop_bodies(
    source: str,
) -> list[tuple[int, int, int, str, str, str, str]]:
    """
    Find for-loops and their bodies using brace matching.

    Returns list of (line_no, start, end, init, cond, incr, body).
    """
    clean_source = _strip_comments_preserve_layout(source)
    loops: list[tuple[int, int, int, str, str, str, str]] = []
    for match in FOR_LOOP_RE.finditer(clean_source):
        line_no = clean_source[: match.start()].count("\n") + 1
        init, cond, incr = [part.strip() for part in match.group(1, 2, 3)]
        brace_start = match.end() - 1
        depth = 0
        body_start = brace_start + 1
        body_end = body_start
        for idx in range(brace_start, len(clean_source)):
            char = clean_source[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    body_end = idx
                    break
        body = clean_source[body_start:body_end]
        loops.append((line_no, match.start(), body_end, init, cond, incr, body))
    return loops


def _immediate_parent_index(
    index: int,
    raw_loops: list[tuple[int, int, int, str, str, str, str]],
) -> Optional[int]:
    """Return index of the smallest enclosing loop, if any."""
    _, start, end, *_ = raw_loops[index]
    candidates = [
        j
        for j, (_, outer_start, outer_end, *__) in enumerate(raw_loops)
        if j != index and outer_start < start and end < outer_end
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda j: raw_loops[j][2] - raw_loops[j][1])


def _parent_chain(
    index: int,
    raw_loops: list[tuple[int, int, int, str, str, str, str]],
    loops: list[LoopInfo],
) -> list[LoopInfo]:
    """Return enclosing loops from immediate parent outward."""
    chain: list[LoopInfo] = []
    current = index
    while True:
        parent_idx = _immediate_parent_index(current, raw_loops)
        if parent_idx is None:
            break
        chain.append(loops[parent_idx])
        current = parent_idx
    return chain


def _has_child_loop(
    index: int,
    raw_loops: list[tuple[int, int, int, str, str, str, str]],
) -> bool:
    _, start, end, *_ = raw_loops[index]
    return any(
        j != index and start < child_start and child_end < end
        for j, (_, child_start, child_end, *__) in enumerate(raw_loops)
    )


def _effective_trip_count(loop: LoopInfo) -> Optional[int]:
    """Product of this loop's trip count and all enclosing loop trip counts."""
    if loop.trip_count is None:
        return None
    product = loop.trip_count
    for parent in loop.nested:
        if parent.trip_count is None:
            return None
        product *= parent.trip_count
    return product


def parse_loops(source: str, constants_source: str = "") -> list[LoopInfo]:
    """Parse for-loops and estimate trip counts where bounds are static."""
    constants = parse_constants(constants_source + "\n" + source)
    raw_loops = _extract_loop_bodies(source)
    loops: list[LoopInfo] = []

    for line_no, _start, _end, init, cond, incr, body in raw_loops:
        var, trip = _trip_count_from_for_header(init, cond, incr, constants)
        body_stmts = _count_body_statements(body)
        loops.append(
            LoopInfo(
                variable=var,
                trip_count=trip,
                body_statements=body_stmts,
                body_instruction_estimate=body_stmts * AVG_INST_PER_STATEMENT,
                line=line_no,
                condition=cond,
                increment=incr,
            )
        )

    for i, loop in enumerate(loops):
        loop.nested = _parent_chain(i, raw_loops, loops)
        loop.depth = len(loop.nested)

    return loops


def _is_leaf_loop(loop: LoopInfo, loops: list[LoopInfo]) -> bool:
    return not any(loop in other.nested for other in loops)


def estimate_dynamic_instructions(
    static_count: int, loops: list[LoopInfo]
) -> int:
    """
    Estimate dynamic instructions from static count and loop trip counts.

    Static disassembly counts each loop body once. Leaf loops add body
    re-execution cost; enclosing loops add branch/setup overhead per extra trip.
    Unknown-trip loops are left visible in the report but excluded from the
    arithmetic estimate to avoid inventing precision.
    """
    extra = 0
    for loop in loops:
        if loop.trip_count is None:
            continue
        if _is_leaf_loop(loop, loops):
            effective_trips = _effective_trip_count(loop)
            if effective_trips and effective_trips > 1:
                extra += (effective_trips - 1) * loop.body_instruction_estimate
        elif loop.trip_count > 1:
            extra += (loop.trip_count - 1) * LOOP_OVERHEAD_INSTRUCTIONS
    return static_count + extra


def _load_related_headers(source_path: str) -> str:
    """Read small local headers for constants; skip generated weight arrays."""
    root = Path(source_path).resolve().parent
    seen: set[Path] = set()
    chunks: list[str] = []

    def visit(path: Path) -> None:
        if path in seen or not path.exists() or path.name == "weights.h":
            return
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        chunks.append(text)
        for include in INCLUDE_RE.findall(text):
            child = (path.parent / include).resolve()
            if child.parent == root:
                visit(child)

    source = Path(source_path).read_text(encoding="utf-8")
    for include in INCLUDE_RE.findall(source):
        child = (root / include).resolve()
        if child.parent == root:
            visit(child)
    return "\n".join(chunks)


def format_report(estimate: EnergyEstimate) -> str:
    """Build human-readable report text."""
    unknown_loops = sum(1 for loop in estimate.loops if loop.trip_count is None)
    lines = [
        "Energy Estimation Report",
        "",
        f"ELF: {estimate.elf_path}",
        f"Source: {estimate.source_path}",
        "",
        f"Static instructions: {estimate.static_instructions}",
        f"Estimated dynamic instructions: {estimate.estimated_dynamic_instructions}",
        f"Assumed CPI: {estimate.assumed_cpi}",
        f"Estimated cycles: {estimate.estimated_cycles:.0f}",
        f"Frequency Hz: {estimate.frequency_hz:.0f}",
        f"Runtime seconds: {estimate.runtime_seconds:.6e}",
        f"OpenROAD power watts: {estimate.openroad_power_watts}",
        f"Energy joules: {estimate.energy_joules:.6e}",
        "",
        "Notes:",
        "- Runtime is estimated, not measured.",
        "- Power is from OpenROAD.",
        "- Dynamic instruction count is based on static loop analysis.",
        "- This is not Hazard3 VCD-based workload power.",
    ]
    if unknown_loops:
        lines.append(f"- {unknown_loops} loop(s) had non-constant bounds and were not included in the dynamic estimate.")

    if estimate.loops:
        lines.extend(["", "Detected loops:"])
        for loop in estimate.loops:
            trip = "unknown" if loop.trip_count is None else str(loop.trip_count)
            lines.append(
                f"- line {loop.line}: `{loop.variable}` "
                f"trip_count={trip}, depth={loop.depth}, "
                f"body_inst≈{loop.body_instruction_estimate}, "
                f"condition=`{loop.condition}`, increment=`{loop.increment}`"
            )

    if estimate.error:
        lines.extend(["", f"Warning: {estimate.error}"])

    return "\n".join(lines)


def estimate_energy(
    elf_path: str,
    source_path: str,
    report_path: str = "",
    disasm_path: str = "",
    objdump: Optional[str] = None,
    frequency_hz: Optional[float] = None,
    assumed_cpi: Optional[float] = None,
    openroad_power_watts: Optional[float] = None,
) -> EnergyEstimate:
    """
    Run full energy estimation pipeline.
    """
    freq = frequency_hz if frequency_hz is not None else read_env_float(
        "FREQUENCY_HZ", DEFAULT_FREQUENCY_HZ
    )
    cpi = assumed_cpi if assumed_cpi is not None else read_env_float(
        "ASSUMED_CPI", DEFAULT_ASSUMED_CPI
    )
    power = (
        openroad_power_watts
        if openroad_power_watts is not None
        else read_env_float("OPENROAD_POWER_WATTS", DEFAULT_OPENROAD_POWER_WATTS)
    )

    try:
        if freq <= 0 or cpi <= 0 or power <= 0:
            raise ValueError("FREQUENCY_HZ, ASSUMED_CPI, and OPENROAD_POWER_WATTS must be positive")
        disasm = run_objdump(elf_path, objdump=objdump)
        if disasm_path:
            Path(disasm_path).parent.mkdir(parents=True, exist_ok=True)
            Path(disasm_path).write_text(disasm, encoding="utf-8")

        static_count = count_static_instructions(disasm)
        source = Path(source_path).read_text(encoding="utf-8")
        constants_source = _load_related_headers(source_path)
        loops = parse_loops(source, constants_source=constants_source)
        dynamic_count = estimate_dynamic_instructions(static_count, loops)

        cycles = dynamic_count * cpi
        runtime = cycles / freq
        energy = power * runtime

        result = EnergyEstimate(
            elf_path=elf_path,
            source_path=source_path,
            static_instructions=static_count,
            estimated_dynamic_instructions=dynamic_count,
            assumed_cpi=cpi,
            estimated_cycles=cycles,
            frequency_hz=freq,
            runtime_seconds=runtime,
            openroad_power_watts=power,
            energy_joules=energy,
            loops=loops,
            disasm_path=disasm_path,
            report_path=report_path,
            success=True,
        )
    except Exception as exc:
        result = EnergyEstimate(
            elf_path=elf_path,
            source_path=source_path,
            static_instructions=0,
            estimated_dynamic_instructions=0,
            assumed_cpi=cpi,
            estimated_cycles=0,
            frequency_hz=freq,
            runtime_seconds=0,
            openroad_power_watts=power,
            energy_joules=0,
            success=False,
            error=str(exc),
        )

    report_text = format_report(result)
    print(report_text)

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(report_text, encoding="utf-8")
        result.report_path = report_path

    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate RISC-V inference runtime and energy from ELF + source."
    )
    parser.add_argument("--elf", required=True, help="Path to RISC-V ELF binary")
    parser.add_argument("--source", required=True, help="Path to model.c")
    parser.add_argument(
        "--report",
        default="",
        help="Path to write energy report (default: print only)",
    )
    parser.add_argument(
        "--disasm",
        default="",
        help="Optional path to save objdump disassembly",
    )
    args = parser.parse_args(argv)

    result = estimate_energy(
        elf_path=args.elf,
        source_path=args.source,
        report_path=args.report,
        disasm_path=args.disasm,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
