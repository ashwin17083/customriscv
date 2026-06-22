"""
Compile Tool — Wraps RISC-V GCC cross-compiler.

Supports:
- Syntax checking (-fsyntax-only)
- Compilation to object file (-c)
- Compilation to ELF binary (full link)
- Fallback to host gcc/clang if cross-compiler not available

Target ISA: rv32imac
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class CompilerUnavailableError(Exception):
    """Raised when no RISC-V compiler is found and the operation cannot proceed."""
    pass


# ── Compiler search order ───────────────────────────────────────
RISCV_COMPILERS = [
    "riscv32-unknown-elf-gcc",
    "riscv64-unknown-elf-gcc",  # Can target rv32 with -march
    "riscv-none-elf-gcc",
    "riscv-none-embed-gcc",
]

HOST_COMPILERS = [
    "gcc",
    "cc",
    "clang",
]

# ── RISC-V compilation flags ───────────────────────────────────
RISCV_CFLAGS = [
    "-march=rv32imac",
    "-mabi=ilp32",
    "-O2",
    "-Wall",
    "-Wextra",
    "-ffreestanding",
    "-nostdlib",
    "-fno-builtin",
    "-std=c99",
]

# Minimal startup code for bare-metal
STARTUP_CODE = '''
/* Minimal bare-metal startup for Hazard3 RISC-V */
extern void model_inference(const float* input, float* output);

/* Minimal soft-float math stubs if needed */
#ifndef __riscv_float_abi_soft
/* Use compiler builtins */
#endif

/* Simple entry point */
void _start(void) {
    /* Placeholder: in real deployment, load inputs from memory-mapped I/O */
    static float input[1] = {0.0f};
    static float output[1] = {0.0f};

    model_inference(input, output);

    /* Halt: write to test-finish register or infinite loop */
    while(1) {
#ifdef __riscv
        __asm__ volatile ("wfi");
#else
        /* Host compilation: plain spin loop (for syntax/link testing) */
        __asm__ volatile ("");
#endif
    }
}
'''


def find_compiler(prefer_riscv: bool = True) -> Optional[str]:
    """
    Find an available C compiler.

    Args:
        prefer_riscv: If True, search for RISC-V cross-compiler first.

    Returns:
        Path to compiler executable, or None if not found.
    """
    search_order = (
        RISCV_COMPILERS + HOST_COMPILERS
        if prefer_riscv
        else HOST_COMPILERS
    )

    for compiler in search_order:
        path = shutil.which(compiler)
        if path:
            logger.info(f"Found compiler: {compiler} at {path}")
            return compiler

    logger.warning("No C compiler found in PATH")
    return None


def _is_riscv_compiler(compiler: str) -> bool:
    """Check if the given compiler is a RISC-V cross-compiler."""
    return "riscv" in compiler.lower()


def _run_compiler(
    compiler: str,
    args: list[str],
    timeout: int = 60,
) -> tuple[bool, str]:
    """
    Run the compiler with given arguments.

    Returns:
        (success, output) tuple.
    """
    cmd = [compiler] + args
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0

        if not success:
            logger.debug(f"Compiler returned {result.returncode}: {output}")

        return success, output.strip()

    except FileNotFoundError:
        return False, f"Compiler not found: {compiler}"
    except subprocess.TimeoutExpired:
        return False, f"Compilation timed out after {timeout}s"
    except Exception as e:
        return False, f"Compilation error: {str(e)}"


def check_syntax(
    source_path: str,
    include_dir: str = "",
    compiler: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Check C syntax without generating output.

    Args:
        source_path: Path to the .c source file.
        include_dir: Directory containing header files.
        compiler: Specific compiler to use (auto-detect if None).

    Returns:
        (success, output) tuple.
    """
    if compiler is None:
        compiler = find_compiler(prefer_riscv=False)  # Host compiler OK for syntax
        if compiler is None:
            return False, "No compiler available for syntax checking"

    args = ["-fsyntax-only", "-std=c99"]

    if include_dir:
        args.extend(["-I", include_dir])

    # Add RISC-V flags only for cross-compiler
    if _is_riscv_compiler(compiler):
        args.extend(["-march=rv32imac", "-mabi=ilp32"])

    args.append(source_path)

    return _run_compiler(compiler, args)


def compile_to_object(
    source_path: str,
    output_path: str = "",
    include_dir: str = "",
    compiler: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Compile C source to object file (.o).

    Args:
        source_path: Path to the .c source file.
        output_path: Path for the output .o file.
        include_dir: Directory containing header files.
        compiler: Specific compiler to use.

    Returns:
        (success, output) tuple.
    """
    if compiler is None:
        compiler = find_compiler(prefer_riscv=True)
        if compiler is None:
            return False, "No compiler available"

    if not output_path:
        output_path = source_path.replace(".c", ".o")

    args = ["-c"]

    if _is_riscv_compiler(compiler):
        args.extend(RISCV_CFLAGS)
    else:
        args.extend(["-std=c99", "-Wall", "-Wextra", "-O2"])

    if include_dir:
        args.extend(["-I", include_dir])

    args.extend(["-o", output_path, source_path])

    return _run_compiler(compiler, args)


def compile_to_elf(
    source_path: str,
    output_path: str = "",
    include_dir: str = "",
    linker_script: str = "",
    compiler: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Compile and link C source to RISC-V ELF binary.

    Generates a minimal startup if needed and links everything together.

    Args:
        source_path: Path to the .c source file.
        output_path: Path for the output ELF file.
        include_dir: Directory containing header files.
        linker_script: Optional linker script path.
        compiler: Specific compiler to use.

    Returns:
        (success, output) tuple.
    """
    if compiler is None:
        compiler = find_compiler(prefer_riscv=True)
        if compiler is None:
            return False, "No RISC-V compiler available for ELF generation"

    if not output_path:
        output_path = source_path.replace(".c", ".elf")

    # Write startup code to a temporary file
    startup_path = os.path.join(
        os.path.dirname(source_path), "_startup.c"
    )
    with open(startup_path, "w", encoding="utf-8") as f:
        f.write(STARTUP_CODE)

    args = []

    if _is_riscv_compiler(compiler):
        args.extend(RISCV_CFLAGS)
    else:
        args.extend(["-std=c99", "-O2"])

    if include_dir:
        args.extend(["-I", include_dir])

    if linker_script:
        args.extend(["-T", linker_script])

    # Link math library
    args.extend([
        "-o", output_path,
        source_path,
        startup_path,
        "-lm",
    ])

    success, output = _run_compiler(compiler, args)

    # Clean up startup file
    try:
        os.remove(startup_path)
    except OSError:
        pass

    return success, output
