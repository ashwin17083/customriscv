"""Tests for RISC-V energy estimation helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.estimate_energy import (
    count_static_instructions,
    estimate_dynamic_instructions,
    parse_constants,
    parse_loops,
)


SAMPLE_DISASM = """
output/model.elf:     file format elf32-little

Disassembly of section .text:

80000000 <_start>:
80000000:   00000013    addi    zero,zero,0
80000004:   00100093    li      ra,1
80000008:   00200113    li      sp,2
"""

NESTED_LOOP_SOURCE = """
#define OUT 4
#define IN 8

void model_inference(const float* input, float* output) {
    float acc = 0.0f;
    for (int i = 0; i < OUT; i++) {
        for (int j = 0; j < IN; j++) {
            acc += input[j];
        }
    }
    output[0] = acc;
}
"""


def test_count_static_instructions():
    assert count_static_instructions(SAMPLE_DISASM) == 3


def test_parse_constants():
    source = "#define N 128\nconst int M = 64;\nstatic const int K = 32;"
    constants = parse_constants(source)
    assert constants == {"N": 128, "M": 64, "K": 32}


def test_parse_simple_loop():
    source = """
#define N 10
void foo(void) {
    for (int i = 0; i < N; i++) {
        x = i;
    }
}
"""
    loops = parse_loops(source)
    assert len(loops) == 1
    assert loops[0].trip_count == 10


def test_parse_nested_loops():
    loops = parse_loops(NESTED_LOOP_SOURCE)
    assert len(loops) == 2
    outer = next(loop for loop in loops if loop.variable == "i")
    inner = next(loop for loop in loops if loop.variable == "j")
    assert outer.trip_count == 4
    assert inner.trip_count == 8
    assert inner.depth == 1
    assert inner.nested and inner.nested[0].variable == "i"


def test_estimate_dynamic_instructions_nested():
    loops = parse_loops(NESTED_LOOP_SOURCE)
    static = 100
    dynamic = estimate_dynamic_instructions(static, loops)
    inner = next(loop for loop in loops if loop.variable == "j")
    effective = 4 * 8
    expected = static + (effective - 1) * inner.body_instruction_estimate
    expected += (4 - 1) * 8  # outer loop overhead
    assert dynamic == expected


def test_parse_downward_loop():
    source = """
#define N 5
void foo(void) {
    for (int i = N - 1; i >= 0; i--) {
        x = i;
    }
}
"""
    loops = parse_loops(source)
    assert len(loops) == 1
    assert loops[0].trip_count == 5



def test_parse_expression_constants_with_precedence():
    source = "#define N (2 + 3 * 4)\n#define M (N / 2)\n"
    constants = parse_constants(source)
    assert constants["N"] == 14
    assert constants["M"] == 7


def test_estimate_dynamic_instructions_counts_standalone_loops():
    source = """
void foo(void) {
    for (int i = 0; i < 10; i++) {
        x += i;
    }
    for (int j = 0; j < 5; j++) {
        y += j;
    }
}
"""
    loops = parse_loops(source)
    static = 100
    dynamic = estimate_dynamic_instructions(static, loops)
    expected = static
    for loop in loops:
        expected += (loop.trip_count - 1) * loop.body_instruction_estimate
    assert dynamic == expected


def test_parse_loop_uses_external_header_constants():
    source = """
void foo(void) {
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        x += i;
    }
}
"""
    constants_source = "#define MODEL_SEQ_LEN 32\n"
    loops = parse_loops(source, constants_source=constants_source)
    assert len(loops) == 1
    assert loops[0].trip_count == 32


def test_unknown_loop_is_reported_but_not_scaled():
    source = """
void foo(int n) {
    for (int i = 0; i < n; i++) {
        x += i;
    }
}
"""
    loops = parse_loops(source)
    assert len(loops) == 1
    assert loops[0].trip_count is None
    assert estimate_dynamic_instructions(100, loops) == 100
