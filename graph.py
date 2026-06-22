"""
LangGraph Workflow Definition.

Pipeline flow:

  parse_fx → generate_code → verify
    verify (pass)     → human_review        [INTERRUPT]
    verify (fail)     → generate_code (retry, up to 5×)
    verify (exhausted)→ human_review        [INTERRUPT]

  human_review (approve, optimize=on)  → hw_optimize
  human_review (approve, optimize=off) → simulate
  human_review (reject + feedback)     → generate_code
  human_review (re-verify)             → verify

  hw_optimize → verify_optimized
    verify_optimized (pass)               → human_review_2   [INTERRUPT]
    verify_optimized (fail, ≤3×)          → hw_optimize (retry)
    verify_optimized (exhausted)          → human_review_2   [INTERRUPT]
    verify_optimized (compiler_unavail)   → compiler_decision [INTERRUPT]

  human_review_2 (approve)  → simulate
  human_review_2 (reject)   → hw_optimize (retry with feedback)

  compiler_decision (proceed) → simulate   (mock, no cross-compiler)
  compiler_decision (skip)    → report     (jump directly)

  simulate (output match)    → synthesize
  simulate (output mismatch) → generate_code   ← back-edge

  synthesize → report → END
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from agents.fx_parser import parse_fx_graph
from agents.code_generator import generate_code
from agents.verifier import verify_code
from agents.human_review import human_review
from agents.hw_optimizer import hw_optimize
from agents.opt_verifier import verify_optimized_code
from agents.human_review_2 import human_review_2
from agents.compiler_decision import compiler_decision
from agents.simulator import simulate
from agents.synthesis import synthesize
from agents.report import generate_report

logger = logging.getLogger(__name__)


# ── Routing Functions ───────────────────────────────────────────

def route_after_verification(
    state: AgentState,
) -> Literal["human_review", "generate_code"]:
    """Decide where to go after code verification."""
    result = state.get("verification_result", {})
    attempts = state.get("verification_attempts", 0)

    if result.get("passed", False):
        logger.info("Routing: Verification passed → Human Review #1")
        return "human_review"

    if attempts >= 5:  # MAX_VERIFICATION_ATTEMPTS
        logger.warning("Routing: Max verification attempts → Human Review #1")
        state["verification_exhausted"] = True
        return "human_review"

    logger.info("Routing: Verification failed → Generate Code (retry)")
    return "generate_code"


def route_after_human_review(
    state: AgentState,
) -> Literal["hw_optimize", "simulate", "generate_code", "verify"]:
    """
    Decide where to go after Human Review #1.

    Approve + --optimize  → hw_optimize (hardware-aware rewriting loop)
    Approve (no optimize) → simulate
    Reject + feedback     → generate_code  (feedback set in state by main.py)
    Re-verify             → verify
    """
    action = state.get("human_action", "")

    if action == "approve" or state.get("human_approved", False):
        if state.get("enable_optimization", False):
            logger.info("Routing: Human approved + --optimize → HW Optimize")
            return "hw_optimize"
        logger.info("Routing: Human approved → Simulate")
        return "simulate"

    if action == "verify":
        logger.info("Routing: Human requested re-verification → Verify")
        state["verification_attempts"] = 0
        state["verification_exhausted"] = False
        return "verify"

    logger.info("Routing: Human rejected → Generate Code")
    state["verification_attempts"] = 0
    state["verification_exhausted"] = False
    return "generate_code"


def route_after_opt_verification(
    state: AgentState,
) -> Literal["hw_optimize", "human_review_2", "compiler_decision"]:
    """
    Decide where to go after Verification #2 (optimized code).

    compiler_unavailable = True → compiler_decision [INTERRUPT]
    passed = True               → human_review_2   [INTERRUPT]
    failed, attempts < 3        → hw_optimize (retry)
    failed, attempts >= 3       → human_review_2 (exhausted)
    """
    result = state.get("opt_verification_result", {})
    attempts = state.get("opt_verification_attempts", 0)
    compiler_unavail = state.get("compiler_unavailable", False)

    if compiler_unavail:
        logger.warning("Routing: No RISC-V compiler → Compiler Decision")
        return "compiler_decision"

    if result.get("passed", False):
        logger.info("Routing: Opt Verification passed → Human Review #2")
        return "human_review_2"

    if attempts >= 3:
        logger.warning("Routing: Max opt verification attempts → Human Review #2 (exhausted)")
        state["opt_verification_exhausted"] = True
        return "human_review_2"

    logger.info("Routing: Opt Verification failed → HW Optimize (retry)")
    return "hw_optimize"


def route_after_human_review_2(
    state: AgentState,
) -> Literal["simulate", "hw_optimize"]:
    """
    Decide where to go after Human Review #2.

    Approve → simulate
    Reject  → hw_optimize (retry with human feedback)
    """
    action = state.get("human2_action", "")

    if action == "approve" or state.get("human2_approved", False):
        logger.info("Routing: Human #2 approved → Simulate")
        return "simulate"

    logger.info("Routing: Human #2 rejected → HW Optimize (retry)")
    state["opt_verification_attempts"] = 0
    state["opt_verification_exhausted"] = False
    return "hw_optimize"


def route_after_compiler_decision(
    state: AgentState,
) -> Literal["simulate", "report"]:
    """
    Decide where to go after the compiler_decision interrupt.

    proceed → simulate (mock mode — cross-compiler absent)
    skip    → report   (jump directly, skip sim+synth)
    """
    action = state.get("compiler_decision_action", "skip")

    if action == "proceed":
        logger.info("Routing: Compiler Decision → Simulate (mock mode)")
        return "simulate"

    logger.info("Routing: Compiler Decision → Report (skip sim+synth)")
    return "report"


def route_after_simulation(
    state: AgentState,
) -> Literal["synthesize", "generate_code"]:
    """
    Decide where to go after simulation.

    output_match = True  → synthesize
    output_match = False → generate_code (re-generate from scratch)
    """
    sim = state.get("simulation_result", {})

    if sim.get("output_match", True):
        logger.info("Routing: Simulation passed → Synthesize")
        return "synthesize"

    logger.info("Routing: Simulation output mismatch → Generate Code (retry)")
    return "generate_code"


# ── Graph Construction ──────────────────────────────────────────

def build_graph(entry_point: str = "parse_fx"):
    """Build and compile the LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────
    workflow.add_node("parse_fx",          parse_fx_graph)
    workflow.add_node("generate_code",     generate_code)
    workflow.add_node("verify",            verify_code)
    workflow.add_node("human_review",      human_review)
    workflow.add_node("hw_optimize",       hw_optimize)
    workflow.add_node("verify_optimized",  verify_optimized_code)
    workflow.add_node("human_review_2",    human_review_2)
    workflow.add_node("compiler_decision", compiler_decision)
    workflow.add_node("simulate",          simulate)
    workflow.add_node("synthesize",        synthesize)
    workflow.add_node("report",            generate_report)

    # ── Entry Point ──────────────────────────────────────────────
    workflow.set_entry_point(entry_point)

    # ── Edges ────────────────────────────────────────────────────

    # parse_fx → generate_code → verify
    workflow.add_edge("parse_fx",      "generate_code")
    workflow.add_edge("generate_code", "verify")

    # verify: pass → human_review | fail → generate_code
    workflow.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "human_review":  "human_review",
            "generate_code": "generate_code",
        },
    )

    # human_review: approve+opt → hw_optimize | approve → simulate | reject → generate_code
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "hw_optimize":   "hw_optimize",
            "simulate":      "simulate",
            "generate_code": "generate_code",
            "verify":        "verify",
        },
    )

    # hw_optimize → verify_optimized
    workflow.add_edge("hw_optimize", "verify_optimized")

    # verify_optimized → compiler_decision | human_review_2 | hw_optimize
    workflow.add_conditional_edges(
        "verify_optimized",
        route_after_opt_verification,
        {
            "compiler_decision": "compiler_decision",
            "human_review_2":    "human_review_2",
            "hw_optimize":       "hw_optimize",
        },
    )

    # human_review_2: approve → simulate | reject → hw_optimize
    workflow.add_conditional_edges(
        "human_review_2",
        route_after_human_review_2,
        {
            "simulate":    "simulate",
            "hw_optimize": "hw_optimize",
        },
    )

    # compiler_decision: proceed → simulate | skip → report
    workflow.add_conditional_edges(
        "compiler_decision",
        route_after_compiler_decision,
        {
            "simulate": "simulate",
            "report":   "report",
        },
    )

    # simulate: pass → synthesize | mismatch → generate_code
    workflow.add_conditional_edges(
        "simulate",
        route_after_simulation,
        {
            "synthesize":    "synthesize",
            "generate_code": "generate_code",
        },
    )

    # synthesize → report → END
    workflow.add_edge("synthesize", "report")
    workflow.add_edge("report",     END)

    # ── Compile ──────────────────────────────────────────────────
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=[
            "human_review",      # Human Approval #1 (after Verifier #1)
            "human_review_2",    # Human Approval #2 (after Verifier #2 / HW optimizer)
            "compiler_decision", # No RISC-V compiler — human decides proceed/skip
        ],
    )

    return app
