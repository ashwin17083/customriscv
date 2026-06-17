"""
LangGraph Workflow Definition.

New pipeline flow:
  parse_fx → generate_code → verify
    verify (pass)     → human_review
    verify (fail)     → generate_code (retry, up to 5×)

  human_review (approve, optimize=on)  → optimize → simulate
  human_review (approve, optimize=off) → simulate
  human_review (reject / retry)        → generate_code

  simulate (output match)    → synthesize
  simulate (output mismatch) → generate_code  ← NEW back-edge

  synthesize → report  ← always (no optimize fork after synthesis)

The Optimizer agent is a hardware-aware hints pass that injects
knowledge of the target micro-architecture (e.g. systolic arrays,
VPUs) into the code-generator prompt.  It runs once, before
simulation, and does not loop.
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
from agents.simulator import simulate
from agents.synthesis import synthesize
from agents.optimizer import optimize
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
        logger.info("Routing: Verification passed → Human Review")
        return "human_review"

    if attempts >= 5:  # MAX_VERIFICATION_ATTEMPTS
        logger.warning(
            "Routing: Max verification attempts reached → Human Review"
        )
        state["verification_exhausted"] = True
        return "human_review"

    logger.info("Routing: Verification failed → Generate Code (Retry)")
    return "generate_code"


def route_after_human_review(
    state: AgentState,
) -> Literal["optimize", "simulate", "generate_code", "verify"]:
    """Decide where to go after human review.

    If the human approves:
      - with --optimize  → optimize (hardware-aware hints) → simulate
      - without --optimize → simulate directly
    If rejected: back to generate_code.
    """
    action = state.get("human_action", "")

    if action == "approve" or state.get("human_approved", False):
        if state.get("enable_optimization", False):
            logger.info("Routing: Human approved + optimization enabled → Optimize")
            return "optimize"
        logger.info("Routing: Human approved → Simulate")
        return "simulate"

    if action == "verify":
        logger.info("Routing: Human requested re-verification → Verify")
        state["verification_attempts"] = 0
        state["verification_exhausted"] = False
        return "verify"

    logger.info("Routing: Human rejected / retry → Generate Code")
    state["verification_attempts"] = 0
    state["verification_exhausted"] = False
    return "generate_code"


def route_after_simulation(
    state: AgentState,
) -> Literal["synthesize", "generate_code"]:
    """Decide where to go after simulation.

    If the simulated output does NOT match the PyTorch reference,
    route back to the code generator with feedback so the model can
    be fixed.  Otherwise proceed to synthesis.
    """
    sim = state.get("simulation_result", {})

    # output_match is True when outputs agree (or when there is no
    # reference to compare against — mock path sets it True too).
    if sim.get("output_match", True):
        logger.info("Routing: Simulation passed → Synthesize")
        return "synthesize"

    logger.info("Routing: Simulation output mismatch → Generate Code (Retry)")
    return "generate_code"


# ── Graph Construction ──────────────────────────────────────────

def build_graph(entry_point: str = "parse_fx"):
    """Build and compile the LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────
    workflow.add_node("parse_fx",      parse_fx_graph)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("verify",        verify_code)
    workflow.add_node("human_review",  human_review)
    workflow.add_node("optimize",      optimize)
    workflow.add_node("simulate",      simulate)
    workflow.add_node("synthesize",    synthesize)
    workflow.add_node("report",        generate_report)

    # ── Entry Point ────────────────────────────────────────────
    workflow.set_entry_point(entry_point)

    # ── Edges ──────────────────────────────────────────────────

    # parse_fx → generate_code → verify
    workflow.add_edge("parse_fx",      "generate_code")
    workflow.add_edge("generate_code", "verify")

    # verify: pass → human_review | fail → generate_code (retry)
    workflow.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "human_review": "human_review",
            "generate_code": "generate_code",
        },
    )

    # human_review: approve+opt → optimize | approve → simulate | reject → generate_code
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "optimize":      "optimize",
            "simulate":      "simulate",
            "generate_code": "generate_code",
            "verify":        "verify",
        },
    )

    # optimize runs once (hardware-aware hints pass), then → simulate
    workflow.add_edge("optimize", "simulate")

    # simulate: pass → synthesize | mismatch → generate_code
    workflow.add_conditional_edges(
        "simulate",
        route_after_simulation,
        {
            "synthesize":    "synthesize",
            "generate_code": "generate_code",
        },
    )

    # synthesize always → report
    workflow.add_edge("synthesize", "report")
    workflow.add_edge("report",     END)

    # ── Compile ────────────────────────────────────────────────
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],   # pause before human_review node
    )

    return app
