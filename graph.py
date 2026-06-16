"""
LangGraph Workflow Definition.

Connects the agents into a cyclic graph representing the
compiler pipeline. Handles conditional edges for the
verification loop and human-in-the-loop pause.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState, ENABLE_HARDWARE_PIPELINE
from agents.fx_parser import parse_fx_graph
from agents.code_generator import generate_code
from agents.verifier import verify_code
from agents.human_review import human_review
from agents.energy_estimator import estimate_energy_node
from agents.simulator import simulate
from agents.synthesis import synthesize
from agents.optimizer import optimize
from agents.report import generate_report

logger = logging.getLogger(__name__)


def _hardware_pipeline_enabled(state: AgentState) -> bool:
    """Return True when the original Hazard3/OpenROAD pipeline should run."""
    return state.get("enable_hardware_pipeline", ENABLE_HARDWARE_PIPELINE)

# ── Routing Functions ───────────────────────────────────────────

def route_after_verification(state: AgentState) -> Literal["human_review", "generate_code", "report"]:
    """Decide where to go after code verification."""
    result = state.get("verification_result", {})
    attempts = state.get("verification_attempts", 0)

    if result.get("passed", False):
        logger.info("Routing: Verification passed -> Human Review")
        return "human_review"
    
    if attempts >= 5: # MAX_VERIFICATION_ATTEMPTS
        logger.warning("Routing: Max verification attempts reached -> Human Review")
        state["verification_exhausted"] = True
        return "human_review"
    
    logger.info("Routing: Verification failed -> Generate Code (Retry)")
    return "generate_code"


def route_after_human_review(state: AgentState) -> Literal["estimate_energy", "generate_code", "verify", "report"]:
    """Decide where to go after human review."""
    action = state.get("human_action", "")
    
    if action == "approve" or state.get("human_approved", False):
        logger.info("Routing: Human approved -> Energy Estimation")
        return "estimate_energy"
    elif action == "verify":
        logger.info("Routing: Human requested re-verification -> Verify")
        state["verification_attempts"] = 0
        state["verification_exhausted"] = False
        return "verify"
    
    logger.info("Routing: Human rejected/retry -> Generate Code")
    # Reset verification attempts so it can try again
    state["verification_attempts"] = 0
    state["verification_exhausted"] = False
    return "generate_code"


def route_after_energy_estimation(
    state: AgentState,
) -> Literal["simulate", "report"]:
    """After energy estimation, optionally continue to hardware pipeline."""
    if _hardware_pipeline_enabled(state):
        logger.info("Routing: Hardware pipeline enabled -> Hazard3 Simulation")
        return "simulate"
    logger.info(
        "Routing: Hardware pipeline disabled -> Report "
        "(skipping Hazard3 simulation and OpenROAD synthesis)"
    )
    return "report"


def route_after_synthesis(state: AgentState) -> Literal["optimize", "report"]:
    """Decide whether to run the optimization loop (hardware pipeline only)."""
    if not _hardware_pipeline_enabled(state):
        return "report"

    enable_opt = state.get("enable_optimization", False)
    iteration = state.get("optimization_iteration", 0)
    
    # Check if synthesis was successful before optimizing
    synth_success = state.get("synthesis_result", {}).get("success", False)
    
    if enable_opt and synth_success and iteration < 3: # MAX_OPTIMIZATION_ITERATIONS
        logger.info(f"Routing: Optimization enabled (Iter {iteration}) -> Optimize")
        return "optimize"
    
    logger.info("Routing: Optimization disabled or max iterations -> Report")
    return "report"


# ── Graph Construction ──────────────────────────────────────────

def build_graph(entry_point: str = "parse_fx"):
    """Build and compile the LangGraph workflow."""
    
    # Initialize StateGraph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("parse_fx", parse_fx_graph)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("verify", verify_code)
    workflow.add_node("human_review", human_review)
    workflow.add_node("estimate_energy", estimate_energy_node)
    workflow.add_node("report", generate_report)

    # Original Hazard3 + OpenROAD pipeline (disabled for now).
    workflow.add_node("simulate", simulate)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("optimize", optimize)
    
    # Set Entry Point
    workflow.set_entry_point(entry_point)
    
    # Add Edges
    workflow.add_edge("parse_fx", "generate_code")
    workflow.add_edge("generate_code", "verify")
    
    # Conditional edge after verification
    workflow.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "human_review": "human_review",
            "generate_code": "generate_code",
            "report": "report"
        }
    )
    
    # Conditional edge after human review
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "estimate_energy": "estimate_energy",
            "generate_code": "generate_code",
            "verify": "verify",
            "report": "report" # Fallback
        }
    )
    
    workflow.add_conditional_edges(
        "estimate_energy",
        route_after_energy_estimation,
        {
            "simulate": "simulate",
            "report": "report",
        },
    )
    workflow.add_edge("simulate", "synthesize")
    workflow.add_conditional_edges(
        "synthesize",
        route_after_synthesis,
        {
            "optimize": "optimize",
            "report": "report",
        },
    )
    workflow.add_edge("optimize", "generate_code")
    
    workflow.add_edge("report", END)
    
    # Compile graph with a checkpointer for human-in-the-loop
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"] # Pause *before* executing the human_review node
    )
    
    return app
