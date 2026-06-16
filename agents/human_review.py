"""
Human Review Agent — Pauses the pipeline for human approval.

Uses LangGraph's interrupt mechanism to pause execution and
wait for the user to approve or reject the generated code.
"""

from __future__ import annotations

import logging

from state import AgentState

logger = logging.getLogger(__name__)


def human_review(state: AgentState) -> dict:
    """
    LangGraph node function: Present code for human review.

    This node is configured with `interrupt_before` in the graph
    definition, so LangGraph will pause BEFORE executing this node.
    The main.py entry point handles the actual user interaction
    and resumes the graph with updated state.

    Reads: state["generated_code"], state["verification_result"]
    Writes: state["human_approved"], state["human_feedback"]
    """
    logger.info("=" * 60)
    logger.info("HUMAN REVIEW CHECKPOINT")
    logger.info("=" * 60)

    code = state.get("generated_code", "")
    header = state.get("generated_header", "")
    verification = state.get("verification_result", {})
    ir_summary = state.get("ir_summary", "")

    # Display summary for the human
    logger.info(f"\nModel Summary:\n{ir_summary}")
    logger.info(f"\nVerification: {'PASSED ✅' if verification.get('passed') else 'FAILED ❌'}")
    logger.info(f"\nGenerated code length: {len(code)} chars")
    logger.info(f"Generated header length: {len(header)} chars")

    if verification.get("warnings"):
        logger.info("\nWarnings:")
        for w in verification["warnings"]:
            logger.info(f"  ⚠ {w}")

    # The actual approval comes from the interrupt handler in main.py
    # This node just marks that we've reached the review point.
    # The state update with human_approved=True/False is done by the
    # interrupt handler before resuming the graph.

    return {
        "human_approved": state.get("human_approved", False),
        "human_feedback": state.get("human_feedback", ""),
    }
