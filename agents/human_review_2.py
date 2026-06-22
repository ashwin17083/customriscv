"""
Human Review Agent #2 — Pauses the pipeline for human approval of HW-optimized code.

Mirrors human_review.py but operates on the optimized artifacts
(model_optimized.h / model_optimized.c) produced by hw_optimizer.py.
"""

from __future__ import annotations

import logging

from state import AgentState

logger = logging.getLogger(__name__)


def human_review_2(state: AgentState) -> dict:
    """
    LangGraph node function: Present HW-optimized code for human review.

    This node is configured with 'interrupt_before' in the graph definition,
    so LangGraph will pause BEFORE executing this node.
    The main.py entry point handles the actual user interaction and resumes
    the graph with updated state (human2_approved, human2_action).

    Reads:  state["optimized_code_path"], state["opt_verification_result"]
    Writes: state["human2_approved"], state["human2_feedback"]
    """
    logger.info("=" * 60)
    logger.info("HUMAN REVIEW #2 CHECKPOINT (HW-Optimized Code)")
    logger.info("=" * 60)

    opt_code = state.get("optimized_code", "")
    opt_header = state.get("optimized_header", "")
    opt_c_path = state.get("optimized_code_path", "output/model_optimized.c")
    opt_verification = state.get("opt_verification_result", {})
    exhausted = state.get("opt_verification_exhausted", False)

    logger.info(f"Optimized code path   : {opt_c_path}")
    logger.info(f"Optimized code length : {len(opt_code)} chars")
    logger.info(f"Optimized header len  : {len(opt_header)} chars")
    logger.info(
        f"Opt Verification      : "
        f"{'PASSED ✅' if opt_verification.get('passed') else 'FAILED ❌'}"
    )

    if exhausted:
        logger.warning(
            "⚠️  Max opt verification attempts reached — manual inspection recommended."
        )

    if opt_verification.get("warnings"):
        logger.info("Warnings from Verifier #2:")
        for w in opt_verification.get("warnings", []):
            logger.info(f"  ⚠ {w}")

    # The actual approval input is handled by main.py before the graph resumes.
    return {
        "human2_approved": state.get("human2_approved", False),
        "human2_feedback":  state.get("human2_feedback", ""),
    }
