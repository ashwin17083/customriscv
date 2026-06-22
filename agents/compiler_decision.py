"""
Compiler Decision Agent — Interrupt node for RISC-V compiler unavailability.

This node is triggered when opt_verifier detects no RISC-V cross-compiler
is available to compile model_optimized.c.

Like human_review.py, this node is configured with interrupt_before so
LangGraph pauses here. main.py handles user interaction and sets:
  - compiler_decision_action = 'proceed'  → continue to simulate + synthesize
  - compiler_decision_action = 'skip'     → jump directly to report
"""

from __future__ import annotations

import logging

from state import AgentState

logger = logging.getLogger(__name__)


def compiler_decision(state: AgentState) -> dict:
    """
    LangGraph node function: Compiler unavailability decision checkpoint.

    Pauses the pipeline (via interrupt_before) and lets main.py present
    the user with a choice:
      (p)roceed  — continue to simulation (mock) + synthesis
      (s)kip     — skip simulation/synthesis and go directly to report

    Reads:  state["compiler_unavailable"]
    Writes: (nothing — main.py updates state before resume)
    """
    logger.info("=" * 60)
    logger.info("⚠️  COMPILER DECISION CHECKPOINT")
    logger.info("=" * 60)
    logger.info(
        "No RISC-V cross-compiler was found. "
        "model_optimized.c cannot be compiled for RISC-V."
    )
    logger.info(
        "Options:\n"
        "  (p)roceed — run simulation (mock mode) + synthesis\n"
        "  (s)kip    — skip simulation/synthesis and go directly to report"
    )

    # The actual user interaction is handled by main.py before resume.
    return {}
