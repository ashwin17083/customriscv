"""
Entry Point / CLI for Agentic RISC-V Compiler.

Initializes the model, runs FX tracing, and drives the LangGraph workflow.
Handles three human-in-the-loop interrupts:
  1. human_review      — After Verifier #1 (approve/reject original code)
  2. human_review_2    — After Verifier #2 (approve/reject HW-optimized code)
  3. compiler_decision — No RISC-V compiler; choose proceed (mock sim) or skip

Bug fixes from previous version:
  - Human feedback is now correctly forwarded to the code generator as
    verification_feedback so it enters REPAIR MODE with the user's text
    and the current model.h + model.c included as context.
  - The resume loop now runs until the graph is fully complete (not just
    one stream pass), so simulation + synthesis always execute after approval.

Non-serializable PyTorch objects (nn.Module, GraphModule, Tensor) are placed
into state.pytorch_object_store (keyed by thread_id) rather than directly into
the LangGraph state so that LangGraph's MemorySaver can safely checkpoint the
state using msgpack at every interrupt boundary.

LLM Backend:
  LLM_BACKEND=vllm   (default) → Qwen2.5-Coder-32B via vLLM
  LLM_BACKEND=ollama            → deepseek-coder-v2:16b-lite-instruct-q4_K_M via Ollama
"""

import argparse
import importlib.util
import logging
import struct
import sys
import time
from pathlib import Path

import torch
import torch.fx

from graph import build_graph
from examples.demo_model import create_demo_model, get_reference_output
from state import store_pytorch_objects

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")


def load_model_from_file(filepath: str, module_name: str = "custom_model"):
    """Dynamically load a PyTorch model from a Python file."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for func_name in ['get_model', 'create_model', 'build_model']:
            if hasattr(module, func_name):
                return getattr(module, func_name)()

        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, torch.nn.Module):
                return obj

    raise ValueError(f"Could not find a PyTorch model in {filepath}")


def _save_sample_input(sample_input: torch.Tensor, out_dir: Path) -> None:
    """
    Save sample input as a flat float32 binary file for use by opt_verifier's
    Python output comparison test harness.
    """
    try:
        flat = sample_input.detach().cpu().numpy().flatten().astype('float32')
        bin_path = out_dir / "sample_input.bin"
        with open(bin_path, "wb") as f:
            f.write(struct.pack(f"{len(flat)}f", *flat))
        logger.info(f"Saved sample input ({len(flat)} floats) → {bin_path}")
    except Exception as e:
        logger.warning(f"Could not save sample_input.bin: {e}")


def _print_telemetry_summary(final_state: dict) -> None:
    """Print a formatted summary of token usage and latency to stdout."""
    print("\n" + "=" * 60)
    print("📊 PIPELINE TELEMETRY SUMMARY")
    print("=" * 60)

    total_in  = final_state.get("total_input_tokens",  0) or 0
    total_out = final_state.get("total_output_tokens", 0) or 0
    total_llm = final_state.get("total_llm_latency_s", 0.0) or 0.0

    print(f"  Total Input  Tokens : {total_in:>10,}")
    print(f"  Total Output Tokens : {total_out:>10,}")
    print(f"  Total Tokens        : {total_in + total_out:>10,}")
    print(f"  Total LLM Latency   : {total_llm:>10.2f}s")

    agent_latencies: dict = final_state.get("agent_latencies") or {}
    if agent_latencies:
        print("\n  Per-Agent Latency:")
        for agent, lat in sorted(agent_latencies.items()):
            print(f"    {agent:<20} {lat:>8.2f}s")

    call_stats: list = final_state.get("llm_call_stats") or []
    if call_stats:
        print("\n  Per-LLM-Call Breakdown:")
        header = f"    {'Agent':<18} {'Call':<25} {'In':>8} {'Out':>8} {'Total':>8} {'Latency':>10}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for s in call_stats:
            if isinstance(s, dict):
                agent = s.get("agent", "?")
                label = s.get("call_label", "?")
                inp   = s.get("input_tokens",  0)
                out   = s.get("output_tokens", 0)
                tot   = s.get("total_tokens",  0)
                lat   = s.get("latency_s", 0.0)
            else:
                agent = getattr(s, "agent", "?")
                label = getattr(s, "call_label", "?")
                inp   = getattr(s, "input_tokens",  0)
                out   = getattr(s, "output_tokens", 0)
                tot   = getattr(s, "total_tokens",  0)
                lat   = getattr(s, "latency_s", 0.0)
            print(f"    {agent:<18} {label:<25} {inp:>8,} {out:>8,} {tot:>8,} {lat:>9.2f}s")

    print("=" * 60)


# ── Interrupt handlers ──────────────────────────────────────────

def _handle_human_review_1(app, run_config, state_snapshot, out_dir: Path) -> bool:
    """
    Handle the human_review interrupt (Approval #1 — original generated code).

    Returns True if the user wants to continue, False if they quit.
    """
    current_state = state_snapshot.values

    print("\n" + "=" * 60)
    print("⏸️  PIPELINE PAUSED: HUMAN REVIEW #1 REQUIRED")
    print("=" * 60)

    is_exhausted = current_state.get('verification_exhausted', False)
    if is_exhausted:
        print("\n⚠️  Max verification attempts reached! You may edit the code manually.")
        print(f"  Code: {current_state.get('code_path', 'output/model.c')}")
        options_text = "Action [ (e)dit+verify, (r)etry generation, (a)pprove, (q)uit ]: "
    else:
        print(f"\n  Code    : {current_state.get('code_path', 'output/model.c')}")
        print(f"  Header  : {current_state.get('model_header_path', 'output/model.h')}")
        options_text = "Action [ (a)pprove, (r)eject with feedback, (q)uit ]: "

    while True:
        action = input(f"\n{options_text}").strip().lower()

        if action in ['a', 'approve']:
            print("✅ Code approved! Continuing pipeline...")
            app.update_state(run_config, {
                "human_approved": True,
                "human_action": "approve",
            })
            return True

        elif action in ['r', 'reject', 'retry']:
            if is_exhausted:
                print("Resetting counters and retrying code generation...")
                app.update_state(run_config, {
                    "human_approved": False,
                    "human_action": "retry",
                    "verification_attempts": 0,
                    "verification_exhausted": False,
                })
            else:
                feedback = input(
                    "Enter feedback for the Code Generator\n"
                    "(describe what to change — current code will be shown to the LLM):\n> "
                ).strip()
                print("Code rejected. Routing back to code generator with your feedback...")
                # BUG FIX: set verification_feedback so code_generator enters REPAIR MODE
                # with the human's message. The repair mode already reads model.h + model.c
                # from disk, so the LLM will see: feedback + current code.
                app.update_state(run_config, {
                    "human_approved": False,
                    "human_feedback": feedback,
                    "human_action": "retry",
                    # This is what code_generator checks to enter repair mode:
                    "verification_feedback": (
                        f"[HUMAN REVIEWER FEEDBACK — address this before anything else]\n"
                        f"{feedback}"
                    ),
                    "verification_attempts": 0,
                    "verification_exhausted": False,
                })
            return True

        elif is_exhausted and action in ['e', 'edit']:
            print("Proceeding to re-verify your manual edits...")
            app.update_state(run_config, {
                "human_approved": False,
                "human_action": "verify",
                "verification_attempts": 0,
                "verification_exhausted": False,
            })
            return True

        elif action in ['q', 'quit']:
            print("Exiting pipeline.")
            return False

        else:
            print("Invalid option. Please try again.")


def _handle_human_review_2(app, run_config, state_snapshot, out_dir: Path) -> bool:
    """
    Handle the human_review_2 interrupt (Approval #2 — HW-optimized code).

    Returns True to continue, False to quit.
    """
    current_state = state_snapshot.values

    print("\n" + "=" * 60)
    print("⏸️  PIPELINE PAUSED: HUMAN REVIEW #2 (HW-Optimized Code)")
    print("=" * 60)

    is_exhausted = current_state.get('opt_verification_exhausted', False)
    opt_c_path   = current_state.get('optimized_code_path', 'output/model_optimized.c')
    opt_h_path   = current_state.get('optimized_header_path', 'output/model_optimized.h')
    opt_result   = current_state.get('opt_verification_result', {})

    if is_exhausted:
        print(f"\n⚠️  Max opt verification attempts reached! Please inspect the code.")
        print(f"  model_optimized.c: {opt_c_path}")
    else:
        print(f"\n  model_optimized.c : {opt_c_path}")
        print(f"  model_optimized.h : {opt_h_path}")
        print(f"  Verification      : {'PASSED ✅' if opt_result.get('passed') else 'FAILED ❌'}")

    warnings = opt_result.get("warnings", [])
    if warnings:
        print("\n  Warnings:")
        for w in warnings[:3]:
            print(f"    ⚠ {w[:100]}")

    while True:
        action = input(
            "\nAction [ (a)pprove, (r)eject with feedback, (q)uit ]: "
        ).strip().lower()

        if action in ['a', 'approve']:
            print("✅ HW-optimized code approved! Continuing to simulation...")
            app.update_state(run_config, {
                "human2_approved": True,
                "human2_action": "approve",
            })
            return True

        elif action in ['r', 'reject', 'retry']:
            feedback = input(
                "Enter feedback for the HW Optimizer\n"
                "(describe what to fix — optimized code will be shown to the LLM):\n> "
            ).strip()
            print("HW-optimized code rejected. Routing back to HW optimizer...")
            app.update_state(run_config, {
                "human2_approved": False,
                "human2_feedback": feedback,
                "human2_action": "retry",
                # Feed feedback into opt_verification_feedback so optimizer sees it
                "opt_verification_feedback": (
                    f"[HUMAN REVIEWER FEEDBACK — address this before anything else]\n"
                    f"{feedback}"
                ),
                "opt_verification_attempts": 0,
                "opt_verification_exhausted": False,
            })
            return True

        elif action in ['q', 'quit']:
            print("Exiting pipeline.")
            return False

        else:
            print("Invalid option. Please try again.")


def _handle_compiler_decision(app, run_config, state_snapshot) -> bool:
    """
    Handle the compiler_decision interrupt (no RISC-V compiler available).

    Returns True to continue, False to quit.
    """
    print("\n" + "=" * 60)
    print("⚠️  COMPILER UNAVAILABLE: No RISC-V Cross-Compiler Found")
    print("=" * 60)
    print("\nmodel_optimized.c could not be compiled for RISC-V because no")
    print("riscv32-unknown-elf-gcc (or equivalent) was found in your PATH.")
    print("\nOptions:")
    print("  (p)roceed — continue to simulation (mock mode) + synthesis")
    print("  (s)kip    — skip simulation/synthesis and go directly to report")
    print("  (q)uit    — exit the pipeline")

    while True:
        action = input("\nYour choice [ (p)roceed, (s)kip, (q)uit ]: ").strip().lower()

        if action in ['p', 'proceed']:
            print("Proceeding with mock simulation (no RISC-V binary execution)...")
            app.update_state(run_config, {
                "compiler_decision_action": "proceed",
            })
            return True

        elif action in ['s', 'skip']:
            print("Skipping simulation and synthesis. Generating final report...")
            app.update_state(run_config, {
                "compiler_decision_action": "skip",
            })
            return True

        elif action in ['q', 'quit']:
            print("Exiting pipeline.")
            return False

        else:
            print("Invalid option. Please try again.")


def _stream_until_interrupt(app, run_config, initial_state=None) -> bool:
    """
    Stream the graph until it either completes or hits an interrupt.

    Returns True if the graph is still running (hit interrupt or has next nodes),
    False if it finished or failed.
    """
    try:
        source = app.stream(initial_state, run_config) if initial_state is not None \
                 else app.stream(None, run_config)
        for event in source:
            for node_name in event:
                logger.info(f"--- Finished node: {node_name} ---")
        return True
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline(model: torch.nn.Module, sample_input: torch.Tensor, config: dict):
    """Run the Agentic RISC-V compiler pipeline."""

    logger.info("Initializing Agentic RISC-V Pipeline...")

    # Unwrap compiled or parallelized models
    if hasattr(model, "_orig_mod"):
        logger.info("Unwrapping compiled model (_orig_mod)...")
        model = model._orig_mod
    if hasattr(model, "module"):
        logger.info("Unwrapping module wrapper (DataParallel/DDP)...")
        model = model.module

    # 1. Trace the model
    logger.info("Tracing model with torch.fx...")
    try:
        class CustomTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, module_qualified_name):
                name = m.__class__.__name__.lower()
                if any(x in name for x in ["rmsnorm", "rms_norm", "swiglu", "rotary", "rope", "attention"]):
                    return True
                return super().is_leaf_module(m, module_qualified_name)

        tracer = CustomTracer()
        graph = tracer.trace(model)
        traced_model = torch.fx.GraphModule(model, graph)
        logger.info(f"Trace successful. Found {len(list(traced_model.graph.nodes))} nodes.")
    except Exception as e:
        logger.error(f"Failed to trace model: {e}")
        return

    # 2. Get reference outputs (for simulator + opt_verifier comparison)
    logger.info("Running forward pass to get reference outputs...")
    with torch.no_grad():
        out = model(sample_input)
        if isinstance(out, torch.Tensor):
            reference_outputs = out.flatten()[:10].tolist()
        else:
            reference_outputs = []

    # 3. Save sample input for opt_verifier's Python output comparison
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_sample_input(sample_input, out_dir)

    # 4. Initialize State
    start_from = config.get("start_from", "parse_fx")
    thread_id = "agentic_riscv_run_01"

    initial_state = {
        "thread_id": thread_id,
        "model_name": config.get("name", "model"),
        "fx_graph_str": str(traced_model.graph),
        "reference_outputs": reference_outputs,
        "enable_optimization": config.get("optimize", False),
        "weight_precision": config.get("precision", "f32"),
        "weight_mode": config.get("weight_mode", "embedded"),
        "verification_attempts": 0,
        "opt_verification_attempts": 0,
        "optimization_iteration": 0,
        "human_approved": False,
        "human_feedback": "",
        "human2_approved": False,
        "human2_feedback": "",
        "compiler_unavailable": False,
        "skip_to_report": False,
        # Telemetry
        "llm_call_stats": [],
        "total_input_tokens":  0,
        "total_output_tokens": 0,
        "total_llm_latency_s": 0.0,
        "agent_latencies": {},
    }

    if start_from == "parse_fx":
        store_pytorch_objects(
            thread_id=thread_id,
            model=model,
            fx_graph=traced_model,
            sample_input=sample_input,
        )
        logger.info(f"PyTorch objects stored (thread_id='{thread_id}')")

    # 5. Load existing state for start-from modes
    if start_from != "parse_fx":
        import json
        logger.info(f"Loading existing state from {out_dir} to start from '{start_from}'")

        if (out_dir / "model.c").exists():
            initial_state["generated_code"] = (out_dir / "model.c").read_text()
            initial_state["code_path"] = str(out_dir / "model.c")
        if (out_dir / "weights.h").exists():
            initial_state["generated_header"] = (out_dir / "weights.h").read_text()
            initial_state["header_path"] = str(out_dir / "weights.h")
        if (out_dir / "model.h").exists():
            initial_state["generated_model_header"] = (out_dir / "model.h").read_text()
            initial_state["model_header_path"] = str(out_dir / "model.h")

        # Load optimized artifacts for hw_optimize / verify_optimized start points
        if start_from in ("hw_optimize", "verify_optimized"):
            if (out_dir / "model_optimized.c").exists():
                initial_state["optimized_code"] = (out_dir / "model_optimized.c").read_text()
                initial_state["optimized_code_path"] = str(out_dir / "model_optimized.c")
            if (out_dir / "model_optimized.h").exists():
                initial_state["optimized_header"] = (out_dir / "model_optimized.h").read_text()
                initial_state["optimized_header_path"] = str(out_dir / "model_optimized.h")

        if (out_dir / "ir_graph.json").exists():
            with open(out_dir / "ir_graph.json", "r") as f:
                ir_graph_data = json.load(f)
                initial_state["ir_graph"] = ir_graph_data
                initial_state["weights_metadata"] = ir_graph_data.get("weight_metadata", {})
                from ir import IRGraph
                try:
                    ir_obj = IRGraph.from_dict(ir_graph_data)
                    initial_state["total_params"] = ir_obj.total_params()
                    initial_state["model_memory_bytes"] = ir_obj.total_weight_memory()
                    initial_state["ir_summary"] = ir_obj.layer_summary()
                except Exception as e:
                    logger.warning(f"Could not compute IR metrics: {e}")

        if (out_dir / "weights_manifest.json").exists():
            with open(out_dir / "weights_manifest.json", "r") as f:
                initial_state["weights_manifest"] = json.load(f)

        initial_state["weights_path"] = str(out_dir / "weights.npz")
        initial_state["weights_bin_path"] = str(out_dir / "weights.bin")

        if not initial_state.get("weights_metadata") and model is not None:
            state_dict = model.state_dict()
            initial_state["weights_metadata"] = {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "numel": tensor.numel(),
                }
                for name, tensor in state_dict.items()
            }

    # 6. Build Graph
    app = build_graph(entry_point=start_from)
    run_config = {"configurable": {"thread_id": thread_id}}

    # 7. Run pipeline loop — handles multiple interrupt points
    logger.info("Starting graph execution...")
    t_pipeline_start = time.perf_counter()

    # Initial stream
    ok = _stream_until_interrupt(app, run_config, initial_state)
    if not ok:
        return

    # Loop to handle all possible interrupts until graph completes
    while True:
        state_snapshot = app.get_state(run_config)

        if not state_snapshot.next:
            # Graph finished
            break

        next_nodes = set(state_snapshot.next)
        logger.info(f"Graph paused at: {next_nodes}")

        if "human_review" in next_nodes:
            should_continue = _handle_human_review_1(app, run_config, state_snapshot, out_dir)
            if not should_continue:
                return

        elif "human_review_2" in next_nodes:
            should_continue = _handle_human_review_2(app, run_config, state_snapshot, out_dir)
            if not should_continue:
                return

        elif "compiler_decision" in next_nodes:
            should_continue = _handle_compiler_decision(app, run_config, state_snapshot)
            if not should_continue:
                return

        else:
            logger.warning(f"Unknown interrupt at: {next_nodes}. Attempting to resume...")

        # Resume graph after handling interrupt
        ok = _stream_until_interrupt(app, run_config, initial_state=None)
        if not ok:
            return

    # 8. Final result
    final_state = app.get_state(run_config).values
    t_pipeline_end = time.perf_counter()

    if "final_report" in final_state:
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED")
        print("=" * 60)
        print("Report saved to output/report.md")

    _print_telemetry_summary(final_state)
    logger.info(
        f"Total pipeline wall-clock time: "
        f"{t_pipeline_end - t_pipeline_start:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Agentic RISC-V Compiler")
    parser.add_argument("--demo",  action="store_true", help="Run the TinyLlama demo model")
    parser.add_argument("--model", type=str,  help="Path to Python file with PyTorch model")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable HW-aware optimization (requires hw_config.yaml)")
    parser.add_argument("--name", type=str, default="custom_model", help="Model name")
    parser.add_argument(
        "--precision", type=str, default="f32",
        choices=["f32", "f16", "bf16", "mxfp8"],
        help="Weight precision (default: f32)"
    )
    parser.add_argument(
        "--weight-mode", type=str, default="embedded",
        choices=["embedded", "binary"],
        help="Weight storage: embedded (bare-metal) or binary (hosted)"
    )
    parser.add_argument(
        "--start-from", type=str, default="parse_fx",
        choices=[
            "parse_fx", "generate_code", "verify",
            "hw_optimize", "verify_optimized",        # HW optimization stages
            "simulate", "synthesize",
        ],
        help="Start pipeline from a specific stage (loads artifacts from output/)"
    )

    args = parser.parse_args()

    if args.demo:
        logger.info("Using built-in TinyLlama demo model")
        model, sample_input = create_demo_model()
        config = {
            "name": "tiny_llama_demo",
            "optimize": args.optimize,
            "precision": args.precision,
            "weight_mode": args.weight_mode,
            "start_from": args.start_from,
        }
        run_pipeline(model, sample_input, config)

    elif args.model:
        logger.info(f"Loading model from {args.model}")
        try:
            model = load_model_from_file(args.model)
            sample_input = torch.randn(1, 10)
            config = {
                "name": args.name,
                "optimize": args.optimize,
                "precision": args.precision,
                "weight_mode": args.weight_mode,
                "start_from": args.start_from,
            }
            run_pipeline(model, sample_input, config)
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
