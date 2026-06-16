"""
Entry Point / CLI for Agentic RISC-V Compiler.

Initializes the model, runs FX tracing, and drives the LangGraph workflow.
Handles human-in-the-loop interaction.
"""

import argparse
import importlib.util
import logging
import sys
import torch
import torch.fx

from graph import build_graph
from examples.demo_model import create_demo_model, get_reference_output

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
        
        # Look for a function like 'get_model()' or 'create_model()'
        for func_name in ['get_model', 'create_model', 'build_model']:
            if hasattr(module, func_name):
                return getattr(module, func_name)()
                
        # If not found, look for instances of nn.Module
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, torch.nn.Module):
                return obj
    
    raise ValueError(f"Could not find a PyTorch model in {filepath}")


def run_pipeline(model: torch.nn.Module, sample_input: torch.Tensor, config: dict):
    """Run the Agentic RISC-V compiler pipeline."""
    
    logger.info("Initializing Agentic RISC-V Pipeline...")
    
    # Unwrap compiled or parallelized models to ensure standard FX symbolic tracing
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
            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
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
        logger.error("Model must be FX-traceable. Avoid dynamic control flow.")
        return

    # 2. Get reference outputs (for simulator verification)
    logger.info("Running forward pass to get reference outputs...")
    with torch.no_grad():
        out = model(sample_input)
        if isinstance(out, torch.Tensor):
             # Just grab a flattened slice for reference comparison
            reference_outputs = out.flatten()[:10].tolist()
        else:
            reference_outputs = []
            
    # 3. Initialize State
    start_from = config.get("start_from", "parse_fx")

    initial_state = {
        "model_name": config.get("name", "model"),
        "fx_graph_str": str(traced_model.graph),
        "reference_outputs": reference_outputs,
        "enable_optimization": config.get("optimize", False),
        "weight_precision": config.get("precision", "f32"),
        "weight_mode": config.get("weight_mode", "embedded"),
        "verification_attempts": 0,
        "optimization_iteration": 0,
        "human_approved": False,
        "human_feedback": "",
    }

    if start_from == "parse_fx":
        # parse_fx needs the live PyTorch objects; it will clear them
        # from state once it has extracted all info into serializable fields.
        initial_state["model"] = model
        initial_state["fx_graph"] = traced_model
        initial_state["sample_input"] = sample_input
    
    if start_from != "parse_fx":
        import json
        from pathlib import Path
        out_dir = Path("output")
        logger.info(f"Loading existing state from {out_dir} to start from {start_from}")
        
        if (out_dir / "model.c").exists():
            initial_state["generated_code"] = (out_dir / "model.c").read_text()
            initial_state["code_path"] = str(out_dir / "model.c")
        if (out_dir / "weights.h").exists():
            initial_state["generated_header"] = (out_dir / "weights.h").read_text()
            initial_state["header_path"] = str(out_dir / "weights.h")
        if (out_dir / "model.h").exists():
            initial_state["generated_model_header"] = (out_dir / "model.h").read_text()
            initial_state["model_header_path"] = str(out_dir / "model.h")
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
            weights_metadata = {}
            for param_name, param_tensor in state_dict.items():
                weights_metadata[param_name] = {
                    "shape": list(param_tensor.shape),
                    "dtype": str(param_tensor.dtype).replace("torch.", ""),
                    "numel": param_tensor.numel(),
                }
            initial_state["weights_metadata"] = weights_metadata

    # 4. Build Graph
    app = build_graph(entry_point=start_from)
    
    # Configuration for the checkpointer (required for human-in-the-loop)
    thread_id = "agentic_riscv_run_01"
    run_config = {"configurable": {"thread_id": thread_id}}
    
    # 5. Run Graph (until interrupt)
    logger.info("Starting graph execution...")
    try:
        # stream() lets us observe node execution
        for event in app.stream(initial_state, run_config):
            for node_name, node_state in event.items():
                logger.info(f"--- Finished node: {node_name} ---")
                
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Handle Human-in-the-Loop Interrupt
    # Check if the graph paused
    state_snapshot = app.get_state(run_config)
    if state_snapshot.next and "human_review" in state_snapshot.next:
        current_state = state_snapshot.values
        
        print("\n" + "="*60)
        print("⏸️  PIPELINE PAUSED: HUMAN REVIEW REQUIRED")
        print("="*60)
        is_exhausted = current_state.get('verification_exhausted', False)
        if is_exhausted:
            print("\n⚠️  Max verification attempts reached! You must manually fix the code.")
            print(f"Edit the code at: {current_state.get('code_path')}")
            options_text = "Action [ (e)dit+verify, (r)etry generation, (a)pprove, (q)uit ]: "
        else:
            print(f"Code generated successfully at: {current_state.get('code_path')}")
            print("Please review the generated code.")
            options_text = "Action [ (a)pprove, (r)eject with feedback, (q)uit ]: "
        
        while True:
            action = input(f"\n{options_text}").strip().lower()
            if action in ['a', 'approve']:
                print("Code approved! Continuing to simulation...")
                app.update_state(run_config, {"human_approved": True, "human_action": "approve"})
                break
            elif action in ['r', 'reject', 'retry']:
                if is_exhausted:
                    print("Resetting counters and retrying code generation...")
                    app.update_state(run_config, {"human_approved": False, "human_action": "retry"})
                else:
                    feedback = input("Enter feedback for the Code Generator: ").strip()
                    print("Code rejected. Routing back to generator...")
                    app.update_state(run_config, {"human_approved": False, "human_feedback": feedback, "human_action": "retry"})
                break
            elif is_exhausted and action in ['e', 'edit']:
                print("Proceeding to re-verify your manual edits...")
                app.update_state(run_config, {"human_approved": False, "human_action": "verify"})
                break
            elif action in ['q', 'quit']:
                print("Exiting pipeline.")
                return
            else:
                print("Invalid option.")

        # 7. Resume Graph execution
        logger.info("Resuming graph execution...")
        for event in app.stream(None, run_config):
            for node_name, node_state in event.items():
                logger.info(f"--- Finished node: {node_name} ---")
                
    # 8. Check Final Result
    final_state = app.get_state(run_config).values
    if "final_report" in final_state:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED")
        print("="*60)
        print("Report saved to output/report.md")


def main():
    parser = argparse.ArgumentParser(description="Agentic RISC-V Compiler")
    parser.add_argument("--demo", action="store_true", help="Run the TinyLlama demo model")
    parser.add_argument("--model", type=str, help="Path to Python file containing a PyTorch model")
    parser.add_argument("--optimize", action="store_true", help="Enable closed-loop optimization")
    parser.add_argument("--name", type=str, default="custom_model", help="Name of the model")
    parser.add_argument(
        "--precision", type=str, default="f32",
        choices=["f32", "f16", "bf16", "mxfp8"],
        help="Weight precision: f32 (default), f16, bf16, or mxfp8"
    )
    parser.add_argument(
        "--weight-mode", type=str, default="embedded",
        choices=["embedded", "binary"],
        help="Weight storage mode: embedded (bare-metal, default) or binary (hosted, uses fopen)"
    )
    parser.add_argument(
        "--start-from", type=str, default="parse_fx",
        choices=["parse_fx", "generate_code", "verify", "simulate", "synthesize"],
        help="Start pipeline from a specific stage"
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
            # Create a dummy input (assuming image format [1, 3, 224, 224] or generic [1, 10])
            # In a real tool, we'd need a way for the user to specify input shapes.
            # Using generic for now to get a reference trace.
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
