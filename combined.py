"""
agentic-riscv/
├── main.py                    # Entry point with CLI args for precision & weight-mode
├── graph.py                   # LangGraph workflow definition
├── state.py                   # Shared AgentState TypedDict state
├── ir.py                      # Custom IR dataclasses
├── agents/
│   ├── __init__.py
│   ├── fx_parser.py           # FX → Custom IR + npz serialization
│   ├── codegen_contract.py    # [NEW] Standardizes required helper signatures dynamically
│   ├── code_generator.py      # IR → C code (model.h / model.c) with deterministic weights header, repair loops
│   ├── verifier.py            # Syntax & structural validation (detects placeholders/errors)
│   ├── human_review.py        # Human-in-the-loop pause
│   ├── simulator.py           # Hazard3 simulation wrapper
│   ├── synthesis.py           # OpenROAD synthesis wrapper
│   ├── optimizer.py           # Optimization suggestions (LLM)
│   └── report.py              # Final report generator
├── prompts/
│   ├── codegen.txt            # Code generator system prompt
│   └── optimizer.txt          # Optimization system prompt
├── tools/
│   ├── __init__.py
│   ├── compile.py             # RISC-V GCC cross-compilation
│   ├── hazard3.py             # Hazard3 simulator wrapper
│   └── openroad.py            # OpenROAD flow wrapper
├── examples/
│   └── demo_model.py          # Simple PyTorch model for testing
├── requirements.txt
└── README.md
"""

#main.py#
"""
Entry Point / CLI for Agentic RISC-V Compiler.

Initializes the model, runs FX tracing, and drives the LangGraph workflow.
Handles human-in-the-loop interaction.

Non-serializable PyTorch objects (nn.Module, GraphModule, Tensor) are placed
into state.pytorch_object_store (keyed by thread_id) rather than directly into
the LangGraph state so that LangGraph's MemorySaver can safely checkpoint the
state using msgpack at every interrupt boundary.
"""

import argparse
import importlib.util
import logging
import sys
import time
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
        header = f"    {'Agent':<18} {'Call':<22} {'In':>8} {'Out':>8} {'Total':>8} {'Latency':>10}"
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
            print(f"    {agent:<18} {label:<22} {inp:>8,} {out:>8,} {tot:>8,} {lat:>9.2f}s")

    print("=" * 60)


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

    # ── Thread ID (used by pytorch_object_store and run_config) ──
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
        "optimization_iteration": 0,
        "human_approved": False,
        "human_feedback": "",
        # Telemetry – initialise accumulators
        "llm_call_stats": [],
        "total_input_tokens":  0,
        "total_output_tokens": 0,
        "total_llm_latency_s": 0.0,
        "agent_latencies": {},
    }

    if start_from == "parse_fx":
        # Place the live PyTorch objects into the out-of-band store so that
        # parse_fx_graph can retrieve them.  They must NOT go into the
        # LangGraph state because msgpack cannot serialise nn.Module,
        # GraphModule or Tensor objects when checkpointing at interrupts.
        store_pytorch_objects(
            thread_id=thread_id,
            model=model,
            fx_graph=traced_model,
            sample_input=sample_input,
        )
        logger.info(
            f"PyTorch objects stored in pytorch_object_store "
            f"(thread_id='{thread_id}')"
        )
    
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
    run_config = {"configurable": {"thread_id": thread_id}}
    
    # 5. Run Graph (until interrupt)
    logger.info("Starting graph execution...")
    t_pipeline_start = time.perf_counter()
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
                print("Code approved! Continuing to optimization/simulation...")
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
    t_pipeline_end = time.perf_counter()

    if "final_report" in final_state:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED")
        print("="*60)
        print("Report saved to output/report.md")

    # 9. Print telemetry summary
    _print_telemetry_summary(final_state)
    logger.info(
        f"Total pipeline wall-clock time: "
        f"{t_pipeline_end - t_pipeline_start:.2f}s"
    )


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
        choices=["parse_fx", "generate_code", "verify", "optimize", "simulate", "synthesize"],
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


#app.py#
"""
Agentic RISC-V Compiler — Streamlit GUI (Hackathon Edition)
============================================================
Run with:
    streamlit run app.py

This single script mirrors every option available via the CLI
(python main.py --demo / --model / --optimize / --precision /
--weight-mode / --start-from) and adds:
  • Live pipeline log streaming
  • Human-review panel with syntax-highlighted C-file tabs
  • Final report rendered in-page
"""

from __future__ import annotations

import importlib.util
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st

# ── Page config (MUST be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Agentic RISC-V Compiler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy imports (heavy; only after path is set) ─────────────────
import torch
import torch.fx
from graph import build_graph
from state import store_pytorch_objects
from examples.demo_model import create_demo_model

# ── Logging bridge → session-state queue ───────────────────────
class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


# ── Custom CSS ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #111128 50%, #0a0a1f 100%);
        color: #e2e2f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13132b 0%, #0e0e24 100%);
        border-right: 1px solid #2a2a5a;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCheckbox label { color: #c0c0e0 !important; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a45 0%, #0f0f30 100%);
        border: 1px solid #2a2a6a;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        text-align: center;
    }
    .hero-banner h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .hero-banner p { color: #8888bb; font-size: 1rem; margin: 0; }

    /* Status pills */
    .pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .pill-idle    { background: #1e1e4a; color: #8888cc; border: 1px solid #3a3a7a; }
    .pill-running { background: #1a3a1a; color: #6fd96f; border: 1px solid #3a6a3a;
                    animation: pulse-green 1.5s infinite; }
    .pill-paused  { background: #3a2a00; color: #f0c060; border: 1px solid #6a5a00; }
    .pill-done    { background: #0a2a0a; color: #50d050; border: 1px solid #2a5a2a; }
    .pill-error   { background: #2a0a0a; color: #f06060; border: 1px solid #5a2a2a; }

    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 0 0 rgba(111,217,111,0.4); }
        50%       { box-shadow: 0 0 0 6px rgba(111,217,111,0); }
    }

    /* Log box */
    .log-box {
        background: #080818;
        border: 1px solid #222244;
        border-radius: 10px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #99aacc;
        max-height: 340px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }

    /* Node stepper */
    .node-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 14px;
        margin: 4px 0;
        border-radius: 8px;
        font-size: 0.85rem;
        border: 1px solid transparent;
    }
    .node-step.done    { background: #0a200a; border-color: #2a5a2a; color: #80d080; }
    .node-step.active  { background: #1a2a00; border-color: #5a6a00; color: #d0d060;
                         animation: pulse-green 1.5s infinite; }
    .node-step.pending { background: #111128; border-color: #222244; color: #606090; }

    /* Review panel */
    .review-header {
        background: linear-gradient(135deg, #2a1a00, #1a1000);
        border: 1px solid #6a4000;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .review-header h2 { color: #f0c060; margin: 0 0 6px 0; font-size: 1.3rem; }
    .review-header p  { color: #aa8840; margin: 0; font-size: 0.9rem; }

    /* Report panel */
    .report-panel {
        background: #0d0d20;
        border: 1px solid #2a2a5a;
        border-radius: 12px;
        padding: 28px 32px;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }
    .metric-card {
        background: #12122a;
        border: 1px solid #2a2a5a;
        border-radius: 10px;
        padding: 16px 20px;
        min-width: 140px;
        flex: 1;
    }
    .metric-card .mlabel { font-size: 0.72rem; color: #7070a0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .mvalue { font-size: 1.6rem; font-weight: 700; color: #60a5fa; margin-top: 4px; }

    /* Buttons */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 8px 24px;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }

    /* Divider */
    hr { border-color: #2a2a5a; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# STATE HELPERS
# ═══════════════════════════════════════════════════════════════

PIPELINE_STATUS_IDLE    = "idle"
PIPELINE_STATUS_RUNNING = "running"
PIPELINE_STATUS_PAUSED  = "paused"   # waiting at human_review interrupt
PIPELINE_STATUS_DONE    = "done"
PIPELINE_STATUS_ERROR   = "error"

PIPELINE_NODES_ORDERED = [
    "parse_fx", "generate_code", "verify",
    "human_review", "optimize", "simulate", "synthesize", "report",
]

def _init_session():
    defaults = {
        "status": PIPELINE_STATUS_IDLE,
        "log_lines": [],
        "completed_nodes": [],
        "active_node": None,
        "app": None,            # compiled LangGraph app
        "run_config": None,
        "thread_id": None,
        "final_state": None,
        "log_queue": queue.Queue(),
        "worker_thread": None,
        "error_msg": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


def _load_model_from_file(filepath: str):
    """Dynamically load a PyTorch model from a Python file (mirrors main.py logic)."""
    spec = importlib.util.spec_from_file_location("custom_model", filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for func_name in ["get_model", "create_model", "build_model"]:
            if hasattr(module, func_name):
                return getattr(module, func_name)()
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, torch.nn.Module):
                return obj
    raise ValueError(f"Could not find a PyTorch model in {filepath}")


# ═══════════════════════════════════════════════════════════════
# BACKGROUND WORKER — runs until human_review interrupt
# ═══════════════════════════════════════════════════════════════

def _pipeline_worker(
    model,
    sample_input,
    config: dict,
    log_q: queue.Queue,
    status_q: queue.Queue,   # sends ("node", name) | ("paused",) | ("done",) | ("error", msg)
):
    """Background thread: traces model, builds graph, streams nodes."""
    try:
        # ── Trace ───────────────────────────────────────────────
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if hasattr(model, "module"):
            model = model.module

        log_q.put("⚙  Tracing model with torch.fx...")
        class CustomTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, module_qualified_name):
                name = m.__class__.__name__.lower()
                if any(x in name for x in ["rmsnorm","rms_norm","swiglu","rotary","rope","attention"]):
                    return True
                return super().is_leaf_module(m, module_qualified_name)

        tracer = CustomTracer()
        graph = tracer.trace(model)
        traced_model = torch.fx.GraphModule(model, graph)
        log_q.put(f"✅ Trace successful — {len(list(traced_model.graph.nodes))} nodes")

        # ── Reference outputs ────────────────────────────────────
        with torch.no_grad():
            out = model(sample_input)
            if isinstance(out, torch.Tensor):
                reference_outputs = out.flatten()[:10].tolist()
            else:
                reference_outputs = []

        # ── Thread / state setup ─────────────────────────────────
        start_from = config.get("start_from", "parse_fx")
        thread_id  = f"gui_run_{int(time.time())}"

        initial_state = {
            "thread_id": thread_id,
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
        else:
            import json
            out_dir = Path("output")
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
                with open(out_dir / "ir_graph.json") as f:
                    ir_data = json.load(f)
                    initial_state["ir_graph"] = ir_data
                    initial_state["weights_metadata"] = ir_data.get("weight_metadata", {})
            initial_state["weights_path"]     = str(out_dir / "weights.npz")
            initial_state["weights_bin_path"] = str(out_dir / "weights.bin")

        app        = build_graph(entry_point=start_from)
        run_config = {"configurable": {"thread_id": thread_id}}

        # stash in status_q so main thread can store in session_state
        status_q.put(("init", app, run_config, thread_id))

        log_q.put("🚀 Starting graph execution...")
        for event in app.stream(initial_state, run_config):
            for node_name in event:
                log_q.put(f"✔  Node finished: {node_name}")
                status_q.put(("node", node_name))

        # After stream: check if paused at human_review
        snap = app.get_state(run_config)
        if snap.next and "human_review" in snap.next:
            status_q.put(("paused",))
        else:
            final = app.get_state(run_config).values
            status_q.put(("done", final))

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log_q.put(f"❌ Error: {exc}\n{tb}")
        status_q.put(("error", str(exc)))


def _resume_worker(
    app,
    run_config: dict,
    log_q: queue.Queue,
    status_q: queue.Queue,
):
    """Background thread: resumes graph after human review."""
    try:
        log_q.put("▶  Resuming graph after human review...")
        for event in app.stream(None, run_config):
            for node_name in event:
                log_q.put(f"✔  Node finished: {node_name}")
                status_q.put(("node", node_name))
        final = app.get_state(run_config).values
        status_q.put(("done", final))
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log_q.put(f"❌ Error: {exc}\n{tb}")
        status_q.put(("error", str(exc)))


# ═══════════════════════════════════════════════════════════════
# DRAIN QUEUES — called on every rerender
# ═══════════════════════════════════════════════════════════════

def _drain_queues():
    """Pull all pending messages from log_q and status_q into session state."""
    ss = st.session_state
    changed = False

    # Drain log queue
    while True:
        try:
            line = ss.log_queue.get_nowait()
            ss.log_lines.append(line)
            changed = True
        except queue.Empty:
            break

    # Drain status queue
    status_q: queue.Queue = ss.get("_status_q")
    if status_q is None:
        return changed
    while True:
        try:
            msg = status_q.get_nowait()
            changed = True
            if msg[0] == "init":
                _, app, run_config, thread_id = msg
                ss.app       = app
                ss.run_config = run_config
                ss.thread_id  = thread_id
            elif msg[0] == "node":
                node_name = msg[1]
                if ss.active_node:
                    ss.completed_nodes.append(ss.active_node)
                ss.active_node = node_name
            elif msg[0] == "paused":
                if ss.active_node:
                    ss.completed_nodes.append(ss.active_node)
                    ss.active_node = None
                ss.status = PIPELINE_STATUS_PAUSED
            elif msg[0] == "done":
                if ss.active_node:
                    ss.completed_nodes.append(ss.active_node)
                    ss.active_node = None
                ss.final_state = msg[1]
                ss.status = PIPELINE_STATUS_DONE
            elif msg[0] == "error":
                ss.error_msg = msg[1]
                ss.status = PIPELINE_STATUS_ERROR
        except queue.Empty:
            break

    return changed


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚡ Agentic RISC-V")
    st.markdown("---")

    model_choice = st.selectbox(
        "Model",
        options=["TinyLlama Demo", "Custom Model File"],
        index=0,
        key="sb_model_choice",
        help="Select the PyTorch model to compile",
    )

    uploaded_model = None
    if model_choice == "Custom Model File":
        uploaded_model = st.file_uploader(
            "Upload model Python file",
            type=["py"],
            key="sb_model_file",
        )
        custom_model_name = st.text_input(
            "Model name", value="custom_model", key="sb_model_name"
        )
    else:
        custom_model_name = "tiny_llama_demo"

    st.markdown("---")
    st.markdown("**Pipeline Options**")

    start_from = st.selectbox(
        "Start from stage",
        options=["parse_fx", "generate_code", "verify", "optimize", "simulate", "synthesize"],
        index=0,
        key="sb_start_from",
        help="Useful for resuming a failed run from a specific stage",
    )

    enable_opt = st.checkbox(
        "Enable HW Optimization Pass",
        value=False,
        key="sb_optimize",
        help="Inject hardware-aware hints (e.g. systolic array, VPU) before simulation",
    )

    precision = st.selectbox(
        "Weight Precision",
        options=["f32", "f16", "bf16", "mxfp8"],
        index=0,
        key="sb_precision",
    )

    weight_mode = st.selectbox(
        "Weight Storage Mode",
        options=["embedded", "binary"],
        index=0,
        key="sb_weight_mode",
        help="embedded = bare-metal C arrays; binary = fopen-based binary weights",
    )

    st.markdown("---")
    can_run = (
        st.session_state.status in (PIPELINE_STATUS_IDLE, PIPELINE_STATUS_DONE, PIPELINE_STATUS_ERROR)
        and (model_choice == "TinyLlama Demo" or uploaded_model is not None)
    )

    run_btn = st.button(
        "🚀  Run Pipeline",
        use_container_width=True,
        disabled=not can_run,
    )

    # Status pill
    s = st.session_state.status
    pill_class = {
        PIPELINE_STATUS_IDLE:    "pill-idle",
        PIPELINE_STATUS_RUNNING: "pill-running",
        PIPELINE_STATUS_PAUSED:  "pill-paused",
        PIPELINE_STATUS_DONE:    "pill-done",
        PIPELINE_STATUS_ERROR:   "pill-error",
    }.get(s, "pill-idle")
    pill_label = {
        PIPELINE_STATUS_IDLE:    "● Idle",
        PIPELINE_STATUS_RUNNING: "● Running",
        PIPELINE_STATUS_PAUSED:  "⏸ Human Review",
        PIPELINE_STATUS_DONE:    "✔ Complete",
        PIPELINE_STATUS_ERROR:   "✖ Error",
    }.get(s, "● Idle")
    st.markdown(
        f"<br><div style='text-align:center'><span class='pill {pill_class}'>{pill_label}</span></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.status != PIPELINE_STATUS_IDLE:
        if st.button("🔄  Reset", use_container_width=True):
            for k in ["status","log_lines","completed_nodes","active_node",
                      "app","run_config","thread_id","final_state","error_msg"]:
                del st.session_state[k]
            st.session_state["_status_q"] = None
            _init_session()
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# LAUNCH PIPELINE on button click
# ═══════════════════════════════════════════════════════════════

if run_btn:
    # Reset state
    for k in ["log_lines","completed_nodes","active_node","app",
              "run_config","thread_id","final_state","error_msg"]:
        st.session_state[k] = [] if k in ("log_lines","completed_nodes") else None

    st.session_state.status = PIPELINE_STATUS_RUNNING
    st.session_state.log_queue = queue.Queue()
    status_q = queue.Queue()
    st.session_state["_status_q"] = status_q

    # Load model
    if model_choice == "TinyLlama Demo":
        model, sample_input = create_demo_model()
        m_name = "tiny_llama_demo"
    else:
        # Save uploaded file temporarily
        tmp_path = Path("output") / "_uploaded_model.py"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded_model.read())
        model = _load_model_from_file(str(tmp_path))
        sample_input = torch.randn(1, 10)
        m_name = custom_model_name

    config = {
        "name": m_name,
        "optimize": enable_opt,
        "precision": precision,
        "weight_mode": weight_mode,
        "start_from": start_from,
    }

    t = threading.Thread(
        target=_pipeline_worker,
        args=(model, sample_input, config, st.session_state.log_queue, status_q),
        daemon=True,
    )
    t.start()
    st.session_state.worker_thread = t
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# DRAIN QUEUES each render cycle
# ═══════════════════════════════════════════════════════════════
changed = _drain_queues()


# ═══════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════

# ── Hero Banner ─────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>⚡ Agentic RISC-V Compiler</h1>
        <p>PyTorch → Custom IR → Verified C → Hazard3 RISC-V Simulation → OpenROAD Synthesis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── IDLE ────────────────────────────────────────────────────────
if st.session_state.status == PIPELINE_STATUS_IDLE:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="mlabel">Step 1</div>
                <div style="font-size:1.8rem;margin:6px 0">🔍</div>
                <div style="color:#c0c0e0;font-size:0.85rem">FX Trace + IR Parse</div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="mlabel">Step 2</div>
                <div style="font-size:1.8rem;margin:6px 0">🤖</div>
                <div style="color:#c0c0e0;font-size:0.85rem">LLM Code Generation + Verify</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="mlabel">Step 3</div>
                <div style="font-size:1.8rem;margin:6px 0">🏭</div>
                <div style="color:#c0c0e0;font-size:0.85rem">Sim + Synthesis + Report</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Configure options in the sidebar, then click **Run Pipeline** to start.", icon="ℹ️")


# ── RUNNING ─────────────────────────────────────────────────────
if st.session_state.status in (PIPELINE_STATUS_RUNNING, PIPELINE_STATUS_PAUSED, PIPELINE_STATUS_DONE, PIPELINE_STATUS_ERROR):

    left, right = st.columns([3, 1])

    with right:
        st.markdown("#### 🗺 Pipeline Steps")
        completed = st.session_state.completed_nodes
        active    = st.session_state.active_node
        for node in PIPELINE_NODES_ORDERED:
            if node in completed:
                icon, cls = "✅", "done"
            elif node == active:
                icon, cls = "⚡", "active"
            else:
                icon, cls = "○", "pending"
            st.markdown(
                f"<div class='node-step {cls}'>{icon} <code>{node}</code></div>",
                unsafe_allow_html=True,
            )

    with left:
        st.markdown("#### 📋 Live Logs")
        log_text = "\n".join(st.session_state.log_lines[-200:])  # last 200 lines
        st.markdown(
            f"<div class='log-box'>{log_text if log_text else '(waiting for output...)'}</div>",
            unsafe_allow_html=True,
        )

    # Auto-refresh while running
    if st.session_state.status == PIPELINE_STATUS_RUNNING:
        time.sleep(0.8)
        st.rerun()


# ── HUMAN REVIEW PANEL ──────────────────────────────────────────
if st.session_state.status == PIPELINE_STATUS_PAUSED:
    app        = st.session_state.app
    run_config = st.session_state.run_config

    snap   = app.get_state(run_config)
    cstate = snap.values

    st.markdown("---")
    st.markdown(
        """
        <div class="review-header">
            <h2>⏸ Human Review Required</h2>
            <p>Review the generated C code below, then approve or reject to continue the pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Verification badge
    verif = cstate.get("verification_result", {})
    if verif.get("passed"):
        st.success("✅ Verification PASSED — code compiled and passed static checks.")
    else:
        st.warning("⚠️ Verification did not fully pass. Review warnings below.")

    if verif.get("warnings"):
        with st.expander("⚠️ Verification Warnings"):
            for w in verif["warnings"]:
                st.markdown(f"- {w}")

    # IR Summary
    ir_summary = cstate.get("ir_summary", "")
    if ir_summary:
        with st.expander("📊 IR Graph Summary"):
            st.code(ir_summary, language="text")

    # C file tabs
    model_c    = cstate.get("generated_code", "")
    weights_h  = cstate.get("generated_header", "")
    model_h    = cstate.get("generated_model_header", "")

    tab_labels = []
    if model_c:   tab_labels.append("model.c")
    if weights_h: tab_labels.append("weights.h")
    if model_h:   tab_labels.append("model.h")

    if tab_labels:
        tabs = st.tabs(tab_labels)
        tab_idx = 0
        if model_c:
            with tabs[tab_idx]:
                st.markdown(f"**Lines:** {len(model_c.splitlines())}  |  **Size:** {len(model_c.encode())//1024} KB")
                st.code(model_c, language="c")
            tab_idx += 1
        if weights_h:
            with tabs[tab_idx]:
                st.markdown(f"**Lines:** {len(weights_h.splitlines())}  |  **Size:** {len(weights_h.encode())//1024} KB")
                st.code(weights_h, language="c")
            tab_idx += 1
        if model_h:
            with tabs[tab_idx]:
                st.markdown(f"**Lines:** {len(model_h.splitlines())}  |  **Size:** {len(model_h.encode())//1024} KB")
                st.code(model_h, language="c")
    else:
        st.info("No generated code files found in state.")

    st.markdown("---")
    st.markdown("#### Action")

    col_a, col_r, col_v = st.columns([1, 1, 1])

    with col_a:
        if st.button("✅  Approve", use_container_width=True, key="btn_approve", type="primary"):
            app.update_state(run_config, {"human_approved": True, "human_action": "approve"})
            st.session_state.status = PIPELINE_STATUS_RUNNING
            st.session_state.log_lines.append("👤 Human approved — resuming pipeline...")

            status_q = queue.Queue()
            st.session_state["_status_q"] = status_q
            t = threading.Thread(
                target=_resume_worker,
                args=(app, run_config, st.session_state.log_queue, status_q),
                daemon=True,
            )
            t.start()
            st.session_state.worker_thread = t
            st.rerun()

    with col_r:
        with st.expander("❌  Reject / Send Feedback"):
            feedback = st.text_area(
                "Feedback for Code Generator",
                placeholder="e.g. The matmul loop is not unrolled. Please unroll by factor 4.",
                key="review_feedback",
                height=100,
            )
            if st.button("Submit Rejection", key="btn_reject"):
                app.update_state(run_config, {
                    "human_approved": False,
                    "human_feedback": feedback,
                    "human_action": "retry",
                    "verification_attempts": 0,
                    "verification_exhausted": False,
                })
                st.session_state.status = PIPELINE_STATUS_RUNNING
                st.session_state.log_lines.append(f"👤 Human rejected — feedback: {feedback[:60]}...")

                status_q = queue.Queue()
                st.session_state["_status_q"] = status_q
                t = threading.Thread(
                    target=_resume_worker,
                    args=(app, run_config, st.session_state.log_queue, status_q),
                    daemon=True,
                )
                t.start()
                st.session_state.worker_thread = t
                st.rerun()

    with col_v:
        if st.button("🔁  Re-verify", use_container_width=True, key="btn_reverify"):
            app.update_state(run_config, {
                "human_approved": False,
                "human_action": "verify",
                "verification_attempts": 0,
                "verification_exhausted": False,
            })
            st.session_state.status = PIPELINE_STATUS_RUNNING
            st.session_state.log_lines.append("👤 Human requested re-verification...")

            status_q = queue.Queue()
            st.session_state["_status_q"] = status_q
            t = threading.Thread(
                target=_resume_worker,
                args=(app, run_config, st.session_state.log_queue, status_q),
                daemon=True,
            )
            t.start()
            st.session_state.worker_thread = t
            st.rerun()


# ── ERROR ────────────────────────────────────────────────────────
if st.session_state.status == PIPELINE_STATUS_ERROR:
    st.markdown("---")
    st.error(f"**Pipeline failed:** {st.session_state.error_msg}")


# ── FINAL REPORT ─────────────────────────────────────────────────
if st.session_state.status == PIPELINE_STATUS_DONE and st.session_state.final_state:
    fs = st.session_state.final_state
    report_md = fs.get("final_report", "")

    st.markdown("---")
    st.markdown("## 🎉 Pipeline Complete!")

    # Quick metrics row
    sim   = fs.get("simulation_result", {})
    synth = fs.get("synthesis_result", {})
    tok_in  = fs.get("total_input_tokens",  0) or 0
    tok_out = fs.get("total_output_tokens", 0) or 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Sim Cycles",     f"{sim.get('cycles', 'N/A'):,}" if isinstance(sim.get('cycles'), int) else "N/A")
    c2.metric("🔋 Power",          f"{synth.get('power_watts', 0):.4f} W" if synth.get("success") else "N/A")
    c3.metric("📐 Area",           f"{synth.get('area_mm2', 0):.4f} mm²" if synth.get("success") else "N/A")
    c4.metric("🪙 Total Tokens",   f"{tok_in + tok_out:,}")

    st.markdown("---")

    if report_md:
        st.markdown('<div class="report-panel">', unsafe_allow_html=True)
        st.markdown(report_md)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="⬇️  Download Report (Markdown)",
            data=report_md,
            file_name="riscv_compiler_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.info("Report not generated yet.")


#graph.py#
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


#ir.py#
"""
Custom Hardware-Friendly Intermediate Representation (IR).

Bridges PyTorch FX graph semantics to C code generation.
Supports both CNN operations (Conv2D, Pool, etc.) and
LLM operations (Attention, RMSNorm, RoPE, SiLU) for TinyLlama.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import math


class IROpType(str, Enum):
    """All supported IR operation types."""

    # ── Tensor I/O ──────────────────────────────────────────────
    TENSOR_INPUT = "TENSOR_INPUT"
    TENSOR_OUTPUT = "TENSOR_OUTPUT"

    # ── CNN Operations ──────────────────────────────────────────
    CONV2D = "CONV2D"
    RELU = "RELU"
    BATCHNORM = "BATCHNORM"
    MAXPOOL2D = "MAXPOOL2D"
    AVGPOOL2D = "AVGPOOL2D"
    FLATTEN = "FLATTEN"

    # ── Common Operations ───────────────────────────────────────
    LINEAR = "LINEAR"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    SQRT = "SQRT"
    MEAN = "MEAN"
    SOFTMAX = "SOFTMAX"
    DROPOUT = "DROPOUT"          # No-op in inference

    # ── LLM Operations (for TinyLlama / LLaMA-style models) ────
    EMBEDDING = "EMBEDDING"           # Token embedding lookup
    RMSNORM = "RMSNORM"               # Root Mean Square Layer Norm
    LAYERNORM = "LAYERNORM"           # Standard Layer Norm
    ATTENTION = "ATTENTION"           # Multi-head self-attention block
    MATMUL = "MATMUL"                 # Raw matrix multiplication
    SILU = "SILU"                     # SiLU activation (x * sigmoid(x))
    SWIGLU = "SWIGLU"                 # SwiGLU gating: SiLU(xW1) * (xW2)
    ROTARY_EMBEDDING = "ROTARY_EMBEDDING"  # Rotary Positional Encoding
    RESHAPE = "RESHAPE"               # Tensor reshape
    TRANSPOSE = "TRANSPOSE"           # Tensor transpose/permute
    SPLIT = "SPLIT"                   # Tensor split (for Q/K/V)
    CONCAT = "CONCAT"                 # Tensor concatenation

    # ── Quantization ────────────────────────────────────────────
    QUANTIZE = "QUANTIZE"             # Float → Int8/Int4
    DEQUANTIZE = "DEQUANTIZE"         # Int8/Int4 → Float


# ── Default parameters for each op type ─────────────────────────
OP_DEFAULT_PARAMS: dict[str, dict] = {
    IROpType.CONV2D: {
        "kernel_size": [3, 3], "stride": [1, 1],
        "padding": [0, 0], "groups": 1, "bias": True
    },
    IROpType.LINEAR: {"bias": True},
    IROpType.MAXPOOL2D: {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
    IROpType.AVGPOOL2D: {"output_size": [1, 1]},
    IROpType.ATTENTION: {
        "num_heads": 4, "head_dim": 64, "causal": True
    },
    IROpType.EMBEDDING: {"num_embeddings": 32000, "embedding_dim": 2048},
    IROpType.RMSNORM: {"eps": 1e-5},
    IROpType.LAYERNORM: {"eps": 1e-5},
    IROpType.ROTARY_EMBEDDING: {"max_seq_len": 2048, "base": 10000.0},
    IROpType.SOFTMAX: {"dim": -1},
    IROpType.SUB: {},
    IROpType.DIV: {},
    IROpType.SQRT: {},
    IROpType.MEAN: {"dim": -1, "keepdim": True},
    IROpType.FLATTEN: {"start_dim": 1, "end_dim": -1},
    IROpType.RESHAPE: {"target_shape": []},
    IROpType.TRANSPOSE: {"dim0": 0, "dim1": 1},
    IROpType.SPLIT: {"num_splits": 3, "dim": -1},
    IROpType.CONCAT: {"dim": -1},
    IROpType.QUANTIZE: {"bits": 8, "scheme": "symmetric"},
    IROpType.DEQUANTIZE: {"bits": 8, "scheme": "symmetric"},
}


@dataclass
class IRNode:
    """A single operation node in the IR graph."""

    id: str                                   # Unique node identifier
    op: str                                   # IROpType value
    inputs: list[str] = field(default_factory=list)  # Input node IDs
    params: dict[str, Any] = field(default_factory=dict)  # Op-specific parameters
    shape: tuple = ()                         # Output tensor shape
    dtype: str = "float32"                    # Output data type
    weight_key: str = ""                      # Key into weights dict (if any)
    bias_key: str = ""                        # Key into weights dict for bias

    def num_elements(self) -> int:
        """Number of elements in the output tensor."""
        if not self.shape:
            return 0
        result = 1
        for s in self.shape:
            result *= s
        return result

    def memory_bytes(self) -> int:
        """Memory footprint of the output tensor in bytes."""
        dtype_sizes = {"float32": 4, "float16": 2, "int8": 1, "int4": 1}
        return self.num_elements() * dtype_sizes.get(self.dtype, 4)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage in LangGraph state."""
        return {
            "id": self.id,
            "op": self.op,
            "inputs": self.inputs,
            "params": self.params,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "weight_key": self.weight_key,
            "bias_key": self.bias_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IRNode:
        """Deserialize from dictionary."""
        return cls(
            id=d["id"],
            op=d["op"],
            inputs=d.get("inputs", []),
            params=d.get("params", {}),
            shape=tuple(d.get("shape", ())),
            dtype=d.get("dtype", "float32"),
            weight_key=d.get("weight_key", ""),
            bias_key=d.get("bias_key", ""),
        )


@dataclass
class IRGraph:
    """
    The complete IR graph — a DAG of IRNodes.

    Designed to be serializable (for LangGraph state) and
    pretty-printable (for LLM consumption).
    """

    nodes: list[IRNode] = field(default_factory=list)
    input_shapes: dict[str, tuple] = field(default_factory=dict)
    weight_metadata: dict[str, dict] = field(default_factory=dict)
    model_name: str = ""

    # ── Graph Operations ────────────────────────────────────────

    def add_node(self, node: IRNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def get_node(self, node_id: str) -> IRNode | None:
        """Look up a node by ID."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def topological_order(self) -> list[IRNode]:
        """Return nodes in topological order (inputs before consumers)."""
        id_to_node = {n.id: n for n in self.nodes}
        visited: set[str] = set()
        order: list[IRNode] = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = id_to_node.get(node_id)
            if node is None:
                return
            for inp in node.inputs:
                visit(inp)
            order.append(node)

        for n in self.nodes:
            visit(n.id)
        return order

    def validate(self) -> list[str]:
        """Validate the graph structure. Returns list of issues."""
        issues = []
        node_ids = {n.id for n in self.nodes}

        for node in self.nodes:
            # Check inputs exist
            for inp in node.inputs:
                if inp not in node_ids:
                    issues.append(
                        f"Node '{node.id}' references unknown input '{inp}'"
                    )

            # Check op type is valid
            try:
                IROpType(node.op)
            except ValueError:
                issues.append(
                    f"Node '{node.id}' has unknown op type '{node.op}'"
                )

            # Check shapes are non-empty (except for output nodes)
            if node.op != IROpType.TENSOR_OUTPUT and not node.shape:
                issues.append(
                    f"Node '{node.id}' ({node.op}) has empty shape"
                )

        return issues

    # ── Memory Estimation ───────────────────────────────────────

    def total_activation_memory(self) -> int:
        """Total activation memory (all intermediate tensors)."""
        return sum(n.memory_bytes() for n in self.nodes)

    def total_weight_memory(self) -> int:
        """Total weight memory from metadata."""
        total = 0
        for meta in self.weight_metadata.values():
            numel = 1
            for s in meta.get("shape", []):
                numel *= s
            dtype_size = {"float32": 4, "float16": 2, "int8": 1}.get(
                meta.get("dtype", "float32"), 4
            )
            total += numel * dtype_size
        return total

    def total_params(self) -> int:
        """Total number of parameters."""
        total = 0
        for meta in self.weight_metadata.values():
            numel = 1
            for s in meta.get("shape", []):
                numel *= s
            total += numel
        return total

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the entire graph to a dictionary."""
        return {
            "model_name": self.model_name,
            "nodes": [n.to_dict() for n in self.nodes],
            "input_shapes": {
                k: list(v) for k, v in self.input_shapes.items()
            },
            "weight_metadata": self.weight_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IRGraph:
        """Deserialize from dictionary."""
        return cls(
            model_name=d.get("model_name", ""),
            nodes=[IRNode.from_dict(n) for n in d.get("nodes", [])],
            input_shapes={
                k: tuple(v) for k, v in d.get("input_shapes", {}).items()
            },
            weight_metadata=d.get("weight_metadata", {}),
        )

    # ── Pretty Printing ────────────────────────────────────────

    def pretty_print(self) -> str:
        """
        Format the IR graph for LLM consumption.

        Produces a clear, structured text representation
        that the code generation LLM can parse.
        """
        lines = []
        lines.append(f"# IR Graph: {self.model_name}")
        lines.append(f"# Total Parameters: {self.total_params():,}")
        lines.append(
            f"# Weight Memory: {self.total_weight_memory() / 1024:.1f} KB"
        )
        lines.append(
            f"# Activation Memory: "
            f"{self.total_activation_memory() / 1024:.1f} KB"
        )
        lines.append("")

        # Input shapes
        lines.append("## Inputs:")
        for name, shape in self.input_shapes.items():
            lines.append(f"  {name}: shape={shape}")
        lines.append("")

        # Node list
        lines.append("## Operations (topological order):")
        for node in self.topological_order():
            inputs_str = ", ".join(node.inputs) if node.inputs else "none"
            line = (
                f"  [{node.id}] {node.op}"
                f"  inputs=({inputs_str})"
                f"  shape={node.shape}"
            )
            if node.weight_key:
                wmeta = self.weight_metadata.get(node.weight_key, {})
                wshape = wmeta.get("shape", "?")
                line += f"  weight={node.weight_key}{wshape}"
            if node.bias_key:
                line += f"  bias={node.bias_key}"
            if node.params:
                # Show non-default params only
                param_str = ", ".join(
                    f"{k}={v}" for k, v in node.params.items()
                )
                line += f"  params={{{param_str}}}"
            lines.append(line)

        lines.append("")

        # Weight summary table
        lines.append("## Weight Tensors:")
        for name, meta in self.weight_metadata.items():
            shape = meta.get("shape", "?")
            dtype = meta.get("dtype", "float32")
            numel = meta.get("numel", "?")
            lines.append(f"  {name}: shape={shape} dtype={dtype} numel={numel}")

        return "\n".join(lines)

    def layer_summary(self) -> str:
        """Concise layer summary for human review."""
        lines = [f"Model: {self.model_name}", ""]

        op_counts: dict[str, int] = {}
        for node in self.nodes:
            op_counts[node.op] = op_counts.get(node.op, 0) + 1

        lines.append("Layer counts:")
        for op, count in sorted(op_counts.items()):
            lines.append(f"  {op}: {count}")

        lines.append(f"\nTotal parameters: {self.total_params():,}")
        lines.append(
            f"Weight memory: {self.total_weight_memory() / (1024*1024):.2f} MB"
        )
        lines.append(
            f"Activation memory: "
            f"{self.total_activation_memory() / 1024:.1f} KB"
        )

        return "\n".join(lines)


# ── C Code Pattern Hints ────────────────────────────────────────
# These help the code generator understand what each op maps to in C.

C_PATTERNS: dict[str, str] = {
    IROpType.TENSOR_INPUT: "float* {id}  /* function parameter */",
    IROpType.TENSOR_OUTPUT: "return {id};",
    IROpType.CONV2D: (
        "for (oc) for (oh) for (ow) for (ic) for (kh) for (kw)\n"
        "  out[oc][oh][ow] += in[ic][ih+kh][iw+kw] * weight[oc][ic][kh][kw];"
    ),
    IROpType.LINEAR: (
        "for (i) for (j) out[i] += in[j] * weight[i][j];\n"
        "out[i] += bias[i];"
    ),
    IROpType.RELU: "out[i] = (in[i] > 0) ? in[i] : 0;",
    IROpType.SILU: "out[i] = in[i] * (1.0f / (1.0f + expf(-in[i])));",
    IROpType.RMSNORM: (
        "float rms = sqrt(mean(x[i]*x[i]) + eps);\n"
        "out[i] = (x[i] / rms) * weight[i];"
    ),
    IROpType.EMBEDDING: "for (i) out[i] = embed_table[token_id][i];",
    IROpType.ATTENTION: (
        "Q = X @ Wq; K = X @ Wk; V = X @ Wv;\n"
        "scores = (Q @ K^T) / sqrt(head_dim);\n"
        "if (causal) apply_causal_mask(scores);\n"
        "attn = softmax(scores);\n"
        "out = attn @ V;"
    ),
    IROpType.MATMUL: "for (i) for (j) for (k) C[i][j] += A[i][k] * B[k][j];",
    IROpType.SOFTMAX: (
        "float max_val = max(x);\n"
        "float sum = 0;\n"
        "for (i) { x[i] = expf(x[i] - max_val); sum += x[i]; }\n"
        "for (i) x[i] /= sum;"
    ),
    IROpType.ROTARY_EMBEDDING: (
        "for (i in 0..head_dim/2):\n"
        "  cos_θ = cos(pos * freq[i]);\n"
        "  sin_θ = sin(pos * freq[i]);\n"
        "  out[2i]   = x[2i]*cos_θ - x[2i+1]*sin_θ;\n"
        "  out[2i+1] = x[2i]*sin_θ + x[2i+1]*cos_θ;"
    ),
    IROpType.SUB: "out[i] = in0[i] - in1[i];",
    IROpType.DIV: "out[i] = in0[i] / in1[i];",
    IROpType.SQRT: "out[i] = sqrtf(in[i]);",
    IROpType.MEAN: "out[0] = mean(in);",

}


#state.py#
"""
Shared LangGraph state definition for the Agentic RISC-V Compiler pipeline.

All agents read from and write to this shared TypedDict state.
LangGraph merges partial updates automatically.

Non-serializable PyTorch objects (nn.Module, GraphModule, Tensor) are
intentionally kept OUT of the LangGraph state to avoid msgpack serialisation
errors at checkpoint boundaries.  They live in a module-level object store
(pytorch_object_store) keyed by a thread_id string that IS stored in state.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


# ── Out-of-band store for non-serializable PyTorch objects ────────
# LangGraph's MemorySaver uses msgpack to checkpoint state before every
# interrupt.  PyTorch objects (nn.Module, GraphModule, Tensor) are NOT
# msgpack-serializable, so we keep them here and reference them by key.
pytorch_object_store: dict[str, dict[str, Any]] = {}


def store_pytorch_objects(thread_id: str, model: Any, fx_graph: Any, sample_input: Any) -> None:
    """Store non-serializable PyTorch objects keyed by thread_id."""
    pytorch_object_store[thread_id] = {
        "model": model,
        "fx_graph": fx_graph,
        "sample_input": sample_input,
    }


def retrieve_pytorch_objects(thread_id: str) -> dict[str, Any]:
    """Retrieve stored PyTorch objects for a thread_id (returns empty dict if not found)."""
    return pytorch_object_store.get(thread_id, {})


def clear_pytorch_objects(thread_id: str) -> None:
    """Remove PyTorch objects from the store once they have been consumed."""
    pytorch_object_store.pop(thread_id, None)


class VerificationResult(TypedDict, total=False):
    """Structured result from the verification agent."""
    passed: bool
    errors: list[str]
    warnings: list[str]
    compiler_output: str


class SimulationResult(TypedDict, total=False):
    """Structured result from the Hazard3 simulation agent."""
    success: bool
    cycles: int
    execution_trace: str
    output_values: list[float]
    output_match: bool
    raw_log: str


class SynthesisResult(TypedDict, total=False):
    """Structured result from the OpenROAD synthesis agent."""
    success: bool
    power_watts: float
    area_mm2: float
    frequency_mhz: float
    cell_count: int
    detailed_report: str


class LLMCallStats(TypedDict, total=False):
    """Token and latency stats for a single LLM call."""
    agent: str           # Which agent made the call
    call_label: str      # E.g. "model.h", "model.c", "optimize"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_s: float     # Wall-clock seconds for the LLM call


class AgentState(TypedDict, total=False):
    """
    Shared state for the LangGraph pipeline.

    All agents receive the full state and return partial updates.
    LangGraph handles merging.

    NOTE: Non-serializable PyTorch objects (nn.Module, GraphModule, Tensor)
    are intentionally absent.  Use pytorch_object_store + thread_id instead.
    """

    # ── Thread identity (used to look up pytorch_object_store) ──
    thread_id: str                           # Matches run_config thread_id

    # ── Input ───────────────────────────────────────────────────
    model_name: str                          # Human-readable model name
    fx_graph_str: str                        # String representation of FX graph
    sample_input_shape: list                 # Shape of the sample input tensor

    # ── IR (FX Parser → Code Generator) ────────────────────────
    ir_graph: dict                           # Serialized IRGraph (dict form)
    ir_summary: str                          # Human-readable layer summary

    # ── Weights ─────────────────────────────────────────────────
    weights_metadata: dict[str, dict]        # param_name → {shape, dtype, numel}
    weights_path: str                        # Path to saved .npz weights file
    total_params: int                        # Total parameter count
    model_memory_bytes: int                  # Estimated memory footprint
    weights_bin_path: str                    # Path to saved weights.bin binary file
    weights_manifest: dict[str, dict]        # param_name → {c_name, offset, size_bytes, numel, shape, c_type, precision}
    weight_precision: str                    # Precision mode: 'f32', 'f16', 'bf16', 'mxfp8'
    weight_mode: str                         # 'embedded' (bare-metal) or 'binary' (hosted)

    # ── Generated Code (Code Generator → Verifier) ─────────────
    generated_code: str                      # model.c content
    generated_header: str                    # weights.h content
    generated_model_header: str              # model.h content
    code_path: str                           # Path to saved model.c
    header_path: str                         # Path to saved weights.h
    model_header_path: str                   # Path to saved model.h

    # ── Verification ────────────────────────────────────────────
    verification_result: VerificationResult
    verification_attempts: int               # Counter (max 5)
    verification_feedback: str               # Formatted feedback for LLM
    verification_exhausted: bool             # True if max verification attempts reached

    # ── Human-in-the-loop ───────────────────────────────────────
    human_approved: bool                     # True if user approved
    human_feedback: str                      # Optional user comments
    human_action: str                        # 'approve', 'retry', 'verify', 'quit'

    # ── Simulation (Hazard3) ────────────────────────────────────
    simulation_result: SimulationResult
    reference_outputs: list[float]           # PyTorch reference output values

    # ── Synthesis (OpenROAD) ────────────────────────────────────
    synthesis_result: SynthesisResult

    # ── Optimization Loop ───────────────────────────────────────
    optimization_suggestions: list[str]      # List of optimization suggestions
    optimization_iteration: int              # Current iteration (max 3)
    enable_optimization: bool                # Toggle optimization loop

    # ── Final Report ────────────────────────────────────────────
    final_report: str                        # Markdown report

    # ── Error Handling ──────────────────────────────────────────
    error: Optional[str]                     # Fatal error message

    # ── Token & Latency Telemetry ───────────────────────────────
    llm_call_stats: list[LLMCallStats]       # Per-call token/latency records
    total_input_tokens: int                  # Running total input tokens
    total_output_tokens: int                 # Running total output tokens
    total_llm_latency_s: float               # Running total GPU/LLM wall-clock seconds
    agent_latencies: dict[str, float]        # agent_name → total wall-clock seconds


#visualize_ir.py#
"""
IR Visualization Utility — Render the Custom IR graph visually.

Supports multiple output formats:
1. Terminal ASCII (always available)
2. Mermaid diagram (copy-paste into any Mermaid renderer)
3. Graphviz DOT (requires graphviz installed)
4. HTML (self-contained, opens in browser)

Usage:
    # From Python
    from visualize_ir import visualize
    visualize(ir_graph, format="html", output="ir_graph.html")

    # From command line
    python visualize_ir.py output/ir_graph.json --format html
    python visualize_ir.py output/ir_graph.json --format mermaid
    python visualize_ir.py output/ir_graph.json --format terminal
    python visualize_ir.py output/ir_graph.json --format dot
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from pathlib import Path

from ir import IRGraph, IRNode, IROpType


# ── Color scheme for different op types ─────────────────────────
OP_COLORS = {
    # I/O
    IROpType.TENSOR_INPUT: {"bg": "#4CAF50", "text": "#fff", "mermaid": ":::input"},
    IROpType.TENSOR_OUTPUT: {"bg": "#F44336", "text": "#fff", "mermaid": ":::output"},

    # CNN ops
    IROpType.CONV2D: {"bg": "#2196F3", "text": "#fff"},
    IROpType.MAXPOOL2D: {"bg": "#03A9F4", "text": "#fff"},
    IROpType.AVGPOOL2D: {"bg": "#03A9F4", "text": "#fff"},
    IROpType.BATCHNORM: {"bg": "#00BCD4", "text": "#fff"},
    IROpType.FLATTEN: {"bg": "#607D8B", "text": "#fff"},

    # Common ops
    IROpType.LINEAR: {"bg": "#9C27B0", "text": "#fff"},
    IROpType.RELU: {"bg": "#FF9800", "text": "#fff"},
    IROpType.SOFTMAX: {"bg": "#FF5722", "text": "#fff"},
    IROpType.ADD: {"bg": "#795548", "text": "#fff"},
    IROpType.MUL: {"bg": "#795548", "text": "#fff"},
    IROpType.DROPOUT: {"bg": "#9E9E9E", "text": "#fff"},

    # LLM ops
    IROpType.EMBEDDING: {"bg": "#E91E63", "text": "#fff"},
    IROpType.RMSNORM: {"bg": "#00BCD4", "text": "#fff"},
    IROpType.LAYERNORM: {"bg": "#00BCD4", "text": "#fff"},
    IROpType.ATTENTION: {"bg": "#673AB7", "text": "#fff"},
    IROpType.MATMUL: {"bg": "#3F51B5", "text": "#fff"},
    IROpType.SILU: {"bg": "#FF9800", "text": "#fff"},
    IROpType.SWIGLU: {"bg": "#FF9800", "text": "#fff"},
    IROpType.ROTARY_EMBEDDING: {"bg": "#009688", "text": "#fff"},
    IROpType.RESHAPE: {"bg": "#607D8B", "text": "#fff"},
    IROpType.TRANSPOSE: {"bg": "#607D8B", "text": "#fff"},
    IROpType.SPLIT: {"bg": "#607D8B", "text": "#fff"},
    IROpType.CONCAT: {"bg": "#607D8B", "text": "#fff"},

    # Quantization
    IROpType.QUANTIZE: {"bg": "#FFEB3B", "text": "#000"},
    IROpType.DEQUANTIZE: {"bg": "#FFEB3B", "text": "#000"},
}

DEFAULT_COLOR = {"bg": "#9E9E9E", "text": "#fff"}


# ═══════════════════════════════════════════════════════════════
# 1. TERMINAL ASCII VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def to_terminal(ir_graph: IRGraph) -> str:
    """
    Render the IR graph as a terminal-friendly ASCII diagram.

    Example output:
    ┌────────────────────────────────────────┐
    │ [input_0] TENSOR_INPUT                 │
    │ shape: (1, 32)  dtype: float32         │
    └──────────────────┬─────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────┐
    │ [embed] EMBEDDING                      │
    │ shape: (1, 32, 64)  weight: embed.wt   │
    └──────────────────┬─────────────────────┘
    """
    lines = []
    nodes = ir_graph.topological_order()
    box_width = 52

    lines.append(f"  IR Graph: {ir_graph.model_name}")
    lines.append(f"  Params: {ir_graph.total_params():,}  "
                 f"Nodes: {len(nodes)}")
    lines.append("")

    for i, node in enumerate(nodes):
        # Build content lines for this node
        title = f"[{node.id}] {node.op}"
        detail = f"shape: {node.shape}  dtype: {node.dtype}"

        extras = []
        if node.weight_key:
            extras.append(f"weight: {node.weight_key}")
        if node.bias_key:
            extras.append(f"bias: {node.bias_key}")
        if node.inputs:
            extras.append(f"inputs: {', '.join(node.inputs)}")

        # Compute box width
        content_lines = [title, detail] + extras
        max_content = max(len(l) for l in content_lines)
        w = max(box_width, max_content + 4)

        # Draw box
        lines.append("    ┌" + "─" * w + "┐")
        for cl in content_lines:
            padding = w - len(cl)
            lines.append(f"    │ {cl}" + " " * (padding - 1) + "│")
        lines.append("    └" + "─" * (w // 2) + "┬" + "─" * (w - w // 2 - 1) + "┘")

        # Draw connector to next node
        if i < len(nodes) - 1:
            lines.append(" " * (5 + w // 2) + "│")
            lines.append(" " * (5 + w // 2) + "▼")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 2. MERMAID DIAGRAM
# ═══════════════════════════════════════════════════════════════

def to_mermaid(ir_graph: IRGraph) -> str:
    """
    Render the IR graph as a Mermaid flowchart.

    Can be pasted into:
    - GitHub markdown (```mermaid ... ```)
    - mermaid.live
    - Any Mermaid-compatible renderer
    """
    lines = []
    lines.append("graph TD")

    # Define style classes
    lines.append("    classDef input fill:#4CAF50,stroke:#2E7D32,color:#fff")
    lines.append("    classDef output fill:#F44336,stroke:#C62828,color:#fff")
    lines.append("    classDef compute fill:#2196F3,stroke:#1565C0,color:#fff")
    lines.append("    classDef activation fill:#FF9800,stroke:#E65100,color:#fff")
    lines.append("    classDef norm fill:#00BCD4,stroke:#00838F,color:#fff")
    lines.append("    classDef attention fill:#673AB7,stroke:#4527A0,color:#fff")
    lines.append("    classDef embedding fill:#E91E63,stroke:#AD1457,color:#fff")
    lines.append("    classDef reshape fill:#607D8B,stroke:#37474F,color:#fff")
    lines.append("")

    # Map ops to style classes
    op_class_map = {
        IROpType.TENSOR_INPUT: "input",
        IROpType.TENSOR_OUTPUT: "output",
        IROpType.CONV2D: "compute",
        IROpType.LINEAR: "compute",
        IROpType.MATMUL: "compute",
        IROpType.RELU: "activation",
        IROpType.SILU: "activation",
        IROpType.SWIGLU: "activation",
        IROpType.SOFTMAX: "activation",
        IROpType.RMSNORM: "norm",
        IROpType.LAYERNORM: "norm",
        IROpType.BATCHNORM: "norm",
        IROpType.ATTENTION: "attention",
        IROpType.EMBEDDING: "embedding",
        IROpType.ROTARY_EMBEDDING: "embedding",
        IROpType.FLATTEN: "reshape",
        IROpType.RESHAPE: "reshape",
        IROpType.TRANSPOSE: "reshape",
        IROpType.SPLIT: "reshape",
        IROpType.CONCAT: "reshape",
    }

    nodes = ir_graph.topological_order()

    # Define nodes
    for node in nodes:
        safe_id = node.id.replace(".", "_").replace("-", "_")
        shape_str = str(node.shape) if node.shape else "?"
        label = f"{node.op}\\n{node.id}\\nshape={shape_str}"

        if node.weight_key:
            label += f"\\nweight={node.weight_key}"

        # Choose node shape based on type
        if node.op == IROpType.TENSOR_INPUT:
            lines.append(f'    {safe_id}(["{label}"])')
        elif node.op == IROpType.TENSOR_OUTPUT:
            lines.append(f'    {safe_id}[["{label}"]]')
        elif node.op in (IROpType.ADD, IROpType.MUL):
            lines.append(f'    {safe_id}{{"{label}"}}')
        else:
            lines.append(f'    {safe_id}["{label}"]')

        # Apply style class
        cls = op_class_map.get(node.op, "")
        if cls:
            lines.append(f"    class {safe_id} {cls}")

    lines.append("")

    # Define edges
    for node in nodes:
        safe_id = node.id.replace(".", "_").replace("-", "_")
        for inp in node.inputs:
            safe_inp = inp.replace(".", "_").replace("-", "_")
            lines.append(f"    {safe_inp} --> {safe_id}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 3. GRAPHVIZ DOT
# ═══════════════════════════════════════════════════════════════

def to_dot(ir_graph: IRGraph) -> str:
    """
    Render the IR graph as a Graphviz DOT diagram.

    Usage:
        dot -Tpng ir_graph.dot -o ir_graph.png
        dot -Tsvg ir_graph.dot -o ir_graph.svg
    """
    lines = []
    lines.append(f'digraph "{ir_graph.model_name}" {{')
    lines.append('    rankdir=TB;')
    lines.append('    node [fontname="Helvetica", fontsize=10];')
    lines.append('    edge [color="#666666"];')
    lines.append('')

    nodes = ir_graph.topological_order()

    for node in nodes:
        safe_id = node.id.replace(".", "_").replace("-", "_")
        color_info = OP_COLORS.get(node.op, DEFAULT_COLOR)
        bg = color_info["bg"]
        text = color_info["text"]

        shape_str = str(node.shape) if node.shape else "?"

        label_parts = [node.op, node.id, f"shape={shape_str}"]
        if node.weight_key:
            label_parts.append(f"wt={node.weight_key}")

        label = "\\n".join(label_parts)

        # Node shape based on type
        if node.op == IROpType.TENSOR_INPUT:
            dot_shape = "oval"
        elif node.op == IROpType.TENSOR_OUTPUT:
            dot_shape = "doubleoctagon"
        elif node.op in (IROpType.ADD, IROpType.MUL):
            dot_shape = "diamond"
        else:
            dot_shape = "box"

        lines.append(
            f'    {safe_id} ['
            f'label="{label}", '
            f'shape={dot_shape}, '
            f'style=filled, '
            f'fillcolor="{bg}", '
            f'fontcolor="{text}"'
            f'];'
        )

    lines.append('')

    # Edges
    for node in nodes:
        safe_id = node.id.replace(".", "_").replace("-", "_")
        for inp in node.inputs:
            safe_inp = inp.replace(".", "_").replace("-", "_")
            lines.append(f'    {safe_inp} -> {safe_id};')

    lines.append('}')
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 4. HTML VISUALIZATION (self-contained, interactive)
# ═══════════════════════════════════════════════════════════════

def to_html(ir_graph: IRGraph) -> str:
    """
    Render an interactive HTML visualization using embedded Mermaid.js.

    Opens in any browser — no dependencies needed.
    """
    mermaid_code = to_mermaid(ir_graph)
    summary = ir_graph.layer_summary()
    pretty = ir_graph.pretty_print()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IR Graph — {ir_graph.model_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            min-height: 100vh;
        }}

        .header {{
            background: linear-gradient(135deg, #1a1a3e 0%, #0f0f23 100%);
            border-bottom: 1px solid #2a2a5a;
            padding: 24px 40px;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}

        .header .subtitle {{
            color: #8888aa;
            font-size: 14px;
        }}

        .container {{
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 0;
            min-height: calc(100vh - 90px);
        }}

        .graph-panel {{
            padding: 30px;
            overflow: auto;
            background: #12122a;
        }}

        .graph-panel .mermaid {{
            display: flex;
            justify-content: center;
        }}

        .info-panel {{
            background: #1a1a3e;
            border-left: 1px solid #2a2a5a;
            padding: 24px;
            overflow-y: auto;
        }}

        .info-panel h2 {{
            font-size: 16px;
            font-weight: 600;
            color: #a78bfa;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .info-section {{
            margin-bottom: 28px;
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}

        .stat-card {{
            background: #22224a;
            border-radius: 8px;
            padding: 14px;
            border: 1px solid #2a2a5a;
        }}

        .stat-card .label {{
            font-size: 11px;
            color: #8888aa;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-card .value {{
            font-size: 20px;
            font-weight: 700;
            color: #60a5fa;
            margin-top: 4px;
        }}

        .op-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .op-table th {{
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #2a2a5a;
            color: #8888aa;
            font-weight: 500;
        }}

        .op-table td {{
            padding: 8px;
            border-bottom: 1px solid #1a1a2e;
        }}

        .op-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}

        pre.ir-text {{
            background: #0a0a1a;
            border: 1px solid #2a2a5a;
            border-radius: 8px;
            padding: 16px;
            font-size: 11px;
            font-family: 'Cascadia Code', 'Fira Code', monospace;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 400px;
            overflow-y: auto;
            color: #b0b0d0;
        }}

        .tabs {{
            display: flex;
            gap: 4px;
            margin-bottom: 12px;
        }}

        .tab {{
            padding: 6px 16px;
            border-radius: 6px;
            border: 1px solid #2a2a5a;
            background: transparent;
            color: #8888aa;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}

        .tab:hover {{ background: #22224a; color: #e0e0e0; }}
        .tab.active {{ background: #2a2a6a; color: #60a5fa; border-color: #60a5fa; }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #8888aa;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>🔬 IR Graph Visualization</h1>
    <div class="subtitle">{ir_graph.model_name} — Custom Hardware-Friendly Intermediate Representation</div>
</div>

<div class="container">
    <div class="graph-panel">
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>

    <div class="info-panel">
        <div class="info-section">
            <h2>📊 Model Stats</h2>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="label">Parameters</div>
                    <div class="value">{ir_graph.total_params():,}</div>
                </div>
                <div class="stat-card">
                    <div class="label">IR Nodes</div>
                    <div class="value">{len(ir_graph.nodes)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Weight Memory</div>
                    <div class="value">{ir_graph.total_weight_memory() / 1024:.0f} KB</div>
                </div>
                <div class="stat-card">
                    <div class="label">Activation Mem</div>
                    <div class="value">{ir_graph.total_activation_memory() / 1024:.0f} KB</div>
                </div>
            </div>
        </div>

        <div class="info-section">
            <h2>🧩 Layer Breakdown</h2>
            <table class="op-table">
                <tr><th>Operation</th><th>Count</th></tr>
                {"".join(
                    f'<tr><td><span class="op-badge" style="background:{OP_COLORS.get(op, DEFAULT_COLOR)["bg"]}">{op}</span></td><td>{count}</td></tr>'
                    for op, count in sorted(
                        _count_ops(ir_graph).items()
                    )
                )}
            </table>
        </div>

        <div class="info-section">
            <h2>🎨 Legend</h2>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background:#4CAF50"></div>Input</div>
                <div class="legend-item"><div class="legend-color" style="background:#F44336"></div>Output</div>
                <div class="legend-item"><div class="legend-color" style="background:#2196F3"></div>Compute</div>
                <div class="legend-item"><div class="legend-color" style="background:#FF9800"></div>Activation</div>
                <div class="legend-item"><div class="legend-color" style="background:#00BCD4"></div>Norm</div>
                <div class="legend-item"><div class="legend-color" style="background:#673AB7"></div>Attention</div>
                <div class="legend-item"><div class="legend-color" style="background:#E91E63"></div>Embedding</div>
                <div class="legend-item"><div class="legend-color" style="background:#607D8B"></div>Reshape</div>
            </div>
        </div>

        <div class="info-section">
            <h2>📝 IR Details</h2>
            <div class="tabs">
                <button class="tab active" onclick="showTab('summary')">Summary</button>
                <button class="tab" onclick="showTab('full')">Full IR</button>
            </div>
            <div id="tab-summary" class="tab-content active">
                <pre class="ir-text">{summary}</pre>
            </div>
            <div id="tab-full" class="tab-content">
                <pre class="ir-text">{pretty}</pre>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
    mermaid.initialize({{
        startOnLoad: true,
        theme: 'dark',
        themeVariables: {{
            darkMode: true,
            background: '#12122a',
            primaryColor: '#2a2a6a',
            primaryTextColor: '#e0e0e0',
            lineColor: '#4a4a8a',
        }}
    }});

    function showTab(name) {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        event.target.classList.add('active');
        document.getElementById('tab-' + name).classList.add('active');
    }}
</script>

</body>
</html>"""
    return html


def _count_ops(ir_graph: IRGraph) -> dict[str, int]:
    """Count occurrences of each op type."""
    counts: dict[str, int] = {}
    for node in ir_graph.nodes:
        counts[node.op] = counts.get(node.op, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def visualize(
    ir_graph: IRGraph,
    format: str = "terminal",
    output: str = "",
    open_browser: bool = True,
) -> str:
    """
    Visualize the IR graph in the specified format.

    Args:
        ir_graph: The IRGraph to visualize.
        format: One of "terminal", "mermaid", "dot", "html".
        output: Output file path. If empty, prints to stdout.
        open_browser: If True and format is "html", opens in browser.

    Returns:
        The rendered string.
    """
    renderers = {
        "terminal": to_terminal,
        "ascii": to_terminal,
        "mermaid": to_mermaid,
        "dot": to_dot,
        "graphviz": to_dot,
        "html": to_html,
    }

    renderer = renderers.get(format.lower())
    if renderer is None:
        raise ValueError(
            f"Unknown format: {format}. "
            f"Choose from: {', '.join(renderers.keys())}"
        )

    result = renderer(ir_graph)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(result, encoding="utf-8")
        print(f"IR visualization saved to: {output}")

        if format.lower() == "html" and open_browser:
            webbrowser.open(f"file://{os.path.abspath(output)}")
    else:
        print(result)

    return result


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a Custom IR graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_ir.py output/ir_graph.json --format terminal
  python visualize_ir.py output/ir_graph.json --format mermaid
  python visualize_ir.py output/ir_graph.json --format html --output graph.html
  python visualize_ir.py output/ir_graph.json --format dot --output graph.dot

  # Then render DOT to PNG:
  dot -Tpng graph.dot -o graph.png
        """,
    )
    parser.add_argument(
        "ir_json",
        help="Path to the IR graph JSON file (serialized IRGraph.to_dict())",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["terminal", "mermaid", "dot", "html"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output file path (prints to stdout if not specified)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser for HTML output",
    )

    args = parser.parse_args()

    # Load IR graph from JSON
    with open(args.ir_json, "r", encoding="utf-8") as f:
        ir_dict = json.load(f)

    ir_graph = IRGraph.from_dict(ir_dict)

    visualize(
        ir_graph,
        format=args.format,
        output=args.output,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()


#agents/__init__.py#
"""Agentic RISC-V Compiler — Agent modules."""


#agents/code_generator.py#
"""
Code Generator Agent — Converts Custom IR → RISC-V C code.

Uses Qwen2.5-Coder-32B via local vLLM server (OpenAI-compatible API)
to generate model.h/model.c from the IR graph.

weights.h is generated deterministically (not by the LLM) using
the export_weights utility. The LLM then produces a model.h contract
before implementing model.c against that contract.

On verification retries, the previously generated header and code are
fed back to the LLM in "repair mode" (Codex-inspired) so it can make
targeted fixes rather than regenerating from scratch.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agents.codegen_contract import required_helper_signatures
from ir import IRGraph
from state import AgentState, LLMCallStats
from tools.export_weights import (
    export_weights_binary,
    generate_weights_header,
    generate_weights_loader,
)

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")  # vLLM doesn't need real key
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "200000"))

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "codegen.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
LLM_MAX_TOKENS = 200_000


def _read_generated_artifact_from_output(
    state: AgentState,
    path_key: str,
    state_key: str,
    default_filename: str,
) -> str:
    """Read a generated artifact from disk, falling back to state content."""
    code_path = state.get(path_key, "")
    if code_path:
        path = Path(code_path)
        if path.exists():
            return path.read_text(encoding="utf-8")

    default_path = OUTPUT_DIR / default_filename
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")

    return state.get(state_key, "")


def _read_generated_code_from_output(state: AgentState) -> str:
    """Read the generated model.c from disk, falling back to state content."""
    return _read_generated_artifact_from_output(
        state, "code_path", "generated_code", "model.c"
    )


def _read_generated_model_header_from_output(state: AgentState) -> str:
    """Read generated model.h from disk, falling back to state content."""
    return _read_generated_artifact_from_output(
        state, "model_header_path", "generated_model_header", "model.h"
    )


def _load_system_prompt() -> str:
    """Load the code generation system prompt."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def extract_weight_variable_names(state: AgentState) -> list[str]:
    """
    Return the C variable names exported by weights.h for LLM prompt context.

    This helper centralizes weight-name extraction for both normal models and
    LLM-style checkpoints with dotted parameter names such as
    ``model.layers.0.self_attn.q_proj.weight``.
    """
    manifest = state.get("weights_manifest", {})
    if manifest:
        return [
            str(info.get("c_name", name.replace(".", "_")))
            for name, info in sorted(manifest.items())
        ]

    weight_metadata = state.get("weights_metadata", {})
    if not weight_metadata:
        weight_metadata = state.get("ir_graph", {}).get("weight_metadata", {})

    return [name.replace(".", "_") for name in sorted(weight_metadata.keys())]


def _build_user_prompt(state: AgentState) -> str:
    """
    Build the user prompt containing IR graph, weight metadata,
    detailed implementation guidance, any previous feedback, and
    optimization suggestions.

    On retries (when verification_feedback is set), reads the
    previously generated code from the output file and includes it
    in a REPAIR MODE section so the LLM
    can make targeted fixes instead of regenerating from scratch.
    """
    ir_dict = state.get("ir_graph", {})
    ir_graph = IRGraph.from_dict(ir_dict)
    is_retry = bool(state.get("verification_feedback", ""))

    sections = []

    # ── IR Graph ────────────────────────────────────────────────
    sections.append("=" * 60)
    sections.append("IR GRAPH")
    sections.append("=" * 60)
    sections.append(ir_graph.pretty_print())

    # ── Weight Summary (C names for #include "weights.h") ───────
    sections.append("")
    sections.append("=" * 60)
    sections.append("WEIGHT TENSORS (available in weights.h — use these C names)")
    sections.append("=" * 60)

    # Prefer manifest (has c_name), fall back to weight_metadata
    manifest = state.get("weights_manifest", {})
    if manifest:
        for name in sorted(manifest.keys()):
            info = manifest[name]
            c_name = info["c_name"]
            numel = info["numel"]
            shape = info["shape"]
            c_type = info.get("c_type", "float")
            sections.append(
                f"  {c_type} {c_name}[{numel}];  "
                f"// shape={shape}, originally: {name}"
            )
    else:
        for name, meta in ir_graph.weight_metadata.items():
            shape = meta.get("shape", [])
            numel = meta.get("numel", 0)
            c_name = name.replace(".", "_")
            sections.append(
                f"  float {c_name}[{numel}];  "
                f"// shape={shape}, originally: {name}"
            )

    # ── FUNCTION HEADER ─────────────────────────────────────────
    functions_h = state.get("generated_functions_header", "")
    if functions_h:
        sections.append("")
        sections.append("=" * 60)
        sections.append("FUNCTION PROTOTYPES (from model_functions.h)")
        sections.append("=" * 60)
        sections.append(functions_h)

    # ── REPAIR MODE: Current Code + Errors ──────────────────────
    if is_retry:
        current_code = _read_generated_code_from_output(state)
        feedback = state.get("verification_feedback", "")

        if current_code:
            sections.append("")
            sections.append("=" * 60)
            sections.append(
                "⚠️  CURRENT CODE (contains errors — you are in REPAIR MODE)"
            )
            sections.append("=" * 60)
            sections.append(current_code)

        sections.append("")
        sections.append("=" * 60)
        sections.append("⚠️  VERIFICATION ERRORS — FIX THESE")
        sections.append("=" * 60)
        sections.append(feedback)
        sections.append("")
        sections.append(
            "You are in REPAIR MODE. Fix ALL the errors listed above "
            "in the current code. Output the COMPLETE fixed model.c file. "
            "Keep the overall structure intact. Mark fixes with "
            "// FIX: <description> comments."
        )
    else:
        # ── First attempt: generate from scratch ────────────────
        # (No verification feedback yet)
        pass

    # ── Optimization Suggestions (if in optimization loop) ──────
    suggestions = state.get("optimization_suggestions", [])
    if suggestions:
        sections.append("")
        sections.append("=" * 60)
        sections.append("🔧 OPTIMIZATION SUGGESTIONS — APPLY THESE")
        sections.append("=" * 60)
        for i, s in enumerate(suggestions, 1):
            sections.append(f"  {i}. {s}")
        sections.append("")
        sections.append(
            "Apply the above optimizations to the generated code. "
            "Mark optimized sections with '// OPTIMIZED: <description>'."
        )

    # ── Detailed generation guidance ────────────────────────────
    sections.append("")
    sections.append("=" * 60)
    sections.append("IMPLEMENTATION DETAILING REQUIREMENTS")
    sections.append("=" * 60)
    sections.append(
        "Before emitting code, internally map every IR node to exact "
        "buffer names, tensor extents, helper calls, loop bounds, and "
        "weight arrays. The final answer must still contain only the "
        "single requested model.c code block, but the implementation "
        "should reflect this detailed plan with clear constants, explicit "
        "shape comments, and operation-by-operation comments."
    )
    sections.append(
        "Use the full available context to preserve all generated code "
        "during repairs and optimizations; do not omit helper functions "
        "or unrelated model_inference steps while fixing localized issues."
    )

    # ── Task Instruction ────────────────────────────────────────
    sections.append("")
    sections.append("=" * 60)
    sections.append("TASK")
    sections.append("=" * 60)
    if is_retry:
        sections.append(
            "Fix the errors in the current code and output the COMPLETE "
            "fixed model.c file. Do NOT rewrite from scratch — make "
            "targeted fixes. Output exactly ONE ```c model.c code block."
        )
    else:
        sections.append(
            "Context for model generation follows."
            "The actual generation instructions are provided later."
            "Do not generate code based only on this section."
            "model targeting RISC-V rv32imac. "
            "Do NOT generate weights.h or model.h in this response — both "
            "are already available. Just #include \"model.h\" and use the "
            "weight array names listed above through the model.h -> weights.h "
            "include chain. "
            "Output exactly ONE ```c model.c code block."
        )

    return "\n".join(sections)

def _build_header_prompt(state: AgentState) -> str:
    """Prompt for generating the model_functions.h header."""
    ir_dict = state.get("ir_graph", {})
    ir_graph = IRGraph.from_dict(ir_dict)
    
    sections = []
    sections.append("=" * 60)
    sections.append("IR GRAPH")
    sections.append("=" * 60)
    sections.append(ir_graph.pretty_print())
    sections.append("")
    sections.append("=" * 60)
    sections.append("TASK")
    sections.append("=" * 60)
    sections.append(
        "Generate a C header file named `model_functions.h` containing ONLY the function "
        "prototypes (declarations) needed to implement this neural network on bare-metal RISC-V. "
        "Include `void model_inference(const float* input, float* output);` "
        "Do NOT implement the functions. Output exactly ONE ```c model_functions.h code block."
    )
    return "\n".join(sections)

def _build_weight_context(state: AgentState) -> str:
    """Build deterministic weight context shared by model.h/model.c prompts."""
    lines = [
        "=" * 60,
        "WEIGHT TENSORS (available in weights.h — use these C names)",
        "=" * 60,
    ]
    manifest = state.get("weights_manifest", {})
    if manifest:
        for name in sorted(manifest.keys()):
            info = manifest[name]
            lines.append(
                f"  {info.get('c_type', 'float')} {info['c_name']}[{info['numel']}]; "
                f"// shape={info['shape']}, originally: {name}"
            )
    else:
        weight_metadata = state.get("weights_metadata", {})
        if not weight_metadata:
            weight_metadata = state.get("ir_graph", {}).get("weight_metadata", {})
        for name, meta in sorted(weight_metadata.items()):
            lines.append(
                f"  float {name.replace('.', '_')}[{meta.get('numel', 0)}]; "
                f"// shape={meta.get('shape', [])}, originally: {name}"
            )
    lines.append("")
    lines.append("Weight variable names only:")
    lines.append(", ".join(extract_weight_variable_names(state)) or "(none)")
    return "\n".join(lines)


def _build_model_header_prompt(state: AgentState) -> str:
    """
    Build step-2 prompt: generate model.h with includes, interface contracts,
    dependencies, tensor block declarations, and function stubs only.
    """
    ir_graph = IRGraph.from_dict(state.get("ir_graph", {}))
    required_helpers = required_helper_signatures(state.get("ir_graph", {}))
    is_retry = bool(state.get("verification_feedback", ""))
    sections = [
        "=" * 60,
        "IR GRAPH",
        "=" * 60,
        ir_graph.pretty_print(),
        "",
        _build_weight_context(state),
        "",
        "=" * 60,
        "STEP 2 TASK: CREATE model.h",
        "=" * 60,
        "Create a C99 header file named model.h. It must include weights.h, "
        "define input/output tensor contracts, document dependencies, declare "
        "static-size tensor block macros/constants, and provide prototypes for "
        "all helper functions and model_inference(). Do not implement function "
        "bodies and do not define storage in model.h.",
        "",
        "REQUIRED HELPER PROTOTYPES:",
        *(f"  {signature}" for signature in required_helpers),
        "If the implementation needs any additional non-static helper function, "
        "declare its prototype in model.h as well.",
        "Output exactly ONE ```c model.h code block.",
    ]

    if is_retry:
        current_header = _read_generated_model_header_from_output(state)
        feedback = state.get("verification_feedback", "")
        if current_header:
            sections.extend([
                "",
                "=" * 60,
                "CURRENT model.h (contains errors — REPAIR MODE)",
                "=" * 60,
                current_header,
            ])
        sections.extend([
            "",
            "=" * 60,
            "VERIFICATION ERRORS — FIX HEADER-RELEVANT ISSUES",
            "=" * 60,
            feedback,
            "Make targeted fixes and output the COMPLETE fixed model.h.",
        ])

    return "\n".join(sections)


def _build_model_c_prompt(state: AgentState, model_h: str) -> str:
    """Build step-3 prompt: implement model.c against the generated model.h."""
    base = _build_user_prompt(state)
    required_helpers = required_helper_signatures(state.get("ir_graph", {}))
    sections = [
        base,
        "",
        "=" * 60,
        "STEP 2 model.h CONTRACT TO IMPLEMENT",
        "=" * 60,
        model_h,
        "",
        "=" * 60,
        "STEP 3 TASK: IMPLEMENT model.c",
        "=" * 60,
        "Implement every prototype and tensor block declared in model.h. "
        "Include \"model.h\" (which includes weights.h) and keep model.c "
        "consistent with the inputs, outputs, dependencies, and stubs above. "
        "The following IR-required helpers must have matching non-static "
        "definitions in model.c:",
        *(f"  {signature}" for signature in required_helpers),
        "Output exactly ONE ```c model.c code block.",
    ]
    return "\n".join(sections)


def _extract_c_artifact(response, filename):

    # Prefer explicitly labelled block
    pattern = rf"```(?:c|C)?\s*{re.escape(filename)}\s*\n(.*?)```"

    m = re.search(pattern, response, re.DOTALL)

    if m:
        return m.group(1).strip()

    # fallback: choose largest code block
    blocks = re.findall(
        r"```(?:c|C)?\s*\n(.*?)```",
        response,
        re.DOTALL,
    )

    if not blocks:
        raise ValueError("No code block found.")

    logger.warning(
        "Multiple code blocks detected (%d). Using largest.",
        len(blocks),
    )

    return max(blocks, key=len).strip()


def _generate_deterministic_header(state: AgentState) -> str:
    """
    Generate weights.h deterministically from the weight data.

    This replaces the old approach of having the LLM generate weights.h.
    The header contains actual weight values (not placeholders) and is
    guaranteed to match the model's state_dict.

    Works for any model type (LLMs, CNNs, etc.) since it operates
    on the raw numpy arrays from model.state_dict().
    """
    npz_path = state.get("weights_path", "")
    manifest = state.get("weights_manifest", {})
    precision = state.get("weight_precision", "f32")
    mode = state.get("weight_mode", "embedded")

    if not npz_path or not os.path.exists(npz_path):
        logger.error(f"Weights file not found: {npz_path}")
        # Fall back to generating from weight_metadata
        return _generate_fallback_header(state)

    if not manifest:
        # Generate manifest from npz if not already in state
        logger.warning("No manifest in state — generating from .npz")
        output_dir = os.path.dirname(npz_path)
        _, manifest = export_weights_binary(
            npz_path, output_dir, precision
        )

    return generate_weights_header(
        manifest=manifest,
        npz_path=npz_path,
        precision=precision,
        mode=mode,
    )


def _generate_deterministic_loader(state: AgentState) -> str:
    """
    Generate weights_loader.c for binary mode.
    Only needed when weight_mode is 'binary'.
    """
    manifest = state.get("weights_manifest", {})
    precision = state.get("weight_precision", "f32")

    return generate_weights_loader(
        manifest=manifest,
        precision=precision,
    )


def _generate_fallback_header(state: AgentState) -> str:
    """
    Generate a minimal weights.h if no .npz file is available.
    Uses zero-initialized arrays (legacy fallback).
    """
    weight_metadata = state.get("weights_metadata", {})
    ir_dict = state.get("ir_graph", {})
    if not weight_metadata:
        weight_metadata = ir_dict.get("weight_metadata", {})

    lines = [
        "#pragma once",
        "/* Auto-generated weight declarations */",
        "/* WARNING: Fallback mode — weights are zero-initialized */",
        "",
    ]
    for name, meta in sorted(weight_metadata.items()):
        c_name = name.replace(".", "_")
        numel = meta.get("numel", 1)
        shape = meta.get("shape", [])
        lines.append(f"// {name}: shape={shape}")
        lines.append(f"static const float {c_name}[{numel}] = {{0}};")
        lines.append("")
    return "\n".join(lines)


def _write_generated_artifacts(
    model_c: str,
    model_h: str,
    weights_h: str,
    loader_c: str = "",
) -> dict[str, str]:
    """
    Write generated code artifacts and return their state path fields.

    Keeping all file writes in one helper avoids legacy loose variables and
    makes the generation write path easier to test end-to-end.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    code_path = OUTPUT_DIR / "model.c"
    model_header_path = OUTPUT_DIR / "model.h"
    header_path = OUTPUT_DIR / "weights.h"

    code_path.write_text(model_c, encoding="utf-8")
    model_header_path.write_text(model_h, encoding="utf-8")
    header_path.write_text(weights_h, encoding="utf-8")

    logger.info(f"Generated code written to: {code_path}")
    logger.info(f"Generated model header written to: {model_header_path}")
    logger.info(f"Generated header written to: {header_path}")

    paths = {
        "code_path": str(code_path),
        "model_header_path": str(model_header_path),
        "header_path": str(header_path),
    }

    if loader_c:
        loader_path = OUTPUT_DIR / "weights_loader.c"
        loader_path.write_text(loader_c, encoding="utf-8")
        logger.info(f"Generated loader written to: {loader_path}")
        paths["loader_path"] = str(loader_path)

    return paths


def generate_code(state: AgentState) -> dict:
    """
    LangGraph node function: Generate C code from IR.

    Reads: state["ir_graph"], state["verification_feedback"],
           state["optimization_suggestions"], state["generated_code"],
           state["code_path"],
           state["weights_path"], state["weights_manifest"],
           state["weight_precision"], state["weight_mode"]
    Writes: state["generated_code"], state["generated_header"],
            state["code_path"], state["header_path"]
    """
    logger.info("Generating C code from IR graph...")

    attempt = state.get("verification_attempts", 0)
    opt_iter = state.get("optimization_iteration", 0)
    is_retry = bool(state.get("verification_feedback", ""))
    logger.info(
        f"  Verification attempt: {attempt}, "
        f"Optimization iteration: {opt_iter}, "
        f"Repair mode: {is_retry}"
    )

    # ── Telemetry accumulators ───────────────────────────────
    call_stats: list[LLMCallStats] = []
    total_input = 0
    total_output = 0
    total_latency = 0.0

    # ── Generate deterministic weights.h ────────────────────────
    weights_h = _generate_deterministic_header(state)
    mode = state.get("weight_mode", "embedded")

    # For binary mode, also generate the loader
    loader_c = ""
    if mode == "binary":
        loader_c = _generate_deterministic_loader(state)

    # ── Build LLM messages ──────────────────────────────────────
    system_prompt = _load_system_prompt()

    # ── Call vLLM ───────────────────────────────────────────────
    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        model=VLLM_MODEL,
        temperature=0.2 if not is_retry else 0.0,  # Lower temp on retries
        max_tokens=LLM_MAX_TOKENS,
    )

    # Step 2: ask the LLM for model.h, which defines includes, tensor
    # contracts, dependencies, and all function stubs to be implemented.
    header_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_model_header_prompt(state)),
    ]

    logger.info(f"Calling LLM ({VLLM_MODEL}) for step 2 model.h ...")
    if is_retry:
        logger.info("  → REPAIR MODE: feeding back current code + errors")
    t0 = time.perf_counter()
    header_response = llm.invoke(header_messages)
    header_latency = time.perf_counter() - t0
    raw_header_response = header_response.content
    logger.info(f"LLM model.h response length: {len(raw_header_response)} chars")

    # ── Collect model.h token stats ────────────────────────
    h_usage = getattr(header_response, "usage_metadata", None) or {}
    h_in  = h_usage.get("input_tokens",  0) if isinstance(h_usage, dict) else getattr(h_usage, "input_tokens",  0)
    h_out = h_usage.get("output_tokens", 0) if isinstance(h_usage, dict) else getattr(h_usage, "output_tokens", 0)
    call_stats.append(LLMCallStats(
        agent="code_generator",
        call_label="model.h",
        input_tokens=h_in,
        output_tokens=h_out,
        total_tokens=h_in + h_out,
        latency_s=header_latency,
    ))
    total_input   += h_in
    total_output  += h_out
    total_latency += header_latency
    logger.info(
        f"  model.h — input_tokens={h_in}, output_tokens={h_out}, "
        f"latency={header_latency:.2f}s"
    )

    # ── Extract model.h ─────────────────────────────────────────
    model_h = _extract_c_artifact(raw_header_response, "model.h")

    # Step 3: ask the LLM to implement model.c against the model.h contract.
    c_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_model_c_prompt(state, model_h)),
    ]

    logger.info(f"Calling LLM ({VLLM_MODEL}) for step 3 model.c ...")
    t0 = time.perf_counter()
    c_response = llm.invoke(c_messages)
    c_latency = time.perf_counter() - t0
    raw_c_response = c_response.content
    logger.info(f"LLM model.c response length: {len(raw_c_response)} chars")

    # ── Collect model.c token stats ────────────────────────
    c_usage = getattr(c_response, "usage_metadata", None) or {}
    c_in  = c_usage.get("input_tokens",  0) if isinstance(c_usage, dict) else getattr(c_usage, "input_tokens",  0)
    c_out = c_usage.get("output_tokens", 0) if isinstance(c_usage, dict) else getattr(c_usage, "output_tokens", 0)
    call_stats.append(LLMCallStats(
        agent="code_generator",
        call_label="model.c",
        input_tokens=c_in,
        output_tokens=c_out,
        total_tokens=c_in + c_out,
        latency_s=c_latency,
    ))
    total_input   += c_in
    total_output  += c_out
    total_latency += c_latency
    logger.info(
        f"  model.c — input_tokens={c_in}, output_tokens={c_out}, "
        f"latency={c_latency:.2f}s"
    )

    # ── Extract model.c ─────────────────────────────────────────
    model_c = _extract_c_artifact(raw_c_response, "model.c")

    # ── Write files ─────────────────────────────────────────────
    artifact_paths = _write_generated_artifacts(
        model_c=model_c,
        model_h=model_h,
        weights_h=weights_h,
        loader_c=loader_c,
    )

    # ── Merge new telemetry into existing state accumulators ─────
    existing_stats: list[LLMCallStats] = list(state.get("llm_call_stats") or [])
    existing_stats.extend(call_stats)
    prev_input   = state.get("total_input_tokens",  0) or 0
    prev_output  = state.get("total_output_tokens", 0) or 0
    prev_latency = state.get("total_llm_latency_s", 0.0) or 0.0
    prev_agent_latencies: dict = dict(state.get("agent_latencies") or {})
    prev_agent_latencies["code_generator"] = (
        prev_agent_latencies.get("code_generator", 0.0) + total_latency
    )

    return {
        "generated_code": model_c,
        "generated_header": weights_h,
        "generated_model_header": model_h,
        **artifact_paths,
        # Telemetry
        "llm_call_stats": existing_stats,
        "total_input_tokens":  prev_input  + total_input,
        "total_output_tokens": prev_output + total_output,
        "total_llm_latency_s": prev_latency + total_latency,
        "agent_latencies": prev_agent_latencies,
    }



#agents/codegen_contract.py#
"""Shared code-generation contract helpers.

These helpers define the C helper-function contract that both the code
prompts and verifier use. Keeping the required helper signatures in one
place prevents the LLM header prompt and deterministic verifier from
drifting apart.
"""

from __future__ import annotations

from ir import IRGraph, IROpType


HELPER_SIGNATURES_BY_OP: dict[str, tuple[str, ...]] = {
    IROpType.CONV2D: (
        "void conv2d(const float* in, const float* weight, const float* bias, float* out, int IC, int OC, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.LINEAR: (
        "void linear(const float* in, const float* weight, const float* bias, float* out, int in_features, int out_features);",
    ),
    IROpType.RELU: ("void relu(float* data, int size);",),
    IROpType.SILU: ("void silu(float* data, int size);",),
    IROpType.RMSNORM: (
        "void rmsnorm(const float* in, const float* weight, float* out, int size, float eps);",
    ),
    IROpType.LAYERNORM: (
        "void layernorm(const float* in, const float* weight, const float* bias, float* out, int size, float eps);",
    ),
    IROpType.SOFTMAX: ("void softmax(float* data, int size);",),
    IROpType.MATMUL: (
        "void matmul(const float* A, const float* B, float* C, int M, int K, int N);",
    ),
    IROpType.EMBEDDING: (
        "void embedding(const float* table, int token_id, float* out, int dim);",
    ),
    IROpType.ROTARY_EMBEDDING: (
        "void rope(float* q, float* k, int head_dim, int pos, int num_heads);",
    ),
    IROpType.ATTENTION: (
        "void attention(const float* x, const float* wq, const float* wk, const float* wv, const float* wo, float* out, int seq_len, int dim, int num_heads, int head_dim);",
    ),
    IROpType.BATCHNORM: (
        "void batchnorm2d(const float* in, const float* weight, const float* bias, float* out, int C, int H, int W, float eps);",
    ),
    IROpType.MAXPOOL2D: (
        "void maxpool2d(const float* in, float* out, int C, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.AVGPOOL2D: (
        "void avgpool2d(const float* in, float* out, int C, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.FLATTEN: (
        "void flatten(const float* in, float* out, int size);",
    ),
    IROpType.ADD: (
        "void add_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.SUB: (
        "void sub_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.MUL: (
        "void mul_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.DIV: (
        "void div_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.SQRT: (
        "void sqrt_tensor(const float* in, float* out, int size);",
    ),
    IROpType.MEAN: (
        "void mean_tensor(const float* in, float* out, int outer, int reduce, int inner);",
    ),
    IROpType.RESHAPE: (
        "void reshape_copy(const float* in, float* out, int size);",
    ),
    IROpType.TRANSPOSE: (
        "void transpose2d(const float* in, float* out, int rows, int cols);",
    ),
    IROpType.SPLIT: (
        "void split_tensor(const float* in, float* out0, float* out1, float* out2, int part_size);",
    ),
    IROpType.CONCAT: (
        "void concat_tensors(const float* a, const float* b, float* out, int a_size, int b_size);",
    ),
    IROpType.QUANTIZE: (
        "void quantize_f32_to_i8(const float* in, signed char* out, int size, float scale);",
    ),
    IROpType.DEQUANTIZE: (
        "void dequantize_i8_to_f32(const signed char* in, float* out, int size, float scale);",
    ),
}


def helper_name_from_signature(signature: str) -> str:
    """Return the C function name from a simple helper prototype."""
    before_paren = signature.split("(", 1)[0].strip()
    return before_paren.split()[-1]


def required_helper_signatures(ir_dict: dict) -> list[str]:
    """Return required helper prototypes for all non-I/O operations in IR order."""
    ir_graph = IRGraph.from_dict(ir_dict)
    seen: set[str] = set()
    signatures: list[str] = []
    for node in ir_graph.topological_order():
        for signature in HELPER_SIGNATURES_BY_OP.get(node.op, ()):  # no helper for I/O/dropout
            if signature not in seen:
                seen.add(signature)
                signatures.append(signature)
    return signatures


#agents/fx_parser.py#
"""
FX Parser Agent — Converts a PyTorch FX graph into the Custom IR.

This agent requires NO LLM. It performs deterministic analysis of the
torch.fx.GraphModule and builds an IRGraph with all layer info, shapes,
and weight metadata.

PyTorch objects (nn.Module, GraphModule, Tensor) are retrieved from the
module-level pytorch_object_store rather than from the LangGraph state
to avoid msgpack serialisation failures at checkpoint boundaries.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ir import IRGraph, IRNode, IROpType
from state import AgentState, retrieve_pytorch_objects, clear_pytorch_objects
from tools.export_weights import export_weights_binary

logger = logging.getLogger(__name__)

# ── Mapping from torch.nn module types to IR ops ────────────────
MODULE_TYPE_MAP: dict[type, str] = {
    nn.Conv2d: IROpType.CONV2D,
    nn.Linear: IROpType.LINEAR,
    nn.ReLU: IROpType.RELU,
    nn.BatchNorm2d: IROpType.BATCHNORM,
    nn.MaxPool2d: IROpType.MAXPOOL2D,
    nn.AdaptiveAvgPool2d: IROpType.AVGPOOL2D,
    nn.AvgPool2d: IROpType.AVGPOOL2D,
    nn.Dropout: IROpType.DROPOUT,
    nn.Flatten: IROpType.FLATTEN,
    nn.Softmax: IROpType.SOFTMAX,
    nn.Embedding: IROpType.EMBEDDING,
    nn.LayerNorm: IROpType.LAYERNORM,
}

# ── Mapping from torch functions to IR ops ──────────────────────
FUNCTION_MAP: dict[Any, str] = {}
METHOD_MAP: dict[str, str] = {
    "relu": IROpType.RELU,
    "flatten": IROpType.FLATTEN,
    "view": IROpType.RESHAPE,
    "reshape": IROpType.RESHAPE,
    "transpose": IROpType.TRANSPOSE,
    "permute": IROpType.TRANSPOSE,
    "contiguous": None,  # no-op, skip
    "softmax": IROpType.SOFTMAX,
    "chunk": IROpType.SPLIT,
    "split": IROpType.SPLIT,
    "mean": None,  # handled inline
    "size": None,   # metadata, skip
}

# Populate FUNCTION_MAP after import (torch.nn.functional may not
# be available at module-level in all environments)
def _init_function_map():
    import torch.nn.functional as F
    import operator

    FUNCTION_MAP.update({
        F.relu: IROpType.RELU,
        F.silu: IROpType.SILU,
        F.softmax: IROpType.SOFTMAX,
        F.linear: IROpType.LINEAR,
        F.conv2d: IROpType.CONV2D,
        F.embedding: IROpType.EMBEDDING,
        F.layer_norm: IROpType.LAYERNORM,
        F.dropout: IROpType.DROPOUT,
        torch.matmul: IROpType.MATMUL,
        torch.bmm: IROpType.MATMUL,
        torch.cat: IROpType.CONCAT,
        torch.split: IROpType.SPLIT,
        operator.add: IROpType.ADD,
        operator.mul: IROpType.MUL,
        torch.add: IROpType.ADD,
        torch.mul: IROpType.MUL,
        torch.flatten: IROpType.FLATTEN,
        torch.reshape: IROpType.RESHAPE,
        torch.transpose: IROpType.TRANSPOSE,
    })

_init_function_map()


def _get_module_for_target(model: nn.Module, target: str) -> nn.Module | None:
    """Resolve a dotted target path to the actual nn.Module."""
    parts = target.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current if isinstance(current, nn.Module) else None


def _extract_conv2d_params(module: nn.Conv2d) -> dict:
    """Extract Conv2D parameters for the IR node."""
    return {
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": list(module.kernel_size),
        "stride": list(module.stride),
        "padding": list(module.padding),
        "groups": module.groups,
        "bias": module.bias is not None,
    }


def _extract_linear_params(module: nn.Linear) -> dict:
    """Extract Linear layer parameters for the IR node."""
    return {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
    }


def _extract_embedding_params(module: nn.Embedding) -> dict:
    """Extract Embedding parameters for the IR node."""
    return {
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
    }


def _extract_pool_params(module: nn.Module) -> dict:
    """Extract pooling parameters."""
    if isinstance(module, nn.MaxPool2d):
        ks = module.kernel_size
        if isinstance(ks, int):
            ks = [ks, ks]
        st = module.stride
        if isinstance(st, int):
            st = [st, st]
        pd = module.padding
        if isinstance(pd, int):
            pd = [pd, pd]
        return {"kernel_size": list(ks), "stride": list(st), "padding": list(pd)}
    elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
        if isinstance(module, nn.AdaptiveAvgPool2d):
            os = module.output_size
            if isinstance(os, int):
                os = [os, os]
            return {"output_size": list(os)}
        return {}
    return {}


def _extract_norm_params(module: nn.Module) -> dict:
    """Extract normalization parameters."""
    if isinstance(module, nn.LayerNorm):
        ns = module.normalized_shape
        return {
            "normalized_shape": list(ns),
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
        }
    return {}


def _infer_shape(node, known_shapes: dict[str, tuple]) -> tuple:
    """
    Try to infer output shape from the node's metadata.
    Falls back to shape propagation heuristics.
    """
    # Try torch.fx metadata first
    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
        meta = node.meta['tensor_meta']
        if hasattr(meta, 'shape'):
            return tuple(meta.shape)
    if hasattr(node, 'meta') and 'val' in node.meta:
        val = node.meta['val']
        if hasattr(val, 'shape'):
            return tuple(val.shape)

    # Fallback: check if any input shapes are known
    for arg in node.args:
        if hasattr(arg, 'name') and arg.name in known_shapes:
            return known_shapes[arg.name]

    return ()


def _node_inputs(node) -> list[str]:
    """Extract input node names from FX node args."""
    inputs = []
    for arg in node.args:
        if hasattr(arg, 'name'):
            inputs.append(arg.name)
        elif isinstance(arg, (list, tuple)):
            for a in arg:
                if hasattr(a, 'name'):
                    inputs.append(a.name)
    return inputs


def parse_fx_graph(state: AgentState) -> dict:
    """
    LangGraph node function: Parse FX graph → Custom IR.

    PyTorch objects are retrieved from pytorch_object_store (keyed by
    thread_id) rather than from the LangGraph state so that msgpack
    serialisation at checkpoint boundaries always succeeds.

    Writes: state["ir_graph"], state["ir_summary"], state["weights_metadata"],
            state["weights_path"], state["total_params"], state["model_memory_bytes"]
    """
    t_start = time.perf_counter()

    # ── Retrieve non-serializable objects from out-of-band store ──
    thread_id = state.get("thread_id", "default")
    pytorch_objects = retrieve_pytorch_objects(thread_id)
    if not pytorch_objects:
        raise RuntimeError(
            f"No PyTorch objects found in pytorch_object_store for "
            f"thread_id='{thread_id}'. "
            "Did you call state.store_pytorch_objects() before running the graph?"
        )

    fx_module = pytorch_objects["fx_graph"]
    model = pytorch_objects["model"]
    sample_input_from_store = pytorch_objects.get("sample_input")
    model_name = state.get("model_name", "unnamed_model")

    logger.info(f"Parsing FX graph for model: {model_name}")

    ir_graph = IRGraph(model_name=model_name)
    known_shapes: dict[str, tuple] = {}
    weight_metadata: dict[str, dict] = {}

    # ── Collect all weight tensors ──────────────────────────────
    state_dict = model.state_dict()
    for param_name, param_tensor in state_dict.items():
        weight_metadata[param_name] = {
            "shape": list(param_tensor.shape),
            "dtype": str(param_tensor.dtype).replace("torch.", ""),
            "numel": param_tensor.numel(),
        }

    ir_graph.weight_metadata = weight_metadata

    # ── Run shape propagation if possible ───────────────────────
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        # Create dummy input for shape propagation
        input_shapes = state.get("input_shapes", {})
        if not input_shapes:
            # Try to infer from the first placeholder
            for node in fx_module.graph.nodes:
                if node.op == "placeholder":
                    # Default: assume batch=1
                    # For LLM models: [1, seq_len] integer input
                    # For CNN models: [1, C, H, W] float input
                    break

        # Attempt shape propagation with a sample input
        sample_input = sample_input_from_store
        if sample_input is not None:
            ShapeProp(fx_module).propagate(sample_input)
    except Exception as e:
        logger.warning(f"Shape propagation failed: {e}. Using heuristic shapes.")

    # ── Iterate FX graph nodes ──────────────────────────────────
    for node in fx_module.graph.nodes:

        if node.op == "placeholder":
            # Input tensor
            shape = _infer_shape(node, known_shapes)
            ir_node = IRNode(
                id=node.name,
                op=IROpType.TENSOR_INPUT,
                inputs=[],
                shape=shape,
            )
            ir_graph.input_shapes[node.name] = shape
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "call_module":
            # nn.Module call (e.g., self.conv1)
            target_module = _get_module_for_target(model, node.target)
            if target_module is None:
                logger.warning(f"Could not resolve module: {node.target}")
                continue

            module_type = type(target_module)
            ir_op = MODULE_TYPE_MAP.get(module_type)

            if ir_op is None:
                # Try to detect custom modules (RMSNorm, SwiGLU, etc.)
                class_name = module_type.__name__.lower()
                if "rmsnorm" in class_name or "rms_norm" in class_name:
                    ir_op = IROpType.RMSNORM
                elif "swiglu" in class_name:
                    ir_op = IROpType.SWIGLU
                elif "rotary" in class_name or "rope" in class_name:
                    ir_op = IROpType.ROTARY_EMBEDDING
                elif "attention" in class_name:
                    ir_op = IROpType.ATTENTION
                else:
                    logger.warning(
                        f"Unsupported module type: {module_type.__name__} "
                        f"at {node.target}. Skipping."
                    )
                    continue

            # Extract op-specific parameters
            params = {}
            weight_key = ""
            bias_key = ""

            if ir_op == IROpType.CONV2D:
                params = _extract_conv2d_params(target_module)
                weight_key = f"{node.target}.weight"
                if params.get("bias"):
                    bias_key = f"{node.target}.bias"

            elif ir_op == IROpType.LINEAR:
                params = _extract_linear_params(target_module)
                weight_key = f"{node.target}.weight"
                if params.get("bias"):
                    bias_key = f"{node.target}.bias"

            elif ir_op == IROpType.EMBEDDING:
                params = _extract_embedding_params(target_module)
                weight_key = f"{node.target}.weight"

            elif ir_op in (IROpType.MAXPOOL2D, IROpType.AVGPOOL2D):
                params = _extract_pool_params(target_module)

            elif ir_op in (IROpType.LAYERNORM, IROpType.RMSNORM):
                params = _extract_norm_params(target_module)
                if hasattr(target_module, 'weight') and target_module.weight is not None:
                    weight_key = f"{node.target}.weight"
                if hasattr(target_module, 'bias') and target_module.bias is not None:
                    bias_key = f"{node.target}.bias"

            elif ir_op == IROpType.ATTENTION:
                # Try to extract attention params from submodules
                if hasattr(target_module, 'num_heads'):
                    params["num_heads"] = target_module.num_heads
                if hasattr(target_module, 'head_dim'):
                    params["head_dim"] = target_module.head_dim
                # Collect weight keys for Q, K, V, O projections
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'wq', 'wk', 'wv', 'wo']:
                    full_key = f"{node.target}.{proj}.weight"
                    if full_key in weight_metadata:
                        params[f"{proj}_weight"] = full_key

            shape = _infer_shape(node, known_shapes)
            inputs = _node_inputs(node)

            ir_node = IRNode(
                id=node.name,
                op=ir_op,
                inputs=inputs,
                params=params,
                shape=shape,
                weight_key=weight_key,
                bias_key=bias_key,
            )
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "call_function":
            # Functional call (e.g., torch.relu, operator.add)
            ir_op = FUNCTION_MAP.get(node.target)

            if ir_op is None:
                target_name = getattr(node.target, '__name__', str(node.target))
                # Try matching by name
                name_lower = target_name.lower()
                if "silu" in name_lower:
                    ir_op = IROpType.SILU
                elif "gelu" in name_lower:
                    ir_op = IROpType.RELU  # Approximate
                elif "embedding" in name_lower:
                    ir_op = IROpType.EMBEDDING
                elif "add" in name_lower:
                    ir_op = IROpType.ADD
                elif "sub" in name_lower:
                    ir_op = IROpType.SUB
                elif "mul" in name_lower:
                    ir_op = IROpType.MUL
                elif "div" in name_lower:
                    ir_op = IROpType.DIV
                elif "sqrt" in name_lower:
                    ir_op = IROpType.SQRT
                elif "mean" in name_lower:
                    ir_op = IROpType.MEAN
                elif "softmax" in name_lower:
                    ir_op = IROpType.SOFTMAX
                elif "permute" in name_lower or "transpose" in name_lower:
                    ir_op = IROpType.TRANSPOSE
                elif "view" in name_lower or "reshape" in name_lower or "unsqueeze" in name_lower or "slice" in name_lower:
                    ir_op = IROpType.RESHAPE
                elif "mm" in name_lower or "matmul" in name_lower or "bmm" in name_lower:
                    ir_op = IROpType.MATMUL
                elif "cat" in name_lower:
                    ir_op = IROpType.CONCAT
                elif "split" in name_lower or "chunk" in name_lower:
                    ir_op = IROpType.SPLIT
                else:
                    logger.warning(f"Unsupported function: {target_name}")
                    continue

            shape = _infer_shape(node, known_shapes)
            inputs = _node_inputs(node)

            params = {}
            # Extract function-specific params from kwargs
            if ir_op == IROpType.SOFTMAX:
                params["dim"] = node.kwargs.get("dim", -1)
            elif ir_op == IROpType.FLATTEN:
                if len(node.args) >= 2:
                    params["start_dim"] = node.args[1] if not hasattr(node.args[1], 'name') else 1
                if len(node.args) >= 3:
                    params["end_dim"] = node.args[2] if not hasattr(node.args[2], 'name') else -1

            ir_node = IRNode(
                id=node.name,
                op=ir_op,
                inputs=inputs,
                params=params,
                shape=shape,
            )
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "call_method":
            # Method call (e.g., x.view(), x.flatten())
            method_name = node.target
            ir_op = METHOD_MAP.get(method_name)

            if ir_op is None:
                # Skip no-op methods
                if method_name in ("contiguous", "size", "dim", "float", "half"):
                    continue
                logger.warning(f"Unsupported method: {method_name}")
                continue

            shape = _infer_shape(node, known_shapes)
            inputs = _node_inputs(node)

            params = {}
            if ir_op == IROpType.RESHAPE:
                # Try to extract target shape from args
                shape_args = []
                for arg in node.args[1:]:
                    if isinstance(arg, int):
                        shape_args.append(arg)
                    elif hasattr(arg, 'name'):
                        shape_args.append(-1)  # dynamic dim
                if shape_args:
                    params["target_shape"] = shape_args

            elif ir_op == IROpType.TRANSPOSE:
                if len(node.args) >= 3:
                    dim_args = node.args[1:3]
                    params["dim0"] = dim_args[0] if isinstance(dim_args[0], int) else 0
                    params["dim1"] = dim_args[1] if isinstance(dim_args[1], int) else 1

            ir_node = IRNode(
                id=node.name,
                op=ir_op,
                inputs=inputs,
                params=params,
                shape=shape,
            )
            known_shapes[node.name] = shape
            ir_graph.add_node(ir_node)

        elif node.op == "output":
            inputs = _node_inputs(node)
            ir_node = IRNode(
                id="output",
                op=IROpType.TENSOR_OUTPUT,
                inputs=inputs,
                shape=(),
            )
            ir_graph.add_node(ir_node)

        elif node.op == "get_attr":
            # Constant attribute access — skip (handled via weight_key)
            continue

    # ── Validate the IR graph ───────────────────────────────────
    issues = ir_graph.validate()
    if issues:
        logger.warning(f"IR validation issues:\n" + "\n".join(issues))

    # ── Save weights to .npz file ───────────────────────────────
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save IR graph to JSON for visualization
    import json
    ir_graph_path = os.path.join(output_dir, "ir_graph.json")
    with open(ir_graph_path, "w") as f:
        json.dump(ir_graph.to_dict(), f, indent=2)
    logger.info(f"IR Graph saved to: {ir_graph_path}")

    weights_path = os.path.join(output_dir, "weights.npz")

    weight_arrays = {}
    for name, tensor in state_dict.items():
        weight_arrays[name] = tensor.detach().cpu().numpy()
    np.savez(weights_path, **weight_arrays)
    logger.info(f"Weights saved to: {weights_path}")

    # ── Export weights to binary + manifest ──────────────────────
    precision = state.get("weight_precision", "f32")
    bin_path, manifest = export_weights_binary(
        weights_path, output_dir, precision
    )
    logger.info(f"Binary weights exported to: {bin_path}")

    # ── Build summary ───────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    model_memory = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )

    # ── Release PyTorch objects from out-of-band store ──────────
    # All information has been extracted into serializable dicts/strings.
    # Clear from store so the GC can reclaim memory.
    clear_pytorch_objects(thread_id)

    t_elapsed = time.perf_counter() - t_start
    logger.info(f"parse_fx_graph completed in {t_elapsed:.2f}s")

    # Build sample_input_shape for informational display
    sample_input_shape = (
        list(sample_input_from_store.shape)
        if sample_input_from_store is not None
        else []
    )

    return {
        "ir_graph": ir_graph.to_dict(),
        "ir_summary": ir_graph.layer_summary(),
        "weights_metadata": weight_metadata,
        "weights_path": weights_path,
        "weights_bin_path": bin_path,
        "weights_manifest": manifest,
        "weight_precision": precision,
        "weight_mode": state.get("weight_mode", "embedded"),
        "total_params": total_params,
        "model_memory_bytes": model_memory,
        "fx_graph_str": str(fx_module.graph),
        "sample_input_shape": sample_input_shape,
        # Telemetry
        "agent_latencies": {"parse_fx": t_elapsed},
    }


#agents/human_review.py#
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


#agents/optimizer.py#
"""
Optimization Agent — Hardware-aware hints pass.

Injects knowledge of the target micro-architecture (e.g. systolic arrays,
VPUs, SIMD units) into a new code-generator run so the output C code is
structured to exploit available hardware resources.

Runs ONCE, after human approval, BEFORE simulation.  It does not loop.
Uses Qwen2.5-Coder-32B via local vLLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ir import IRGraph
from state import AgentState, LLMCallStats

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "optimizer.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
LLM_MAX_TOKENS = 200_000

MAX_OPTIMIZATION_ITERATIONS = 3


def _read_generated_code_from_output(state: AgentState) -> str:
    """Read model.c from disk so optimization uses verified output content."""
    code_path = state.get("code_path", "")
    if code_path:
        path = Path(code_path)
        if path.exists():
            return path.read_text(encoding="utf-8")

    default_path = OUTPUT_DIR / "model.c"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")

    return state.get("generated_code", "")


def _load_system_prompt() -> str:
    """Load the optimization system prompt."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_optimization_prompt(state: AgentState) -> str:
    """Build the prompt with current metrics and code."""
    sections = []

    # ── Current Code ────────────────────────────────────────────
    code = _read_generated_code_from_output(state)
    sections.append("=" * 60)
    sections.append("CURRENT GENERATED C CODE")
    sections.append("=" * 60)
    sections.append(code)

    # ── IR Graph ────────────────────────────────────────────────
    ir_dict = state.get("ir_graph", {})
    if ir_dict:
        ir_graph = IRGraph.from_dict(ir_dict)
        sections.append("")
        sections.append("=" * 60)
        sections.append("IR GRAPH SUMMARY")
        sections.append("=" * 60)
        sections.append(ir_graph.layer_summary())

    # ── Simulation Results ──────────────────────────────────────
    sim = state.get("simulation_result", {})
    if sim:
        sections.append("")
        sections.append("=" * 60)
        sections.append("SIMULATION RESULTS (Hazard3 RISC-V)")
        sections.append("=" * 60)
        sections.append(f"  Clock Cycles: {sim.get('cycles', 'N/A'):,}")
        sections.append(f"  Output Match: {sim.get('output_match', 'N/A')}")

    # ── Synthesis Results ───────────────────────────────────────
    synth = state.get("synthesis_result", {})
    if synth:
        sections.append("")
        sections.append("=" * 60)
        sections.append("SYNTHESIS RESULTS (OpenROAD)")
        sections.append("=" * 60)
        sections.append(f"  Power:     {synth.get('power_watts', 'N/A')} W")
        sections.append(f"  Area:      {synth.get('area_mm2', 'N/A')} mm²")
        sections.append(f"  Frequency: {synth.get('frequency_mhz', 'N/A')} MHz")
        sections.append(f"  Cells:     {synth.get('cell_count', 'N/A')}")

    # ── Previous Optimizations ──────────────────────────────────
    prev_suggestions = state.get("optimization_suggestions", [])
    if prev_suggestions:
        sections.append("")
        sections.append("=" * 60)
        sections.append("PREVIOUSLY APPLIED OPTIMIZATIONS")
        sections.append("=" * 60)
        for s in prev_suggestions:
            sections.append(f"  • {s}")
        sections.append(
            "\nSuggest NEW optimizations that are different from the above."
        )

    sections.append("")
    sections.append("=" * 60)
    sections.append("TASK")
    sections.append("=" * 60)
    sections.append(
        "Analyze the simulation and synthesis results above in detail. "
        "Reference concrete functions, buffers, loops, weight arrays, and "
        "IR nodes from the current output file wherever possible. "
        "Suggest up to 5 concrete code optimizations to improve "
        "performance (reduce cycles), power, and/or area. "
        "Return your suggestions as a JSON array."
    )

    return "\n".join(sections)


def _parse_suggestions(response: str) -> list[str]:
    """Extract optimization suggestions from LLM response."""
    suggestions = []

    # Try to parse JSON array from response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            for item in items:
                if isinstance(item, dict):
                    suggestion = item.get("suggestion", "")
                    category = item.get("category", "")
                    target = item.get("target", "")
                    if suggestion:
                        s = f"[{category}] {target}: {suggestion}"
                        suggestions.append(s)
                elif isinstance(item, str):
                    suggestions.append(item)
        except json.JSONDecodeError:
            pass

    # Fallback: extract bullet points
    if not suggestions:
        for line in response.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•")
                         or line.startswith("*")):
                suggestions.append(line.lstrip("-•* "))
            elif re.match(r'^\d+[\.\)]\s', line):
                suggestions.append(re.sub(r'^\d+[\.\)]\s*', '', line))

    return suggestions[:5]  # Max 5 suggestions


def optimize(state: AgentState) -> dict:
    """
    LangGraph node function: Hardware-aware optimization hints pass.

    Reads: state["generated_code"], state["ir_graph"]
           (simulation_result / synthesis_result not yet available at this
           point in the new flow — optimizer works from IR + code only).
    Writes: state["optimization_suggestions"], state["optimization_iteration"]
    """
    iteration = state.get("optimization_iteration", 0) + 1
    logger.info(f"Optimization iteration {iteration}/{MAX_OPTIMIZATION_ITERATIONS}")

    # ── Build and send LLM prompt ───────────────────────────────
    system_prompt = _load_system_prompt()
    user_prompt = _build_optimization_prompt(state)

    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        model=VLLM_MODEL,
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    logger.info(f"Calling LLM for optimization suggestions...")
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    llm_latency = time.perf_counter() - t0
    raw_response = response.content

    # ── Parse suggestions ────────────────────────────────
    suggestions = _parse_suggestions(raw_response)

    logger.info(f"Generated {len(suggestions)} optimization suggestions:")
    for i, s in enumerate(suggestions, 1):
        logger.info(f"  {i}. {s}")

    # ── Token stats ────────────────────────────────────────
    usage = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  0) if isinstance(usage, dict) else getattr(usage, "input_tokens",  0)
    out_tok = usage.get("output_tokens", 0) if isinstance(usage, dict) else getattr(usage, "output_tokens", 0)
    logger.info(
        f"  optimizer — input_tokens={in_tok}, output_tokens={out_tok}, "
        f"latency={llm_latency:.2f}s"
    )

    new_stat = LLMCallStats(
        agent="optimizer",
        call_label=f"optimize_iter_{iteration}",
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        latency_s=llm_latency,
    )

    existing_stats: list[LLMCallStats] = list(state.get("llm_call_stats") or [])
    existing_stats.append(new_stat)
    prev_input   = state.get("total_input_tokens",  0) or 0
    prev_output  = state.get("total_output_tokens", 0) or 0
    prev_latency = state.get("total_llm_latency_s", 0.0) or 0.0
    prev_agent_latencies: dict = dict(state.get("agent_latencies") or {})
    prev_agent_latencies["optimizer"] = (
        prev_agent_latencies.get("optimizer", 0.0) + llm_latency
    )

    return {
        "optimization_suggestions": suggestions,
        "optimization_iteration": iteration,
        # Telemetry
        "llm_call_stats": existing_stats,
        "total_input_tokens":  prev_input  + in_tok,
        "total_output_tokens": prev_output + out_tok,
        "total_llm_latency_s": prev_latency + llm_latency,
        "agent_latencies": prev_agent_latencies,
    }


#agents/report.py#
"""
Report Agent — Generates the final summary report.

Aggregates all pipeline results into a formatted Markdown report
including model info, code stats, simulation, synthesis, and
optimization history.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from ir import IRGraph
from state import AgentState

logger = logging.getLogger(__name__)


def generate_report(state: AgentState) -> dict:
    """
    LangGraph node function: Generate final report.

    Reads: all state fields
    Writes: state["final_report"]
    """
    logger.info("=" * 60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 60)

    model_name = state.get("model_name", "Unknown Model")
    ir_dict = state.get("ir_graph", {})
    ir_graph = IRGraph.from_dict(ir_dict) if ir_dict else None
    sim = state.get("simulation_result", {})
    synth = state.get("synthesis_result", {})
    verification = state.get("verification_result", {})
    opt_suggestions = state.get("optimization_suggestions", [])
    opt_iteration = state.get("optimization_iteration", 0)

    lines = []

    # ── Header ──────────────────────────────────────────────────
    lines.append("# 🚀 Agentic RISC-V Compiler — Final Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Target ISA:** rv32imac (RISC-V)")
    lines.append(f"**Processor:** Hazard3")
    lines.append("")

    # ── Model Summary ───────────────────────────────────────────
    lines.append("## 📊 Model Summary")
    lines.append("")
    if ir_graph:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(
            f"| Total Parameters | {state.get('total_params', 0):,} |"
        )
        lines.append(
            f"| Weight Memory | "
            f"{state.get('model_memory_bytes', 0) / (1024*1024):.2f} MB |"
        )
        lines.append(
            f"| Activation Memory | "
            f"{ir_graph.total_activation_memory() / 1024:.1f} KB |"
        )
        lines.append(f"| IR Nodes | {len(ir_graph.nodes)} |")
        lines.append("")

        # Op breakdown
        op_counts: dict[str, int] = {}
        for node in ir_graph.nodes:
            op_counts[node.op] = op_counts.get(node.op, 0) + 1

        lines.append("### Layer Breakdown")
        lines.append("")
        lines.append("| Operation | Count |")
        lines.append("|-----------|-------|")
        for op, count in sorted(op_counts.items()):
            lines.append(f"| {op} | {count} |")
        lines.append("")

    # ── Verification ────────────────────────────────────────────
    lines.append("## ✅ Verification")
    lines.append("")
    v_attempts = state.get("verification_attempts", 0)
    v_passed = verification.get("passed", False)
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Status | {'PASSED ✅' if v_passed else 'FAILED ❌'} |")
    lines.append(f"| Attempts | {v_attempts} |")
    lines.append(f"| Errors | {len(verification.get('errors', []))} |")
    lines.append(f"| Warnings | {len(verification.get('warnings', []))} |")
    lines.append("")

    if verification.get("warnings"):
        lines.append("### Warnings")
        for w in verification["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    # ── Generated Code Stats ────────────────────────────────────
    lines.append("## 💻 Generated Code")
    lines.append("")
    code = state.get("generated_code", "")
    header = state.get("generated_header", "")
    lines.append(f"| File | Lines | Size |")
    lines.append(f"|------|-------|------|")
    lines.append(
        f"| model.c | {len(code.splitlines())} | "
        f"{len(code.encode('utf-8')) / 1024:.1f} KB |"
    )
    lines.append(
        f"| weights.h | {len(header.splitlines())} | "
        f"{len(header.encode('utf-8')) / 1024:.1f} KB |"
    )
    lines.append("")

    # ── Simulation Results ──────────────────────────────────────
    lines.append("## ⚡ Simulation Results (Hazard3)")
    lines.append("")
    if sim and sim.get("success"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Clock Cycles | {sim.get('cycles', 'N/A'):,} |")
        lines.append(
            f"| Output Correctness | "
            f"{'Match ✅' if sim.get('output_match') else 'Mismatch ⚠️'} |"
        )
        lines.append("")
    else:
        error = sim.get("raw_log", "Simulation not run or failed")
        lines.append(f"⚠️ Simulation did not complete: {error}")
        lines.append("")

    # ── Synthesis Results ───────────────────────────────────────
    lines.append("## 🔧 Synthesis Results (OpenROAD)")
    lines.append("")
    if synth and synth.get("success"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Power | {synth.get('power_watts', 0):.4f} W |")
        lines.append(f"| Area | {synth.get('area_mm2', 0):.4f} mm² |")
        lines.append(f"| Max Frequency | {synth.get('frequency_mhz', 0):.1f} MHz |")
        lines.append(f"| Cell Count | {synth.get('cell_count', 0):,} |")
        lines.append("")

        # Derived metrics
        cycles = sim.get("cycles", 0)
        freq = synth.get("frequency_mhz", 0)
        if cycles > 0 and freq > 0:
            exec_time_ms = (cycles / (freq * 1e6)) * 1000
            lines.append("### Derived Metrics")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Estimated Execution Time | {exec_time_ms:.2f} ms |")
            lines.append(
                f"| Energy per Inference | "
                f"{synth['power_watts'] * exec_time_ms / 1000:.6f} J |"
            )
            lines.append(
                f"| Throughput | "
                f"{1000 / exec_time_ms:.1f} inferences/sec |"
            )
            lines.append("")
    else:
        lines.append("⚠️ Synthesis did not complete or was not run.")
        lines.append("")

    # ── Optimization History ────────────────────────────────────
    if opt_iteration > 0:
        lines.append("## 🔄 Optimization History")
        lines.append("")
        lines.append(f"**Iterations completed:** {opt_iteration}")
        lines.append("")
        if opt_suggestions:
            lines.append("### Applied Optimizations")
            for i, s in enumerate(opt_suggestions, 1):
                lines.append(f"{i}. {s}")
            lines.append("")

    # ── Telemetry ────────────────────────────────────────────────
    lines.append("## 📈 Telemetry")
    lines.append("")
    total_in  = state.get("total_input_tokens",  0) or 0
    total_out = state.get("total_output_tokens", 0) or 0
    total_llm = state.get("total_llm_latency_s", 0.0) or 0.0
    lines.append("### Token Usage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Input Tokens  | {total_in:,} |")
    lines.append(f"| Total Output Tokens | {total_out:,} |")
    lines.append(f"| Total Tokens        | {total_in + total_out:,} |")
    lines.append(f"| Total LLM Latency   | {total_llm:.2f}s |")
    lines.append("")

    agent_latencies: dict = state.get("agent_latencies") or {}
    if agent_latencies:
        lines.append("### Per-Agent Wall-Clock Latency")
        lines.append("")
        lines.append("| Agent | Latency (s) |")
        lines.append("|-------|-------------|")
        for agent, lat in sorted(agent_latencies.items()):
            lines.append(f"| {agent} | {lat:.2f} |")
        lines.append("")

    call_stats: list = state.get("llm_call_stats") or []
    if call_stats:
        lines.append("### LLM Call Breakdown")
        lines.append("")
        lines.append("| Agent | Call | Input Tokens | Output Tokens | Total | Latency (s) |")
        lines.append("|-------|------|-------------|--------------|-------|-------------|")
        for s in call_stats:
            if isinstance(s, dict):
                a, lbl = s.get("agent","?"), s.get("call_label","?")
                inp, out = s.get("input_tokens",0), s.get("output_tokens",0)
                tot, lat = s.get("total_tokens",0), s.get("latency_s",0.0)
            else:
                a, lbl = getattr(s,"agent","?"), getattr(s,"call_label","?")
                inp, out = getattr(s,"input_tokens",0), getattr(s,"output_tokens",0)
                tot, lat = getattr(s,"total_tokens",0), getattr(s,"latency_s",0.0)
            lines.append(f"| {a} | {lbl} | {inp:,} | {out:,} | {tot:,} | {lat:.2f} |")
        lines.append("")

    # ── Pipeline Summary ────────────────────────────────────────

    lines.append("## 📋 Pipeline Summary")
    lines.append("")
    lines.append("```")
    lines.append("PyTorch Model")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("FX Graph Trace ────── ✅")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("Custom IR ─────────── ✅")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"C Code Generation ─── ✅ ({v_attempts} attempt(s))")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"Verification ──────── {'✅' if v_passed else '❌'}")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"Human Approval ────── {'✅' if state.get('human_approved') else '⏳'}")
    if opt_iteration > 0:
        lines.append("    │")
        lines.append("    ▼")
        lines.append(f"HW Optimization ───── ✅ ({opt_iteration} hint(s) injected)")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"Hazard3 Simulation ── {'✅' if sim.get('success') else '⏳'}")
    lines.append("    │")
    lines.append("    ▼")
    lines.append(f"OpenROAD Synthesis ── {'✅' if synth.get('success') else '⏳'}")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("Final Report ──────── ✅")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by Agentic RISC-V Compiler*")

    report = "\n".join(lines)

    # Save report to file
    output_dir = Path(os.getcwd()) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {report_path}")

    return {"final_report": report}


#agents/simulator.py#
"""
Simulation Agent — Runs the generated C code on Hazard3 RISC-V simulator.

Wraps the Hazard3 CXXRTL simulation flow:
1. Cross-compile C code → RISC-V ELF binary
2. Run on Hazard3 simulator
3. Parse output for cycle counts and execution trace
4. Compare outputs against PyTorch reference values

Falls back to mock simulation when toolchains are unavailable.
"""

from __future__ import annotations

import logging
import os
import time

from state import AgentState
from tools.compile import compile_to_elf, find_compiler, _is_riscv_compiler
from tools.hazard3 import run_hazard3_simulation, _mock_simulation

logger = logging.getLogger(__name__)


def simulate(state: AgentState) -> dict:
    """
    LangGraph node function: Run Hazard3 simulation.

    Reads: state["code_path"], state["header_path"],
           state["reference_outputs"]
    Writes: state["simulation_result"]
    """
    code_path = state.get("code_path", "")
    header_path = state.get("header_path", "")
    reference_outputs = state.get("reference_outputs", [])

    logger.info("=" * 60)
    logger.info("HAZARD3 RISC-V SIMULATION")
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # ── Step 1: Cross-compile to RISC-V ELF ─────────────────────
    output_dir = os.path.dirname(code_path) if code_path else "output"
    elf_path = os.path.join(output_dir, "firmware.elf")

    logger.info("Step 1: Cross-compiling to RISC-V ELF...")
    compile_ok, compile_output = compile_to_elf(
        source_path=code_path,
        output_path=elf_path,
        include_dir=os.path.dirname(header_path) if header_path else output_dir,
    )

    if not compile_ok:
        logger.warning(f"Cross-compilation failed: {compile_output}")
        logger.info("Falling back to mock simulation (toolchain unavailable)")

        # Use mock simulation so the pipeline can continue
        sim_result = _mock_simulation(elf_path)
        sim_result["raw_log"] = (
            f"[MOCK — cross-compilation failed]\n"
            f"Compiler output: {compile_output}\n\n"
            f"{sim_result.get('raw_log', '')}"
        )

        sim_elapsed_mock = time.perf_counter() - t_start
        prev_al: dict = dict(state.get("agent_latencies") or {})
        prev_al["simulator"] = prev_al.get("simulator", 0.0) + sim_elapsed_mock
        return {"simulation_result": sim_result, "agent_latencies": prev_al}


    logger.info(f"Cross-compilation successful: {elf_path}")

    # ── Step 2: Run Hazard3 simulator ───────────────────────────
    logger.info("Step 2: Running Hazard3 simulation...")
    sim_result = run_hazard3_simulation(elf_path)

    if not sim_result["success"]:
        logger.error(f"Simulation failed: {sim_result.get('error', 'Unknown error')}")
        return {"simulation_result": sim_result}

    logger.info(f"Simulation completed in {sim_result['cycles']:,} cycles")

    # ── Step 3: Compare outputs ─────────────────────────────────
    if reference_outputs and sim_result.get("output_values"):
        output_values = sim_result["output_values"]
        match = True
        tolerance = 1e-3  # Allow small floating-point differences

        if len(output_values) != len(reference_outputs):
            match = False
            logger.warning(
                f"Output size mismatch: got {len(output_values)}, "
                f"expected {len(reference_outputs)}"
            )
        else:
            for i, (actual, expected) in enumerate(
                zip(output_values, reference_outputs)
            ):
                if abs(actual - expected) > tolerance:
                    match = False
                    logger.warning(
                        f"Output mismatch at index {i}: "
                        f"got {actual:.6f}, expected {expected:.6f}"
                    )
                    break

        sim_result["output_match"] = match
        if match:
            logger.info("✅ Output values match PyTorch reference")
        else:
            logger.warning("⚠️ Output values differ from PyTorch reference")
    else:
        sim_result["output_match"] = True  # No reference to compare against
        logger.info("No reference outputs for comparison — skipping")

    # ── Step 4: Log summary ─────────────────────────────────────
    logger.info(f"  Cycles: {sim_result['cycles']:,}")
    logger.info(f"  Output match: {sim_result['output_match']}")

    sim_elapsed = time.perf_counter() - t_start
    logger.info(f"  Simulation wall-clock: {sim_elapsed:.2f}s")
    prev_agent_latencies: dict = dict(state.get("agent_latencies") or {})
    prev_agent_latencies["simulator"] = (
        prev_agent_latencies.get("simulator", 0.0) + sim_elapsed
    )

    return {
        "simulation_result": sim_result,
        "agent_latencies": prev_agent_latencies,
    }


#agents/synthesis.py#
"""
Synthesis Agent — Runs OpenROAD to synthesize the Hazard3 SoC
and measure power, area, and frequency.

Wraps the OpenROAD-flow-scripts (ORFS) pipeline:
1. Yosys logic synthesis
2. OpenROAD physical design (floorplan → place → CTS → route)
3. Parse timing/power/area reports
"""

from __future__ import annotations

import logging
import os
import time

from state import AgentState
from tools.openroad import run_openroad_flow

logger = logging.getLogger(__name__)


def synthesize(state: AgentState) -> dict:
    """
    LangGraph node function: Run OpenROAD synthesis.

    Reads: state["simulation_result"]
    Writes: state["synthesis_result"]
    """
    logger.info("=" * 60)
    logger.info("OPENROAD SYNTHESIS")
    logger.info("=" * 60)

    t_start = time.perf_counter()
    sim_result = state.get("simulation_result", {})

    # ── Run the OpenROAD flow ───────────────────────────────────
    logger.info("Running OpenROAD synthesis flow...")
    logger.info("  Steps: Yosys → Floorplan → Place → CTS → Route → Report")

    synthesis_result = run_openroad_flow(
        design_name=state.get("model_name", "model"),
        sim_cycles=sim_result.get("cycles", 0),
    )

    if synthesis_result["success"]:
        logger.info("✅ Synthesis completed successfully")
        logger.info(f"  Power:     {synthesis_result['power_watts']:.4f} W")
        logger.info(f"  Area:      {synthesis_result['area_mm2']:.4f} mm²")
        logger.info(f"  Frequency: {synthesis_result['frequency_mhz']:.1f} MHz")
        logger.info(f"  Cells:     {synthesis_result['cell_count']:,}")
    else:
        logger.error(
            f"Synthesis failed: "
            f"{synthesis_result.get('detailed_report', 'Unknown error')}"
        )

    synth_elapsed = time.perf_counter() - t_start
    logger.info(f"  Synthesis wall-clock: {synth_elapsed:.2f}s")
    prev_al: dict = dict(state.get("agent_latencies") or {})
    prev_al["synthesizer"] = prev_al.get("synthesizer", 0.0) + synth_elapsed

    return {
        "synthesis_result": synthesis_result,
        "agent_latencies": prev_al,
    }


#agents/verifier.py#
"""
Verification Agent — Validates the generated C code.

Performs:
1. Syntax check (compile with -fsyntax-only)
2. Full compilation to object file
3. Structural validation (all IR ops mapped to C)
4. Static analysis for common issues

No LLM required — fully deterministic.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from agents.codegen_contract import (
    helper_name_from_signature,
    required_helper_signatures,
)
from ir import IRGraph, IROpType
from state import AgentState
from tools.compile import (
    check_syntax,
    compile_to_object,
    find_compiler,
)

logger = logging.getLogger(__name__)

MAX_VERIFICATION_ATTEMPTS = 5


def _read_text_file(path_value: str, fallback: str) -> str:
    """Read generated artifact text from disk when available."""
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return fallback


def _check_structural_completeness(
    code: str, ir_dict: dict, model_header: str = ""
) -> list[str]:
    """
    Check that the generated C code covers all IR operations.
    Returns a list of error/warning messages.
    """
    issues = []
    ir_graph = IRGraph.from_dict(ir_dict)

    # Check that model_inference function exists
    if "model_inference" not in code:
        issues.append(
            "ERROR: Missing 'model_inference' function. "
            "The entry point must be: "
            "void model_inference(const float* input, float* output);"
        )

    # Check that weights.h is included
    if '#include "weights.h"' not in code and '#include "model.h"' not in code:
        issues.append(
            'ERROR: Missing #include "model.h" or #include "weights.h". '
            "Weight arrays must be imported through the generated headers."
        )

    # Check for each weight tensor referenced in the IR
    for node in ir_graph.nodes:
        if node.weight_key:
            c_name = node.weight_key.replace(".", "_")
            if c_name not in code:
                issues.append(
                    f"WARNING: Weight '{node.weight_key}' (C name: {c_name}) "
                    f"referenced by node '{node.id}' ({node.op}) "
                    f"not found in generated code."
                )

    # Check for common operation implementations
    op_patterns = {
        IROpType.CONV2D: ["conv2d", "convolution", "kernel"],
        IROpType.LINEAR: ["linear", "gemm", "matmul", "weight"],
        IROpType.RELU: ["relu", "> 0", "max("],
        IROpType.SILU: ["silu", "sigmoid", "expf"],
        IROpType.RMSNORM: ["rmsnorm", "rms", "sqrt"],
        IROpType.SOFTMAX: ["softmax", "expf", "sum"],
        IROpType.EMBEDDING: ["embed", "token"],
        IROpType.ATTENTION: ["attention", "query", "score"],
    }

    ops_in_graph = {node.op for node in ir_graph.nodes}
    code_lower = code.lower()

    for op in ops_in_graph:
        if op in (IROpType.TENSOR_INPUT, IROpType.TENSOR_OUTPUT,
                  IROpType.DROPOUT):
            continue
        patterns = op_patterns.get(op, [])
        if patterns and not any(p in code_lower for p in patterns):
            issues.append(
                f"WARNING: IR operation '{op}' may not be implemented — "
                f"none of the expected patterns {patterns} found in code."
            )

    return issues


def _check_model_header(model_header: str, ir_dict: dict | None = None) -> list[str]:
    """Validate the LLM-generated model.h contract."""
    issues: list[str] = []
    if not model_header.strip():
        issues.append("ERROR: model.h is empty.")
        return issues

    if "#pragma once" not in model_header and "#ifndef" not in model_header:
        issues.append("WARNING: model.h has no include guard.")
    if '#include "weights.h"' not in model_header:
        issues.append('ERROR: model.h must include "weights.h".')
    if "model_inference" not in model_header:
        issues.append(
            "ERROR: model.h must declare model_inference(const float* input, float* output)."
        )
    if re.search(r'\bstatic\s+float\s+\w+\s*\[', model_header):
        issues.append(
            "ERROR: model.h must not define activation storage arrays; define storage in model.c."
        )
    if ir_dict is not None:
        for signature in required_helper_signatures(ir_dict):
            helper_name = helper_name_from_signature(signature)
            if not re.search(rf"\b{re.escape(helper_name)}\s*\(", model_header):
                issues.append(
                    f"ERROR: model.h is missing required helper prototype "
                    f"for IR operation helper '{helper_name}': {signature}"
                )
    return issues


def _check_required_helper_definitions(code: str, ir_dict: dict) -> list[str]:
    """Validate that model.c implements every IR-required helper function."""
    issues: list[str] = []
    for signature in required_helper_signatures(ir_dict):
        helper_name = helper_name_from_signature(signature)
        if not re.search(
            rf"(^|\n)\s*(?:static\s+)?(?:inline\s+)?\w[\w\s\*]*\b"
            rf"{re.escape(helper_name)}\s*\([^;]*\)\s*\{{",
            code,
            re.DOTALL,
        ):
            issues.append(
                f"ERROR: model.c is missing required helper implementation "
                f"for '{helper_name}' declared by the IR contract."
            )
    return issues


def _check_common_errors(code: str) -> list[str]:
    """Check for common C code errors."""
    issues = []

    # Check for malloc/calloc/free (forbidden in bare-metal)
    if re.search(r'\b(malloc|calloc|realloc|free)\b', code):
        issues.append(
            "ERROR: Dynamic memory allocation detected (malloc/calloc/free). "
            "All arrays must be statically allocated for bare-metal RISC-V."
        )

    # Check for missing semicolons after closing braces (common LLM error)
    # This is a heuristic — just flag obvious patterns
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Check for unterminated string literals
        if stripped.count('"') % 2 != 0 and '//' not in stripped.split('"')[0]:
            issues.append(
                f"WARNING: Possible unterminated string literal on line {i}: "
                f"{stripped[:60]}..."
            )

    # Check for C++ features
    cpp_patterns = [
        (r'\bclass\b', "C++ 'class' keyword"),
        (r'\btemplate\b', "C++ 'template' keyword"),
        (r'\bnew\b\s+\w+', "C++ 'new' operator"),
        (r'\bstd::', "C++ std:: namespace"),
        (r'\bcout\b', "C++ cout"),
        (r'\bvector\b', "C++ vector"),
    ]
    for pattern, description in cpp_patterns:
        if re.search(pattern, code):
            issues.append(
                f"ERROR: C++ feature detected: {description}. "
                "Code must be pure C99."
            )

    # Check for reasonable buffer sizes
    # Flag arrays larger than 100MB (likely a mistake)
    array_decls = re.findall(
        r'(?:static\s+)?(?:const\s+)?float\s+\w+\[(\d+)\]', code
    )
    for size_str in array_decls:
        size = int(size_str)
        mem_mb = (size * 4) / (1024 * 1024)
        if mem_mb > 100:
            issues.append(
                f"WARNING: Very large array ({size} floats = {mem_mb:.1f} MB). "
                "This may exceed RISC-V memory. Consider quantization."
            )

    return issues


def _check_header(header: str, weight_metadata: dict) -> list[str]:
    """
    Validate the weights.h header file.

    Since weights.h is now deterministically generated by the pipeline
    (not by the LLM), this check is simpler — it mainly validates
    that the header was generated correctly and contains all expected
    weight tensors with actual values (not zero placeholders).
    """
    issues = []

    if not header.strip():
        issues.append("ERROR: weights.h is empty.")
        return issues

    # Check include guard
    if "#pragma once" not in header and "#ifndef" not in header:
        issues.append(
            "WARNING: weights.h has no include guard. "
            "Add '#pragma once' or '#ifndef WEIGHTS_H'."
        )

    # Check that each weight tensor is declared
    for name, meta in weight_metadata.items():
        c_name = name.replace(".", "_")
        if c_name not in header:
            issues.append(
                f"WARNING: Weight '{name}' (C name: {c_name}) "
                f"not declared in weights.h."
            )

    # Check that header has actual weight values, not just placeholders
    # (The deterministic generator should embed real values)
    if "Auto-generated" in header and "Fallback mode" in header:
        issues.append(
            "WARNING: weights.h is using fallback zero-initialized values. "
            "The weight export may have failed. Check weights.npz exists."
        )

    return issues


def verify_code(state: AgentState) -> dict:
    """
    LangGraph node function: Verify the generated C code.

    Reads: state["generated_code"], state["generated_header"],
           state["ir_graph"], state["weights_metadata"],
           state["code_path"], state["header_path"]
    Writes: state["verification_result"], state["verification_attempts"],
            state["verification_feedback"]
    """
    code_path = state.get("code_path", "")
    header_path = state.get("header_path", "")
    model_header_path = state.get("model_header_path", "")
    code = _read_text_file(code_path, state.get("generated_code", ""))
    header = _read_text_file(header_path, state.get("generated_header", ""))
    model_header = _read_text_file(
        model_header_path, state.get("generated_model_header", "")
    )
    ir_dict = state.get("ir_graph", {})
    weight_metadata = state.get("weights_metadata", {})
    attempt = state.get("verification_attempts", 0) + 1

    logger.info(f"Verification attempt {attempt}/{MAX_VERIFICATION_ATTEMPTS}")
    t_start = time.perf_counter()

    all_errors: list[str] = []
    all_warnings: list[str] = []
    compiler_output = ""

    # ── 1. Structural completeness ──────────────────────────────
    structural = _check_structural_completeness(code, ir_dict, model_header)
    for issue in structural:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    # ── 2. Common error patterns ────────────────────────────────
    common = _check_common_errors(code)
    for issue in common:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    helper_definition_issues = _check_required_helper_definitions(code, ir_dict)
    for issue in helper_definition_issues:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    # ── 3. Header validation ────────────────────────────────────
    model_header_issues = _check_model_header(model_header, ir_dict)
    for issue in model_header_issues:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    header_issues = _check_header(header, weight_metadata)
    for issue in header_issues:
        if issue.startswith("ERROR"):
            all_errors.append(issue)
        else:
            all_warnings.append(issue)

    # ── 4. Compilation check ────────────────────────────────────
    if code_path and os.path.exists(code_path):
        compiler = find_compiler()
        if compiler:
            # Syntax check first
            syntax_ok, syntax_output = check_syntax(
                code_path, include_dir=os.path.dirname(header_path)
            )
            if not syntax_ok:
                all_errors.append(
                    f"COMPILATION ERROR (syntax check):\n{syntax_output}"
                )
                compiler_output = syntax_output
            else:
                # Full compilation
                compile_ok, compile_output = compile_to_object(
                    code_path, include_dir=os.path.dirname(header_path)
                )
                if not compile_ok:
                    all_errors.append(
                        f"COMPILATION ERROR:\n{compile_output}"
                    )
                    compiler_output = compile_output
                else:
                    compiler_output = "Compilation successful."
                    logger.info("✓ Compilation successful")
        else:
            all_warnings.append(
                "WARNING: No C compiler found. Skipping compilation check. "
                "Install riscv32-unknown-elf-gcc or gcc."
            )
    else:
        all_warnings.append(
            "WARNING: Code file not found on disk. "
            "Skipping compilation check."
        )

    # ── Build result ────────────────────────────────────────────
    passed = len(all_errors) == 0

    if passed:
        logger.info("✅ Verification PASSED")
    else:
        logger.info(
            f"❌ Verification FAILED with {len(all_errors)} error(s)"
        )

    # Build feedback string for the LLM (used on retry)
    feedback_lines = []
    if all_errors:
        feedback_lines.append("ERRORS (must fix):")
        for e in all_errors:
            feedback_lines.append(f"  • {e}")
    if all_warnings:
        feedback_lines.append("\nWARNINGS (should fix):")
        for w in all_warnings:
            feedback_lines.append(f"  • {w}")

    return {
        "verification_result": {
            "passed": passed,
            "errors": all_errors,
            "warnings": all_warnings,
            "compiler_output": compiler_output,
        },
        "verification_attempts": attempt,
        "verification_feedback": "\n".join(feedback_lines) if not passed else "",
        # Telemetry
        "agent_latencies": {
            **dict(state.get("agent_latencies") or {}),
            "verifier": (
                (state.get("agent_latencies") or {}).get("verifier", 0.0)
                + (time.perf_counter() - t_start)
            ),
        },
    }


#tools/__init__.py#
"""Agentic RISC-V Compiler — Tool wrappers for external toolchains."""


#tools/compile.py#
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


#tools/export_weights.py#
"""
Weight Export Utility — Converts .npz weights to C-embeddable formats.

Supports multiple precision modes:
  - f32   : Full 32-bit float (default, bit-exact)
  - f16   : IEEE 754 half-precision (16-bit)
  - bf16  : Brain floating-point 16-bit
  - mxfp8 : Microscaling FP8 (E4M3 variant)

Outputs:
  - weights.bin  : Flat binary file (all tensors concatenated)
  - weights_manifest.json : Maps param names → offset, size, shape

Model-agnostic — works for any PyTorch model (LLMs, CNNs, etc.)
since it operates on raw numpy arrays from model.state_dict().
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ── Precision Modes ─────────────────────────────────────────────

PrecisionMode = Literal["f32", "f16", "bf16", "mxfp8"]

# C type names for each precision
C_TYPE_MAP: dict[str, str] = {
    "f32": "float",
    "f16": "uint16_t",   # stored as raw bits, decoded in C
    "bf16": "uint16_t",  # stored as raw bits, decoded in C
    "mxfp8": "uint8_t",  # stored as raw bits, decoded in C
}

# Bytes per element for each precision
BYTES_PER_ELEMENT: dict[str, int] = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "mxfp8": 1,
}


def _convert_to_f16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to float16 (IEEE 754 half)."""
    return arr.astype(np.float16)


def _convert_to_bf16(arr: np.ndarray) -> np.ndarray:
    """
    Convert float32 array to bfloat16.
    
    BF16 is the upper 16 bits of f32 (sign + 8-bit exponent + 7-bit mantissa).
    We store as uint16 raw bits.
    """
    # View f32 as uint32, shift right 16 bits to get bf16
    f32_bits = arr.astype(np.float32).view(np.uint32)
    # Round-to-nearest-even: add rounding bias
    rounding_bias = (f32_bits >> 16) & 1  # LSB of bf16
    rounding_bias += 0x7FFF  # bias toward even
    bf16_bits = ((f32_bits + rounding_bias) >> 16).astype(np.uint16)
    return bf16_bits


def _convert_to_mxfp8_e4m3(arr: np.ndarray) -> np.ndarray:
    """
    Convert float32 array to MXFP8 E4M3 format.
    
    E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits.
    Range: [-448, 448], special: NaN (0x7F and 0xFF), no inf.
    
    This is a simplified conversion; for production use, 
    per-block scaling (microscaling) should be applied.
    """
    result = np.zeros(arr.shape, dtype=np.uint8)
    flat_in = arr.astype(np.float32).ravel()
    flat_out = result.ravel()
    
    for i in range(len(flat_in)):
        val = float(flat_in[i])
        
        # Handle special cases
        if np.isnan(val):
            flat_out[i] = 0x7F  # E4M3 NaN
            continue
        
        # Determine sign
        sign = 0
        if val < 0:
            sign = 1
            val = -val
        
        # Clamp to E4M3 max range
        max_val = 448.0
        if val > max_val:
            val = max_val
        
        if val == 0.0:
            flat_out[i] = sign << 7
            continue
        
        # Find exponent (bias = 7 for E4M3)
        import math
        exp = int(math.floor(math.log2(val))) if val > 0 else 0
        exp_biased = exp + 7  # E4M3 bias = 7
        
        if exp_biased <= 0:
            # Subnormal
            exp_biased = 0
            mantissa = val / (2.0 ** (-6))  # 2^(1-bias) = 2^(-6)
            mantissa_bits = int(round(mantissa * 8)) & 0x07  # 3 mantissa bits
        elif exp_biased >= 15:
            # Clamp to max normal (exponent=14, mantissa=111)
            exp_biased = 14
            mantissa_bits = 7
        else:
            # Normal
            mantissa = val / (2.0 ** exp) - 1.0  # Remove leading 1
            mantissa_bits = int(round(mantissa * 8)) & 0x07
            if mantissa_bits > 7:
                mantissa_bits = 7
        
        flat_out[i] = (sign << 7) | (exp_biased << 3) | mantissa_bits
    
    return result


def convert_array(arr: np.ndarray, precision: PrecisionMode) -> bytes:
    """
    Convert a numpy array to the specified precision and return raw bytes.
    
    Args:
        arr: Input array (any dtype, will be cast to float32 first)
        precision: Target precision mode
        
    Returns:
        Raw bytes in the target precision
    """
    arr = arr.astype(np.float32)
    
    if precision == "f32":
        return arr.tobytes()
    elif precision == "f16":
        return _convert_to_f16(arr).tobytes()
    elif precision == "bf16":
        return _convert_to_bf16(arr).tobytes()
    elif precision == "mxfp8":
        return _convert_to_mxfp8_e4m3(arr).tobytes()
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def export_weights_binary(
    npz_path: str | Path,
    output_dir: str | Path,
    precision: PrecisionMode = "f32",
) -> tuple[str, dict]:
    """
    Export weights from .npz to a flat binary file + JSON manifest.
    
    Args:
        npz_path: Path to the .npz weights file
        output_dir: Directory for output files
        precision: Weight precision mode
        
    Returns:
        (bin_path, manifest) tuple where manifest maps param names
        to {offset, size_bytes, numel, shape, c_name, c_type}
    """
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bin_path = output_dir / "weights.bin"
    manifest_path = output_dir / "weights_manifest.json"
    
    data = np.load(npz_path)
    manifest: dict[str, dict] = {}
    
    c_type = C_TYPE_MAP[precision]
    elem_size = BYTES_PER_ELEMENT[precision]
    
    offset = 0
    with open(bin_path, "wb") as f:
        for name in sorted(data.files):
            arr = data[name]
            raw_bytes = convert_array(arr, precision)
            f.write(raw_bytes)
            
            c_name = name.replace(".", "_")
            numel = int(np.prod(arr.shape))
            size_bytes = numel * elem_size
            
            manifest[name] = {
                "c_name": c_name,
                "offset": offset,
                "size_bytes": size_bytes,
                "numel": numel,
                "shape": list(arr.shape),
                "c_type": c_type,
                "precision": precision,
            }
            
            offset += size_bytes
    
    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(
        f"Exported weights: {bin_path} "
        f"({offset:,} bytes, {precision}, {len(manifest)} tensors)"
    )
    
    return str(bin_path), manifest


def _float_to_c_literal(val: float) -> str:
    """Sanitize float values for C code, replacing inf/nan."""
    if np.isnan(val):
        return "0.0f"
    if np.isposinf(val):
        return "3.402823466e+38f" # FLT_MAX
    if np.isneginf(val):
        return "-3.402823466e+38f" # -FLT_MAX
    return f"{val:.9e}f"


def generate_weights_header(
    manifest: dict[str, dict],
    npz_path: str | Path,
    precision: PrecisionMode = "f32",
    mode: Literal["embedded", "binary"] = "embedded",
) -> str:
    """
    Generate a C header file declaring/defining weight arrays.
    
    Args:
        manifest: Weight manifest from export_weights_binary()
        npz_path: Path to the .npz file (needed for embedded mode)
        precision: Weight precision
        mode: 
            "embedded" — weight values baked into header as C array initializers
                         (works on bare-metal, no filesystem needed)
            "binary"   — extern declarations + load_weights() from .bin file
                         (smaller header, requires fopen/fread)
    
    Returns:
        Complete weights.h content as a string
    """
    c_type = C_TYPE_MAP[precision]
    lines = [
        "#pragma once",
        f"/* Auto-generated weight declarations — precision: {precision} */",
        f"/* Mode: {mode} */",
        "",
    ]
    
    if precision != "f32":
        lines.append("#include <stdint.h>")
        lines.append("")
        # Add decode helpers for non-f32 types
        if precision == "bf16":
            lines.extend([
                "/* BF16 → float32 decode helper */",
                "static inline float bf16_to_f32(uint16_t bf16) {",
                "    uint32_t f32_bits = ((uint32_t)bf16) << 16;",
                "    float result;",
                "    __builtin_memcpy(&result, &f32_bits, sizeof(float));",
                "    return result;",
                "}",
                "",
            ])
        elif precision == "f16":
            lines.extend([
                "/* FP16 → float32 decode helper */",
                "static inline float f16_to_f32(uint16_t h) {",
                "    uint32_t sign = (h >> 15) & 0x1;",
                "    uint32_t exp  = (h >> 10) & 0x1F;",
                "    uint32_t mant = h & 0x3FF;",
                "    uint32_t f32;",
                "    if (exp == 0) {",
                "        if (mant == 0) { f32 = sign << 31; }",
                "        else {",
                "            exp = 1;",
                "            while (!(mant & 0x400)) { mant <<= 1; exp--; }",
                "            mant &= 0x3FF;",
                "            f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);",
                "        }",
                "    } else if (exp == 31) {",
                "        f32 = (sign << 31) | 0x7F800000 | (mant << 13);",
                "    } else {",
                "        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);",
                "    }",
                "    float result;",
                "    __builtin_memcpy(&result, &f32, sizeof(float));",
                "    return result;",
                "}",
                "",
            ])
        elif precision == "mxfp8":
            lines.extend([
                "/* MXFP8 E4M3 → float32 decode helper */",
                "static inline float mxfp8_to_f32(uint8_t v) {",
                "    uint32_t sign = (v >> 7) & 0x1;",
                "    uint32_t exp  = (v >> 3) & 0xF;",
                "    uint32_t mant = v & 0x7;",
                "    float result;",
                "    if (exp == 0 && mant == 0) { return sign ? -0.0f : 0.0f; }",
                "    if (exp == 15 && mant == 7) {",
                "        /* NaN */",
                "        uint32_t nan_bits = 0x7FC00000 | (sign << 31);",
                "        __builtin_memcpy(&result, &nan_bits, sizeof(float));",
                "        return result;",
                "    }",
                "    if (exp == 0) {",
                "        /* Subnormal: value = (-1)^sign * 2^(-6) * (mant/8) */",
                "        result = (mant / 8.0f) * (1.0f / 64.0f);",
                "    } else {",
                "        /* Normal: value = (-1)^sign * 2^(exp-7) * (1 + mant/8) */",
                "        float m = 1.0f + mant / 8.0f;",
                "        int e = (int)exp - 7;",
                "        result = m;",
                "        if (e > 0) { for (int i = 0; i < e; i++) result *= 2.0f; }",
                "        else { for (int i = 0; i < -e; i++) result /= 2.0f; }",
                "    }",
                "    return sign ? -result : result;",
                "}",
                "",
            ])
    
    if mode == "embedded":
        # Load actual weight values from .npz and embed as C array initializers
        data = np.load(str(npz_path))
        
        for name in sorted(manifest.keys()):
            info = manifest[name]
            c_name = info["c_name"]
            numel = info["numel"]
            shape = info["shape"]
            arr = data[name].astype(np.float32).ravel()
            
            lines.append(f"/* {name}: shape={shape}, numel={numel} */")
            
            if precision == "f32":
                # Embed as float array with full precision
                values = ", ".join(_float_to_c_literal(v) for v in arr)
                lines.append(
                    f"static const float {c_name}[{numel}] = {{{values}}};"
                )
            elif precision == "bf16":
                bf16_arr = _convert_to_bf16(arr.reshape(-1))
                values = ", ".join(f"0x{v:04X}" for v in bf16_arr)
                lines.append(
                    f"static const uint16_t {c_name}[{numel}] = {{{values}}};"
                )
            elif precision == "f16":
                f16_arr = arr.astype(np.float16).view(np.uint16)
                values = ", ".join(f"0x{v:04X}" for v in f16_arr)
                lines.append(
                    f"static const uint16_t {c_name}[{numel}] = {{{values}}};"
                )
            elif precision == "mxfp8":
                fp8_arr = _convert_to_mxfp8_e4m3(arr.reshape(-1))
                values = ", ".join(f"0x{v:02X}" for v in fp8_arr.ravel())
                lines.append(
                    f"static const uint8_t {c_name}[{numel}] = {{{values}}};"
                )
            
            lines.append("")
        
    elif mode == "binary":
        # Extern declarations + load_weights() function
        lines.append("/* Weight arrays — loaded from weights.bin at runtime */")
        lines.append("")
        
        for name in sorted(manifest.keys()):
            info = manifest[name]
            c_name = info["c_name"]
            numel = info["numel"]
            shape = info["shape"]
            lines.append(f"/* {name}: shape={shape}, numel={numel} */")
            lines.append(f"extern {c_type} {c_name}[{numel}];")
            lines.append("")
        
        lines.extend([
            "/* Load all weights from a binary file. Returns 0 on success. */",
            "int load_weights(const char* filepath);",
            "",
        ])
    
    return "\n".join(lines)


def generate_weights_loader(
    manifest: dict[str, dict],
    precision: PrecisionMode = "f32",
) -> str:
    """
    Generate weights_loader.c — implements load_weights() for binary mode.
    
    Only needed when mode="binary".
    
    Args:
        manifest: Weight manifest from export_weights_binary()
        precision: Weight precision
        
    Returns:
        Complete weights_loader.c content as a string
    """
    c_type = C_TYPE_MAP[precision]
    elem_size = BYTES_PER_ELEMENT[precision]
    
    lines = [
        '#include "weights.h"',
        "#include <stdio.h>",
        "#include <string.h>",
        "",
        f"/* Weight storage — precision: {precision} */",
        "",
    ]
    
    # Define storage arrays
    for name in sorted(manifest.keys()):
        info = manifest[name]
        c_name = info["c_name"]
        numel = info["numel"]
        shape = info["shape"]
        lines.append(f"/* {name}: shape={shape} */")
        lines.append(f"{c_type} {c_name}[{numel}];")
        lines.append("")
    
    # load_weights function
    lines.extend([
        "int load_weights(const char* filepath) {",
        '    FILE* f = fopen(filepath, "rb");',
        "    if (!f) return -1;",
        "",
    ])
    
    for name in sorted(manifest.keys()):
        info = manifest[name]
        c_name = info["c_name"]
        numel = info["numel"]
        lines.append(
            f"    if (fread({c_name}, {elem_size}, {numel}, f) != {numel}) "
            f"{{ fclose(f); return -2; }}"
        )
    
    lines.extend([
        "",
        "    fclose(f);",
        "    return 0;",
        "}",
        "",
    ])
    
    return "\n".join(lines)


#tools/hazard3.py#
"""
Hazard3 Simulation Tool — Wraps the Hazard3 RISC-V CXXRTL simulator.

Flow:
1. Build the CXXRTL simulator from Hazard3 Verilog (cached)
2. Run the firmware ELF on the simulator
3. Parse stdout for cycle counts and output values

Supports a mock mode for demo/testing when Hazard3 is not installed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────
HAZARD3_DIR = os.environ.get("HAZARD3_DIR", "")
HAZARD3_SIM = os.environ.get("HAZARD3_SIM", "")  # Pre-built simulator path
MOCK_MODE = os.environ.get("MOCK_SIMULATION", "true").lower() == "true"


def _find_hazard3() -> Optional[str]:
    """Find the Hazard3 simulator binary."""
    # Check explicit path
    if HAZARD3_SIM and os.path.isfile(HAZARD3_SIM):
        return HAZARD3_SIM

    # Check in HAZARD3_DIR
    if HAZARD3_DIR:
        candidates = [
            os.path.join(HAZARD3_DIR, "build", "tb_cxxrtl"),
            os.path.join(HAZARD3_DIR, "sim", "tb_cxxrtl"),
            os.path.join(HAZARD3_DIR, "tb_cxxrtl"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    # Check PATH
    sim_path = shutil.which("hazard3_sim")
    if sim_path:
        return sim_path

    return None


def _build_hazard3_simulator() -> tuple[bool, str]:
    """
    Build the Hazard3 CXXRTL simulator from source.

    Requires: Yosys, Clang, Hazard3 source code.
    """
    if not HAZARD3_DIR:
        return False, "HAZARD3_DIR not set — cannot build simulator"

    if not os.path.isdir(HAZARD3_DIR):
        return False, f"HAZARD3_DIR does not exist: {HAZARD3_DIR}"

    logger.info("Building Hazard3 CXXRTL simulator...")

    try:
        result = subprocess.run(
            ["make", "sim"],
            cwd=HAZARD3_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, "Simulator built successfully"
        else:
            return False, f"Build failed: {result.stderr}"
    except Exception as e:
        return False, f"Build error: {str(e)}"


def _parse_simulation_output(output: str) -> dict:
    """
    Parse the Hazard3 simulator stdout for results.

    Expected output format (from a modified testbench):
    ```
    [CYCLES] 1234567
    [OUTPUT] 0.123 0.456 0.789 ...
    [DONE]
    ```
    """
    cycles = 0
    output_values: list[float] = []

    for line in output.split("\n"):
        line = line.strip()

        # Parse cycle count
        cycle_match = re.search(r'\[CYCLES?\]\s*(\d+)', line)
        if cycle_match:
            cycles = int(cycle_match.group(1))

        # Parse output values
        output_match = re.search(r'\[OUTPUT\]\s*([\d\.\-\s]+)', line)
        if output_match:
            values_str = output_match.group(1).strip()
            for v in values_str.split():
                try:
                    output_values.append(float(v))
                except ValueError:
                    pass

        # Look for cycle count in other common formats
        if not cycles:
            alt_match = re.search(
                r'(?:cycles?|clk|clock)\s*[=:]\s*(\d+)', line, re.IGNORECASE
            )
            if alt_match:
                cycles = int(alt_match.group(1))

    return {
        "cycles": cycles,
        "output_values": output_values,
    }


def _mock_simulation(elf_path: str) -> dict:
    """
    Generate mock simulation results for demo purposes.

    Produces realistic-looking results based on file size
    as a rough proxy for code complexity.
    """
    import random
    random.seed(42)  # Deterministic for demo

    # Estimate cycles based on ELF size
    try:
        elf_size = os.path.getsize(elf_path)
    except OSError:
        elf_size = 10000

    # Heuristic: ~100 cycles per byte of code (rough estimate)
    base_cycles = max(10000, elf_size * 100)
    cycles = base_cycles + random.randint(-base_cycles // 10, base_cycles // 10)

    # Generate mock output values
    output_values = [
        round(random.uniform(-1.0, 1.0), 6)
        for _ in range(10)
    ]

    logger.info(f"[MOCK] Simulated {cycles:,} cycles")
    logger.info(f"[MOCK] Generated {len(output_values)} output values")

    return {
        "success": True,
        "cycles": cycles,
        "execution_trace": (
            f"[MOCK SIMULATION]\n"
            f"ELF: {elf_path}\n"
            f"Cycles: {cycles:,}\n"
            f"Output: {output_values[:5]}..."
        ),
        "output_values": output_values,
        "output_match": True,
        "raw_log": "[MOCK] Simulation completed successfully",
    }


def run_hazard3_simulation(elf_path: str) -> dict:
    """
    Run the Hazard3 CXXRTL simulation.

    Args:
        elf_path: Path to the compiled RISC-V ELF binary.

    Returns:
        SimulationResult dict with cycles, output values, etc.
    """
    # ── Check if we should use mock mode ────────────────────────
    if MOCK_MODE:
        logger.info("Running in MOCK simulation mode")
        return _mock_simulation(elf_path)

    # ── Find or build the simulator ─────────────────────────────
    sim_path = _find_hazard3()

    if sim_path is None:
        # Try building
        build_ok, build_msg = _build_hazard3_simulator()
        if build_ok:
            sim_path = _find_hazard3()
        if sim_path is None:
            logger.warning(
                f"Hazard3 simulator not available: {build_msg}. "
                "Falling back to mock mode."
            )
            return _mock_simulation(elf_path)

    # ── Run the simulation ──────────────────────────────────────
    logger.info(f"Running Hazard3 simulator: {sim_path}")
    logger.info(f"Firmware: {elf_path}")

    try:
        result = subprocess.run(
            [sim_path, elf_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        raw_log = result.stdout + result.stderr

        if result.returncode != 0:
            return {
                "success": False,
                "cycles": 0,
                "execution_trace": "",
                "output_values": [],
                "output_match": False,
                "raw_log": f"Simulator exited with code {result.returncode}:\n{raw_log}",
            }

        # Parse output
        parsed = _parse_simulation_output(raw_log)

        return {
            "success": True,
            "cycles": parsed["cycles"],
            "execution_trace": raw_log[:2000],  # Truncate for state
            "output_values": parsed["output_values"],
            "output_match": True,  # Will be checked by simulator agent
            "raw_log": raw_log,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "cycles": 0,
            "execution_trace": "",
            "output_values": [],
            "output_match": False,
            "raw_log": "Simulation timed out after 600 seconds",
        }
    except Exception as e:
        return {
            "success": False,
            "cycles": 0,
            "execution_trace": "",
            "output_values": [],
            "output_match": False,
            "raw_log": f"Simulation error: {str(e)}",
        }


#tools/openroad.py#
"""
OpenROAD Synthesis Tool — Wraps the OpenROAD-flow-scripts pipeline.

Flow:
1. Yosys logic synthesis
2. OpenROAD physical design (floorplan → place → CTS → route)
3. Parse timing/power/area reports

Supports mock mode for demo/testing when OpenROAD is not installed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────
OPENROAD_FLOW_DIR = os.environ.get("OPENROAD_FLOW_DIR", "")
OPENROAD_BIN = os.environ.get("OPENROAD_BIN", "")
YOSYS_BIN = os.environ.get("YOSYS_BIN", "")
PDK = os.environ.get("OPENROAD_PDK", "sky130hd")
MOCK_MODE = os.environ.get("MOCK_SYNTHESIS", "true").lower() == "true"


def _find_openroad() -> Optional[str]:
    """Find the OpenROAD executable."""
    if OPENROAD_BIN and os.path.isfile(OPENROAD_BIN):
        return OPENROAD_BIN
    return shutil.which("openroad")


def _find_yosys() -> Optional[str]:
    """Find the Yosys executable."""
    if YOSYS_BIN and os.path.isfile(YOSYS_BIN):
        return YOSYS_BIN
    return shutil.which("yosys")


def _parse_power_report(report: str) -> float:
    """Extract total power from OpenROAD power report."""
    # Look for pattern: "Total ... <number> W" or "total_power: <number>"
    patterns = [
        r'[Tt]otal\s+(?:\w+\s+)*(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*[Ww]',
        r'total_power\s*[=:]\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Total\s+Power\s*[=:]\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, report)
        if match:
            return float(match.group(1))
    return 0.0


def _parse_area_report(report: str) -> tuple[float, int]:
    """Extract area and cell count from OpenROAD area report."""
    area = 0.0
    cells = 0

    # Area patterns
    area_patterns = [
        r'[Tt]otal\s+[Aa]rea\s*[=:]\s*(\d+\.?\d*)',
        r'[Dd]ie\s+[Aa]rea\s*[=:]\s*(\d+\.?\d*)',
        r'area\s*[=:]\s*(\d+\.?\d*)',
    ]
    for pattern in area_patterns:
        match = re.search(pattern, report)
        if match:
            area = float(match.group(1))
            break

    # Cell count patterns
    cell_patterns = [
        r'[Nn]umber\s+of\s+[Cc]ells?\s*[=:]\s*(\d+)',
        r'[Cc]ell\s+[Cc]ount\s*[=:]\s*(\d+)',
        r'(\d+)\s+cells?',
    ]
    for pattern in cell_patterns:
        match = re.search(pattern, report)
        if match:
            cells = int(match.group(1))
            break

    return area, cells


def _parse_timing_report(report: str) -> float:
    """Extract maximum frequency from OpenROAD timing report."""
    # Look for clock period or frequency
    period_match = re.search(
        r'[Cc]lock\s+[Pp]eriod\s*[=:]\s*(\d+\.?\d*)', report
    )
    if period_match:
        period_ns = float(period_match.group(1))
        if period_ns > 0:
            return 1000.0 / period_ns  # Convert ns to MHz

    freq_match = re.search(
        r'[Ff]requency\s*[=:]\s*(\d+\.?\d*)\s*[Mm][Hh]z', report
    )
    if freq_match:
        return float(freq_match.group(1))

    # Slack-based: f = 1 / (target_period - slack)
    slack_match = re.search(r'[Ss]lack\s*[=:]\s*(-?\d+\.?\d*)', report)
    if slack_match:
        slack = float(slack_match.group(1))
        # Assume 10ns target period (100 MHz)
        effective_period = 10.0 - slack
        if effective_period > 0:
            return 1000.0 / effective_period

    return 0.0


def _mock_synthesis(design_name: str, sim_cycles: int) -> dict:
    """
    Generate mock synthesis results for demo purposes.

    Produces realistic-looking results for a Hazard3 SoC
    on Sky130 process.
    """
    import random
    random.seed(hash(design_name) % 2**32)

    # Hazard3 baseline: ~15k cells, ~0.5mm² on Sky130
    # Application code adds some overhead
    base_cells = 15000
    app_cells = random.randint(2000, 8000)
    total_cells = base_cells + app_cells

    # Area scales with cell count (~33 um² per cell on Sky130)
    area_um2 = total_cells * 33.0
    area_mm2 = area_um2 / 1e6

    # Power: ~50-200 mW typical for Hazard3 at moderate frequency
    power_mw = 50 + random.uniform(30, 150)
    power_w = power_mw / 1000

    # Frequency: 50-150 MHz on Sky130
    freq_mhz = random.uniform(50, 150)

    detailed_report = (
        f"[MOCK SYNTHESIS REPORT]\n"
        f"Design: {design_name}\n"
        f"PDK: {PDK}\n"
        f"Target Clock: 10.0 ns (100 MHz)\n"
        f"\n"
        f"=== Synthesis (Yosys) ===\n"
        f"  Cells: {total_cells:,}\n"
        f"  Hazard3 Core: {base_cells:,} cells\n"
        f"  Application Logic: {app_cells:,} cells\n"
        f"\n"
        f"=== Physical Design (OpenROAD) ===\n"
        f"  Die Area: {area_mm2:.4f} mm²\n"
        f"  Utilization: {random.uniform(40, 75):.1f}%\n"
        f"\n"
        f"=== Timing ===\n"
        f"  Max Frequency: {freq_mhz:.1f} MHz\n"
        f"  WNS (Worst Negative Slack): {random.uniform(-0.5, 0.5):.3f} ns\n"
        f"\n"
        f"=== Power ===\n"
        f"  Total Power: {power_w:.4f} W ({power_mw:.1f} mW)\n"
        f"  Dynamic: {power_mw * 0.7:.1f} mW\n"
        f"  Leakage: {power_mw * 0.3:.1f} mW\n"
    )

    if sim_cycles > 0 and freq_mhz > 0:
        exec_time_ms = (sim_cycles / (freq_mhz * 1e6)) * 1000
        energy_j = power_w * exec_time_ms / 1000
        detailed_report += (
            f"\n=== Derived Metrics ===\n"
            f"  Execution Time: {exec_time_ms:.2f} ms "
            f"({sim_cycles:,} cycles @ {freq_mhz:.0f} MHz)\n"
            f"  Energy/Inference: {energy_j:.6f} J\n"
            f"  Throughput: {1000 / exec_time_ms:.1f} inf/s\n"
        )

    logger.info(f"[MOCK] Synthesis completed: {total_cells} cells, "
                f"{area_mm2:.4f} mm², {power_w:.4f} W, {freq_mhz:.1f} MHz")

    return {
        "success": True,
        "power_watts": round(power_w, 4),
        "area_mm2": round(area_mm2, 4),
        "frequency_mhz": round(freq_mhz, 1),
        "cell_count": total_cells,
        "detailed_report": detailed_report,
    }


def run_openroad_flow(
    design_name: str = "model",
    sim_cycles: int = 0,
    rtl_dir: str = "",
) -> dict:
    """
    Run the OpenROAD synthesis flow.

    Args:
        design_name: Name of the design.
        sim_cycles: Clock cycles from simulation (for derived metrics).
        rtl_dir: Directory containing RTL source files.

    Returns:
        SynthesisResult dict with power, area, frequency, etc.
    """
    # ── Check for mock mode ─────────────────────────────────────
    if MOCK_MODE:
        logger.info("Running in MOCK synthesis mode")
        return _mock_synthesis(design_name, sim_cycles)

    # ── Find tools ──────────────────────────────────────────────
    openroad = _find_openroad()
    yosys = _find_yosys()

    if not openroad or not yosys:
        logger.warning(
            f"OpenROAD or Yosys not found "
            f"(openroad={openroad}, yosys={yosys}). "
            "Falling back to mock mode."
        )
        return _mock_synthesis(design_name, sim_cycles)

    # ── Run ORFS Make flow ──────────────────────────────────────
    if OPENROAD_FLOW_DIR and os.path.isdir(OPENROAD_FLOW_DIR):
        logger.info("Running OpenROAD-flow-scripts...")

        try:
            # Run full flow
            result = subprocess.run(
                [
                    "make",
                    f"DESIGN_CONFIG={design_name}/config.mk",
                ],
                cwd=os.path.join(OPENROAD_FLOW_DIR, "flow"),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "power_watts": 0,
                    "area_mm2": 0,
                    "frequency_mhz": 0,
                    "cell_count": 0,
                    "detailed_report": (
                        f"ORFS failed:\n{result.stderr[:2000]}"
                    ),
                }

            # Parse reports
            flow_output = result.stdout + result.stderr
            power = _parse_power_report(flow_output)
            area, cells = _parse_area_report(flow_output)
            freq = _parse_timing_report(flow_output)

            return {
                "success": True,
                "power_watts": power,
                "area_mm2": area / 1e6,  # Convert from um² to mm²
                "frequency_mhz": freq,
                "cell_count": cells,
                "detailed_report": flow_output[:5000],
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "power_watts": 0,
                "area_mm2": 0,
                "frequency_mhz": 0,
                "cell_count": 0,
                "detailed_report": "Synthesis timed out after 30 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "power_watts": 0,
                "area_mm2": 0,
                "frequency_mhz": 0,
                "cell_count": 0,
                "detailed_report": f"Synthesis error: {str(e)}",
            }
    else:
        logger.warning("OPENROAD_FLOW_DIR not set. Falling back to mock mode.")
        return _mock_synthesis(design_name, sim_cycles)


#examples/demo_model.py#
"""
Demo Model — Simplified TinyLlama for end-to-end pipeline testing.

This is a scaled-down LLaMA-style transformer that can be traced
with torch.fx.symbolic_trace(). Uses:
- RMSNorm (custom)
- Rotary Positional Embedding (precomputed, no dynamic control flow)
- Multi-head Self-Attention (no KV cache, fixed seq_len)
- SwiGLU MLP
- Embedding + LM Head

Parameters are intentionally tiny (~200K) so the generated C code
is manageable and the pipeline runs quickly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ───────────────────────────────────────────────

@dataclass
class TinyLlamaConfig:
    """Tiny config for demo purposes."""
    vocab_size: int = 512        # Small vocabulary
    hidden_dim: int = 64         # Small hidden dimension
    num_heads: int = 4           # 4 attention heads
    head_dim: int = 16           # hidden_dim // num_heads
    num_layers: int = 2          # Only 2 transformer layers
    max_seq_len: int = 32        # Short sequences
    intermediate_dim: int = 128  # MLP intermediate size (2x hidden)
    rms_norm_eps: float = 1e-5
    rope_base: float = 10000.0


# ── Custom Modules (FX-traceable) ───────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Precomputed Rotary Positional Embedding.

    Stores sin/cos tables as buffers (not parameters) to avoid
    dynamic computation that breaks FX tracing.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        # Precompute frequency table
        freqs = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim/2]

        # Store as buffers
        self.register_buffer("cos_cached", torch.cos(angles))  # [seq, dim/2]
        self.register_buffer("sin_cached", torch.sin(angles))  # [seq, dim/2]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to q and k.
        q, k: [batch, num_heads, seq_len, head_dim]
        """
        cos = self.cos_cached  # [seq, dim/2]
        sin = self.sin_cached  # [seq, dim/2]

        # Reshape for broadcasting: [1, 1, seq, dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split q and k into even/odd pairs
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        # Apply rotation
        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rot, k_rot


class Attention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # Q, K, V, O projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Rotary embeddings
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_base
        )

        # Causal mask (precomputed)
        mask = torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

        self.scale = 1.0 / math.sqrt(config.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_dim]
        returns: [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # [B, S, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: [B, num_heads, S, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, seq_len)

        # Attention scores: [B, heads, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        scores = scores + self.causal_mask

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum: [B, heads, S, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [B, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch, seq_len, self.hidden_dim
        )

        # Output projection
        return self.o_proj(attn_output)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP block (LLaMA-style)."""

    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_dim, config.intermediate_dim, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_dim, config.intermediate_dim, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_dim, config.hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """Single transformer block (LLaMA-style)."""

    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        h = x + self.attention(self.attn_norm(x))
        # Pre-norm MLP with residual
        out = h + self.mlp(self.mlp_norm(h))
        return out


class TinyLlamaDemo(nn.Module):
    """
    Simplified TinyLlama model for demo purposes.

    Architecture:
    - Token embedding
    - N transformer blocks (RMSNorm + Attention + SwiGLU MLP)
    - Final RMSNorm
    - Linear LM head (tied with embeddings optionally)

    ~200K parameters with default config.
    """

    def __init__(self, config: TinyLlamaConfig | None = None):
        super().__init__()
        if config is None:
            config = TinyLlamaConfig()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # Final norm
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch, seq_len] (integer token IDs)
        returns: [batch, seq_len, vocab_size] (logits)
        """
        # Embed tokens
        h = self.embed_tokens(input_ids)  # [B, S, D]

        # Transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Final norm + LM head
        h = self.norm(h)
        logits = self.lm_head(h)  # [B, S, vocab_size]

        return logits


# ── Helper functions ────────────────────────────────────────────

def create_demo_model() -> tuple[TinyLlamaDemo, torch.Tensor]:
    """
    Create the demo model and a sample input for tracing.

    Returns:
        (model, sample_input) tuple.
    """
    config = TinyLlamaConfig()
    model = TinyLlamaDemo(config)
    model.eval()

    # Sample input: batch=1, seq_len=32
    sample_input = torch.randint(0, config.vocab_size, (1, config.max_seq_len))

    return model, sample_input


def get_reference_output(
    model: TinyLlamaDemo, sample_input: torch.Tensor
) -> list[float]:
    """Run the model and get reference output values for comparison."""
    with torch.no_grad():
        output = model(sample_input)
        # Return the logits for the last token
        last_logits = output[0, -1, :].tolist()
        return last_logits


if __name__ == "__main__":
    # Quick test
    model, sample = create_demo_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sample input shape: {sample.shape}")

    # Test forward pass
    with torch.no_grad():
        out = model(sample)
    print(f"Output shape: {out.shape}")
    print(f"Output sample: {out[0, -1, :5].tolist()}")

    # Test FX tracing
    try:
        import torch.fx
        traced = torch.fx.symbolic_trace(model)
        print(f"\nFX trace successful!")
        print(f"Graph nodes: {len(list(traced.graph.nodes))}")
        print(f"\nFX Graph:\n{traced.graph}")
    except Exception as e:
        print(f"\nFX trace failed: {e}")
        print("This is expected for models with dynamic control flow.")
        print("Use torch.compile or manual tracing instead.")


#tests/test_code_generator.py#
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.code_generator as code_generator
from agents.codegen_contract import required_helper_signatures
from agents.verifier import _check_model_header, _check_required_helper_definitions


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        prompt = messages[-1].content
        if "STEP 2 TASK: CREATE model.h" in prompt:
            return _FakeResponse(
                '```c model.h\n'
                '#pragma once\n'
                '#include "weights.h"\n'
                'void model_inference(const float* input, float* output);\n'
                '```'
            )
        if "STEP 3 TASK: IMPLEMENT model.c" in prompt:
            assert 'STEP 2 model.h CONTRACT TO IMPLEMENT' in prompt
            assert '#include "weights.h"' in prompt
            return _FakeResponse(
                '```c model.c\n'
                '#include "model.h"\n'
                'void model_inference(const float* input, float* output) {\n'
                '    output[0] = input[0];\n'
                '}\n'
                '```'
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:200]}")


def _minimal_state() -> dict:
    return {
        "ir_graph": {
            "model_name": "unit_model",
            "nodes": [
                {
                    "id": "input",
                    "op": "TENSOR_INPUT",
                    "inputs": [],
                    "shape": [1],
                },
                {
                    "id": "output",
                    "op": "TENSOR_OUTPUT",
                    "inputs": ["input"],
                    "shape": [],
                },
            ],
            "input_shapes": {"input": [1]},
            "weight_metadata": {},
        },
        "weights_metadata": {},
        "weight_precision": "f32",
        "weight_mode": "embedded",
        "verification_attempts": 0,
        "optimization_iteration": 0,
    }


def _linear_ir_graph() -> dict:
    return {
        "model_name": "linear_model",
        "nodes": [
            {
                "id": "input",
                "op": "TENSOR_INPUT",
                "inputs": [],
                "shape": [4],
            },
            {
                "id": "fc",
                "op": "LINEAR",
                "inputs": ["input"],
                "shape": [2],
                "weight_key": "fc.weight",
                "bias_key": "fc.bias",
            },
            {
                "id": "output",
                "op": "TENSOR_OUTPUT",
                "inputs": ["fc"],
                "shape": [],
            },
        ],
        "input_shapes": {"input": [4]},
        "weight_metadata": {
            "fc.weight": {"shape": [2, 4], "dtype": "float32", "numel": 8},
            "fc.bias": {"shape": [2], "dtype": "float32", "numel": 2},
        },
    }


def test_extract_c_artifact_requires_exactly_one_named_block():
    assert (
        code_generator._extract_c_artifact(
            '```c model.h\n#pragma once\n```', "model.h"
        )
        == "#pragma once"
    )

    with pytest.raises(ValueError, match="Expected exactly one fenced code block"):
        code_generator._extract_c_artifact("#pragma once", "model.h")

    with pytest.raises(ValueError, match="Expected exactly one fenced code block"):
        code_generator._extract_c_artifact(
            '```c model.h\n#pragma once\n```\n```c model.h\n#pragma once\n```',
            "model.h",
        )


def test_generate_code_writes_model_header_and_implementation(monkeypatch, tmp_path):
    monkeypatch.setattr(code_generator, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(code_generator, "ChatOpenAI", _FakeChatOpenAI)

    result = code_generator.generate_code(_minimal_state())

    model_h_path = Path(result["model_header_path"])
    model_c_path = Path(result["code_path"])
    weights_h_path = Path(result["header_path"])

    assert model_h_path == tmp_path / "model.h"
    assert model_c_path == tmp_path / "model.c"
    assert weights_h_path == tmp_path / "weights.h"
    assert model_h_path.read_text(encoding="utf-8").startswith("#pragma once")
    assert '#include "model.h"' in model_c_path.read_text(encoding="utf-8")
    assert result["generated_model_header"] == model_h_path.read_text(encoding="utf-8")
    assert result["generated_code"] == model_c_path.read_text(encoding="utf-8")


def test_generate_code_write_path_has_no_legacy_functions_header_variables():
    source = Path(code_generator.__file__).read_text(encoding="utf-8")

    assert "functions_h" not in source
    assert "funcs_path" not in source


def test_required_helpers_are_declared_and_implemented_for_ir_ops():
    ir_graph = _linear_ir_graph()

    assert required_helper_signatures(ir_graph) == [
        "void linear(const float* in, const float* weight, const float* bias, float* out, int in_features, int out_features);"
    ]

    missing_header = (
        '#pragma once\n'
        '#include "weights.h"\n'
        'void model_inference(const float* input, float* output);\n'
    )
    assert any(
        "missing required helper prototype" in issue
        for issue in _check_model_header(missing_header, ir_graph)
    )

    complete_header = missing_header + required_helper_signatures(ir_graph)[0] + "\n"
    assert not _check_model_header(complete_header, ir_graph)

    missing_impl = (
        '#include "model.h"\n'
        'void model_inference(const float* input, float* output) { output[0] = input[0]; }\n'
    )
    assert any(
        "missing required helper implementation" in issue
        for issue in _check_required_helper_definitions(missing_impl, ir_graph)
    )

    complete_impl = (
        missing_impl
        + "void linear(const float* in, const float* weight, const float* bias, "
        + "float* out, int in_features, int out_features) { out[0] = 0.0f; }\n"
    )
    assert not _check_required_helper_definitions(complete_impl, ir_graph)




#requirements.txt#
"""
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-core>=0.3.0
torch>=2.0.0
numpy>=1.24.0

"""

#README.md#
"""
RISCify
"Turn PyTorch Models into RISC-V Executables Automatically"
A multi-agent LangGraph system that converts PyTorch models into optimized bare-metal C code targeting the RISC-V (`rv32imac`) architecture, simulates it, and synthesizes it for hardware metrics.

## Features

- **Hardware-Friendly IR**: Bridges PyTorch semantics to C-level loops.
- **LLM Code Generation**: Uses Qwen 2.5 Coder to generate pure C99 inference code without dynamic memory allocation.
- **Automated Verification**: Syntax checking, static analysis, and full cross-compilation feedback loop.
- **Human-in-the-Loop**: Execution pauses for user review of the generated C code.
- **Hardware Simulation**: Wraps Hazard3 CXXRTL to extract exact cycle counts.
- **RTL Synthesis**: Wraps OpenROAD-flow-scripts to extract power (W), area (mm²), and max frequency.
- **Self-Optimization**: Feeds hardware metrics back into an LLM optimizer to suggest loop tiling, fusion, and unrolling.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set LLM environment variables (defaults to local vLLM endpoint):
   ```bash
   export VLLM_BASE_URL="http://localhost:8000/v1"
   export VLLM_API_KEY="your_api_key"
   export VLLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
   ```

## Usage

### Run the Demo (TinyLlama scaled-down)
```bash
python main.py --demo --optimize
```

### Visualizing the IR Graph
The tool generates a Custom IR from your PyTorch model. You can visualize it in various formats:

```bash
# Terminal ASCII
python visualize_ir.py output/ir_graph.json --format terminal

# Interactive HTML (opens in browser, dark theme, includes Mermaid diagram)
python visualize_ir.py output/ir_graph.json --format html --output output/ir_graph.html
```

## System Requirements
- **LLM**: Local or cloud endpoint serving Qwen2.5-Coder-32B.
- **Simulation/Synthesis Tools** *(Optional, will mock if unavailable)*:
  - `riscv32-unknown-elf-gcc`
  - Hazard3 RISC-V Core
  - Yosys & OpenROAD

## Pipeline Workflow

1. `parse_fx_graph`: No-LLM parsing of `torch.fx` to Custom IR.
2. `generate_code`: deterministically writes `weights.h`, asks the LLM for
   a `model.h` contract, then asks the LLM to implement `model.c`.
3. `verify_code`: Compilation checks (loops back to generation if failed).
4. **Human Review**: System pauses. User approves or rejects.
5. `simulate`: Run binary on Hazard3.
6. `synthesize`: Run OpenROAD for area/power.
7. `optimize`: (Optional) Analyze metrics and loop back to generation.
8. `report`: Generate `report.md`.

"""
