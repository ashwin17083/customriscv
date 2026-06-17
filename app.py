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
