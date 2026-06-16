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
