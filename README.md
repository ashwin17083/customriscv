# Agentic RISC-V Compiler

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
