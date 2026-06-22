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

2. Set LLM environment variables based on your chosen backend:

   **For local vLLM endpoint (default):**
   ```bash
   export LLM_BACKEND="vllm"
   export VLLM_BASE_URL="http://localhost:8000/v1"
   export VLLM_API_KEY="your_api_key"
   export VLLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
   ```

   **For Ollama endpoint:**
   ```bash
   export LLM_BACKEND="ollama"
   export OLLAMA_BASE_URL="http://localhost:11434"
   export OLLAMA_MODEL="deepseek-coder-v2:16b-lite-instruct-q4_K_M"
   ```

## Usage

### Run the Demo (TinyLlama scaled-down)
```bash
python main.py --demo
```

### Run the Demo with Hardware-Aware Optimization
```bash
python main.py --demo --optimize
```
When running with `--optimize`, the pipeline loads `hw_config.yaml` from the root directory. This config details available custom instructions (e.g. systolic array matrix multipliers, vector processing units), multi-core distribution strategies, and vector extension widths. The system will invoke a hardware-aware optimization agent to rewrite the generated C code into optimized versions (`model_optimized.c` / `model_optimized.h`) using these hardware primitives.

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

1. `parse_fx_graph`: Extracts the `torch.fx` GraphModule to a Custom hardware-friendly IR.
2. `generate_code`: Deterministically exports weights, generates a `model.h` contract, and prompts the LLM to implement `model.c`.
3. `verify_code` (Verifier #1): Performs cross-compiler syntax checks (auto-retries up to 5 times on failures).
4. **Human Review #1**: Pipeline pauses to let the user inspect and approve the original C code.
5. `hw_optimize` (Optional, when `--optimize` is enabled): Rewrites code targeting primitives in `hw_config.yaml`.
6. `verify_optimized` (Verifier #2): Verifies the optimized code's syntax and functional correctness.
7. **Human Review #2 / Compiler Decision**: Pauses to approve the hardware-optimized code or decide how to proceed if cross-compilers are missing.
8. `simulate`: Compiles and runs model inference on Hazard3 (or runs mock simulation).
9. `synthesize`: Generates layout gate count, power, and area metrics with OpenROAD.
10. `report`: Generates the final telemetry and performance report.
