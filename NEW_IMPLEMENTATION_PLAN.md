# Fix Pipeline: Code Persistence, Two-Phase LLM, Routing & Weight Sanitization

## Background

Seven issues identified in the current agentic RISC-V compiler pipeline need to be resolved. They span code generation, weight export, graph routing, and CLI ergonomics.

---

## Issue 1: `generated_code` Is Empty on Retries

### Root Cause

LangGraph's `TypedDict` state merging works by **replacing** keys that the returning node provides. The `generate_code` node returns `generated_code` correctly on its first run, but the `verify_code` node does **not** return `generated_code` — so LangGraph may drop it depending on the reducer configuration. Additionally, when the graph loops back to `generate_code`, the state snapshot may not carry the previous `generated_code` forward because `TypedDict` states in LangGraph use *last-writer-wins* semantics without a reducer for `str` fields.

### Fix: File-Backed Code Persistence

Instead of relying on LangGraph state alone, persist the generated code to a temp file (`output/_latest_model.c`) and read it back on retries.

#### Files Changed

##### [MODIFY] [code_generator.py](file:///c:/Coding%20projects/agentic-riscv/agents/code_generator.py)
- After writing `model.c`, also write to `output/_latest_model.c` (the persistent temp file)
- In `_build_user_prompt()`, when `is_retry` is True, read `current_code` from `output/_latest_model.c` **instead of** `state.get("generated_code", "")`
- Each successful generation overwrites `_latest_model.c` incrementally
- Log the length of the code read back to confirm it's not empty

---

## Issue 2: Two-Phase LLM Code Generation & Codegen Contracts

### Current State
One LLM call generates `model.c`. The LLM must invent function signatures and implement them in one shot.

### New Design: Two LLM Calls

**LLM Call 1 — Generate `model.h`** (Contract Definition):
- Receives: IR graph + weight tensor list + required helper signatures derived from `ir.py`.
- Outputs: A complete `model.h` header containing includes (`#include "weights.h"`), tensor contracts, dependencies, static-size macros, and function prototypes.
- This creates a rigid and verifiable contract for the next step.

**LLM Call 2 — Generate `model.c`** (Implementation):
- Receives: IR graph + weight tensor list + the **full text** of `model.h` from Call 1.
- Outputs: `model.c` that `#include "model.h"` and implements all functions declared in the header exactly as defined.
- On repair mode: also receives current `model.c` code + errors.

#### Files Changed

##### [MODIFY] [code_generator.py](file:///c:/Coding%20projects/agentic-riscv/agents/code_generator.py)
- Add `_build_model_header_prompt(state)` — prompt for LLM Call 1.
- Add `_extract_c_artifact(response, filename)` — robust extraction method.
- Modify `_build_model_c_prompt(state, model_h)` — include `model.h` content.
- Write `model.h` to `output/`.
- Return in state: `generated_model_header`, `model_header_path`.

##### [MODIFY] [codegen.txt](file:///c:/Coding%20projects/agentic-riscv/prompts/codegen.txt)
- Update instructions for Step 2 (`model.h` generation) and Step 3 (`model.c` implementation).

##### [MODIFY] [state.py](file:///c:/Coding%20projects/agentic-riscv/state.py)
- Add `generated_model_header: str` field.
- Add `model_header_path: str` field.

##### [MODIFY] [verifier.py](file:///c:/Coding%20projects/agentic-riscv/agents/verifier.py)
- Update compilation checks to validate `model.h` inclusion and enforce IR-derived helper compliance.

---

## Issue 3: Increase `max_tokens` to 200K for MI300x GPU

### Fix

##### [MODIFY] [code_generator.py](file:///c:/Coding%20projects/agentic-riscv/agents/code_generator.py)
- Change `max_tokens=8192` → `max_tokens=200000` (or read from env var `VLLM_MAX_TOKENS`)
- Make it configurable:
  ```python
  MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "200000"))
  ```

---

## Issue 4: Fix `inff` Undeclared Error in `weights.h`

### Root Cause

When weight values contain `inf` or `-inf` (common in some initialized or overflowed models), Python's `f"{v:.9e}f"` formatting produces `inff` or `-inff`, which is not valid C. The C standard uses `INFINITY` from `<math.h>`, but for a bare-metal header, we need a self-contained representation.

### Fix

##### [MODIFY] [export_weights.py](file:///c:/Coding%20projects/agentic-riscv/tools/export_weights.py)
- In `generate_weights_header()`, sanitize float values before emitting C literals:
  - `inf` → `3.402823466e+38f` (FLT_MAX)
  - `-inf` → `-3.402823466e+38f` (-FLT_MAX)
  - `nan` → `0.0f` (replace NaN with zero)
- Add a helper function `_float_to_c_literal(val: float) -> str` that handles all edge cases
- Apply this sanitization for f32 embedded mode where values are formatted as C float literals

---

## Issue 5: Route to Human Review After 5 Verification Failures (Instead of Stopping)

### Current Behavior
[graph.py:L40-L43](file:///c:/Coding%20projects/agentic-riscv/graph.py#L40-L43): When `attempts >= 5`, the pipeline routes directly to `report` with an error, ending the run.

### New Behavior
After 5 failures, route to `human_review` instead. The human can:
- Manually edit the generated files in `output/`
- Then choose to (a) retry code generation, (b) approve and proceed to simulation, or (c) quit

#### Files Changed

##### [MODIFY] [graph.py](file:///c:/Coding%20projects/agentic-riscv/graph.py)
- In `route_after_verification()`: change `attempts >= 5` branch from `"report"` → `"human_review"`
- Add a flag `"verification_exhausted"` to state so the human review handler knows this came from max-retries
- Update `route_after_human_review()` to support a new option: route to `"verify"` (skipping code generation) if the human edited files manually and wants to re-verify

##### [MODIFY] [main.py](file:///c:/Coding%20projects/agentic-riscv/main.py)
- Update the human-in-the-loop interrupt handler to show different options when coming from verification exhaustion:
  - `(e)dit` — user edits files manually, then re-verify
  - `(r)etry` — reset verification counter and regenerate
  - `(a)pprove` — force-approve and continue to simulation
  - `(q)uit` — exit

##### [MODIFY] [state.py](file:///c:/Coding%20projects/agentic-riscv/state.py)
- Add `verification_exhausted: bool` field

---

## Issue 6: Allow Starting the Pipeline from a Specific Agent

### Use Case
The user manually edits `output/model.c` and wants to continue from `verify` or `simulate` without re-running code generation (which would overwrite their edits).

### Design
Add a `--start-from` CLI argument that accepts a node name (`parse_fx`, `generate_code`, `verify`, `simulate`, `synthesize`). When specified, the pipeline initializes state from existing output files and enters the graph at the specified node.

#### Files Changed

##### [MODIFY] [main.py](file:///c:/Coding%20projects/agentic-riscv/main.py)
- Add `--start-from` CLI argument with choices: `parse_fx`, `generate_code`, `verify`, `simulate`, `synthesize`
- State recovery reads from `output/` directory:
  - `model.c` → `generated_code`
  - `weights.h` → `generated_header`
  - `model.h` → `generated_model_header`
  - `ir_graph.json` → `ir_graph`
  - `weights.npz` → `weights_path`
  - `weights.bin` → `weights_bin_path`
  - `weights_manifest.json` → `weights_manifest`
- Extract `weights_metadata` and IR metrics (`total_params`, `model_memory_bytes`, `ir_summary`) directly from `ir_graph.json`. If missing, fall back to parsing the loaded PyTorch `model.state_dict()`. This ensures robust metric reporting and verification when bypassing `parse_fx`.
- When `--start-from` is provided, build the graph with a different entry point

##### [MODIFY] [graph.py](file:///c:/Coding%20projects/agentic-riscv/graph.py)
- Modify `build_graph()` to accept an optional `entry_point` parameter
- When `entry_point` is specified, use `workflow.set_entry_point(entry_point)` instead of always starting at `parse_fx`

---

## Issue 7: Update `IMPLEMENTATION_PLAN.md`

##### [MODIFY] [IMPLEMENTATION_PLAN.md](file:///c:/Coding%20projects/agentic-riscv/IMPLEMENTATION_PLAN.md)
- Integrate all changes from issues 1–6 into the unified plan

---

## Summary of All File Changes

| File | Action | Purpose |
|------|--------|---------|
| [code_generator.py](file:///c:/Coding%20projects/agentic-riscv/agents/code_generator.py) | MODIFY | File-backed code persistence, two-phase `model.h`/`model.c` generation, 200K max_tokens |
| [codegen.txt](file:///c:/Coding%20projects/agentic-riscv/prompts/codegen.txt) | MODIFY | Add `model.h` contract generation instructions |
| [export_weights.py](file:///c:/Coding%20projects/agentic-riscv/tools/export_weights.py) | MODIFY | Sanitize inf/nan values in C literals |
| [graph.py](file:///c:/Coding%20projects/agentic-riscv/graph.py) | MODIFY | Route max-retries to human review, configurable entry point |
| [main.py](file:///c:/Coding%20projects/agentic-riscv/main.py) | MODIFY | Mid-pipeline state recovery of `model.h` and robust extraction of `weights_metadata` from IR/model |
| [state.py](file:///c:/Coding%20projects/agentic-riscv/state.py) | MODIFY | Replace legacy header fields with `generated_model_header`, add `verification_exhausted` |
| [test_code_generator.py](file:///c:/Coding%20projects/agentic-riscv/tests/test_code_generator.py) | NEW | Added unit tests to verify the two-phase `model.h` contract generator behavior |
| [verifier.py](file:///c:/Coding%20projects/agentic-riscv/agents/verifier.py) | MODIFY | Validate `model.h` inclusion and IR-derived helper enforcement |
| [IMPLEMENTATION_PLAN.md](file:///c:/Coding%20projects/agentic-riscv/IMPLEMENTATION_PLAN.md) | MODIFY | Update with all new changes |

---

## Verification Plan

### Automated Tests
1. Run pipeline end-to-end with `--demo` and verify:
   - `output/_latest_model.c` is created and persists across retries
   - `output/model.h` is generated by LLM Call 1
   - `output/model.c` includes `#include "model.h"`
   - No `inff` or `nanf` in `output/weights.h`
2. Simulate a verification failure and confirm:
   - Repair mode reads code from `_latest_model.c` (not empty)
   - After 5 failures, pipeline routes to human review (not report)
3. Test `--start-from verify` after manually editing `output/model.c`

### Manual Verification
- Check LLM prompts in logs to confirm `model.h` content and IR-derived helper signatures are included in Call 2
- Verify that `--start-from verify` correctly populates reporting metrics (no empty dictionaries/tables)
- Verify that the human review interrupt shows the correct options after verification exhaustion
