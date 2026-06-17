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


def test_extract_c_artifact_prefers_named_block_and_recovers_common_llm_drift():
    assert (
        code_generator._extract_c_artifact(
            '```c model.h\n#pragma once\n```', "model.h"
        )
        == "#pragma once"
    )

    assert (
        code_generator._extract_c_artifact(
            '```c\n#pragma once\n```', "model.h"
        )
        == "#pragma once"
    )

    assert code_generator._extract_c_artifact("#pragma once", "model.h") == "#pragma once"

    assert (
        code_generator._extract_c_artifact(
            '```c model.h\n#pragma once\n```\n```c model.h\n#pragma once\n#include "weights.h"\n```',
            "model.h",
        )
        == '#pragma once\n#include "weights.h"'
    )

    with pytest.raises(ValueError, match="Could not extract model.h"):
        code_generator._extract_c_artifact("not C code", "model.h")

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
