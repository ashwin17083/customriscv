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
