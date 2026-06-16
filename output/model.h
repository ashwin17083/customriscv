#ifndef MODEL_H
#define MODEL_H

#include "weights.h"
#include <stdint.h>
#include <math.h>

/*
 * ============================================================
 * model.h — Tiny LLaMA Demo (2-layer, 64-dim)
 * Target: RISC-V rv32imac (soft-float, C99, no dynamic alloc)
 * ============================================================
 *
 * Tensor Contracts:
 *   Input:  input_ids[MODEL_SEQ_LEN]       — token IDs (passed as float, cast to int)
 *   Output: logits[MODEL_SEQ_LEN * MODEL_VOCAB_SIZE] — raw per-token logits
 *
 * Dependencies:
 *   weights.h  — all weight/bias arrays (auto-generated)
 *   math.h     — expf, sqrtf, fabsf
 *   string.h   — memset, memcpy (used in model.c)
 * ============================================================
 */

/* Model dimension constants */
#define MODEL_SEQ_LEN       32
#define MODEL_DIM           64
#define MODEL_NUM_HEADS     4
#define MODEL_HEAD_DIM      16
#define MODEL_INTER_DIM     128
#define MODEL_VOCAB_SIZE    512
#define MODEL_NUM_LAYERS    2

/* RMSNorm epsilon */
#define RMSNORM_EPS         1e-5f

/* Causal mask sentinel for future positions */
#define CAUSE_MASK_SENTINEL -1e9f

/*
 * Helper function prototypes
 *
 * embedding:
 *   Look up a single token_id in the embedding table.
 *   table: [num_embeddings, embedding_dim]
 *   out:   [embedding_dim]
 *
 * rmsnorm:
 *   Root Mean Square Layer Normalization.
 *   in:    [size]
 *   weight:[size]
 *   out:   [size]
 *
 * attention:
 *   Multi-head causal self-attention with RoPE.
 *   x:           [seq_len, dim]
 *   wq, wk, wv:  [dim, dim]  (Q/K/V projection weights)
 *   wo:          [dim, dim]  (output projection weight)
 *   causal_mask: [seq_len, seq_len] (precomputed mask)
 *   rope_cos:    [seq_len, head_dim/2] (cached cos values)
 *   rope_sin:    [seq_len, head_dim/2] (cached sin values)
 *   out:         [seq_len, dim]
 *
 * add_tensors:
 *   Element-wise addition of two tensors.
 *   a, b, out: [size]
 *
 * linear:
 *   Fully-connected layer (GEMM).
 *   in:     [in_features]
 *   weight: [out_features, in_features]
 *   bias:   [out_features] or NULL
 *   out:    [out_features]
 *
 * swiglu:
 *   SwiGLU MLP block.
 *   x:           [seq_len, dim]
 *   gate_weight: [intermediate_dim, dim]
 *   up_weight:   [intermediate_dim, dim]
 *   down_weight: [dim, intermediate_dim]
 *   out:         [seq_len, dim]
 */

void embedding(const float* table, int token_id, float* out, int dim);

void rmsnorm(const float* in, const float* weight, float* out, int size, float eps);

void attention(const float* x, const float* wq, const float* wk, const float* wv, const float* wo,
               const float* causal_mask, const float* rope_cos, const float* rope_sin,
               float* out, int seq_len, int dim, int num_heads, int head_dim);

void add_tensors(const float* a, const float* b, float* out, int size);

void linear(const float* in, const float* weight, const float* bias, float* out,
            int in_features, int out_features);

void swiglu(const float* x, const float* gate_weight, const float* up_weight,
            const float* down_weight, float* out, int seq_len, int dim, int intermediate_dim);

/*
 * Main inference entry point
 *
 * input:  pointer to input_ids[MODEL_SEQ_LEN] (float, cast to int for embedding)
 * output: pointer to logits[MODEL_SEQ_LEN * MODEL_VOCAB_SIZE]
 */
void model_inference(const float* input, float* output);

#endif /* MODEL_H */