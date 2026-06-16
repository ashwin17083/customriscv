#include "model.h"
#include <string.h>

/* Helper function implementations */

void embedding(const float* table, int token_id, float* out, int dim) {
    const float* row = table + token_id * dim;
    for (int i = 0; i < dim; i++) {
        out[i] = row[i];
    }
}

void rmsnorm(const float* in, const float* weight, float* out, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += in[i] * in[i];
    }
    float rms = sqrtf(sum_sq / size + eps);
    for (int i = 0; i < size; i++) {
        out[i] = (in[i] / rms) * weight[i];
    }
}

void add_tensors(const float* a, const float* b, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

void linear(const float* in, const float* weight, const float* bias, float* out,
            int in_features, int out_features) {
    for (int i = 0; i < out_features; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; j++) {
            sum += in[j] * weight[i * in_features + j];
        }
        if (bias) sum += bias[i];
        out[i] = sum;
    }
}

void attention(const float* x, const float* wq, const float* wk, const float* wv, const float* wo,
               const float* causal_mask, const float* rope_cos, const float* rope_sin,
               float* out, int seq_len, int dim, int num_heads, int head_dim) {
    static float q_buf[32 * 64];
    static float k_buf[32 * 64];
    static float v_buf[32 * 64];
    static float scores[32 * 32];
    static float attn_out[32 * 64];

    // Project Q, K, V
    for (int i = 0; i < seq_len; i++) {
        linear(x + i * dim, wq, NULL, q_buf + i * dim, dim, dim);
        linear(x + i * dim, wk, NULL, k_buf + i * dim, dim, dim);
        linear(x + i * dim, wv, NULL, v_buf + i * dim, dim, dim);
    }

    // Apply RoPE to Q and K
    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d += 2) {
                int idx_q = pos * dim + h * head_dim + d;
                int idx_k = pos * dim + h * head_dim + d;
                int rope_idx = pos * (head_dim / 2) + d / 2;
                float cos_val = rope_cos[rope_idx];
                float sin_val = rope_sin[rope_idx];

                float q0 = q_buf[idx_q];
                float q1 = q_buf[idx_q + 1];
                q_buf[idx_q]     = q0 * cos_val - q1 * sin_val;
                q_buf[idx_q + 1] = q0 * sin_val + q1 * cos_val;

                float k0 = k_buf[idx_k];
                float k1 = k_buf[idx_k + 1];
                k_buf[idx_k]     = k0 * cos_val - k1 * sin_val;
                k_buf[idx_k + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; h++) {
        // Compute scores: Q @ K^T
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    sum += q_buf[i * dim + h * head_dim + d] * k_buf[j * dim + h * head_dim + d];
                }
                scores[i * seq_len + j] = sum * inv_sqrt_head_dim;
            }
        }

        // Apply causal mask
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] += causal_mask[i * seq_len + j];
            }
        }

        // Softmax
        for (int i = 0; i < seq_len; i++) {
            float max_val = scores[i * seq_len];
            for (int j = 1; j < seq_len; j++) {
                if (scores[i * seq_len + j] > max_val) max_val = scores[i * seq_len + j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                float exp_val = expf(scores[i * seq_len + j] - max_val);
                scores[i * seq_len + j] = exp_val;
                sum_exp += exp_val;
            }
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] /= sum_exp;
            }
        }

        // Output = scores @ V
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    sum += scores[i * seq_len + j] * v_buf[j * dim + h * head_dim + d];
                }
                attn_out[i * dim + h * head_dim + d] = sum;
            }
        }
    }

    // Project output with Wo
    for (int i = 0; i < seq_len; i++) {
        linear(attn_out + i * dim, wo, NULL, out + i * dim, dim, dim);
    }
}

void swiglu(const float* x, const float* gate_weight, const float* up_weight,
            const float* down_weight, float* out, int seq_len, int dim, int intermediate_dim) {
    static float gate_buf[32 * 128];
    static float up_buf[32 * 128];
    static float fused_buf[32 * 128];

    for (int i = 0; i < seq_len; i++) {
        linear(x + i * dim, gate_weight, NULL, gate_buf + i * intermediate_dim, dim, intermediate_dim);
        linear(x + i * dim, up_weight, NULL, up_buf + i * intermediate_dim, dim, intermediate_dim);
    }

    // SiLU activation on gate, then multiply with up
    for (int i = 0; i < seq_len * intermediate_dim; i++) {
        float g = gate_buf[i];
        fused_buf[i] = g * (1.0f / (1.0f + expf(-g))) * up_buf[i];
    }

    // Down projection
    for (int i = 0; i < seq_len; i++) {
        linear(fused_buf + i * intermediate_dim, down_weight, NULL, out + i * dim, intermediate_dim, dim);
    }
}

void model_inference(const float* input, float* output) {
    // Static activation buffers for each intermediate tensor
    static float buf_embed[32 * 64];
    static float buf_attn_norm_0[32 * 64];
    static float buf_attn_0[32 * 64];
    static float buf_add_0[32 * 64];
    static float buf_mlp_norm_0[32 * 64];
    static float buf_mlp_0[32 * 64];
    static float buf_add_1[32 * 64];
    static float buf_attn_norm_1[32 * 64];
    static float buf_attn_1[32 * 64];
    static float buf_add_2[32 * 64];
    static float buf_mlp_norm_1[32 * 64];
    static float buf_mlp_1[32 * 64];
    static float buf_add_3[32 * 64];
    static float buf_norm[32 * 64];
    static float buf_lm_head[32 * 512];

    // 1. Embedding
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        int token_id = (int)input[i];
        embedding(embed_tokens_weight, token_id, buf_embed + i * MODEL_DIM, MODEL_DIM);
    }

    // Layer 0
    // 2. RMSNorm
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        rmsnorm(buf_embed + i * MODEL_DIM, layers_0_attn_norm_weight, buf_attn_norm_0 + i * MODEL_DIM, MODEL_DIM, RMSNORM_EPS);
    }

    // 3. Attention
    attention(buf_attn_norm_0, layers_0_attention_q_proj_weight, layers_0_attention_k_proj_weight,
              layers_0_attention_v_proj_weight, layers_0_attention_o_proj_weight,
              layers_0_attention_causal_mask, layers_0_attention_rope_cos_cached, layers_0_attention_rope_sin_cached,
              buf_attn_0, MODEL_SEQ_LEN, MODEL_DIM, MODEL_NUM_HEADS, MODEL_HEAD_DIM);

    // 4. Add (Residual)
    add_tensors(buf_embed, buf_attn_0, buf_add_0, MODEL_SEQ_LEN * MODEL_DIM);

    // 5. RMSNorm
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        rmsnorm(buf_add_0 + i * MODEL_DIM, layers_0_mlp_norm_weight, buf_mlp_norm_0 + i * MODEL_DIM, MODEL_DIM, RMSNORM_EPS);
    }

    // 6. SwiGLU
    swiglu(buf_mlp_norm_0, layers_0_mlp_gate_proj_weight, layers_0_mlp_up_proj_weight,
           layers_0_mlp_down_proj_weight, buf_mlp_0, MODEL_SEQ_LEN, MODEL_DIM, MODEL_INTER_DIM);

    // 7. Add (Residual)
    add_tensors(buf_add_0, buf_mlp_0, buf_add_1, MODEL_SEQ_LEN * MODEL_DIM);

    // Layer 1
    // 8. RMSNorm
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        rmsnorm(buf_add_1 + i * MODEL_DIM, layers_1_attn_norm_weight, buf_attn_norm_1 + i * MODEL_DIM, MODEL_DIM, RMSNORM_EPS);
    }

    // 9. Attention
    attention(buf_attn_norm_1, layers_1_attention_q_proj_weight, layers_1_attention_k_proj_weight,
              layers_1_attention_v_proj_weight, layers_1_attention_o_proj_weight,
              layers_1_attention_causal_mask, layers_1_attention_rope_cos_cached, layers_1_attention_rope_sin_cached,
              buf_attn_1, MODEL_SEQ_LEN, MODEL_DIM, MODEL_NUM_HEADS, MODEL_HEAD_DIM);

    // 10. Add (Residual)
    add_tensors(buf_add_1, buf_attn_1, buf_add_2, MODEL_SEQ_LEN * MODEL_DIM);

    // 11. RMSNorm
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        rmsnorm(buf_add_2 + i * MODEL_DIM, layers_1_mlp_norm_weight, buf_mlp_norm_1 + i * MODEL_DIM, MODEL_DIM, RMSNORM_EPS);
    }

    // 12. SwiGLU
    swiglu(buf_mlp_norm_1, layers_1_mlp_gate_proj_weight, layers_1_mlp_up_proj_weight,
           layers_1_mlp_down_proj_weight, buf_mlp_1, MODEL_SEQ_LEN, MODEL_DIM, MODEL_INTER_DIM);

    // 13. Add (Residual)
    add_tensors(buf_add_2, buf_mlp_1, buf_add_3, MODEL_SEQ_LEN * MODEL_DIM);

    // Final Norm
    // 14. RMSNorm
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        rmsnorm(buf_add_3 + i * MODEL_DIM, norm_weight, buf_norm + i * MODEL_DIM, MODEL_DIM, RMSNORM_EPS);
    }

    // LM Head
    // 15. Linear
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        linear(buf_norm + i * MODEL_DIM, lm_head_weight, NULL, buf_lm_head + i * MODEL_VOCAB_SIZE, MODEL_DIM, MODEL_VOCAB_SIZE);
    }

    // Output
    memcpy(output, buf_lm_head, MODEL_SEQ_LEN * MODEL_VOCAB_SIZE * sizeof(float));
}