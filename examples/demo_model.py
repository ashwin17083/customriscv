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
