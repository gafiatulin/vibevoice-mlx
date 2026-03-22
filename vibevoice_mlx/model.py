"""VibeVoice model definitions in MLX.

Implements Qwen2.5 backbone with VibeVoice TTS extensions:
diffusion head, VAE decoder, and acoustic/semantic connectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VibeVoiceConfig:
    # Qwen2.5 backbone
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 8960
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    # Diffusion
    vae_dim: int = 64
    diffusion_layers: int = 4
    ddpm_steps: int = 1000
    head_ffn_ratio: float = 3.0
    # Audio
    semantic_dim: int = 128
    sample_rate: int = 24000
    # VAE decoder
    vae_depths: list[int] = field(default_factory=lambda: [8, 3, 3, 3, 3, 3, 3])
    vae_ratios: list[int] = field(default_factory=lambda: [8, 5, 5, 4, 2, 2])
    # Learned scaling
    speech_scaling_factor: float = 0.1962890625
    speech_bias_factor: float = -0.04931640625
    # Special tokens
    speech_start_id: int = 151652
    speech_end_id: int = 151653
    speech_diffusion_id: int = 151654
    eos_id: int = 151643
    # Model variant behavior (auto-detected from config)
    single_segment: bool = False  # True for KugelAudio-family models


# ---------------------------------------------------------------------------
# RoPE (standard, no scaling — theta=1M)
# ---------------------------------------------------------------------------

def compute_rope(positions: mx.array, head_dim: int, rope_theta: float) -> tuple[mx.array, mx.array]:
    """Compute cos/sin for RoPE. positions: (Q,), returns (Q, head_dim)."""
    inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = positions[:, None] * inv_freq[None, :]  # (Q, D/2)
    freqs = mx.concatenate([freqs, freqs], axis=-1)  # (Q, D)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE to x: (B, H, Q, D). cos/sin: (Q, D)."""
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    cos = cos[None, None, :, :]  # (1, 1, Q, D)
    sin = sin[None, None, :, :]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """KV cache with concatenation (MLX lazy eval handles this efficiently)."""

    def __init__(self, num_layers: int):
        self.keys: list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers
        self.offset = 0

    def update(self, layer_idx: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = mx.concatenate([self.keys[layer_idx], k], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx], v], axis=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def advance(self, n: int = 1):
        self.offset += n


# ---------------------------------------------------------------------------
# Qwen2 layers (attention has bias on QKV, not on o_proj)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
        B, Q, _ = x.shape

        q = self.q_proj(x).reshape(B, Q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, Q, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, Q, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, Q, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache, layer_idx)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen2Model(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, mask, cache, i)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Diffusion Head (DDPM v-prediction)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = [
            nn.Linear(freq_dim, hidden_size, bias=False),
            None,  # SiLU activation placeholder
            nn.Linear(hidden_size, hidden_size, bias=False),
        ]

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        emb = emb.astype(self.mlp[0].weight.dtype)
        return self.mlp[2](nn.silu(self.mlp[0](emb)))


class HeadFFN(nn.Module):
    """SwiGLU FFN for diffusion head layers."""
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadLayer(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(embed_dim, eps=eps)
        self.ffn = HeadFFN(embed_dim, ffn_dim)
        self.adaLN_modulation = [None, nn.Linear(cond_dim, 3 * embed_dim, bias=False)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mods = self.adaLN_modulation[1](nn.silu(c))
        H = x.shape[-1]
        shift, scale, gate = mods[..., :H], mods[..., H:2*H], mods[..., 2*H:]
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, cond_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = [None, nn.Linear(cond_size, 2 * hidden_size, bias=False)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mods = self.adaLN_modulation[1](nn.silu(c))
        H = x.shape[-1]
        shift, scale = mods[..., :H], mods[..., H:]
        h = mx.fast.rms_norm(x, mx.ones(H, dtype=x.dtype), self.eps)
        h = h * (1 + scale) + shift
        return self.linear(h)


class DiffusionHead(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        hidden = config.hidden_size
        vae_dim = config.vae_dim

        ffn_dim = int(hidden * config.head_ffn_ratio)

        self.noisy_images_proj = nn.Linear(vae_dim, hidden, bias=False)
        self.cond_proj = nn.Linear(hidden, hidden, bias=False)
        self.t_embedder = TimestepEmbedder(hidden)
        self.layers = [
            HeadLayer(hidden, ffn_dim, hidden) for _ in range(config.diffusion_layers)
        ]
        self.final_layer = FinalLayer(hidden, vae_dim, hidden)

    def __call__(self, noisy: mx.array, timesteps: mx.array, condition: mx.array) -> mx.array:
        x = self.noisy_images_proj(noisy)
        t = self.t_embedder(timesteps)
        c = self.cond_proj(condition) + t
        for layer in self.layers:
            x = layer(x, c)
        return self.final_layer(x, c)


# ---------------------------------------------------------------------------
# VAE Decoder (channels-first convolutional)
# ---------------------------------------------------------------------------

class VAEDecoder:
    """Convolutional VAE decoder: latent (B, 64, 1) -> audio (B, 1, 3200).

    Uses channels-first (B, C, T) layout with manual conv1d calls.
    Weights are populated directly by load_weights._populate_vae_decoder()
    since the complex conv structure doesn't map cleanly to nn.Module paths.

    Each block in self.stages is a dict with keys:
        norm_w, conv_w, conv_b, gamma, ffn_norm_w,
        ffn_l1_w, ffn_l1_b, ffn_l2_w, ffn_l2_b, ffn_gamma
    """
    def __init__(self):
        self.init_conv_w = None  # (C_out, C_in, K)
        self.init_conv_b = None
        self.stages: list[list[dict]] = []
        self.upsample_convs: list[tuple] = []  # (weight, bias, stride)
        self.head_w = None
        self.head_b = None

    def __call__(self, latent: mx.array) -> mx.array:
        """latent: (B, C, T) channels-first -> audio: (B, 1, T_audio)."""
        # Init conv (causal: pad left)
        K = self.init_conv_w.shape[-1]
        x = mx.pad(latent, [(0, 0), (0, 0), (K - 1, 0)])
        x = mx.conv1d(
            x.transpose(0, 2, 1),
            self.init_conv_w.transpose(0, 2, 1),
        ).transpose(0, 2, 1) + self.init_conv_b[:, None]

        for s_idx, blocks in enumerate(self.stages):
            for block in blocks:
                x = self._block_forward(block, x)
            if s_idx < len(self.upsample_convs):
                w, b, stride = self.upsample_convs[s_idx]
                x = mx.conv_transpose1d(
                    x.transpose(0, 2, 1),
                    w.transpose(1, 2, 0),
                    stride=stride,
                ).transpose(0, 2, 1) + b[:, None]
                K_up = w.shape[-1]
                trim = (K_up - stride) // 2
                if trim > 0:
                    x = x[:, :, trim:-trim]

        # Head conv (causal)
        K = self.head_w.shape[-1]
        x = mx.pad(x, [(0, 0), (0, 0), (K - 1, 0)])
        x = mx.conv1d(
            x.transpose(0, 2, 1),
            self.head_w.transpose(0, 2, 1),
        ).transpose(0, 2, 1) + self.head_b[:, None]
        return x

    @staticmethod
    def _block_forward(block, x):
        """Forward pass for a DecoderBlock dict."""
        B, C, T = x.shape
        residual = x

        # RMSNorm (needs channels-last)
        xt = x.transpose(0, 2, 1)
        xt = mx.fast.rms_norm(xt, block["norm_w"], 1e-5)
        x = xt.transpose(0, 2, 1)

        # Depthwise causal conv
        K = block["conv_w"].shape[-1]
        pad = K - 1
        x = mx.pad(x, [(0, 0), (0, 0), (pad, 0)])
        x = mx.conv1d(
            x.transpose(0, 2, 1),
            block["conv_w"].transpose(0, 2, 1),
            groups=C,
        ).transpose(0, 2, 1) + block["conv_b"][:, None]

        x = residual + block["gamma"][:, None] * x
        residual = x

        # FFN (channels-last)
        xt = x.transpose(0, 2, 1)
        xt = mx.fast.rms_norm(xt, block["ffn_norm_w"], 1e-5)
        xt = nn.gelu(xt @ block["ffn_l1_w"].T + block["ffn_l1_b"])
        xt = xt @ block["ffn_l2_w"].T + block["ffn_l2_b"]

        x = residual + block["ffn_gamma"][:, None] * xt.transpose(0, 2, 1)
        return x


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------

class Connector(nn.Module):
    """fc1 (with bias) -> RMSNorm -> fc2 (with bias)."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        self.norm = nn.RMSNorm(output_dim, eps=1e-5)
        self.fc2 = nn.Linear(output_dim, output_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.norm(self.fc1(x)))


# ---------------------------------------------------------------------------
# VibeVoice top-level model
# ---------------------------------------------------------------------------

class VibeVoiceModel(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.diffusion_head = DiffusionHead(config)
        self.vae_decoder = VAEDecoder()
        self.acoustic_connector = Connector(config.vae_dim, config.hidden_size)
        self.semantic_connector = Connector(config.semantic_dim, config.hidden_size)

        # LM head — may be tied or untied
        if config.tie_word_embeddings:
            self.lm_head = None  # use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_logits(self, hidden: mx.array) -> mx.array:
        """Compute logits in float32 to prevent overflow (critical for 7B untied)."""
        if self.lm_head is not None:
            return (hidden.astype(mx.float32) @ self.lm_head.weight.astype(mx.float32).T).astype(hidden.dtype)
        return (hidden.astype(mx.float32) @ self.model.embed_tokens.weight.astype(mx.float32).T).astype(hidden.dtype)
