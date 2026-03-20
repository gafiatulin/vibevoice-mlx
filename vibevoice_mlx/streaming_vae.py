"""Streaming VAE decoder for VibeVoice — pure MLX.

Processes one latent frame (1, 64, 1) per call, producing exactly 3200 audio
samples via conv cache buffers that carry state across frames.

Wraps the non-streaming VAEDecoder's weights with streaming cache logic.
Same architecture: 7 stages of DecoderBlocks + 6 upsample convs + head.
34 total cache buffers.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


DEPTHS = [8, 3, 3, 3, 3, 3, 3]
RATIOS = [8, 5, 5, 4, 2, 2]


class StreamingVAEDecoder:
    """Streaming VAE decoder with explicit conv caches.

    Each causal conv/conv_transpose layer has a cache buffer.
    On each call with T=1 input, caches provide temporal context
    from previous tokens, producing smooth continuous audio.
    """

    def __init__(self, vae_decoder):
        """Wrap an existing VAEDecoder's weights with streaming caches.

        Args:
            vae_decoder: Loaded VAEDecoder with weights populated.
        """
        self.vae = vae_decoder
        self.caches = {}
        self._build_caches()

    def _build_caches(self):
        """Initialize zero cache buffers for all conv layers."""
        dtype = mx.float16
        vae = self.vae

        # Stem (init conv): (C_out, C_in, K) weight
        K = vae.init_conv_w.shape[-1]
        C_in = vae.init_conv_w.shape[1]
        self.caches["stem"] = mx.zeros((1, C_in, K - 1), dtype=dtype)

        # Stage 0 blocks
        for b in range(DEPTHS[0]):
            block = vae.stages[0][b]
            K = block["conv_w"].shape[-1]
            C = block["conv_w"].shape[0]  # depthwise: C groups
            self.caches[f"s0_b{b}"] = mx.zeros((1, C, K - 1), dtype=dtype)

        # Stages 1..6: upsample + blocks
        for s in range(len(RATIOS)):
            w, b, stride = vae.upsample_convs[s]
            C_in = w.shape[0]  # ConvTranspose: (C_in, C_out, K) in PyTorch layout
            ctx = w.shape[-1] - stride  # kernel - stride
            self.caches[f"up{s}"] = mx.zeros((1, C_in, ctx), dtype=dtype)

            for bi in range(DEPTHS[s + 1]):
                block = vae.stages[s + 1][bi]
                K = block["conv_w"].shape[-1]
                C = block["conv_w"].shape[0]
                self.caches[f"s{s+1}_b{bi}"] = mx.zeros((1, C, K - 1), dtype=dtype)

        # Head conv
        K = vae.head_w.shape[-1]
        C_in = vae.head_w.shape[1]
        self.caches["head"] = mx.zeros((1, C_in, K - 1), dtype=dtype)

    def reset(self):
        """Reset all caches to zero (call before a new utterance)."""
        for name in self.caches:
            self.caches[name] = mx.zeros_like(self.caches[name])

    def _conv1d_streaming(self, x, weight, bias, cache_name, stride=1, groups=1):
        """Streaming causal conv1d: prepend cache, conv, update cache.

        x: (1, C_in, T) channels-first
        weight: (C_out, C_in/groups, K) PyTorch layout
        """
        cache = self.caches[cache_name]
        x_full = mx.concatenate([cache, x], axis=2)

        out = mx.conv1d(
            x_full.transpose(0, 2, 1),
            weight.transpose(0, 2, 1),
            stride=stride,
            groups=groups,
        ).transpose(0, 2, 1) + bias[:, None]

        ctx = cache.shape[2]
        if ctx > 0:
            self.caches[cache_name] = x_full[:, :, -ctx:]

        return out

    def _convtr1d_streaming(self, x, weight, bias, stride, cache_name):
        """Streaming causal conv_transpose1d with cache.

        x: (1, C_in, T) channels-first
        weight: (C_in, C_out, K) PyTorch layout
        """
        cache = self.caches[cache_name]
        x_full = mx.concatenate([cache, x], axis=2)

        # MLX conv_transpose1d expects channels-last input, weight (C_out, K, C_in)
        out = mx.conv_transpose1d(
            x_full.transpose(0, 2, 1),
            weight.transpose(1, 2, 0),
            stride=stride,
        ).transpose(0, 2, 1) + bias[:, None]

        # Trim: padding_total = kernel - stride, all trimmed from right (causal)
        K = weight.shape[-1]
        padding_right = K - stride
        if padding_right > 0:
            out = out[:, :, :-padding_right]

        # Take only the new samples (from the new input, not the cached input)
        new_samples = x.shape[2] * stride
        out = out[:, :, -new_samples:]

        # Update cache
        ctx = cache.shape[2]
        if ctx > 0:
            self.caches[cache_name] = x_full[:, :, -ctx:]

        return out

    def _block_forward(self, x, block, cache_name):
        """DecoderBlock with streaming conv cache."""
        B, C, T = x.shape
        residual = x

        # RMSNorm (channels-last)
        xt = x.transpose(0, 2, 1)
        xt = mx.fast.rms_norm(xt, block["norm_w"], 1e-5)
        x_normed = xt.transpose(0, 2, 1)

        # Streaming depthwise conv
        x_conv = self._conv1d_streaming(
            x_normed, block["conv_w"], block["conv_b"],
            cache_name, groups=C,
        )
        x = residual + block["gamma"][:, None] * x_conv

        # FFN
        residual = x
        xt = x.transpose(0, 2, 1)
        xt = mx.fast.rms_norm(xt, block["ffn_norm_w"], 1e-5)
        xt = nn.gelu(xt @ block["ffn_l1_w"].T + block["ffn_l1_b"])
        xt = xt @ block["ffn_l2_w"].T + block["ffn_l2_b"]
        x = residual + block["ffn_gamma"][:, None] * xt.transpose(0, 2, 1)

        return x

    def __call__(self, latent):
        """Decode one latent frame using streaming caches.

        Args:
            latent: (1, 64, 1) channels-first, single frame

        Returns:
            audio: (1, 1, 3200) channels-first
        """
        vae = self.vae
        x = latent

        # Stem conv (streaming)
        x = self._conv1d_streaming(x, vae.init_conv_w, vae.init_conv_b, "stem")

        # Stage 0 blocks
        for b in range(DEPTHS[0]):
            x = self._block_forward(x, vae.stages[0][b], f"s0_b{b}")

        # Stages 1..6: upsample + blocks
        for s in range(len(RATIOS)):
            w, bias, stride = vae.upsample_convs[s]
            x = self._convtr1d_streaming(x, w, bias, stride, f"up{s}")

            for b in range(DEPTHS[s + 1]):
                x = self._block_forward(x, vae.stages[s + 1][b], f"s{s+1}_b{b}")

        # Head conv (streaming)
        x = self._conv1d_streaming(x, vae.head_w, vae.head_b, "head")

        return x
