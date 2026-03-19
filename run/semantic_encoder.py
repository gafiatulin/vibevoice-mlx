"""Pure MLX streaming semantic encoder for VibeVoice.

Processes one audio frame (3200 samples) per call using explicit conv cache
arrays, producing semantic features [1, 128, 1] for the semantic connector.

Architecture: 7 stages of EncoderBlocks + 6 downsample convs + head conv.
Each causal conv layer has a cache buffer that carries state across frames.
34 total cache buffers.

The forward pass is a pure function (caches in, caches out) compiled via
mx.compile for minimal Python dispatch overhead.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Architecture constants (from semantic_tokenizer_config in HF config.json)
N_FILTERS = 32
RATIOS = [2, 2, 4, 5, 5, 8]  # encoder ratios (reversed from decoder)
DEPTHS = [3, 3, 3, 3, 3, 3, 8]
KERNEL_SIZE = 7
LAST_KERNEL_SIZE = 7
OUTPUT_DIM = 128  # semantic feature dimension
RMS_EPS = 1e-5
FRAME_SAMPLES = 3200


def _context_size(kernel_size, stride=1):
    """Causal conv cache size = kernel_size - stride."""
    return kernel_size - stride


class SemanticEncoder:
    """Streaming semantic encoder with compiled forward pass.

    Weights stored as ordered lists for positional access in the compiled
    function. Caches stored as a flat list, updated functionally.
    """

    def __init__(self):
        # Ordered conv params: [(weight, bias, stride, groups), ...]
        self.conv_params = []
        # Ordered block params: [(norm_w, ffn_norm_w, gamma, ffn_gamma, l1_w, l1_b, l2_w, l2_b), ...]
        self.block_params = []
        # Cache shapes for initialization
        self.cache_shapes = []
        # Current caches (flat list)
        self.caches = []
        # Compiled forward function
        self._compiled_fn = None

    def reset_caches(self):
        """Reset all conv caches to zero."""
        self.caches = [mx.zeros(s, dtype=mx.float16) for s in self.cache_shapes]

    def __call__(self, audio):
        """Process one frame. Updates caches in place.

        Args:
            audio: (1, 1, 3200) float32, channels-first
        Returns:
            features: (1, 128, 1) float32, channels-first
        """
        if self._compiled_fn is None:
            self._compiled_fn = mx.compile(_semantic_forward)

        result, new_caches = self._compiled_fn(
            audio, self.caches, self.conv_params, self.block_params,
        )
        self.caches = new_caches
        return result


def _conv_streaming(x, weight, bias, stride, groups, cache):
    """Pure streaming conv: returns (output, new_cache)."""
    if cache.shape[2] > 0:
        x_full = mx.concatenate([cache, x], axis=2)
    else:
        x_full = x

    out = mx.conv1d(
        x_full.transpose(0, 2, 1),
        weight.transpose(0, 2, 1),
        stride=stride,
        groups=groups,
    ).transpose(0, 2, 1) + bias[:, None]

    ctx = cache.shape[2]
    new_cache = x_full[:, :, -ctx:] if ctx > 0 else cache
    return out, new_cache


def _block_forward_t1(x, conv_w, conv_b, cache,
                      norm_w, ffn_norm_w, gamma, ffn_gamma,
                      ffn_l1_w, ffn_l1_b, ffn_l2_w, ffn_l2_b):
    """Fused encoder block for T=1 — avoids conv1d dispatch.

    At T=1, depthwise conv is just a dot product per channel,
    expressed as element-wise multiply + sum.
    """
    residual = x

    # RMSNorm
    xt = x.transpose(0, 2, 1)  # (1, 1, C)
    xt = mx.fast.rms_norm(xt, norm_w, RMS_EPS)
    x_normed = xt.transpose(0, 2, 1)  # (1, C, 1)

    # Depthwise conv via element-wise: concat cache, multiply kernel, sum
    x_full = mx.concatenate([cache, x_normed], axis=2)  # (1, C, K)
    new_cache = x_full[:, :, -(cache.shape[2]):] if cache.shape[2] > 0 else cache
    # conv_w: (C, 1, K) depthwise — squeeze to (C, K), dot product per channel
    conv_out = (x_full * conv_w.squeeze(1)[None, :, :]).sum(axis=2, keepdims=True) + conv_b[:, None]

    x = residual + gamma[:, None] * conv_out

    # FFN
    residual = x
    xt = x.transpose(0, 2, 1)  # (1, 1, C)
    xt = mx.fast.rms_norm(xt, ffn_norm_w, RMS_EPS)
    xt = nn.gelu(xt @ ffn_l1_w.T + ffn_l1_b)
    xt = xt @ ffn_l2_w.T + ffn_l2_b
    x = residual + ffn_gamma[:, None] * xt.transpose(0, 2, 1)

    return x, new_cache


def _block_forward(x, conv_w, conv_b, conv_stride, conv_groups, cache,
                   norm_w, ffn_norm_w, gamma, ffn_gamma,
                   ffn_l1_w, ffn_l1_b, ffn_l2_w, ffn_l2_b):
    """Pure encoder block: returns (output, new_cache)."""
    B, C, T = x.shape

    # For T=1, use fused path that avoids conv1d dispatch overhead
    if T == 1 and conv_stride == 1:
        return _block_forward_t1(x, conv_w, conv_b, cache,
                                 norm_w, ffn_norm_w, gamma, ffn_gamma,
                                 ffn_l1_w, ffn_l1_b, ffn_l2_w, ffn_l2_b)

    residual = x

    # RMSNorm → depthwise conv → gamma → residual
    xt = x.transpose(0, 2, 1)
    xt = mx.fast.rms_norm(xt, norm_w, RMS_EPS)
    x_normed = xt.transpose(0, 2, 1)

    x_conv, new_cache = _conv_streaming(x_normed, conv_w, conv_b, conv_stride, conv_groups, cache)
    x = residual + gamma[:, None] * x_conv

    # FFN
    residual = x
    xt = x.transpose(0, 2, 1)
    xt = mx.fast.rms_norm(xt, ffn_norm_w, RMS_EPS)
    xt = nn.gelu(xt @ ffn_l1_w.T + ffn_l1_b)
    xt = xt @ ffn_l2_w.T + ffn_l2_b
    x = residual + ffn_gamma[:, None] * xt.transpose(0, 2, 1)

    return x, new_cache


def _semantic_forward(audio, caches, conv_params, block_params):
    """Pure forward function for mx.compile.

    All state (caches) passed in and returned — no side effects.
    """
    x = audio
    cache_idx = 0
    new_caches = []
    block_idx = 0

    # Stem conv
    w, b, s, g = conv_params[0]
    x, c = _conv_streaming(x, w, b, s, g, caches[cache_idx])
    new_caches.append(c)
    cache_idx += 1
    conv_param_idx = 1

    # Stages + downsamples
    for i in range(len(DEPTHS)):
        for j in range(DEPTHS[i]):
            bp = block_params[block_idx]
            norm_w, ffn_norm_w, gamma, ffn_gamma, l1_w, l1_b, l2_w, l2_b = bp
            cw, cb, cs, cg = conv_params[conv_param_idx]
            x, c = _block_forward(x, cw, cb, cs, cg, caches[cache_idx],
                                  norm_w, ffn_norm_w, gamma, ffn_gamma,
                                  l1_w, l1_b, l2_w, l2_b)
            new_caches.append(c)
            cache_idx += 1
            conv_param_idx += 1
            block_idx += 1

        if i < len(RATIOS):
            w, b, s, g = conv_params[conv_param_idx]
            x, c = _conv_streaming(x, w, b, s, g, caches[cache_idx])
            new_caches.append(c)
            cache_idx += 1
            conv_param_idx += 1

    # Head conv
    w, b, s, g = conv_params[conv_param_idx]
    x, c = _conv_streaming(x, w, b, s, g, caches[cache_idx])
    new_caches.append(c)

    return x, new_caches


def load_semantic_encoder(weights: dict) -> SemanticEncoder:
    """Load semantic encoder from weight dict.

    Handles both original HF keys (model.semantic_tokenizer.encoder.*)
    and converted MLX keys (semantic_encoder.*).

    Returns:
        Initialized SemanticEncoder with all weights and zero caches.
    """
    enc = SemanticEncoder()
    dtype = mx.float16

    # Detect key prefix
    if any(k.startswith("model.semantic_tokenizer.encoder.") for k in weights):
        p = "model.semantic_tokenizer.encoder."
    elif any(k.startswith("semantic_encoder.") for k in weights):
        p = "semantic_encoder."
    else:
        raise KeyError("No semantic encoder weights found")

    def _w(key):
        return weights[p + key].astype(dtype)

    # Build ordered conv_params and cache_shapes

    # Stem
    stem_w = _w("downsample_layers.0.0.conv.conv.weight")
    stem_b = _w("downsample_layers.0.0.conv.conv.bias")
    ctx = _context_size(KERNEL_SIZE)
    enc.conv_params.append((stem_w, stem_b, 1, 1))
    enc.cache_shapes.append((1, 1, ctx))

    # Stages + downsamples
    for i in range(len(DEPTHS)):
        ch = N_FILTERS * (2 ** i)

        for j in range(DEPTHS[i]):
            bp_prefix = f"stages.{i}.{j}."

            # Conv params for this block's depthwise conv
            conv_w = _w(bp_prefix + "mixer.conv.conv.conv.weight")
            conv_b = _w(bp_prefix + "mixer.conv.conv.conv.bias")
            conv_ctx = _context_size(KERNEL_SIZE)
            enc.conv_params.append((conv_w, conv_b, 1, ch))
            enc.cache_shapes.append((1, ch, conv_ctx))

            # Block params (tuple for positional access)
            enc.block_params.append((
                _w(bp_prefix + "norm.weight"),
                _w(bp_prefix + "ffn_norm.weight"),
                _w(bp_prefix + "gamma"),
                _w(bp_prefix + "ffn_gamma"),
                _w(bp_prefix + "ffn.linear1.weight"),
                _w(bp_prefix + "ffn.linear1.bias"),
                _w(bp_prefix + "ffn.linear2.weight"),
                _w(bp_prefix + "ffn.linear2.bias"),
            ))

        if i < len(RATIOS):
            ratio = RATIOS[i]
            kernel = ratio * 2
            ds_w = _w(f"downsample_layers.{i + 1}.0.conv.conv.weight")
            ds_b = _w(f"downsample_layers.{i + 1}.0.conv.conv.bias")
            ds_ctx = _context_size(kernel, ratio)
            enc.conv_params.append((ds_w, ds_b, ratio, 1))
            enc.cache_shapes.append((1, ch, ds_ctx))

    # Head conv
    last_ch = N_FILTERS * (2 ** len(RATIOS))
    head_w = _w("head.conv.conv.weight")
    head_b = _w("head.conv.conv.bias")
    head_ctx = _context_size(LAST_KERNEL_SIZE)
    enc.conv_params.append((head_w, head_b, 1, 1))
    enc.cache_shapes.append((1, last_ch, head_ctx))

    # Initialize caches
    enc.reset_caches()

    return enc
