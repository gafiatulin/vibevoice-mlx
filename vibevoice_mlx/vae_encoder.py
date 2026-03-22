"""Pure MLX VAE encoder for voice cloning.

Encodes reference audio to VAE latents using the acoustic tokenizer encoder
weights directly, without needing CoreML or PyTorch.

Architecture: same CNN as semantic_encoder (7 stages + 6 downsamples + head),
but outputs vae_dim (64) instead of semantic_dim (128).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Architecture (from acoustic_tokenizer_config — same as semantic)
N_FILTERS = 32
RATIOS = [2, 2, 4, 5, 5, 8]  # encoder order (reversed from decoder's [8,5,5,4,2,2])
DEPTHS = [3, 3, 3, 3, 3, 3, 8]
KERNEL_SIZE = 7
RMS_EPS = 1e-5


def _causal_conv1d(x, weight, bias, stride=1, groups=1):
    """Causal conv1d: pad left by kernel-stride, no future context."""
    K = weight.shape[-1]
    pad = K - stride
    if pad > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (pad, 0)])
    out = mx.conv1d(
        x.transpose(0, 2, 1),
        weight.transpose(0, 2, 1),
        stride=stride,
        groups=groups,
    ).transpose(0, 2, 1) + bias[:, None]
    return out


def _block_forward(x, block):
    """Encoder block: RMSNorm → depthwise conv → residual → FFN → residual."""
    B, C, T = x.shape
    residual = x

    xt = x.transpose(0, 2, 1)
    xt = mx.fast.rms_norm(xt, block["norm_w"], RMS_EPS)
    x = xt.transpose(0, 2, 1)

    x = _causal_conv1d(x, block["conv_w"], block["conv_b"], groups=C)
    x = residual + block["gamma"][:, None] * x

    residual = x
    xt = x.transpose(0, 2, 1)
    xt = mx.fast.rms_norm(xt, block["ffn_norm_w"], RMS_EPS)
    xt = nn.gelu(xt @ block["ffn_l1_w"].T + block["ffn_l1_b"])
    xt = xt @ block["ffn_l2_w"].T + block["ffn_l2_b"]
    x = residual + block["ffn_gamma"][:, None] * xt.transpose(0, 2, 1)

    return x


def encode_audio(audio, weights):
    """Encode audio to VAE latents.

    Args:
        audio: (1, 1, T) float32 channels-first audio at 24kHz
        weights: dict of encoder weights (keyed by short names)

    Returns:
        latents: (1, T_compressed, vae_dim) float32
    """
    x = audio.astype(mx.float16)

    # Stem conv
    x = _causal_conv1d(x, weights["stem_w"], weights["stem_b"])

    # Stages + downsamples
    for i in range(len(DEPTHS)):
        for j in range(DEPTHS[i]):
            x = _block_forward(x, weights["blocks"][i][j])
        if i < len(RATIOS):
            x = _causal_conv1d(
                x, weights["ds"][i]["w"], weights["ds"][i]["b"],
                stride=RATIOS[i],
            )

    # Head conv
    x = _causal_conv1d(x, weights["head_w"], weights["head_b"])

    # x is (1, vae_dim, T_compressed) — transpose to (1, T_compressed, vae_dim)
    return x.transpose(0, 2, 1).astype(mx.float32)


def load_vae_encoder_weights(raw_weights: dict) -> dict:
    """Extract acoustic tokenizer encoder weights into a flat structure.

    Handles both original HF keys (model.acoustic_tokenizer.encoder.*)
    and converted MLX keys (acoustic_encoder.*).

    Returns:
        Structured weight dict for encode_audio()
    """
    if any(k.startswith("model.acoustic_tokenizer.encoder.") for k in raw_weights):
        p = "model.acoustic_tokenizer.encoder."
    elif any(k.startswith("acoustic_encoder.") for k in raw_weights):
        p = "acoustic_encoder."
    else:
        raise KeyError("No acoustic encoder weights found")
    dtype = mx.float16

    def _w(key):
        return raw_weights[p + key].astype(dtype)

    weights = {}

    # Stem
    weights["stem_w"] = _w("downsample_layers.0.0.conv.conv.weight")
    weights["stem_b"] = _w("downsample_layers.0.0.conv.conv.bias")

    # Stages
    weights["blocks"] = []
    for i in range(len(DEPTHS)):
        stage_blocks = []
        for j in range(DEPTHS[i]):
            bp = f"stages.{i}.{j}."
            stage_blocks.append({
                "norm_w": _w(bp + "norm.weight"),
                "conv_w": _w(bp + "mixer.conv.conv.conv.weight"),
                "conv_b": _w(bp + "mixer.conv.conv.conv.bias"),
                "gamma": _w(bp + "gamma"),
                "ffn_norm_w": _w(bp + "ffn_norm.weight"),
                "ffn_l1_w": _w(bp + "ffn.linear1.weight"),
                "ffn_l1_b": _w(bp + "ffn.linear1.bias"),
                "ffn_l2_w": _w(bp + "ffn.linear2.weight"),
                "ffn_l2_b": _w(bp + "ffn.linear2.bias"),
                "ffn_gamma": _w(bp + "ffn_gamma"),
            })
        weights["blocks"].append(stage_blocks)

    # Downsamples
    weights["ds"] = []
    for i in range(len(RATIOS)):
        weights["ds"].append({
            "w": _w(f"downsample_layers.{i + 1}.0.conv.conv.weight"),
            "b": _w(f"downsample_layers.{i + 1}.0.conv.conv.bias"),
        })

    # Head
    weights["head_w"] = _w("head.conv.conv.weight")
    weights["head_b"] = _w("head.conv.conv.bias")

    return weights
