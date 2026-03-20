"""Load VibeVoice model weights from HuggingFace into MLX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .model import VibeVoiceConfig, VibeVoiceModel


def resolve_model_path(model_id_or_path: str) -> Path:
    """Resolve HuggingFace model ID or local path."""
    p = Path(model_id_or_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id_or_path))


def load_config(model_path: Path) -> VibeVoiceConfig:
    """Load VibeVoiceConfig from config.json.

    Handles both original HF format (nested decoder_config/diffusion_head_config)
    and converted MLX format (flat keys).
    """
    cfg_file = model_path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(cfg_file) as f:
        raw = json.load(f)

    # Original HF format nests LLM config under decoder_config
    dec = raw.get("decoder_config", {})
    diff = raw.get("diffusion_head_config", {})

    def _get(*keys, default=None):
        """Look up key in raw, then decoder_config, then diffusion_head_config."""
        for k in keys:
            if k in raw:
                return raw[k]
            if k in dec:
                return dec[k]
            if k in diff:
                return diff[k]
        return default

    return VibeVoiceConfig(
        hidden_size=_get("hidden_size", default=1536),
        num_hidden_layers=_get("num_hidden_layers", default=28),
        num_attention_heads=_get("num_attention_heads", default=12),
        num_key_value_heads=_get("num_key_value_heads", default=2),
        head_dim=_get("head_dim", default=128),
        intermediate_size=_get("intermediate_size", default=8960),
        vocab_size=_get("vocab_size", default=151936),
        rms_norm_eps=_get("rms_norm_eps", default=1e-6),
        rope_theta=_get("rope_theta", default=1_000_000.0),
        tie_word_embeddings=_get("tie_word_embeddings", default=True),
        vae_dim=_get("vae_dim", "acoustic_vae_dim", "speech_vae_dim", "latent_size", default=64),
        diffusion_layers=_get("head_layers", "diffusion_layers", default=4),
        head_ffn_ratio=_get("head_ffn_ratio", default=3.0),
        speech_scaling_factor=_get("speech_scaling_factor", default=0.1962890625),
        speech_bias_factor=_get("speech_bias_factor", default=-0.04931640625),
        speech_start_id=_get("speech_start_id", default=151652),
        speech_end_id=_get("speech_end_id", default=151653),
        speech_diffusion_id=_get("speech_diffusion_id", default=151654),
        eos_id=_get("eos_id", default=151643),
    )


def _load_safetensors(model_path: Path) -> dict[str, mx.array]:
    """Load all safetensors files from a model directory."""
    weights = {}
    for sf_file in sorted(model_path.glob("*.safetensors")):
        weights.update(mx.load(str(sf_file)))
    return weights


# ---------------------------------------------------------------------------
# Weight key mapping: original HF -> MLX nn.Module paths
# ---------------------------------------------------------------------------

_PREFIX_MAP = [
    ("model.language_model.layers.", "model.layers."),
    ("model.language_model.embed_tokens.", "model.embed_tokens."),
    ("model.language_model.norm.", "model.norm."),
    ("model.prediction_head.", "diffusion_head."),
    ("model.acoustic_tokenizer.decoder.", "vae_decoder."),
    ("model.acoustic_connector.", "acoustic_connector."),
    ("model.semantic_connector.", "semantic_connector."),
    ("model.semantic_tokenizer.encoder.", "semantic_encoder."),
]


def _map_hf_key(name: str) -> str | None:
    """Map a single HuggingFace weight key to MLX model path."""
    # lm_head stays as-is (7B only)
    if name == "lm_head.weight":
        return "lm_head.weight"

    # Skip weights we don't need
    if "acoustic_tokenizer.encoder" in name:
        return None
    # Skip non-encoder semantic_tokenizer weights (e.g. decoder parts)
    # but keep semantic_tokenizer.encoder.* for the MLX semantic encoder
    if "semantic_tokenizer" in name and "semantic_tokenizer.encoder." not in name:
        return None
    if name in ("model.speech_scaling_factor", "model.speech_bias_factor"):
        return None

    for hf_prefix, mlx_prefix in _PREFIX_MAP:
        if name.startswith(hf_prefix):
            return mlx_prefix + name[len(hf_prefix):]

    return None


def _map_mlx_vae_weights(raw: dict[str, mx.array]) -> dict[str, mx.array]:
    """Extract and organize VAE decoder weights into the format expected by VAEDecoder."""
    p = "vae_decoder."
    result = {}

    # Init conv
    if p + "upsample_layers.0.0.conv.conv.weight" in raw:
        result["init_conv_w"] = raw[p + "upsample_layers.0.0.conv.conv.weight"]
        result["init_conv_b"] = raw[p + "upsample_layers.0.0.conv.conv.bias"]

    # Stages
    depths = [8, 3, 3, 3, 3, 3, 3]
    for s in range(7):
        for b in range(depths[s]):
            bp = f"{p}stages.{s}.{b}."
            prefix = f"stage_{s}_block_{b}_"
            for key_suffix, result_key in [
                ("norm.weight", "norm_w"),
                ("mixer.conv.conv.conv.weight", "conv_w"),
                ("mixer.conv.conv.conv.bias", "conv_b"),
                ("gamma", "gamma"),
                ("ffn_norm.weight", "ffn_norm_w"),
                ("ffn.linear1.weight", "ffn_l1_w"),
                ("ffn.linear1.bias", "ffn_l1_b"),
                ("ffn.linear2.weight", "ffn_l2_w"),
                ("ffn.linear2.bias", "ffn_l2_b"),
                ("ffn_gamma", "ffn_gamma"),
            ]:
                full_key = bp + key_suffix
                if full_key in raw:
                    result[prefix + result_key] = raw[full_key]

    # Upsample convs
    ratios = [8, 5, 5, 4, 2, 2]
    for i in range(1, 7):
        up = f"{p}upsample_layers.{i}.0.convtr.convtr."
        if up + "weight" in raw:
            result[f"upsample_{i}_w"] = raw[up + "weight"]
            result[f"upsample_{i}_b"] = raw[up + "bias"]
            result[f"upsample_{i}_stride"] = ratios[i - 1]

    # Head conv
    if p + "head.conv.conv.weight" in raw:
        result["head_w"] = raw[p + "head.conv.conv.weight"]
        result["head_b"] = raw[p + "head.conv.conv.bias"]

    return result


def load_model(
    model_id: str,
    quantize_bits: Optional[int] = None,
) -> tuple[VibeVoiceModel, VibeVoiceConfig]:
    """Load VibeVoice model from HuggingFace.

    Args:
        model_id: HuggingFace model ID or local path
        quantize_bits: None or 8 for INT8 quantization on LLM backbone

    Returns:
        (model, config)
    """
    print(f"Loading model from {model_id}...")
    model_path = resolve_model_path(model_id)
    config = load_config(model_path)

    print(f"  Config: {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, heads={config.num_attention_heads}, "
          f"tie_embeddings={config.tie_word_embeddings}")

    raw_weights = _load_safetensors(model_path)

    # Check if weights are already in MLX format (converted) or original HF format
    is_mlx_format = any(k.startswith("model.layers.") for k in raw_weights)
    is_hf_format = any(k.startswith("model.language_model.") for k in raw_weights)

    if is_hf_format:
        print("  Detected original HuggingFace format, remapping keys...")
        mapped = {}
        for name, w in raw_weights.items():
            mlx_name = _map_hf_key(name)
            if mlx_name is not None:
                mapped[mlx_name] = w
        raw_weights = mapped
    elif is_mlx_format:
        print("  Detected MLX format")
    else:
        print("  Warning: Unknown weight format, attempting direct load")

    # Create model
    model = VibeVoiceModel(config)

    # Separate VAE decoder and semantic encoder weights (loaded manually, not via nn.Module)
    vae_keys = [k for k in raw_weights if k.startswith("vae_decoder.")]
    sem_keys = [k for k in raw_weights if k.startswith("semantic_encoder.")]
    non_vae = {k: v for k, v in raw_weights.items()
               if not k.startswith("vae_decoder.") and not k.startswith("semantic_encoder.")}

    # Filter out lm_head.weight for tied-embedding models (1.5B)
    # nn.Module.load_weights can't set attributes on None
    if config.tie_word_embeddings:
        non_vae.pop("lm_head.weight", None)

    # Load non-VAE weights via nn.Module.load_weights
    weight_pairs = list(non_vae.items())

    # Per-layer eval for LLM weights to control memory
    if quantize_bits is not None:
        print(f"  Loading with INT{quantize_bits} quantization (per-layer eval)...")
        # Load weights layer by layer with eval to prevent memory explosion
        lm_layer_weights = {}
        other_weights = []

        for name, w in weight_pairs:
            if name.startswith("model.layers."):
                parts = name.split(".")
                layer_idx = int(parts[2])
                if layer_idx not in lm_layer_weights:
                    lm_layer_weights[layer_idx] = []
                lm_layer_weights[layer_idx].append((name, w))
            else:
                other_weights.append((name, w))

        # Load other weights first (strict=False since LLM layers are loaded separately)
        model.load_weights(other_weights, strict=False)

        # Load LLM layers one at a time with eval
        for layer_idx in sorted(lm_layer_weights.keys()):
            model.load_weights(lm_layer_weights[layer_idx], strict=False)
            mx.eval(model.model.layers[layer_idx].parameters())

        # Quantize LLM backbone
        nn.quantize(
            model.model,
            bits=quantize_bits,
            group_size=64 if quantize_bits == 4 else 32,
            class_predicate=lambda _, m: (
                isinstance(m, nn.Linear)
                and m.weight.shape[0] % 64 == 0
                and m.weight.shape[1] % 64 == 0
            ),
        )
    else:
        model.load_weights(weight_pairs)

    # Load VAE decoder weights manually
    vae_raw = {k: raw_weights[k] for k in vae_keys}
    vae_data = _map_mlx_vae_weights(vae_raw) if vae_keys else {}
    _populate_vae_decoder(model.vae_decoder, vae_data, config)

    # Cast all weights to float16 (HF checkpoints store bfloat16 which is
    # ~2x slower than float16 for repeated small-batch matmul on Apple Silicon)
    def _cast_to_f16(params):
        if isinstance(params, dict):
            return {k: _cast_to_f16(v) for k, v in params.items()}
        if isinstance(params, list):
            return [_cast_to_f16(v) for v in params]
        if isinstance(params, mx.array) and params.dtype == mx.bfloat16:
            return params.astype(mx.float16)
        return params

    model.update(_cast_to_f16(model.parameters()))
    mx.eval(model.parameters())
    print(f"  Model loaded successfully")

    return model, config


def _populate_vae_decoder(vae: "VAEDecoder", data: dict, config: VibeVoiceConfig):
    """Populate VAE decoder with extracted weight data."""
    dtype = mx.float16

    if "init_conv_w" in data:
        vae.init_conv_w = data["init_conv_w"].astype(dtype)
        vae.init_conv_b = data["init_conv_b"].astype(dtype)

    if "head_w" in data:
        vae.head_w = data["head_w"].astype(dtype)
        vae.head_b = data["head_b"].astype(dtype)

    # Build stages as list of lists of dicts
    depths = config.vae_depths
    vae.stages = []
    for s in range(7):
        blocks = []
        for b in range(depths[s]):
            prefix = f"stage_{s}_block_{b}_"
            block = {}
            for key in ["norm_w", "conv_w", "conv_b", "gamma",
                         "ffn_norm_w", "ffn_l1_w", "ffn_l1_b",
                         "ffn_l2_w", "ffn_l2_b", "ffn_gamma"]:
                full_key = prefix + key
                if full_key in data:
                    block[key] = data[full_key].astype(dtype)
            if block:
                blocks.append(block)
        vae.stages.append(blocks)

    # Upsample convs
    vae.upsample_convs = []
    for i in range(1, 7):
        w_key = f"upsample_{i}_w"
        if w_key in data:
            vae.upsample_convs.append((
                data[w_key].astype(dtype),
                data[f"upsample_{i}_b"].astype(dtype),
                data[f"upsample_{i}_stride"],
            ))
