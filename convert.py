"""Convert VibeVoice-compatible HuggingFace weights to standalone MLX format.

Produces HuggingFace-uploadable model repos with MLX safetensors.

Usage:
    uv run python convert.py --output-dir converted/
    uv run python convert.py --output-dir converted/ --models 1.5b
    uv run python convert.py --output-dir converted/ --upload --hf-prefix your-username
    uv run python convert.py --model-id kugelaudio/kugelaudio-0-open --output-dir converted/kugelaudio-mlx
    uv run python convert.py --model-id microsoft/VibeVoice-1.5B --quantize 4 --output-dir converted/vibevoice-1.5b-int4-mlx
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from vibevoice_mlx.load_weights import (
    _load_safetensors, _map_hf_key, _quantize_predicate,
    load_config, resolve_model_path,
)
from vibevoice_mlx.model import VibeVoiceModel

MODEL_IDS = {
    "1.5b": "microsoft/VibeVoice-1.5B",
    "7b": "vibevoice/VibeVoice-7B",
}

TOKENIZER_IDS = {
    "1.5b": "Qwen/Qwen2.5-1.5B",
    "7b": "Qwen/Qwen2.5-7B",
}

# 5GB shard threshold
SHARD_SIZE = 5 * 1024 * 1024 * 1024


def _detect_tokenizer_id(config) -> str:
    """Pick the right Qwen2.5 tokenizer based on vocab size."""
    if config.vocab_size <= 151936:
        return "Qwen/Qwen2.5-1.5B"
    return "Qwen/Qwen2.5-7B"


def convert_model(model_id: str, output_dir: Path, tokenizer_id: str | None = None,
                   quantize_bits: int | None = None):
    """Convert a VibeVoice-compatible model to MLX format."""
    print(f"\n{'='*60}")
    print(f"Converting {model_id}" + (f" (INT{quantize_bits})" if quantize_bits else ""))
    print(f"{'='*60}")

    model_path = resolve_model_path(model_id)
    config = load_config(model_path)

    # Load original weights
    raw = _load_safetensors(model_path)
    print(f"  Loaded {len(raw)} weight tensors")

    # Extract speech scaling/bias from weights (stored as scalar tensors in HF
    # checkpoints but used as config values in the MLX pipeline)
    for wname, attr in [
        ("model.speech_scaling_factor", "speech_scaling_factor"),
        ("model.speech_bias_factor", "speech_bias_factor"),
    ]:
        if wname in raw:
            setattr(config, attr, raw[wname].item())

    # Remap keys
    mapped = {}
    skipped = []
    for name, w in raw.items():
        mlx_name = _map_hf_key(name)
        if mlx_name is not None:
            mapped[mlx_name] = w
        else:
            skipped.append(name)

    print(f"  Mapped {len(mapped)} weights, skipped {len(skipped)}")

    # Quantize if requested: load into model, quantize, then flatten back
    quantization_meta = None
    if quantize_bits is not None:
        group_size = 64 if quantize_bits == 4 else 32
        print(f"  Quantizing to INT{quantize_bits} (group_size={group_size})...")

        model = VibeVoiceModel(config)
        # Separate model backbone weights from VAE/encoder
        manual_prefixes = ("vae_decoder.", "semantic_encoder.", "acoustic_encoder.")
        backbone_weights = {k: v for k, v in mapped.items()
                           if not any(k.startswith(p) for p in manual_prefixes)}
        other_weights = {k: v for k, v in mapped.items()
                        if any(k.startswith(p) for p in manual_prefixes)}

        if config.tie_word_embeddings:
            backbone_weights.pop("lm_head.weight", None)

        model.load_weights(list(backbone_weights.items()), strict=False)
        nn.quantize(
            model.model, bits=quantize_bits, group_size=group_size,
            class_predicate=_quantize_predicate,
        )

        # Flatten quantized parameters back to saveable dict
        quantized = dict(tree_flatten(model.parameters()))
        # Re-add non-backbone weights (VAE, encoders)
        quantized.update(other_weights)
        mapped = quantized
        quantization_meta = {"bits": quantize_bits, "group_size": group_size}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights (potentially sharded for 7B+)
    total_bytes = sum(w.nbytes for w in mapped.values())
    if total_bytes > SHARD_SIZE:
        _save_sharded(output_dir, mapped)
    else:
        mx.save_safetensors(str(output_dir / "model.safetensors"), mapped)

    # Save config
    config_dict = {
        "model_type": "vibevoice",
        "base_model": model_id,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "vae_dim": config.vae_dim,
        "diffusion_layers": config.diffusion_layers,
        "head_ffn_ratio": config.head_ffn_ratio,
        "speech_scaling_factor": config.speech_scaling_factor,
        "speech_bias_factor": config.speech_bias_factor,
        "speech_start_id": config.speech_start_id,
        "speech_end_id": config.speech_end_id,
        "speech_diffusion_id": config.speech_diffusion_id,
        "eos_id": config.eos_id,
    }
    if quantization_meta is not None:
        config_dict["quantization"] = quantization_meta
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy tokenizer
    tok_id = tokenizer_id or _detect_tokenizer_id(config)
    _copy_tokenizer(output_dir, tok_id)

    # Model card
    _write_model_card(output_dir, model_id)

    total_mb = sum(f.stat().st_size for f in output_dir.glob("*")) / 1e6
    print(f"  Saved to {output_dir} ({total_mb:.0f} MB)")


def _save_sharded(output_dir: Path, weights: dict[str, mx.array]):
    """Save weights in shards of ~5GB each."""
    shards = []
    current_shard = {}
    current_size = 0
    shard_idx = 0

    for name, w in sorted(weights.items()):
        if current_size + w.nbytes > SHARD_SIZE and current_shard:
            shard_idx += 1
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = w
        current_size += w.nbytes

    if current_shard:
        shards.append(current_shard)

    n_shards = len(shards)
    weight_map = {}

    for i, shard in enumerate(shards):
        filename = f"model-{i+1:05d}-of-{n_shards:05d}.safetensors"
        mx.save_safetensors(str(output_dir / filename), shard)
        for name in shard:
            weight_map[name] = filename

    index = {
        "metadata": {"total_size": sum(w.nbytes for w in weights.values())},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Saved {n_shards} shards")


def _copy_tokenizer(output_dir: Path, tokenizer_id: str):
    """Copy tokenizer files from Qwen2.5."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"  Tokenizer saved from {tokenizer_id}")


def _write_model_card(output_dir: Path, model_id: str):
    card = f"""---
license: mit
base_model: {model_id}
tags:
  - mlx
  - tts
  - apple-silicon
---

# {model_id} — MLX

MLX-converted fp16 weights for [{model_id}](https://huggingface.co/{model_id}).

For inference code see [vibevoice-mlx](https://github.com/gafiatulin/vibevoice-mlx).
"""
    (output_dir / "README.md").write_text(card)


def upload(output_dir: Path, repo_id: str):
    """Upload converted weights to HuggingFace."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id)
    print(f"  Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice-compatible weights to MLX format")
    parser.add_argument("--output-dir", type=str, default="converted", help="Output directory")
    parser.add_argument("--models", nargs="+", default=None, choices=["1.5b", "7b"],
                        help="Built-in model tags to convert (1.5b, 7b)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Arbitrary HuggingFace model ID to convert")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None,
                        help="Save pre-quantized weights (int4 or int8)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer ID (auto-detected if not specified)")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--hf-prefix", type=str, default="gafiatulin", help="HF username/org prefix")
    args = parser.parse_args()

    base = Path(args.output_dir)

    if args.model_id:
        # Convert arbitrary model
        out = base if base.name != "converted" else base / (args.model_id.replace("/", "-") + "-mlx")
        convert_model(args.model_id, out, tokenizer_id=args.tokenizer,
                      quantize_bits=args.quantize)
        if args.upload:
            repo_id = f"{args.hf_prefix}/{out.name}"
            upload(out, repo_id)
    else:
        # Convert built-in models
        tags = args.models or ["1.5b", "7b"]
        for tag in tags:
            model_id = MODEL_IDS[tag]
            out = base / f"vibevoice-{tag}-mlx"
            convert_model(model_id, out, tokenizer_id=TOKENIZER_IDS[tag],
                          quantize_bits=args.quantize)
            if args.upload:
                upload(out, f"{args.hf_prefix}/vibevoice-{tag}-mlx")

    print("\nDone!")


if __name__ == "__main__":
    main()
