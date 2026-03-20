"""Convert VibeVoice HuggingFace weights to standalone MLX format.

Produces HuggingFace-uploadable model repos with MLX safetensors.

Usage:
    uv run python convert.py --output-dir converted/
    uv run python convert.py --output-dir converted/ --models 1.5b
    uv run python convert.py --output-dir converted/ --upload --hf-prefix your-username
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

from vibevoice_mlx.load_weights import _load_safetensors, _map_hf_key, load_config, resolve_model_path

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


def convert_model(model_id: str, tag: str, output_dir: Path, hf_prefix: str = "gafiatulin"):
    """Convert a VibeVoice model to MLX format."""
    print(f"\n{'='*60}")
    print(f"Converting {model_id}")
    print(f"{'='*60}")

    model_path = resolve_model_path(model_id)
    config = load_config(model_path)

    # Load original weights
    raw = _load_safetensors(model_path)
    print(f"  Loaded {len(raw)} weight tensors")

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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights (potentially sharded for 7B)
    total_bytes = sum(w.nbytes for w in mapped.values())
    if total_bytes > SHARD_SIZE:
        _save_sharded(output_dir, mapped)
    else:
        mx.save_safetensors(str(output_dir / "model.safetensors"), mapped)

    # Save config
    config_dict = {
        "model_type": "vibevoice",
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
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy tokenizer
    _copy_tokenizer(output_dir, TOKENIZER_IDS[tag])

    # Model card
    _write_model_card(output_dir, model_id, tag, hf_prefix)

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


def _write_model_card(output_dir: Path, model_id: str, tag: str, hf_prefix: str):
    model_arg = "" if tag == "1.5b" else f"\n  --model {hf_prefix}/vibevoice-{tag}-mlx "
    card = f"""---
license: mit
base_model: {model_id}
tags:
  - mlx
  - tts
  - vibevoice
  - apple-silicon
  - voice-cloning
---

# VibeVoice {tag.upper()} — MLX

MLX-converted fp16 weights for [{model_id}](https://huggingface.co/{model_id}).

For inference code, benchmarks, and documentation see [vibevoice-mlx](https://github.com/gafiatulin/vibevoice-mlx).

## Quick start

```bash
git clone https://github.com/gafiatulin/vibevoice-mlx && cd vibevoice-mlx
uv sync

# Basic synthesis (weights download automatically)
uv run vibevoice-mlx{model_arg} --text "Hello, world!" --output hello.wav

# Voice cloning
uv run vibevoice-mlx{model_arg} \\
  --ref-audio speaker.wav --text "Clone this voice" --output cloned.wav
```
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
    parser = argparse.ArgumentParser(description="Convert VibeVoice weights to MLX format")
    parser.add_argument("--output-dir", type=str, default="converted", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["1.5b", "7b"], choices=["1.5b", "7b"],
                        help="Which models to convert")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--hf-prefix", type=str, default="gafiatulin", help="HF username/org prefix")
    args = parser.parse_args()

    base = Path(args.output_dir)

    for tag in args.models:
        model_id = MODEL_IDS[tag]
        out = base / f"vibevoice-{tag}-mlx"
        convert_model(model_id, tag, out, hf_prefix=args.hf_prefix)
        if args.upload:
            upload(out, f"{args.hf_prefix}/vibevoice-{tag}-mlx")

    print("\nDone!")


if __name__ == "__main__":
    main()
