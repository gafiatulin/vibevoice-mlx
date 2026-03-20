# VibeVoice MLX

MLX inference for [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) text-to-speech on Apple Silicon.

Zero-shot voice cloning TTS: synthesize speech from text, optionally cloning one or more reference voices. Pure MLX — no PyTorch dependency at inference time.

## Quick start

```bash
# Install
git clone https://github.com/gafiatulin/vibevoice-mlx && cd vibevoice-mlx
uv sync

# Basic synthesis (model downloads automatically)
uv run vibevoice-mlx --text "Hello, world!" --output hello.wav

# Voice cloning
uv run vibevoice-mlx \
  --ref-audio speaker.wav --text "Clone this voice" --output cloned.wav

# Encode a voice for reuse (one-time)
uv run vibevoice-mlx \
  --ref-audio speaker.wav --save-voice voice.safetensors

# Synthesize with saved voice
uv run vibevoice-mlx \
  --voice voice.safetensors --text "Hello again!" --output hello.wav

# Multi-speaker voice cloning
uv run vibevoice-mlx \
  --ref-audio spk1.wav spk2.wav \
  --text "Speaker 1: Hello.\nSpeaker 2: Hi there." --output dialogue.wav

# With quantization for faster generation
uv run vibevoice-mlx --quantize 8 --text "Hello, world!"
```

## Performance

Benchmarked subprocess-isolated on Apple Silicon (M4 Max, 64 GB) with voice cloning, ~30s audio output:

**vibevoice-1.5b-mlx:**

| Config | RTF | Gen | Peak Mem |
|--------|-----|-----|----------|
| fp16 | 1.85x | 16.3s | 6.7 GB |
| fp16, no-semantic | 2.46x | 11.2s | 5.7 GB |
| fp16, coreml-semantic | 2.07x | 12.3s | 6.0 GB |
| int8 | 2.63x | 9.7s | 5.4 GB |
| int8, no-semantic | 4.03x | 6.9s | 4.5 GB |
| int8, coreml-semantic | 3.07x | 8.4s | 4.8 GB |
| int4 | 2.72x | 8.9s | 4.6 GB |
| int4, no-semantic | 4.30x | 5.2s | 3.8 GB |
| int4, coreml-semantic | 3.22x | 7.3s | 3.9 GB |

**vibevoice-7b-mlx:**

| Config | RTF | Gen | Peak Mem |
|--------|-----|-----|----------|
| fp16 | 0.53x | 53.0s | 21.7 GB |
| fp16, no-semantic | 0.59x | 51.7s | 20.3 GB |
| fp16, coreml-semantic | 0.56x | 57.2s | 21.0 GB |
| int8 | 1.06x | 29.6s | 14.9 GB |
| int8, no-semantic | 1.24x | 23.3s | 13.6 GB |
| int8, coreml-semantic | 1.12x | 25.5s | 14.2 GB |
| int4 | 1.16x | 25.8s | 11.2 GB |
| int4, no-semantic | 1.37x | 19.5s | 9.8 GB |
| int4, coreml-semantic | 1.24x | 22.0s | 10.5 GB |

RTF = audio duration / processing time (higher is faster; >1x means faster than real-time).

## Supported models

Pre-converted MLX weights with bundled tokenizer (downloaded automatically):

| Model | Size | Original |
|-------|------|----------|
| [gafiatulin/vibevoice-1.5b-mlx](https://huggingface.co/gafiatulin/vibevoice-1.5b-mlx) | 4.7 GB | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| [gafiatulin/vibevoice-7b-mlx](https://huggingface.co/gafiatulin/vibevoice-7b-mlx) | 18 GB | [vibevoice/VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B) |

Optional CoreML semantic encoder (downloaded automatically when `--coreml-semantic` is used):

| Component | Size |
|-----------|------|
| [gafiatulin/vibevoice-semantic-encoder-mlpackage](https://huggingface.co/gafiatulin/vibevoice-semantic-encoder-mlpackage) | ~657 MB |

The semantic encoder provides acoustic feedback to the LLM during generation, improving speech quality. By default it runs as a pure MLX model on the GPU. The CoreML variant offloads it to the Apple Neural Engine, freeing the GPU for the LLM and diffusion head — this improves throughput without quality loss.

## Architecture

```
Text ──→ Qwen2.5 LLM backbone ──→ control tokens
                │
                └──→ Diffusion head (DDPM v-prediction, DPM-Solver++) ──→ VAE latents
                                                                              │
                                                          VAE decoder ──→ 24kHz audio
                                                              ▲
                                          Semantic encoder ───┘ (optional feedback)
```

## CLI options

```
--text TEXT              Text to synthesize (required)
--model MODEL            HuggingFace model ID (default: gafiatulin/vibevoice-1.5b-mlx)
--output FILE            Output WAV path (default: output.wav)
--ref-audio FILE [FILE]  Reference audio for voice cloning (one per speaker)
--voice FILE [FILE]      Pre-encoded voice (.safetensors) for voice cloning
--save-voice FILE        Save encoded voice embeddings for reuse
--quantize {4,8}         Quantize LLM backbone (int4 or int8)
--quantize-diffusion     Also quantize the diffusion head
--diffusion-steps N      DPM-Solver++ steps (default: 10)
--cfg-scale FLOAT        Classifier-free guidance scale (default: 1.3)
--max-speech-tokens N    Max speech tokens to generate (default: 200)
--seed INT               Random seed (default: 42)
--no-semantic            Disable semantic encoder feedback
--coreml-semantic        Use CoreML semantic encoder for GPU pipelining
--tokenizer MODEL        Override tokenizer (default: bundled)
```

## Optimizations

- **DPM-Solver++ 2M**: Second-order multistep solver — 10 DPM steps > 100 DDPM steps quality
- **Streaming VAE decoder**: Causal conv caches for chunk-by-chunk decoding
- **Streaming semantic encoder**: 34-buffer causal CNN for real-time feedback
- **CoreML semantic pipelining**: Offload semantic encoder to ANE while GPU runs LLM
- **Selective quantization**: LLM backbone quantized (int4/int8), diffusion head stays full precision
- **PyTorch-compatible RNG**: Pure Python MT19937 + Box-Muller matching `torch.randn()` exactly
- **bf16→fp16 conversion**: 2x faster inference on Apple Silicon vs bfloat16

## Project structure

```
vibevoice_mlx/
├── e2e_pipeline.py     CLI entry point and voice cloning
├── model.py            Qwen2.5 backbone + diffusion head + VAE decoder
├── generate.py         Autoregressive generation with DPM-Solver++
├── load_weights.py     HuggingFace weight loading and key mapping
├── streaming_vae.py    Streaming VAE decoder with conv caches
├── semantic_encoder.py Pure MLX streaming semantic encoder
├── fast_forward.py     Optimized LM and diffusion head forward
└── numpy_rng.py        PyTorch-compatible RNG (MT19937 + Box-Muller)
convert.py              Weight conversion and HuggingFace upload
bench_compare.py        Quantization and config benchmark suite
```

## Requirements

- Python >= 3.10
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.22.0

```bash
uv sync
```

## License

This inference code is MIT licensed. See [LICENSE](LICENSE).

The model weights ([microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)) are under the [MIT License](https://huggingface.co/microsoft/VibeVoice-1.5B/blob/main/LICENSE).
