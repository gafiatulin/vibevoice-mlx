"""VibeVoice MLX — end-to-end text-to-speech pipeline.

Usage:
    vibevoice-mlx --text "Hello, world!" --output hello.wav
    vibevoice-mlx --ref-audio speaker.wav --text "Clone this voice" --output cloned.wav
    vibevoice-mlx --ref-audio spk1.wav spk2.wav --text "Speaker 1: Hello\nSpeaker 2: Hi there"
    vibevoice-mlx --quantize 8 --text "Test"
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

import mlx.core as mx

from .generate import GenerationOptions, generate
from .load_weights import load_model, resolve_model_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000
VOICE_CLONE_SAMPLE_SECS = 10
VOICE_CLONE_SAMPLES = SAMPLE_RATE * VOICE_CLONE_SAMPLE_SECS  # 240000
SPEECH_TOK_COMPRESS_RATIO = 3200  # samples per VAE token


# ---------------------------------------------------------------------------
# Voice cloning data
# ---------------------------------------------------------------------------

@dataclass
class SpeakerRef:
    """Per-speaker voice reference data."""
    speaker_id: int
    ref_audio_np: np.ndarray
    num_vae_tokens: int
    speech_embed_positions: list[int]
    cached_embeds: Optional[np.ndarray] = None


@dataclass
class VoiceCloneData:
    """Pre-processed voice cloning data for prompt injection."""
    input_ids: list[int]
    speakers: list[SpeakerRef]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_and_resample(audio_path: str) -> np.ndarray:
    """Load audio file, convert to mono 24kHz float32."""
    import soundfile as sf
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        gcd = math.gcd(SAMPLE_RATE, sr)
        wav = resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Voice save/load
# ---------------------------------------------------------------------------

def save_voice(path: str, embeds: np.ndarray):
    """Save pre-encoded voice embeddings to safetensors."""
    from safetensors.numpy import save_file
    save_file({"embeddings": embeds}, path)
    logger.info("Saved voice embeddings to %s (%s)", path, embeds.shape)


def load_voice(path: str) -> np.ndarray:
    """Load pre-encoded voice embeddings from safetensors."""
    from safetensors.numpy import load_file
    data = load_file(path)
    return data["embeddings"]


# ---------------------------------------------------------------------------
# Voice encoding
# ---------------------------------------------------------------------------

def encode_voice_reference(
    wav: np.ndarray,
    num_vae_tokens: int,
    model,
    config,
    model_id: str,
) -> np.ndarray:
    """Encode reference audio to speech embeddings via pure MLX VAE encoder.

    Uses the model's own acoustic tokenizer encoder weights directly.
    Returns embeddings of shape (num_vae_tokens, hidden_size).
    """
    from .vae_encoder import load_vae_encoder_weights, encode_audio
    from .load_weights import resolve_model_path, _load_safetensors

    # Prefer pre-extracted encoder weights from load_model() to avoid
    # reloading all safetensors (~18 GB for large models).
    enc_weights = getattr(model, "_encoder_weights", None)

    if enc_weights is None:
        # Fallback: load from disk with per-model caching
        cache_key = model_id
        if not hasattr(encode_voice_reference, "_enc_cache"):
            encode_voice_reference._enc_cache = {}
        if cache_key not in encode_voice_reference._enc_cache:
            model_path = resolve_model_path(model_id)
            raw = _load_safetensors(model_path)
            has_enc = (
                any(k.startswith("model.acoustic_tokenizer.encoder.") for k in raw)
                or any(k.startswith("acoustic_encoder.") for k in raw)
            )
            if not has_enc:
                raise RuntimeError(
                    f"No acoustic encoder weights found in {model_path}. "
                    "Re-convert the model to include encoder weights."
                )
            encode_voice_reference._enc_cache[cache_key] = load_vae_encoder_weights(raw)
        enc_weights = encode_voice_reference._enc_cache[cache_key]

    # Pad/trim to VOICE_CLONE_SAMPLES
    audio_padded = np.zeros(VOICE_CLONE_SAMPLES, dtype=np.float32)
    actual_len = min(len(wav), VOICE_CLONE_SAMPLES)
    audio_padded[:actual_len] = wav[:actual_len]

    # Encode: (1, 1, T) → (1, T_compressed, vae_dim)
    audio_mx = mx.array(audio_padded).reshape(1, 1, -1)
    latents = encode_audio(audio_mx, enc_weights)
    mx.eval(latents)

    # Trim to actual token count
    actual_t = min(latents.shape[1], num_vae_tokens)
    latents = latents[:, :actual_t, :]  # (1, T, vae_dim)

    # Apply scaling and bias, then batch through acoustic connector
    features = (latents + config.speech_bias_factor) * config.speech_scaling_factor
    features = features.astype(mx.float16)  # (1, T, vae_dim)
    embeds = model.acoustic_connector(features)  # (1, T, hidden_size)
    mx.eval(embeds)

    logger.info("  Encoded voice: %d tokens", actual_t)
    return np.array(embeds[0])  # (T, hidden_size)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_text(
    text: str,
    tokenizer_name: str,
    config,
    ref_audio: list[str] | None = None,
    tokenizer=None,
    speaker_embeds: list[tuple[int, np.ndarray]] | None = None,
) -> list[int] | VoiceCloneData:
    """Build the full prompt token sequence for TTS.

    If ref_audio or speaker_embeds is provided, returns VoiceCloneData with
    per-speaker embedding positions for voice cloning injection during prefill.

    Args:
        speaker_embeds: Pre-encoded embeddings as list of (num_tokens, embeds)
            where embeds is shape (num_tokens, hidden_size). Alternative to
            ref_audio for batch synthesis with pre-encoded voices.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    system_tokens = tokenizer.encode(system_prompt)

    # Build voice references from either file paths or pre-encoded embeddings
    voice_refs = None
    if speaker_embeds is not None and len(speaker_embeds) > 0:
        voice_refs = []
        for num_tokens, embeds in speaker_embeds:
            voice_refs.append((num_tokens, np.zeros(0, dtype=np.float32), embeds))
    elif ref_audio is not None and len(ref_audio) > 0:
        voice_refs = []
        for audio_path in ref_audio:
            if audio_path.endswith(".safetensors"):
                embeds = load_voice(audio_path)
                num_vae_tokens = embeds.shape[0]
                wav = np.zeros(0, dtype=np.float32)
            else:
                wav = _load_and_resample(audio_path)
                if len(wav) > VOICE_CLONE_SAMPLES:
                    wav = wav[:VOICE_CLONE_SAMPLES]
                num_vae_tokens = math.ceil(len(wav) / SPEECH_TOK_COMPRESS_RATIO)
                embeds = None
            voice_refs.append((num_vae_tokens, wav, embeds))

    if voice_refs is not None:
        voice_prefix = tokenizer.encode(" Voice input:\n", add_special_tokens=False)
        newline_tok = tokenizer.encode("\n", add_special_tokens=False)

        voice_tokens = list(voice_prefix)
        speakers = []
        current_offset = len(system_tokens) + len(voice_prefix)

        for spk_idx, (num_vae_tokens, wav, cached) in enumerate(voice_refs):
            spk_prefix = tokenizer.encode(f" Speaker {spk_idx}:", add_special_tokens=False)
            voice_tokens += spk_prefix
            current_offset += len(spk_prefix)

            voice_tokens.append(config.speech_start_id)
            current_offset += 1

            speech_positions = list(range(current_offset, current_offset + num_vae_tokens))
            voice_tokens += [config.speech_diffusion_id] * num_vae_tokens
            current_offset += num_vae_tokens

            voice_tokens.append(config.speech_end_id)
            voice_tokens += newline_tok
            current_offset += 1 + len(newline_tok)

            speakers.append(SpeakerRef(
                speaker_id=spk_idx,
                ref_audio_np=wav,
                num_vae_tokens=num_vae_tokens,
                speech_embed_positions=speech_positions,
                cached_embeds=cached,
            ))

        # Text section — normalize speaker IDs from 1-based (user) to 0-based (model)
        text_section = tokenizer.encode(" Text input:\n", add_special_tokens=False)
        lines = text.strip().split("\n")
        speaker_tokens_list = []
        for line in lines:
            line = line.strip()
            def _decrement_speaker(m):
                n = int(m.group(1))
                return f"Speaker {max(0, n - 1)}"
            if re.match(r"Speaker\s+\d+", line):
                line = re.sub(r"Speaker\s+(\d+)", _decrement_speaker, line)
            elif config.single_segment:
                line = f"Speaker 0: {line}"
            speaker_tokens_list += tokenizer.encode(f" {line}\n", add_special_tokens=False)

        output_section = tokenizer.encode(" Speech output:\n", add_special_tokens=False)
        all_ids = (system_tokens + voice_tokens + text_section +
                   speaker_tokens_list + output_section + [config.speech_start_id])

        return VoiceCloneData(input_ids=all_ids, speakers=speakers)

    # No voice cloning — simple text prompt
    text_section = tokenizer.encode(" Text input:\n", add_special_tokens=False)

    lines = text.strip().split("\n")
    speaker_tokens = []
    for line in lines:
        line = line.strip()
        # Single-segment models require "Speaker 0:" prefix
        if config.single_segment and not re.match(r"Speaker\s+\d+", line):
            line = f"Speaker 0: {line}"
        speaker_tokens += tokenizer.encode(f" {line}\n", add_special_tokens=False)

    output_section = tokenizer.encode(" Speech output:\n", add_special_tokens=False)

    return system_tokens + text_section + speaker_tokens + output_section + [config.speech_start_id]


def _load_semantic_encoder(model, config, model_id, use_coreml=False):
    """Load semantic encoder + connector, returns (callback, reset_fn) or (None, None).

    Args:
        use_coreml: If True, try CoreML semantic encoder (faster, requires
            coremltools and downloads ~657MB .mlpackage on first use).
            If False (default), use pure MLX.
    """
    if use_coreml:
        result = _try_coreml_semantic(model, config)
        if result is not None:
            return result

    # Pure MLX (no extra dependencies)
    return _try_mlx_semantic(model, config, model_id)


def _try_coreml_semantic(model, config):
    """Try loading CoreML semantic encoder (.mlpackage with stateful conv caches).

    Search order:
    1. Local vibevoice-coreml build dir (development)
    2. Bundled in converted MLX model repo on HuggingFace
    """
    try:
        import coremltools as ct
    except ImportError:
        return None

    sem_enc_path = None

    # 1. Local build dir (development)
    for d in [
        Path("../vibevoice-coreml/python/tts/vibevoice-multispeaker/build/vibevoice-1.5b"),
        Path("../vibevoice-coreml/python/tts/vibevoice-multispeaker/build/vibevoice-7b"),
    ]:
        enc = d / "semantic_encoder_streaming.mlpackage"
        if enc.exists():
            sem_enc_path = enc
            break

    # 2. HuggingFace (auto-download the .mlpackage)
    # CoreML can't load from HF cache (symlinks break compilation),
    # so copy to a local directory on first use.
    if sem_enc_path is None:
        try:
            from huggingface_hub import snapshot_download
            import shutil
            coreml_path = Path(snapshot_download(
                "gafiatulin/vibevoice-semantic-encoder-mlpackage",
            ))
            src = coreml_path / "semantic_encoder_streaming.mlpackage"
            if src.exists():
                # Copy to local cache (resolves symlinks)
                local_cache = Path.home() / ".cache" / "vibevoice-mlx" / "coreml"
                local_enc = local_cache / "semantic_encoder_streaming.mlpackage"
                if not local_enc.exists():
                    local_cache.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(str(src), str(local_enc), copy_function=shutil.copy2)
                sem_enc_path = local_enc
        except Exception:
            pass

    if sem_enc_path is None:
        return None

    try:
        sem_enc = ct.models.MLModel(
            str(sem_enc_path),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        sem_state = sem_enc.make_state()

        # Verify
        sem_enc.predict(
            {"audio": np.zeros((1, 1, 3200), dtype=np.float32)},
            state=sem_state,
        )
        sem_state = sem_enc.make_state()  # reset after test

        # Use MLX semantic connector (correct hidden_size for both 1.5B and 7B)
        def semantic_fn(audio_chunk):
            audio_input = np.zeros((1, 1, 3200), dtype=np.float32)
            audio_input[0, 0, :min(len(audio_chunk), 3200)] = audio_chunk[:3200]
            features = sem_enc.predict(
                {"audio": audio_input}, state=sem_state
            )["features"]
            # features: (1, 128, 1) -> (1, 1, 128) for MLX connector
            feat = mx.array(features.transpose(0, 2, 1)).astype(mx.float16)
            embedding = model.semantic_connector(feat)
            mx.eval(embedding)
            return np.array(embedding)

        def reset_fn():
            nonlocal sem_state
            sem_state = sem_enc.make_state()

        logger.info("Semantic encoder: CoreML + MLX connector")
        return semantic_fn, reset_fn
    except Exception as e:
        logger.debug("  CoreML semantic encoder failed: %s", e)
        return None


def _try_mlx_semantic(model, config, model_id):
    """Load pure MLX semantic encoder from HF weights."""
    try:
        from .semantic_encoder import load_semantic_encoder, FRAME_SAMPLES
        from .load_weights import resolve_model_path, _load_safetensors

        logger.info("Loading semantic encoder weights...")
        model_path = resolve_model_path(model_id)
        raw = _load_safetensors(model_path)

        sem_keys = {k: v for k, v in raw.items()
                    if k.startswith("model.semantic_tokenizer.encoder.")
                    or k.startswith("semantic_encoder.")}
        if not sem_keys:
            logger.info("  No semantic encoder weights found")
            return None

        sem_enc = load_semantic_encoder(raw)

        test_audio = mx.zeros((1, 1, FRAME_SAMPLES), dtype=mx.float32)
        test_out = sem_enc(test_audio)
        mx.eval(test_out)
        sem_enc.reset_caches()

        def semantic_fn(audio_chunk):
            audio_np = np.zeros((1, 1, FRAME_SAMPLES), dtype=np.float32)
            audio_np[0, 0, :min(len(audio_chunk), FRAME_SAMPLES)] = audio_chunk[:FRAME_SAMPLES]
            features = sem_enc(mx.array(audio_np))
            mx.eval(features)
            feat = features.transpose(0, 2, 1).astype(mx.float16)
            embedding = model.semantic_connector(feat)
            mx.eval(embedding)
            return np.array(embedding)

        def reset_fn():
            sem_enc.reset_caches()

        logger.info("Semantic encoder: MLX (%d cache buffers)", len(sem_enc.caches))
        return semantic_fn, reset_fn
    except Exception as e:
        logger.warning("Could not load semantic encoder: %s", e)
        return None


def _detect_tokenizer(model_id: str, config) -> str:
    """Detect the appropriate tokenizer for the model."""
    if config.vocab_size <= 151936:
        return "Qwen/Qwen2.5-1.5B"
    return "Qwen/Qwen2.5-7B"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="VibeVoice MLX Text-to-Speech")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--model", type=str, default="gafiatulin/vibevoice-1.5b-mlx",
                        help="Model ID or local path")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--ref-audio", type=str, nargs="+", default=None,
                        help="Reference audio file(s) for voice cloning (one per speaker)")
    parser.add_argument("--voice", type=str, nargs="+", default=None,
                        help="Pre-encoded voice file(s) (.safetensors) for voice cloning")
    parser.add_argument("--save-voice", type=str, default=None,
                        help="Save encoded voice embeddings to .safetensors (use with --ref-audio)")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None,
                        help="Quantization for LLM backbone (4=INT4, 8=INT8)")
    parser.add_argument("--quantize-diffusion", action="store_true",
                        help="Also INT8 quantize the diffusion head (faster, slight quality loss)")
    parser.add_argument("--solver", type=str, default="dpm", choices=["dpm", "sde", "ddpm"],
                        help="Diffusion solver (sde=stochastic DPM-Solver++, dpm=ODE DPM-Solver++)")
    parser.add_argument("--diffusion-steps", type=int, default=10,
                        help="Number of diffusion steps")
    parser.add_argument("--cfg-scale", type=float, default=1.3,
                        help="Classifier-free guidance scale")
    parser.add_argument("--max-speech-tokens", type=int, default=200,
                        help="Maximum speech tokens to generate")
    parser.add_argument("--silence-detection", action="store_true",
                        help="Detect end of speech via latent energy (for models that don't stop naturally)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip semantic feedback (faster, lower quality)")
    parser.add_argument("--coreml-semantic", action="store_true",
                        help="Use CoreML semantic encoder (faster, downloads ~657MB on first use)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name (auto-detected if not specified)")
    args = parser.parse_args()

    # Encode-only mode: --ref-audio + --save-voice without --text
    if args.save_voice and args.ref_audio and not args.text:
        model, config = load_model(args.model, quantize_bits=None)
        for i, audio_path in enumerate(args.ref_audio):
            wav = _load_and_resample(audio_path)
            if len(wav) > VOICE_CLONE_SAMPLES:
                wav = wav[:VOICE_CLONE_SAMPLES]
            num_vae_tokens = math.ceil(len(wav) / SPEECH_TOK_COMPRESS_RATIO)
            embeds = encode_voice_reference(wav, num_vae_tokens, model, config, args.model)
            save_path = args.save_voice
            if len(args.ref_audio) > 1:
                base, ext = os.path.splitext(save_path)
                save_path = f"{base}_spk{i}{ext}"
            save_voice(save_path, embeds)
        return

    if not args.text:
        parser.error("--text is required for synthesis (or use --ref-audio with --save-voice for voice encoding)")

    # Load model
    model, config = load_model(args.model, quantize_bits=args.quantize)

    # Optionally quantize diffusion head for faster inference
    if args.quantize_diffusion:
        import mlx.nn as nn_mod
        print("Quantizing diffusion head (INT8)...")
        nn_mod.quantize(
            model.diffusion_head, bits=8, group_size=64,
            class_predicate=lambda _, m: (
                isinstance(m, nn_mod.Linear)
                and m.weight.shape[0] % 64 == 0
                and m.weight.shape[1] % 64 == 0
            ),
        )
        mx.eval(model.diffusion_head.parameters())

    # Tokenize (with optional voice cloning)
    tokenizer_name = args.tokenizer or _detect_tokenizer(args.model, config)
    print(f"Using tokenizer: {tokenizer_name}")

    text = args.text.replace("\\n", "\n")

    # Determine voice source: --voice (pre-encoded) or --ref-audio (encode now)
    voice_files = args.voice or args.ref_audio
    result = tokenize_text(text, tokenizer_name, config, ref_audio=voice_files)

    # Handle voice cloning vs simple text
    voice_embeds = None
    if isinstance(result, VoiceCloneData):
        input_ids = result.input_ids
        print(f"Input: {len(input_ids)} tokens ({len(result.speakers)} speaker(s) for voice cloning)")

        # Encode voice references
        voice_embeds = {}
        for spk in result.speakers:
            if spk.cached_embeds is not None:
                pass  # already have embeddings (e.g. from speaker_embeds)
            else:
                voice_path = voice_files[spk.speaker_id] if voice_files else None
                is_preencoded = voice_path and voice_path.endswith(".safetensors")

                if is_preencoded:
                    print(f"  Loading pre-encoded voice for speaker {spk.speaker_id}...")
                    spk.cached_embeds = load_voice(voice_path)[:spk.num_vae_tokens]
                else:
                    print(f"  Encoding speaker {spk.speaker_id} ({spk.num_vae_tokens} tokens, "
                          f"{len(spk.ref_audio_np)/SAMPLE_RATE:.1f}s audio)...")
                    spk.cached_embeds = encode_voice_reference(
                        spk.ref_audio_np, spk.num_vae_tokens, model, config, args.model)

                    # Save if requested
                    if args.save_voice:
                        save_path = args.save_voice
                        if len(result.speakers) > 1:
                            base, ext = os.path.splitext(save_path)
                            save_path = f"{base}_spk{spk.speaker_id}{ext}"
                        save_voice(save_path, spk.cached_embeds)

            # Map position -> embedding
            embeds_mx = mx.array(spk.cached_embeds).astype(mx.float16)
            for i, pos in enumerate(spk.speech_embed_positions):
                if i < embeds_mx.shape[0]:
                    voice_embeds[pos] = embeds_mx[i:i+1]
    else:
        input_ids = result
        print(f"Input: {len(input_ids)} tokens")

    # Semantic encoder callback
    semantic_fn = None
    semantic_reset = None
    if not args.no_semantic:
        result = _load_semantic_encoder(model, config, args.model,
                                         use_coreml=args.coreml_semantic)
        if result is not None:
            semantic_fn, semantic_reset = result
        else:
            print("Semantic encoder: disabled (not available)")

    # Generate
    opts = GenerationOptions(
        solver=args.solver,
        diffusion_steps=args.diffusion_steps,
        cfg_scale=args.cfg_scale,
        max_speech_tokens=args.max_speech_tokens,
        silence_detection=args.silence_detection,
        seed=args.seed,
    )

    print("Generating...")
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()

    audio, metrics = generate(
        model=model,
        input_ids=input_ids,
        opts=opts,
        semantic_encoder_fn=semantic_fn,
        semantic_reset_fn=semantic_reset,
        voice_embeds=voice_embeds,
    )

    gen_time = (time.perf_counter() - t0) * 1000
    summary = metrics.summary()
    print(f"  Generated {metrics.num_speech_tokens} speech tokens in {gen_time:.0f}ms")
    print(f"  Audio: {summary['audio_seconds']:.2f}s ({metrics.audio_samples} samples)")

    if "diffusion_total_ms" in summary:
        print(f"  Diffusion: {summary['diffusion_total_ms']:.0f}ms total "
              f"({summary.get('diffusion_mean_ms', 0):.1f}ms/step)")
    if "vae_total_ms" in summary:
        print(f"  VAE: {summary['vae_total_ms']:.0f}ms total")
    if "lm_step_total_ms" in summary:
        print(f"  LM steps: {summary['lm_step_total_ms']:.0f}ms total "
              f"({summary.get('lm_step_mean_ms', 0):.1f}ms/step)")
    if "gen_rtf" in summary:
        print(f"  RTF: {summary['gen_rtf']:.2f}x realtime")
    peak_mem = mx.metal.get_peak_memory() / 1e9
    print(f"  Peak memory: {peak_mem:.2f} GB")

    # Save
    if len(audio) > 0:
        import soundfile as sf
        sf.write(args.output, audio, SAMPLE_RATE)
        print(f"Saved to {args.output}")
    else:
        print("Warning: No audio generated")


if __name__ == "__main__":
    main()
