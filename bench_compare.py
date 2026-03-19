"""Benchmark VibeVoice MLX across quantization and semantic encoder configurations.

Each config runs in a subprocess for clean GPU state.

Usage:
    uv run python bench.py --model gafiatulin/vibevoice-1.5b-mlx
    uv run python bench.py --model gafiatulin/vibevoice-1.5b-mlx --ref-audio speaker.wav
    uv run python bench.py --model gafiatulin/vibevoice-7b-mlx --ref-audio speaker.wav --save-audio bench_outputs
"""

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

# ~30s of speech
DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "She sells seashells by the seashore while the waves crash against the rocks. "
    "A journey of a thousand miles begins with a single step, they always say. "
    "The old lighthouse keeper watched the storm approach from the distant horizon. "
    "Every morning, the birds would sing their beautiful melodies in the garden. "
    "He carefully arranged the ancient books on the mahogany shelf in the library."
)

CONFIGS = [
    # (label, {quantize, no_semantic, coreml_semantic})
    ("fp16",                     {}),
    ("fp16, no-semantic",        {"no_semantic": True}),
    ("fp16, coreml-semantic",    {"coreml_semantic": True}),
    ("int8",                     {"quantize": 8}),
    ("int8, no-semantic",        {"quantize": 8, "no_semantic": True}),
    ("int8, coreml-semantic",    {"quantize": 8, "coreml_semantic": True}),
    ("int4",                     {"quantize": 4}),
    ("int4, no-semantic",        {"quantize": 4, "no_semantic": True}),
    ("int4, coreml-semantic",    {"quantize": 4, "coreml_semantic": True}),
]


def make_script(model, voice_arg, text, seed, max_tokens, audio_out,
                quantize=None, no_semantic=False, coreml_semantic=False):
    """Build inline Python script for subprocess execution."""
    q = quantize if quantize else "None"
    sem_mode = '"none"' if no_semantic else ('"coreml"' if coreml_semantic else '"mlx"')
    audio_out_repr = repr(str(audio_out)) if audio_out else "None"
    voice_repr = repr(str(voice_arg)) if voice_arg else "None"

    return textwrap.dedent(f"""\
import json, time, sys, os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
sys.path.insert(0, "run")
import mlx.core as mx
from load_weights import load_model
from generate import generate, GenerationOptions
from e2e_pipeline import (tokenize_text, VoiceCloneData, SAMPLE_RATE,
                          _detect_tokenizer, load_voice, encode_voice_reference)

model, config = load_model("{model}", quantize_bits={q})

sem_mode = {sem_mode}
semantic_fn = None
semantic_reset = None

if sem_mode == "coreml":
    from e2e_pipeline import _try_coreml_semantic
    r = _try_coreml_semantic(model, config)
    if r is not None:
        semantic_fn, semantic_reset = r
    else:
        sem_mode = "mlx"

if sem_mode == "mlx":
    from e2e_pipeline import _try_mlx_semantic
    r = _try_mlx_semantic(model, config, "{model}")
    if r is not None:
        semantic_fn, semantic_reset = r

tokenizer_name = _detect_tokenizer("{model}", config)
voice_arg = {voice_repr}
voice_list = [voice_arg] if voice_arg else None

text = {repr(text)}
result = tokenize_text(text, tokenizer_name, config, ref_audio=voice_list)

voice_embeds = None
if isinstance(result, VoiceCloneData):
    input_ids = result.input_ids
    voice_embeds = {{}}
    for spk in result.speakers:
        if voice_arg and voice_arg.endswith(".safetensors"):
            spk.cached_embeds = load_voice(voice_arg)[:spk.num_vae_tokens]
        else:
            spk.cached_embeds = encode_voice_reference(
                spk.ref_audio_np, spk.num_vae_tokens, model, config, "{model}")
        embeds_mx = mx.array(spk.cached_embeds).astype(mx.float16)
        for i, pos in enumerate(spk.speech_embed_positions):
            if i < embeds_mx.shape[0]:
                voice_embeds[pos] = embeds_mx[i:i+1]
else:
    input_ids = result

opts = GenerationOptions(
    solver="dpm",
    diffusion_steps=10,
    cfg_scale=1.3,
    max_speech_tokens={max_tokens},
    seed={seed},
)

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
gen_s = time.perf_counter() - t0

summary = metrics.summary()
peak = mx.metal.get_peak_memory() / 1e9
audio_s = summary.get("audio_seconds", 0)
rtf = audio_s / gen_s if gen_s > 0 else 0

audio_out = {audio_out_repr}
if audio_out and len(audio) > 0:
    import soundfile as sf
    sf.write(audio_out, audio, SAMPLE_RATE)

print("BENCH_RESULT:" + json.dumps({{
    "gen_s": gen_s,
    "audio_s": audio_s,
    "rtf": rtf,
    "peak_mem_gb": peak,
    "speech_tokens": metrics.num_speech_tokens,
}}))
""")


def run_config(label, cfg, args, audio_dir=None):
    audio_out = None
    if audio_dir:
        safe = label.replace(", ", "_").replace(" ", "_")
        audio_out = Path(audio_dir) / f"{safe}.wav"

    script = make_script(
        model=args.model,
        voice_arg=args.voice_arg,
        text=args.text,
        seed=args.seed,
        max_tokens=args.max_tokens,
        audio_out=audio_out,
        **cfg,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=600,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("BENCH_RESULT:"):
                return json.loads(line[len("BENCH_RESULT:"):])
        print(f"FAILED")
        stderr = (result.stdout + result.stderr)[-500:]
        if stderr.strip():
            for l in stderr.strip().split("\n")[-3:]:
                print(f"    {l}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
    return None


def pre_encode_voice(model_path, ref_audio, save_path):
    """Pre-encode voice in a subprocess, return path to saved embeddings."""
    script = textwrap.dedent(f"""\
import sys, os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
sys.path.insert(0, "run")
from load_weights import load_model
from e2e_pipeline import encode_voice_reference, save_voice, SAMPLE_RATE, VOICE_CLONE_SAMPLES, SPEECH_TOK_COMPRESS_RATIO, _load_and_resample
import math

model, config = load_model("{model_path}", quantize_bits=None)
wav = _load_and_resample("{ref_audio}")
if len(wav) > VOICE_CLONE_SAMPLES:
    wav = wav[:VOICE_CLONE_SAMPLES]
num_vae_tokens = math.ceil(len(wav) / SPEECH_TOK_COMPRESS_RATIO)
embeds = encode_voice_reference(wav, num_vae_tokens, model, config, "{model_path}")
save_voice("{save_path}", embeds)
print("DONE")
""")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    if "DONE" in result.stdout:
        return save_path
    print(f"Voice encoding failed:")
    print((result.stdout + result.stderr)[-500:])
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark VibeVoice MLX configurations")
    parser.add_argument("--model", default="gafiatulin/vibevoice-1.5b-mlx", help="Model ID")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=400, help="Max speech tokens (~30s)")
    parser.add_argument("--save-audio", default=None, help="Directory to save audio files")
    args = parser.parse_args()

    # Pre-download model in parent process
    sys.path.insert(0, "run")
    from load_weights import resolve_model_path
    resolved = str(resolve_model_path(args.model))
    args.model = resolved

    tag = "7b" if "vibevoice-7b" in resolved.lower() else "1.5b"
    vc = " + voice clone" if args.ref_audio else ""

    print(f"Model: {tag.upper()}{vc}")
    print(f"Text: {args.text[:60]}...")

    audio_dir = None
    if args.save_audio:
        audio_dir = Path(args.save_audio) / tag
        audio_dir.mkdir(parents=True, exist_ok=True)

    # Pre-encode voice once if ref-audio provided
    args.voice_arg = None
    if args.ref_audio:
        voice_dir = Path(args.save_audio or "bench_outputs") / tag
        voice_dir.mkdir(parents=True, exist_ok=True)
        voice_path = str(voice_dir / "voice.safetensors")
        print(f"Pre-encoding voice from {args.ref_audio}...")
        result = pre_encode_voice(args.model, args.ref_audio, voice_path)
        if result:
            args.voice_arg = voice_path
            print(f"Voice saved to {voice_path}")
        else:
            print("WARNING: Voice encoding failed, running without voice cloning")

    print()
    header = f"{'Config':<30} {'RTF':>6} {'Gen':>8} {'Audio':>7} {'Mem':>8}"
    print(header)
    print("-" * len(header))

    all_results = []
    for label, cfg in CONFIGS:
        print(f"  {label:<28}", end=" ", flush=True)
        r = run_config(label, cfg, args, audio_dir=audio_dir)
        if r:
            all_results.append((label, r))
            print(f"{r['rtf']:>5.2f}x {r['gen_s']*1000:>7.0f}ms {r['audio_s']:>6.1f}s {r['peak_mem_gb']:>6.1f} GB")
        else:
            pass  # error already printed

    # Summary table
    print(f"\n{'='*62}")
    print(f"{'Config':<30} {'RTF':>6} {'Gen':>8} {'Audio':>7} {'Mem':>8}")
    print(f"{'-'*30} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")
    for label, r in all_results:
        print(f"{label:<30} {r['rtf']:>5.2f}x {r['gen_s']*1000:>7.0f}ms {r['audio_s']:>6.1f}s {r['peak_mem_gb']:>6.1f} GB")
    print(f"{'='*62}")

    if args.save_audio:
        results_file = Path(args.save_audio) / tag / "results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
