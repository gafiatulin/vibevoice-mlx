"""VibeVoice generation pipeline — autoregressive TTS with DPM-Solver++ diffusion."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .model import KVCache, VibeVoiceConfig, VibeVoiceModel, compute_rope
from .streaming_vae import StreamingVAEDecoder
from .fast_forward import FastLM, FastDiffusionHead


# ---------------------------------------------------------------------------
# DPM-Solver++ schedule (precomputed in float64)
# ---------------------------------------------------------------------------

DDPM_STEPS = 1000
VAE_DIM = 64

_AC64 = np.cos((np.arange(DDPM_STEPS + 1, dtype=np.float64) / DDPM_STEPS + 0.008) / 1.008 * np.pi / 2) ** 2
_AC64 = (_AC64 / _AC64[0])[:DDPM_STEPS]
_ALPHA_NP = np.sqrt(_AC64)
_SIGMA_NP = np.sqrt(1.0 - _AC64)
_LAMBDA_NP = np.log(_ALPHA_NP / np.maximum(_SIGMA_NP, 1e-10))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class GenerationOptions:
    solver: str = "dpm"            # "ddpm", "dpm", or "sde"
    diffusion_steps: int = 10      # DPM-Solver++ steps
    cfg_scale: float = 1.3         # Classifier-free guidance
    max_speech_tokens: int = 200   # Safety limit
    silence_detection: bool = False  # Stop on sustained silence + trim trailing
    seed: int = 42


@dataclass
class GenerationMetrics:
    name: str = ""
    timings: dict = field(default_factory=dict)
    total_time: float = 0.0
    num_speech_tokens: int = 0
    num_text_tokens: int = 0
    audio_samples: int = 0

    def record(self, component: str, ms: float):
        if component not in self.timings:
            self.timings[component] = []
        self.timings[component].append(ms)

    def summary(self) -> dict:
        result = {"name": self.name}
        for k, v in self.timings.items():
            result[f"{k}_total_ms"] = sum(v)
            result[f"{k}_mean_ms"] = sum(v) / len(v) if v else 0
            result[f"{k}_count"] = len(v)
        result["total_ms"] = self.total_time
        result["speech_tokens"] = self.num_speech_tokens
        result["text_tokens"] = self.num_text_tokens
        result["audio_samples"] = self.audio_samples
        result["audio_seconds"] = self.audio_samples / 24000
        if self.audio_samples > 0 and self.total_time > 0:
            audio_ms = self.audio_samples / 24000 * 1000
            result["rtf"] = audio_ms / self.total_time
        load_ms = result.get("load_total_ms", 0)
        gen_ms = self.total_time - load_ms
        result["gen_ms"] = gen_ms
        if self.audio_samples > 0 and gen_ms > 0:
            audio_ms = self.audio_samples / 24000 * 1000
            result["gen_rtf"] = audio_ms / gen_ms
        return result


# ---------------------------------------------------------------------------
# DPM-Solver++ 2M — ODE and SDE variants (all MLX, batched CFG)
# ---------------------------------------------------------------------------

def _dpm_denoise_step(diff_head, sample, batched_cond, s, cfg_scale, dtype):
    """Run diffusion head with batched CFG, return x0 prediction.

    No mx.eval — relies on MLX lazy evaluation to batch the entire
    diffusion solve into fewer GPU submissions.
    """
    batched_sample = mx.concatenate([sample, sample], axis=0).astype(dtype)
    ts_mx = mx.array([float(s)]).astype(dtype)
    v_batched = diff_head(batched_sample, ts_mx, batched_cond)

    v_cond = v_batched[0:1].astype(mx.float32)
    v_uncond = v_batched[1:2].astype(mx.float32)
    v = v_uncond + cfg_scale * (v_cond - v_uncond)

    alpha_s = float(_ALPHA_NP[s])
    sigma_s = float(_SIGMA_NP[s])
    return alpha_s * sample - sigma_s * v


def dpm_solver_2m(
    diff_head,
    condition: mx.array,
    neg_condition: mx.array,
    cfg_scale: float,
    num_steps: int = 10,
    seed: int = 0,
    dtype=mx.float16,
) -> mx.array:
    """ODE DPM-Solver++ 2M with batched CFG.

    Returns sample of shape (1, VAE_DIM) in float32.
    """
    t_schedule = np.round(np.linspace(DDPM_STEPS - 1, 0, num_steps + 1)).astype(np.int64)

    key = mx.random.key(seed)
    sample = mx.random.normal(shape=(1, VAE_DIM), key=key).astype(mx.float32)

    batched_cond = mx.concatenate([
        condition.astype(dtype), neg_condition.astype(dtype)
    ], axis=0)

    x0_list = []

    for i in range(num_steps):
        s = int(t_schedule[i])
        t = int(t_schedule[i + 1])

        x0 = _dpm_denoise_step(diff_head, sample, batched_cond, s, cfg_scale, dtype)
        x0_list.append(x0)

        sigma_s = float(_SIGMA_NP[s])
        lam_s = float(_LAMBDA_NP[s])
        lam_t = float(_LAMBDA_NP[max(t, 0)])
        h = lam_t - lam_s

        is_last = (i == num_steps - 1)
        is_second_last = (i == num_steps - 2)
        lower_order_final = is_last and num_steps < 15
        lower_order_second = is_second_last and num_steps < 15
        use_first_order = len(x0_list) < 2 or lower_order_final or lower_order_second

        if use_first_order:
            D = x0_list[-1]
        else:
            s_prev = int(t_schedule[i - 1])
            lam_s_prev = float(_LAMBDA_NP[s_prev])
            h_prev = lam_s - lam_s_prev
            r = h_prev / h
            D = x0_list[-1] + 0.5 / r * (x0_list[-1] - x0_list[-2])

        sigma_t = float(_SIGMA_NP[t])
        alpha_t = float(_ALPHA_NP[t])
        sample = (sigma_t / sigma_s) * sample - alpha_t * float(np.expm1(-h)) * D

    return sample


def dpm_solver_sde_2m(
    diff_head,
    condition: mx.array,
    neg_condition: mx.array,
    cfg_scale: float,
    num_steps: int = 20,
    seed: int = 0,
    dtype=mx.float16,
) -> mx.array:
    """SDE DPM-Solver++ 2M (stochastic, midpoint) with batched CFG.

    Matches the sde-dpmsolver++ algorithm from HuggingFace Diffusers:
    - final_sigmas_type="zero": last step targets sigma=0 (perfect denoising)
    - lower_order_final: last step uses first-order update
    - Noise injected at every step (coefficient=0 at last step due to sigma_t=0)

    Returns sample of shape (1, VAE_DIM) in float32.
    """
    # Timestep schedule: N timesteps from 999→~50 (matching diffusers linspace)
    timesteps = np.round(
        np.linspace(0, DDPM_STEPS - 1, num_steps + 1)
    ).astype(np.int64)[::-1][:-1]  # (num_steps,) from high to low

    # Build sigmas array with final sigma=0 (diffusers final_sigmas_type="zero")
    all_sigmas = ((1.0 - _AC64) / _AC64) ** 0.5  # ratio-form sigmas
    sigmas = np.interp(timesteps, np.arange(len(all_sigmas)), all_sigmas)
    sigmas = np.append(sigmas, 0.0)  # (num_steps + 1,) — last entry is 0

    def _sig_to_alpha_sigma(sig):
        a = 1.0 / np.sqrt(sig ** 2 + 1.0)
        s = sig * a
        return float(a), float(s)

    key = mx.random.key(seed)
    # Pre-generate all noise vectors upfront (avoids per-step random split overhead)
    noise_keys = mx.random.split(key, num_steps + 1)
    sample = mx.random.normal(shape=(1, VAE_DIM), key=noise_keys[0]).astype(mx.float32)
    noise_vecs = mx.random.normal(shape=(num_steps, VAE_DIM), key=noise_keys[1])
    mx.eval(sample, noise_vecs)

    # Precompute schedule values (all in numpy, no per-step recomputation)
    alphas = np.array([_sig_to_alpha_sigma(s)[0] for s in sigmas])
    sigma_vals = np.array([_sig_to_alpha_sigma(s)[1] for s in sigmas])
    lambdas = np.log(np.maximum(alphas, 1e-10)) - np.log(np.maximum(sigma_vals, 1e-10))
    lambdas[-1] = np.inf  # sigma=0 at last position

    batched_cond = mx.concatenate([
        condition.astype(dtype), neg_condition.astype(dtype)
    ], axis=0)

    x0_list = []

    for i in range(num_steps):
        s_ts = int(timesteps[i])

        x0 = _dpm_denoise_step(diff_head, sample, batched_cond, s_ts, cfg_scale, dtype)
        x0_list.append(x0)

        h = float(lambdas[i + 1] - lambdas[i])
        is_last = (i == num_steps - 1)
        use_first_order = len(x0_list) < 2 or is_last

        if use_first_order:
            D = x0_list[-1]
        else:
            h_prev = float(lambdas[i] - lambdas[i - 1])
            r = h_prev / h
            D = x0_list[-1] + 0.5 / r * (x0_list[-1] - x0_list[-2])

        # SDE update
        if np.isinf(h):
            sample = D
        else:
            exp_neg_h = float(np.exp(-h))
            exp_neg_2h = float(np.exp(-2.0 * h))
            s_t = float(sigma_vals[i + 1])
            s_s = float(sigma_vals[i])
            a_t = float(alphas[i + 1])
            sample = (s_t / s_s * exp_neg_h) * sample + a_t * (1.0 - exp_neg_2h) * D
            noise = noise_vecs[i:i + 1].astype(mx.float32)
            sample = sample + s_t * float(np.sqrt(max(0.0, 1.0 - exp_neg_2h))) * noise

    return sample


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(
    model: VibeVoiceModel,
    input_ids: list[int],
    opts: GenerationOptions | None = None,
    semantic_encoder_fn: Optional[Callable] = None,
    semantic_reset_fn: Optional[Callable] = None,
    voice_embeds: Optional[dict[int, mx.array]] = None,
) -> tuple[np.ndarray, GenerationMetrics]:
    """Full autoregressive TTS generation.

    Args:
        model: Loaded VibeVoiceModel
        input_ids: Tokenized input sequence (includes special tokens)
        opts: Generation options
        semantic_encoder_fn: Optional callback for semantic encoder.
            Signature: fn(audio_chunk: np.ndarray) -> np.ndarray of shape (1, 1, hidden_size)
        voice_embeds: Optional dict mapping position -> embedding for voice cloning.
            Each value is an mx.array of shape (1, hidden_size).

    Returns:
        (audio_array, metrics)
    """
    if opts is None:
        opts = GenerationOptions()

    config = model.config
    dtype = mx.float16
    metrics = GenerationMetrics(
        name=f"MLX ({opts.solver}-{opts.diffusion_steps}s)"
    )

    t0_total = time.perf_counter()

    # Build fast-path LM and diffusion head (raw matmul, no nn.Module dispatch)
    fast_lm = FastLM(model, config)
    fast_diff = FastDiffusionHead(model, config)
    embed_table = fast_lm.embed_w
    NL = config.num_hidden_layers

    use_evolving_cfg = opts.cfg_scale > 1.0

    # Initialize negative (unconditional) branch for CFG.
    # Only allocate when CFG is active to save memory.
    if use_evolving_cfg:
        neg_embed = embed_table[config.speech_start_id].reshape(1, 1, config.hidden_size)
        neg_pos = mx.arange(1, dtype=mx.float32)
        neg_cos, neg_sin = compute_rope(neg_pos, config.head_dim, config.rope_theta)
        neg_k_cache = [mx.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=dtype)] * NL
        neg_v_cache = [mx.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=dtype)] * NL
        neg_hidden = fast_lm.forward(neg_embed, neg_cos, neg_sin, neg_k_cache, neg_v_cache)
        mx.eval(neg_hidden, *neg_k_cache, *neg_v_cache)
        neg_condition = neg_hidden[:, 0:1, :].reshape(1, config.hidden_size)
        neg_position = 1
    else:
        # Static zero condition for CFG=1.0 (no guidance)
        neg_condition = mx.zeros((1, config.hidden_size), dtype=dtype)
        neg_k_cache = neg_v_cache = None
        neg_position = 0

    # Prefill (use fast_lm.prefill for batched tokens)
    t0 = time.perf_counter()
    n_prefill = len(input_ids)

    if voice_embeds:
        embeds_list = []
        for pos, tok_id in enumerate(input_ids):
            if pos in voice_embeds:
                embeds_list.append(voice_embeds[pos].reshape(1, config.hidden_size))
            else:
                embeds_list.append(embed_table[tok_id].reshape(1, config.hidden_size))
        prefill_embeds = mx.stack(embeds_list, axis=0).reshape(1, n_prefill, config.hidden_size)
    else:
        ids_mx = mx.array(input_ids)
        prefill_embeds = embed_table[ids_mx].reshape(1, n_prefill, config.hidden_size)

    positions = mx.arange(n_prefill, dtype=mx.float32)
    cos_prefill, sin_prefill = compute_rope(positions, config.head_dim, config.rope_theta)
    causal_mask = mx.triu(mx.full((n_prefill, n_prefill), float("-inf"), dtype=dtype), k=1)

    k_cache = [None] * NL
    v_cache = [None] * NL
    hidden = fast_lm.prefill(prefill_embeds, cos_prefill, sin_prefill, causal_mask, k_cache, v_cache)
    mx.eval(hidden, *[k for k in k_cache], *[v for v in v_cache])
    hidden = hidden[:, -1:, :]

    metrics.record("prefill", (time.perf_counter() - t0) * 1000)
    metrics.num_text_tokens = n_prefill

    # Single-segment models (KugelAudio): constrain to speech-structural tokens
    # only to prevent text token hallucination and looping.
    # Multi-segment models (VibeVoice): allow all tokens for speaker turns.
    if config.single_segment:
        allowed_tokens = mx.array([
            config.speech_start_id, config.speech_end_id,
            config.speech_diffusion_id, config.eos_id,
        ])
        logit_mask = mx.full((1, 1, config.vocab_size), float("-inf"), dtype=dtype)
        logit_mask[0, 0, allowed_tokens] = 0.0
    else:
        logit_mask = None

    # First token
    logits = fast_lm.logits(hidden)
    if logit_mask is not None:
        logits = logits + logit_mask
    mx.eval(logits)
    next_token = int(mx.argmax(logits[0, 0]).item())

    # Always batch-decode at end for clean audio.
    # For semantic feedback, use cheap non-streaming per-token VAE
    # (quality doesn't matter — just needs approximate audio for features).

    # Autoregressive generation
    audio_chunks = []
    all_latents = []
    silent_run = 0
    rng = np.random.RandomState(opts.seed)
    position = n_prefill

    # Single-segment models stop on speech_end; multi-segment only on eos
    stop_tokens = {config.eos_id}
    if config.single_segment:
        stop_tokens.add(config.speech_end_id)

    for step in range(opts.max_speech_tokens * 3):
        if next_token in stop_tokens:
            break
        if metrics.num_speech_tokens >= opts.max_speech_tokens:
            break

        if next_token == config.speech_diffusion_id:
            metrics.num_speech_tokens += 1

            # Diffusion (fast path — no nn.Module dispatch)
            t0 = time.perf_counter()
            condition = hidden[:, 0:1, :].reshape(1, config.hidden_size)
            solver_fn = dpm_solver_sde_2m if opts.solver == "sde" else dpm_solver_2m
            sample = solver_fn(
                fast_diff, condition, neg_condition, opts.cfg_scale,
                num_steps=opts.diffusion_steps,
                seed=rng.randint(0, 2**31),
                dtype=dtype,
            )
            metrics.record("diffusion", (time.perf_counter() - t0) * 1000)

            # Scale and accumulate latent for batch re-decode
            latent = (sample / config.speech_scaling_factor - config.speech_bias_factor)
            all_latents.append(latent)

            # Per-token VAE for semantic feedback (diffusion already eval'd, so this is cheap)
            if semantic_encoder_fn is not None:
                t0 = time.perf_counter()
                latent_frame = latent[:, :, None].astype(dtype)
                audio = model.vae_decoder(latent_frame)
                mx.eval(audio)
                audio_chunks.append(np.array(audio).squeeze().astype(np.float32))
                metrics.record("vae", (time.perf_counter() - t0) * 1000)

            # Connectors: acoustic + optional semantic feedback
            t0 = time.perf_counter()
            acoustic_embed = model.acoustic_connector(sample[:, None, :].astype(dtype))

            if semantic_encoder_fn is not None and audio_chunks:
                chunk = audio_chunks[-1][:3200].astype(np.float32)
                sem_embed_np = semantic_encoder_fn(chunk)
                sem_embed = mx.array(sem_embed_np).astype(dtype)
                if sem_embed.ndim == 3:
                    next_embed = acoustic_embed + sem_embed
                else:
                    next_embed = acoustic_embed + sem_embed.reshape(1, 1, config.hidden_size)
            else:
                next_embed = acoustic_embed
            metrics.record("connector", (time.perf_counter() - t0) * 1000)
        else:
            if next_token == config.speech_end_id and semantic_reset_fn is not None:
                semantic_reset_fn()
            next_embed = embed_table[next_token].reshape(1, 1, config.hidden_size)

        # LM step (+ neg branch if evolving CFG)
        t0 = time.perf_counter()
        pos = mx.array([float(position)], dtype=mx.float32)
        cos, sin = compute_rope(pos, config.head_dim, config.rope_theta)

        if use_evolving_cfg:
            # Batched: read weights once for both main+neg passes
            neg_pos = mx.array([float(neg_position)], dtype=mx.float32)
            neg_cos, neg_sin = compute_rope(neg_pos, config.head_dim, config.rope_theta)
            hidden, neg_hidden = fast_lm.forward_dual(
                next_embed, cos, sin, k_cache, v_cache,
                next_embed, neg_cos, neg_sin, neg_k_cache, neg_v_cache,
            )
            neg_condition = neg_hidden[:, 0:1, :].reshape(1, config.hidden_size)
            neg_position += 1
        else:
            hidden = fast_lm.forward(next_embed, cos, sin, k_cache, v_cache)

        logits = fast_lm.logits(hidden)
        if logit_mask is not None:
            logits = logits + logit_mask
        # Silence-aware stop: boost speech_end when generating silence
        if opts.silence_detection and config.single_segment and all_latents:
            lat_rms = float(mx.sqrt(mx.mean(all_latents[-1] ** 2)))
            if lat_rms < 1.0:
                silent_run += 1
            else:
                silent_run = 0
            if silent_run >= 3:
                boost = min((silent_run - 2) * 5.0, 20.0)
                logits[0, 0, config.speech_end_id] += boost
                logits[0, 0, config.eos_id] += boost
        mx.eval(logits)
        next_token = int(mx.argmax(logits[0, 0]).item())
        metrics.record("lm_step", (time.perf_counter() - t0) * 1000)
        position += 1

    # Batch re-decode all latents for temporally continuous audio
    if all_latents:
        t0 = time.perf_counter()
        full_latent = mx.concatenate(all_latents, axis=0).T[None, :, :].astype(dtype)
        full_audio = model.vae_decoder(full_latent)
        mx.eval(full_audio)
        audio_out = np.array(full_audio).squeeze().astype(np.float32)
        if opts.silence_detection:
            audio_out = _trim_trailing_silence(audio_out)
        metrics.record("vae_final", (time.perf_counter() - t0) * 1000)
    else:
        audio_out = np.zeros(0, dtype=np.float32)

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.audio_samples = len(audio_out)

    # Free MLX metal buffer pool so repeated calls don't grow unbounded.
    mx.clear_cache()

    return audio_out, metrics


def _trim_trailing_silence(audio: np.ndarray, sr: int = 24000,
                           threshold: float = 0.05,
                           long_silence_ms: int = 1500,
                           pad_ms: int = 150) -> np.ndarray:
    """Trim audio after speech ends.

    Two-pass approach:
    1. Forward scan: if a long silence gap (>= long_silence_ms) follows
       speech, cut there — this catches model repetition after a pause.
    2. Backward scan: trim trailing silence/noise from the end.
    """
    window = int(sr * 0.05)  # 50ms windows
    pad = int(sr * pad_ms / 1000)
    long_silent_windows = max(1, int(long_silence_ms / 50))

    n_windows = len(audio) // window
    if n_windows == 0:
        return audio

    rms = np.array([
        np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2))
        for i in range(n_windows)
    ])

    # Forward: find first long silence gap after speech starts
    found_speech = False
    silent_count = 0
    for i in range(n_windows):
        if rms[i] >= threshold:
            found_speech = True
            silent_count = 0
        elif found_speech:
            silent_count += 1
            if silent_count >= long_silent_windows:
                cut = (i - silent_count + 1) * window + pad
                audio = audio[:min(cut, len(audio))]
                break

    # Backward: trim trailing silence/noise
    n_windows = len(audio) // window
    if n_windows > 2:
        rms = np.array([
            np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2))
            for i in range(n_windows)
        ])
        for i in range(n_windows - 1, 2, -1):
            if rms[i] >= threshold and rms[i - 1] >= threshold and rms[i - 2] >= threshold:
                end = min((i + 1) * window + pad, len(audio))
                return audio[:end]

    return audio
