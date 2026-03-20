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
    solver: str = "dpm"            # "ddpm" or "dpm"
    diffusion_steps: int = 10      # DPM-Solver++ steps
    cfg_scale: float = 1.3         # Classifier-free guidance
    max_speech_tokens: int = 200   # Safety limit
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
# DPM-Solver++ 2M (all MLX, batched CFG)
# ---------------------------------------------------------------------------

def dpm_solver_2m(
    diff_head,
    condition: mx.array,
    neg_condition: mx.array,
    cfg_scale: float,
    num_steps: int = 10,
    seed: int = 0,
    dtype=mx.float16,
) -> mx.array:
    """DPM-Solver++ 2M entirely in MLX with batched CFG (B=2).

    Returns sample of shape (1, VAE_DIM) in float32.
    """
    t_schedule = np.round(np.linspace(DDPM_STEPS - 1, 0, num_steps + 1)).astype(np.int64)

    key = mx.random.key(seed)
    sample = mx.random.normal(shape=(1, VAE_DIM), key=key).astype(mx.float32)

    batched_cond = mx.concatenate([
        condition.astype(dtype), neg_condition.astype(dtype)
    ], axis=0)  # (2, H)

    x0_list = []

    for i in range(num_steps):
        s = int(t_schedule[i])
        t = int(t_schedule[i + 1])

        batched_sample = mx.concatenate([sample, sample], axis=0).astype(dtype)
        ts_mx = mx.array([float(s)]).astype(dtype)

        v_batched = diff_head(batched_sample, ts_mx, batched_cond)
        mx.eval(v_batched)

        v_cond = v_batched[0:1].astype(mx.float32)
        v_uncond = v_batched[1:2].astype(mx.float32)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        alpha_s = float(_ALPHA_NP[s])
        sigma_s = float(_SIGMA_NP[s])
        x0 = alpha_s * sample - sigma_s * v
        x0_list.append(x0)

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
            D0 = x0_list[-1]
            D1 = (1.0 / r) * (x0_list[-1] - x0_list[-2])
            D = D0 + 0.5 * D1

        sigma_t = float(_SIGMA_NP[t])
        alpha_t = float(_ALPHA_NP[t])
        sample = (sigma_t / sigma_s) * sample - alpha_t * float(np.expm1(-h)) * D

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

    # Compute negative condition for CFG (speech_start token through LM, no cache)
    neg_k = [None] * NL
    neg_v = [None] * NL
    neg_embed = embed_table[config.speech_start_id].reshape(1, 1, config.hidden_size)
    neg_pos = mx.arange(1, dtype=mx.float32)
    neg_cos, neg_sin = compute_rope(neg_pos, config.head_dim, config.rope_theta)
    # Use fast_lm for neg condition (no mask needed for single token)
    neg_k_tmp = [mx.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=dtype)] * NL
    neg_v_tmp = [mx.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=dtype)] * NL
    neg_hidden = fast_lm.forward(neg_embed, neg_cos, neg_sin, neg_k_tmp, neg_v_tmp)
    mx.eval(neg_hidden)
    neg_condition = neg_hidden[:, 0:1, :].reshape(1, config.hidden_size)
    del neg_k_tmp, neg_v_tmp

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

    # First token
    logits = fast_lm.logits(hidden)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[0, 0]).item())

    # Always batch-decode at end for clean audio.
    # For semantic feedback, use cheap non-streaming per-token VAE
    # (quality doesn't matter — just needs approximate audio for features).

    # Autoregressive generation
    audio_chunks = []
    all_latents = []
    rng = np.random.RandomState(opts.seed)
    position = n_prefill

    for step in range(opts.max_speech_tokens * 3):
        if next_token == config.eos_id:
            break
        if metrics.num_speech_tokens >= opts.max_speech_tokens:
            break

        if next_token == config.speech_diffusion_id:
            metrics.num_speech_tokens += 1

            # Diffusion (fast path — no nn.Module dispatch)
            t0 = time.perf_counter()
            condition = hidden[:, 0:1, :].reshape(1, config.hidden_size)
            sample = dpm_solver_2m(
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
            mx.eval(next_embed)
            metrics.record("connector", (time.perf_counter() - t0) * 1000)
        else:
            if next_token == config.speech_end_id and semantic_reset_fn is not None:
                semantic_reset_fn()
            next_embed = embed_table[next_token].reshape(1, 1, config.hidden_size)

        # LM step (fast path — raw quantized matmul)
        t0 = time.perf_counter()
        pos = mx.array([float(position)], dtype=mx.float32)
        cos, sin = compute_rope(pos, config.head_dim, config.rope_theta)
        hidden = fast_lm.forward(next_embed, cos, sin, k_cache, v_cache)
        logits = fast_lm.logits(hidden)
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
        metrics.record("vae_final", (time.perf_counter() - t0) * 1000)
    else:
        audio_out = np.zeros(0, dtype=np.float32)

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.audio_samples = len(audio_out)

    return audio_out, metrics
