"""Fast-path forward functions for VibeVoice generation.

Bypasses nn.Module dispatch by extracting weights into flat dicts
and using raw mx.quantized_matmul / matmul calls. ~30% faster than
nn.Module for the autoregressive loop.

Used by generate.py for the hot path (LM step + diffusion).
Model loading still uses nn.Module for clean weight mapping.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .model import VibeVoiceModel, VibeVoiceConfig, apply_rope


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def _extract_linear(mod) -> dict:
    """Extract weight data from nn.Linear or nn.QuantizedLinear."""
    d = {}
    if hasattr(mod, "scales"):  # quantized
        d["w"] = mod.weight
        d["s"] = mod.scales
        d["b"] = mod.biases
        d["gs"] = mod.group_size
        d["bits"] = mod.bits
        d["q"] = True
    else:
        d["w"] = mod.weight
        d["q"] = False
    if hasattr(mod, "bias") and mod.bias is not None:
        d["bias"] = mod.bias
    return d


def _mm(x, d):
    """x @ w.T — dispatches to quantized or plain matmul."""
    if d["q"]:
        return mx.quantized_matmul(
            x, d["w"], d["s"], d["b"],
            transpose=True, group_size=d["gs"], bits=d["bits"],
        )
    return x @ d["w"].T


class FastLM:
    """Flat-dict LM for fast autoregressive decode."""

    def __init__(self, model: VibeVoiceModel, config: VibeVoiceConfig):
        self.H = config.hidden_size
        self.NH = config.num_attention_heads
        self.NKV = config.num_key_value_heads
        self.HD = config.head_dim
        self.NL = config.num_hidden_layers
        self.scale = config.head_dim ** -0.5
        self.eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta

        # Extract layer weights
        self.layers = []
        for layer in model.model.layers:
            sa, ml = layer.self_attn, layer.mlp
            d = {
                "iln": layer.input_layernorm.weight,
                "pln": layer.post_attention_layernorm.weight,
                "q": _extract_linear(sa.q_proj),
                "k": _extract_linear(sa.k_proj),
                "v": _extract_linear(sa.v_proj),
                "o": _extract_linear(sa.o_proj),
                "g": _extract_linear(ml.gate_proj),
                "u": _extract_linear(ml.up_proj),
                "d": _extract_linear(ml.down_proj),
            }
            self.layers.append(d)

        self.norm_w = model.model.norm.weight
        self.embed_w = model.model.embed_tokens.weight

        # LM head
        if model.lm_head is not None:
            self.lm_head_w = model.lm_head.weight
        else:
            self.lm_head_w = None  # tied

    def forward(self, h, cos, sin, k_cache, v_cache):
        """Single-step LM forward with KV cache.

        h: (1, Q, H)
        cos, sin: (Q, HD) from compute_rope
        k_cache, v_cache: lists of (1, NKV, S, HD) arrays
        Returns: hidden (1, Q, H), updates k_cache/v_cache in place.
        """
        return self._forward_inner(h, cos, sin, k_cache, v_cache)

    def forward_dual(self, h_main, cos_main, sin_main, k_cache, v_cache,
                     h_neg, cos_neg, sin_neg, neg_k_cache, neg_v_cache):
        """Batched main+neg LM forward — reads weights once for both.

        Returns: (hidden_main, hidden_neg), updates both KV caches.
        """
        NH, NKV, HD, H = self.NH, self.NKV, self.HD, self.H
        scale = self.scale
        eps = self.eps

        hm, hn_input = h_main, h_neg

        for li, d in enumerate(self.layers):
            # Batch projections: concat (1,1,H) inputs → (2,1,H), single matmul
            h_cat = mx.concatenate([hm, hn_input], axis=0)  # (2, 1, H)
            res_cat = h_cat

            hn_cat = mx.fast.rms_norm(h_cat, d["iln"], eps)

            q_cat = _mm(hn_cat, d["q"])
            if "bias" in d["q"]:
                q_cat = q_cat + d["q"]["bias"]
            k_cat = _mm(hn_cat, d["k"])
            if "bias" in d["k"]:
                k_cat = k_cat + d["k"]["bias"]
            v_cat = _mm(hn_cat, d["v"])
            if "bias" in d["v"]:
                v_cat = v_cat + d["v"]["bias"]

            # Split for attention (different KV caches)
            q_m, q_n = q_cat[0:1], q_cat[1:2]
            k_m, k_n = k_cat[0:1], k_cat[1:2]
            v_m, v_n = v_cat[0:1], v_cat[1:2]

            # Main attention
            q_m = q_m.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k_m = k_m.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v_m = v_m.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            q_m = apply_rope(q_m, cos_main, sin_main)
            k_m = apply_rope(k_m, cos_main, sin_main)
            k_cache[li] = mx.concatenate([k_cache[li], k_m], axis=2)
            v_cache[li] = mx.concatenate([v_cache[li], v_m], axis=2)
            out_m = mx.fast.scaled_dot_product_attention(
                q_m, k_cache[li], v_cache[li], scale=scale,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            # Neg attention
            q_n = q_n.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k_n = k_n.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v_n = v_n.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            q_n = apply_rope(q_n, cos_neg, sin_neg)
            k_n = apply_rope(k_n, cos_neg, sin_neg)
            neg_k_cache[li] = mx.concatenate([neg_k_cache[li], k_n], axis=2)
            neg_v_cache[li] = mx.concatenate([neg_v_cache[li], v_n], axis=2)
            out_n = mx.fast.scaled_dot_product_attention(
                q_n, neg_k_cache[li], neg_v_cache[li], scale=scale,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            # Batch o_proj + MLP
            out_cat = mx.concatenate([out_m, out_n], axis=0)
            h_cat = res_cat + _mm(out_cat, d["o"])

            res_cat = h_cat
            hn_cat = mx.fast.rms_norm(h_cat, d["pln"], eps)
            h_cat = res_cat + _mm(nn.silu(_mm(hn_cat, d["g"])) * _mm(hn_cat, d["u"]), d["d"])

            hm, hn_input = h_cat[0:1], h_cat[1:2]

        hm = mx.fast.rms_norm(hm, self.norm_w, eps)
        hn_input = mx.fast.rms_norm(hn_input, self.norm_w, eps)
        return hm, hn_input

    def _forward_inner(self, h, cos, sin, k_cache, v_cache):
        NH, NKV, HD, H = self.NH, self.NKV, self.HD, self.H
        scale = self.scale
        eps = self.eps

        for li, d in enumerate(self.layers):
            res = h
            hn = mx.fast.rms_norm(h, d["iln"], eps)

            q = _mm(hn, d["q"])
            if "bias" in d["q"]:
                q = q + d["q"]["bias"]
            k = _mm(hn, d["k"])
            if "bias" in d["k"]:
                k = k + d["k"]["bias"]
            v = _mm(hn, d["v"])
            if "bias" in d["v"]:
                v = v + d["v"]["bias"]

            q = q.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k = k.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v = v.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            k_cache[li] = mx.concatenate([k_cache[li], k], axis=2)
            v_cache[li] = mx.concatenate([v_cache[li], v], axis=2)

            out = mx.fast.scaled_dot_product_attention(
                q, k_cache[li], v_cache[li], scale=scale,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            h = res + _mm(out, d["o"])

            res = h
            hn = mx.fast.rms_norm(h, d["pln"], eps)
            h = res + _mm(nn.silu(_mm(hn, d["g"])) * _mm(hn, d["u"]), d["d"])

        return mx.fast.rms_norm(h, self.norm_w, eps)

    def logits(self, h):
        """Compute logits in float32 for numerical stability."""
        if self.lm_head_w is not None:
            return (h.astype(mx.float32) @ self.lm_head_w.astype(mx.float32).T).astype(h.dtype)
        return (h.astype(mx.float32) @ self.embed_w.astype(mx.float32).T).astype(h.dtype)

    def prefill(self, embeds, cos, sin, mask, k_cache, v_cache):
        """Batched prefill (Q>1) with causal mask. Same as forward but with mask."""
        NH, NKV, HD, H = self.NH, self.NKV, self.HD, self.H
        scale = self.scale
        eps = self.eps
        h = embeds

        for li, d in enumerate(self.layers):
            res = h
            hn = mx.fast.rms_norm(h, d["iln"], eps)

            q = _mm(hn, d["q"])
            if "bias" in d["q"]:
                q = q + d["q"]["bias"]
            k = _mm(hn, d["k"])
            if "bias" in d["k"]:
                k = k + d["k"]["bias"]
            v = _mm(hn, d["v"])
            if "bias" in d["v"]:
                v = v + d["v"]["bias"]

            q = q.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k = k.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v = v.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            k_cache[li] = k
            v_cache[li] = v

            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            h = res + _mm(out, d["o"])

            res = h
            hn = mx.fast.rms_norm(h, d["pln"], eps)
            h = res + _mm(nn.silu(_mm(hn, d["g"])) * _mm(hn, d["u"]), d["d"])

        return mx.fast.rms_norm(h, self.norm_w, eps)


class FastDiffusionHead:
    """Flat-dict diffusion head for fast DPM solver.

    Handles both fp16 and quantized (INT8) diffusion head weights.
    """

    def __init__(self, model: VibeVoiceModel, config: VibeVoiceConfig):
        dh = model.diffusion_head
        self.noisy = _extract_linear(dh.noisy_images_proj)
        self.cond = _extract_linear(dh.cond_proj)
        self.t0 = _extract_linear(dh.t_embedder.mlp[0])
        self.t2 = _extract_linear(dh.t_embedder.mlp[2])
        self.freq_dim = dh.t_embedder.freq_dim

        self.layers = []
        for layer in dh.layers:
            self.layers.append((
                _extract_linear(layer.adaLN_modulation[1]),
                layer.norm.weight,
                _extract_linear(layer.ffn.gate_proj),
                _extract_linear(layer.ffn.up_proj),
                _extract_linear(layer.ffn.down_proj),
            ))

        self.final_adaln = _extract_linear(dh.final_layer.adaLN_modulation[1])
        self.final_linear = _extract_linear(dh.final_layer.linear)
        self.H = config.hidden_size

    def __call__(self, noisy, timestep, condition):
        H = self.H
        x = _mm(noisy, self.noisy)

        # Timestep embedding
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = timestep[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(x.dtype)
        t = _mm(nn.silu(_mm(emb, self.t0)), self.t2)

        c = _mm(condition, self.cond) + t

        for adaln, norm_w, gate, up, down in self.layers:
            mods = _mm(nn.silu(c), adaln)
            shift, scale, g = mods[..., :H], mods[..., H:2*H], mods[..., 2*H:3*H]
            h = mx.fast.rms_norm(x, norm_w, 1e-5) * (1 + scale) + shift
            x = x + g * _mm(nn.silu(_mm(h, gate)) * _mm(h, up), down)

        mods = _mm(nn.silu(c), self.final_adaln)
        shift, scale = mods[..., :H], mods[..., H:]
        h = mx.fast.rms_norm(x, mx.ones(H, dtype=x.dtype), 1e-5) * (1 + scale) + shift
        return _mm(h, self.final_linear)
