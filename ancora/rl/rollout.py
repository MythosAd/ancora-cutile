"""
ancora/rl/rollout.py — autoregressive generation (rollout) for GRPO.

Two paths:
  generate()        — v1, NO KV-cache: re-runs the full model per token (O(S²)). Simple,
                      tile-aligned, correct (causal mask hides post-frontier placeholders).
  generate_cached() — KV-cache decode (FlashInfer/vLLM-style): prefill the prompt once,
                      then decode one token/step attending to the cached K/V (O(S_cur)/step).
                      Reuses the host-API kernels (rmsnorm/gemm/swiglu) at the batch dim +
                      _attn_decode / _append_kv; the single decode position's RoPE is on host.

Decode tile constraints: the projection GEMM needs M % 128, so we pad the batch to Bp
(multiple of 128); padding sequences produce garbage we slice off. All cache staging
uploads go on the SAME stream as the consuming kernel (cross-stream race lesson).
"""
import math
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.kernels.norm import rmsnorm_forward
from ancora.kernels.activation import swiglu_forward
from ancora.kernels.rope import build_cos_sin
from ancora.kernels.attention import _attn_decode, _attn_decode_blk, _append_kv, D as HEAD_D, BKV, BQ
# _attn_decode_blk is the UNIFIED decode (bitwise == prefill, proven test_attn_decode_mma.py); wiring it
# here needs identical-kernel parity for the WHOLE forward (host-glue path differs from training) — that
# belongs in the device-resident decode, not this transitional host path. Host rollout keeps _attn_decode.
from ancora.model.qwen3_layer import linear_bf16, _bf

_GTM = 128   # projection GEMM tile → batch must be a multiple of this


def _f32bf(x):
    u = x.astype(np.float32).view(np.uint32); u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)
def _bitsf(u): return (u.astype(np.uint32) << 16).view(np.float32)


# ── O(S²) generation (no cache) ──────────────────────────────────────────────

def generate(model, prompt_ids, gen_len, stream_int, temp=1.0, seed=0):
    """prompt_ids: (B, P) int. Returns ids (B, P+gen_len). Re-runs full model/token."""
    B, P = prompt_ids.shape
    S = P + gen_len; H, V = model.cfg.hidden, model.V
    ids = np.zeros((B, S), np.int64); ids[:, :P] = prompt_ids
    rng = np.random.default_rng(seed); lm_head = model.lm_head.astype(np.float32)
    for t in range(P, S):
        hidden, _ = model.forward(ids, stream_int)
        logits = hidden.reshape(B, S, H)[:, t - 1].astype(np.float32) @ lm_head
        if temp == 0.0:
            ids[:, t] = logits.argmax(-1)
        else:
            logits = (logits - logits.max(-1, keepdims=True)) / temp
            p = np.exp(logits); p /= p.sum(-1, keepdims=True)
            ids[:, t] = [rng.choice(V, p=p[b]) for b in range(B)]
    return ids


def build_grpo_targets(ids, advantage, prompt_len):
    """Shifted LM targets + per-token advantage masked to the generated span."""
    B, S = ids.shape
    labels = np.zeros((B, S), np.int64); labels[:, :S - 1] = ids[:, 1:]
    adv_tok = np.zeros((B, S), np.float32); adv_tok[:, prompt_len - 1:S - 1] = advantage[:, None]
    return labels.reshape(-1), adv_tok.reshape(-1)


# ── KV-cache decode ──────────────────────────────────────────────────────────

class _DBuf:
    """Device buffer with same-stream async upload + position-offset view (for append)."""
    def __init__(self, shape, dtype):
        self.sh, self.dt = shape, np.dtype(dtype); self.nb = int(np.prod(shape)) * self.dt.itemsize
        _, self.p = cdrv.cuMemAlloc(self.nb)
        self.__cuda_array_interface__ = {"shape": shape, "typestr": self.dt.str,
                                         "data": (int(self.p), False), "version": 3}
    def upload(self, arr, si):
        a = np.ascontiguousarray(arr.astype(self.dt)); cdrv.cuMemcpyHtoDAsync(self.p, a, a.nbytes, si); return a
    def upload_sync(self, arr):
        a = np.ascontiguousarray(arr.astype(self.dt)); cdrv.cuMemcpyHtoD(self.p, a, a.nbytes)
    def numpy(self):
        o = np.empty(self.sh, self.dt); cdrv.cuMemcpyDtoH(o, self.p, self.nb); return o
    def at_pos(self, pos):
        v = type("V", (), {})()
        v.__cuda_array_interface__ = {"shape": self.sh, "typestr": self.dt.str,
                                      "data": (int(self.p) + pos * HEAD_D * self.dt.itemsize, False), "version": 3}
        return v
    def free(self): cdrv.cuMemFree(self.p)


class KVCache:
    """Per-layer sequence-major K/V cache (Bp*Hkv*maxS, HEAD_D) uint16 BF16 bits, plus
    REUSED scratch (Kn,Vn,gQ,gO) — preallocated once, never alloc/free per step (the
    alloc/free churn races the async append/decode kernels → scattered corrupt positions)."""
    def __init__(self, nl, Bp, Hkv, Hq, maxS):
        self.nl, self.Bp, self.Hkv, self.Hq, self.maxS = nl, Bp, Hkv, Hq, maxS
        self.K = [_DBuf((Bp * Hkv * maxS, HEAD_D), np.uint16) for _ in range(nl)]
        self.V = [_DBuf((Bp * Hkv * maxS, HEAD_D), np.uint16) for _ in range(nl)]
        self.Kn = _DBuf((Bp * Hkv, HEAD_D), np.uint16); self.Vn = _DBuf((Bp * Hkv, HEAD_D), np.uint16)
        self.gQ = _DBuf((Bp * Hq, HEAD_D), np.uint16);  self.gO = _DBuf((Bp * Hq, HEAD_D), np.float32)
        self._host = []   # host arrays for async copies, kept alive until the per-layer sync

    def init_layer(self, i, kr_bits, vh_bits, P):
        """kr_bits,vh_bits: (Bp, Hkv, P, HEAD_D) uint16 → write into positions [0,P)."""
        Bp, Hkv, maxS = self.Bp, self.Hkv, self.maxS
        full = np.zeros((Bp, Hkv, maxS, HEAD_D), np.uint16)
        full[:, :, :P] = kr_bits; self.K[i].upload_sync(full.reshape(-1, HEAD_D))
        full[:, :, :P] = vh_bits; self.V[i].upload_sync(full.reshape(-1, HEAD_D))

    def append(self, i, k_bits, v_bits, pos, si):
        """k_bits,v_bits: (Bp*Hkv, HEAD_D) uint16 → reused scratch, append at pos (on si)."""
        self._host += [self.Kn.upload(k_bits, si), self.Vn.upload(v_bits, si)]
        ct.launch(si, (self.Bp, self.Hkv, 1), _append_kv,
                  (self.Kn, self.Vn, self.K[i].at_pos(pos), self.V[i].at_pos(pos), self.maxS, self.Hkv))

    def free(self):
        for b in self.K + self.V + [self.Kn, self.Vn, self.gQ, self.gO]: b.free()


def _rope_single(x, cos_p, sin_p):
    """Apply rotate-half RoPE at one position to x:(rows, Hh, Dh). cos_p,sin_p:(Dh/2,).
    Matches kernels/rope._rope_fwd exactly (fp32 compute on BF16-valued input)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return np.concatenate([x1 * cos_p - x2 * sin_p, x2 * cos_p + x1 * sin_p], -1)


def _decode_layer(layer, x, cache, i, pos, cos_p, sin_p, cfg, si):
    """One token (Bp rows) through one decoder layer using the KV cache. x:(Bp,H)→(Bp,H)."""
    Bp, H = x.shape
    Hq, Hkv, Dh, I = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate
    w = layer.w
    res = x
    h, _ = rmsnorm_forward(x, w["input_ln"], si, cfg.eps)
    q = linear_bf16(h, w["q_proj"], si); k = linear_bf16(h, w["k_proj"], si); v = linear_bf16(h, w["v_proj"], si)
    q = rmsnorm_forward(q.reshape(Bp * Hq,  Dh), w["q_norm"], si, cfg.eps)[0].reshape(Bp, Hq, Dh)
    k = rmsnorm_forward(k.reshape(Bp * Hkv, Dh), w["k_norm"], si, cfg.eps)[0].reshape(Bp, Hkv, Dh)
    q = _bf(_rope_single(q, cos_p, sin_p)); k = _bf(_rope_single(k, cos_p, sin_p))   # host RoPE @ pos
    v = v.reshape(Bp, Hkv, Dh)

    cache.append(i, _f32bf(k.reshape(Bp * Hkv, Dh)), _f32bf(v.reshape(Bp * Hkv, Dh)), pos, si)

    cache._host.append(cache.gQ.upload(_f32bf(q.reshape(Bp * Hq, Dh)), si))   # reused scratch
    ct.launch(si, (Bp, Hq, 1), _attn_decode,
              (cache.gQ, cache.K[i], cache.V[i], cache.gO, cache.maxS // BKV, Hq, Hkv,
               1.0 / math.sqrt(Dh), int(pos + 1)))
    cudart.cudaStreamSynchronize(si)
    attn = cache.gO.numpy().reshape(Bp, Hq * Dh)
    cache._host = []   # async copies done (synced) → host arrays can be released

    x = res + linear_bf16(attn, w["o_proj"], si)
    res = x
    h2, _ = rmsnorm_forward(x, w["post_ln"], si, cfg.eps)
    a = swiglu_forward(linear_bf16(h2, w["gate_proj"], si), linear_bf16(h2, w["up_proj"], si), si)
    return res + linear_bf16(a, w["down_proj"], si)


def _decode_step(model, token_ids, cache, pos, cos_sin, si):
    """token_ids:(Bp,) → final-norm hidden (Bp,H). cos_sin = (cos(maxS,Dh/2), sin)."""
    cfg = model.cfg; H = cfg.hidden
    cos_p, sin_p = cos_sin[0][pos], cos_sin[1][pos]
    x = model.embed[token_ids].astype(np.float32)                    # (Bp, H)
    for i, layer in enumerate(model.layers):
        x = _decode_layer(layer, x, cache, i, pos, cos_p, sin_p, cfg, si)
    h, _ = rmsnorm_forward(x, model.final_norm, si, cfg.eps)
    return h


def generate_cached(model, prompt_ids, gen_len, stream_int, temp=1.0, seed=0,
                    teacher=None, return_hidden=False):
    """KV-cache rollout. prompt_ids:(B,P). Returns ids (B, P+gen_len).
    teacher (B,P+gen_len): feed these tokens instead of sampling (validation).
    return_hidden: also return decode hidden (B, gen_len, H) at positions [P, S)."""
    cfg = model.cfg; B, P = prompt_ids.shape
    S = P + gen_len; H, V, Hkv = cfg.hidden, model.V, cfg.n_kv_heads
    Bp = ((B + _GTM - 1) // _GTM) * _GTM                             # pad batch to GEMM tile
    rng = np.random.default_rng(seed)
    cos_sin = build_cos_sin(S, cfg.head_dim, cfg.rope_theta)
    lm_head = model.lm_head.astype(np.float32)

    # ── prefill the (padded) prompt; extract per-layer RoPE'd K,V into the cache ──
    pad_prompt = np.zeros((Bp, P), np.int64); pad_prompt[:B] = prompt_ids
    hidden_full, pc = model.forward(pad_prompt, stream_int)
    cache = KVCache(len(model.layers), Bp, Hkv, cfg.n_heads, S)
    for i, lc in enumerate(pc["caches"]):
        cache.init_layer(i, _f32bf(lc["kr"]), _f32bf(lc["vh"]), P)        # kr,vh: (Bp,Hkv,P,Dh)

    ids = np.zeros((Bp, S), np.int64); ids[:B, :P] = prompt_ids
    hpre = hidden_full.reshape(Bp, P, H)[:B, P - 1].astype(np.float32)    # last prompt hidden
    ids[:B, P] = teacher[:, P] if teacher is not None else _sample(hpre @ lm_head, temp, rng, V, B)

    hiddens = []
    for pos in range(P, S):
        h = _decode_step(model, ids[:, pos], cache, pos, cos_sin, stream_int)   # (Bp,H)
        hiddens.append(h[:B].copy())
        if pos < S - 1:
            nxt = _sample(h[:B].astype(np.float32) @ lm_head, temp, rng, V, B)
            ids[:B, pos + 1] = teacher[:, pos + 1] if teacher is not None else nxt
    cache.free()
    if return_hidden:
        return ids[:B], np.stack(hiddens, 1)        # (B, gen_len, H)
    return ids[:B]


def _sample(logits, temp, rng, V, B):
    if temp == 0.0:
        return logits.argmax(-1)
    z = (logits - logits.max(-1, keepdims=True)) / temp
    p = np.exp(z); p /= p.sum(-1, keepdims=True)
    return np.array([rng.choice(V, p=p[b]) for b in range(B)], np.int64)
