"""
ancora/kernels/loss.py — fused, batch-invariant cross-entropy / log-prob.

Core primitive: per-token log-probability of the target token
    logprob[m] = logit[m, label[m]] - logsumexp_v(logit[m, :])
where logit = hidden @ W_head is computed STREAMING over vocab tiles and never
materialized in HBM (Qwen3 vocab=151669 → full logits would be ~2.5 GB).

This single primitive serves:
  - SFT cross-entropy:   loss = -mean_m logprob[m]
  - GRPO:                loss = -mean( advantage * logprob )  (+ KL)

Batch invariance (required for on-policy RL — see CLAUDE.md):
  The logsumexp reduction streams vocab tiles vb=0,1,...,V_BLOCKS-1 in FIXED order,
  one token-row owned entirely by one block. No split-V, no atomic accumulation.
  → a token's logprob is bitwise-identical regardless of batch size / how many other
    tokens co-run. Rollout and training therefore agree on logprobs.

Layout:
  hidden : (M, H)  uint16  (BF16 bits)
  w_head : (H, V)  uint16  (BF16 bits, LM-head weight, NOT transposed: y=(H,V))
  labels : (M, 1)  int32
  logprob: (M, 1)  float32  (output)
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # sets CUDA_PATH

# ── tile sizes ──────────────────────────────────────────────────────────────
TM = 64    # token rows per block
TV = 128   # vocab columns per streaming tile


@ct.kernel(occupancy=2)
def _fused_logprob(hidden, w_head, labels, logprob, lse_out,
                   H: ct.Constant[int], V_BLOCKS: ct.Constant[int]):
    """
    logprob[m] = logit[m, label[m]] - logsumexp_v logit[m,:],  logit = hidden @ W_head.
    One block owns TM token-rows; streams vocab in TV-tiles (fixed order → invariant).
    Also stores lse[m] (logsumexp) — the backward needs it to form softmax probs without
    a second forward pass.
    """
    mb = ct.bid(0)

    # Load this block's hidden rows once (kept across the vocab loop). (TM, H) BF16.
    th = ct.bitcast(ct.load(hidden, index=(mb, 0), shape=(TM, H)), ct.bfloat16)
    # Target token id per row. (TM, 1) int32.
    lbl = ct.load(labels, index=(mb, 0), shape=(TM, 1))

    NEG = -1e38
    m_run = ct.full((TM, 1), NEG, ct.float32)   # running max (online logsumexp)
    s_run = ct.zeros((TM, 1),     ct.float32)    # running Σ exp(logit - max)
    tgt   = ct.zeros((TM, 1),     ct.float32)    # gathered target logit

    for vb in range(V_BLOCKS):
        # logit tile = hidden @ W_head[:, vb*TV:(vb+1)*TV]  → (TM, TV), full H reduction
        tw = ct.bitcast(ct.load(w_head, index=(0, vb), shape=(H, TV)), ct.bfloat16)
        logit = ct.mma(th, tw, ct.zeros((TM, TV), ct.float32))

        # online logsumexp (numerically stable, fixed vocab order)
        m_new = ct.maximum(m_run, ct.max(logit, axis=-1, keepdims=True))
        s_run = s_run * ct.exp(m_run - m_new) + ct.sum(
            ct.exp(logit - ct.broadcast_to(m_new, (TM, TV))), axis=-1, keepdims=True)
        m_run = m_new

        # gather target logit: rows whose label falls in this vocab tile.
        # mask[i,j] = (label[i] == vb*TV + j); only one tile matches per row.
        v_idx = ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0)   # (1, TV)
        mask  = ct.equal(ct.broadcast_to(lbl,   (TM, TV)),
                         ct.broadcast_to(v_idx, (TM, TV)))                    # (TM, TV)
        tgt   = tgt + ct.sum(ct.where(mask, logit, 0.0), axis=-1, keepdims=True)

    lse = m_run + ct.log(s_run)        # (TM, 1) logsumexp
    ct.store(logprob, index=(mb, 0), tile=tgt - lse)   # log-prob of target token
    ct.store(lse_out,  index=(mb, 0), tile=lse)        # saved for backward


# ── backward: fused, streamed over vocab ─────────────────────────────────────
# glogit[m,v] = inv_M * adv[m] * (softmax(logit)[m,v] - onehot(label[m])[v])
# dhidden = glogit @ W_headᵀ ;  dW_head = hiddenᵀ @ glogit.
# Both recompute logit per vocab-tile and form p from the SAVED lse → never
# materialize the (M, V) glogit. Fixed vocab/m order → batch-invariant.

@ct.kernel(occupancy=2)
def _bwd_dhidden(hidden, w_head, labels, lse, adv, dhidden,
                 H: ct.Constant[int], V_BLOCKS: ct.Constant[int],
                 inv_M: ct.Constant[float]):
    """dhidden[m,:] = Σ_v glogit[m,v] W_head[:,v].  Grid over m-tiles, stream vocab."""
    mb = ct.bid(0)
    th   = ct.bitcast(ct.load(hidden, index=(mb, 0), shape=(TM, H)), ct.bfloat16)
    tlse = ct.load(lse,    index=(mb, 0), shape=(TM, 1))
    tlbl = ct.load(labels, index=(mb, 0), shape=(TM, 1))
    tadv = ct.load(adv,    index=(mb, 0), shape=(TM, 1))
    scl  = tadv * inv_M                                   # (TM,1) per-token grad scale

    dh = ct.zeros((TM, H), ct.float32)
    for vb in range(V_BLOCKS):
        tw    = ct.bitcast(ct.load(w_head, index=(0, vb), shape=(H, TV)), ct.bfloat16)
        logit = ct.mma(th, tw, ct.zeros((TM, TV), ct.float32))
        p     = ct.exp(logit - ct.broadcast_to(tlse, (TM, TV)))     # softmax probs
        v_idx = ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0)
        oneh  = ct.where(ct.equal(ct.broadcast_to(tlbl,  (TM, TV)),
                                  ct.broadcast_to(v_idx, (TM, TV))), 1.0, 0.0)
        glog  = ct.broadcast_to(scl, (TM, TV)) * (p - oneh)         # (TM, TV)
        # dh += glogit @ W_headᵀ : (TM,TV)@(TV,H)
        dh = ct.mma(ct.astype(glog, ct.bfloat16), ct.transpose(tw), dh)
    ct.store(dhidden, index=(mb, 0), tile=dh)


TH = 64   # h-tile for the dW_head output (keep MMA output rows ≤ 128)

@ct.kernel(occupancy=2)
def _bwd_dwhead(hidden, w_head, labels, lse, adv, dwhead,
                H: ct.Constant[int], M_BLOCKS: ct.Constant[int],
                inv_M: ct.Constant[float]):
    """
    dW_head[hb,vb] = Σ_m hidden[m,hb]ᵀ glogit[m,vb].  Grid (V/TV, H/TH); each block
    outputs a small (TH, TV) tile (a 512-row output crashes the tile compiler).
    Recomputes logit (full H) per h-tile — redundant but correct; optimize later.
    """
    vb = ct.bid(0)   # vocab tile
    hb = ct.bid(1)   # hidden tile (output rows of dW)
    tw    = ct.bitcast(ct.load(w_head, index=(0, vb), shape=(H, TV)), ct.bfloat16)
    v_idx = ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0)

    dW = ct.zeros((TH, TV), ct.float32)
    for mb in range(M_BLOCKS):
        th   = ct.bitcast(ct.load(hidden, index=(mb, 0),  shape=(TM, H)),  ct.bfloat16)
        ths  = ct.bitcast(ct.load(hidden, index=(mb, hb), shape=(TM, TH)), ct.bfloat16)  # h-slice
        tlse = ct.load(lse,    index=(mb, 0), shape=(TM, 1))
        tlbl = ct.load(labels, index=(mb, 0), shape=(TM, 1))
        tadv = ct.load(adv,    index=(mb, 0), shape=(TM, 1))
        logit = ct.mma(th, tw, ct.zeros((TM, TV), ct.float32))
        p     = ct.exp(logit - ct.broadcast_to(tlse, (TM, TV)))
        oneh  = ct.where(ct.equal(ct.broadcast_to(tlbl,  (TM, TV)),
                                  ct.broadcast_to(v_idx, (TM, TV))), 1.0, 0.0)
        glog  = ct.broadcast_to(tadv * inv_M, (TM, TV)) * (p - oneh)
        # dW[hb] += hidden[:,hb]ᵀ @ glogit : (TH,TM)@(TM,TV)
        dW = ct.mma(ct.transpose(ths), ct.astype(glog, ct.bfloat16), dW)
    ct.store(dwhead, index=(hb, vb), tile=dW)


# ── FAST PATH: Liger-style GEMM-based fused linear CE ────────────────────────
# The streaming-vocab fused kernels above are memory-optimal but compute-slow
# (5 TFLOPS fwd / 1 TFLOPS bwd) because of small tiles + redundant logit recompute.
# Instead: materialize logits with a TUNED BF16 GEMM (~80-170 TFLOPS), do a cheap
# memory-bound softmax/grad over the materialized (M,V), and use efficient GEMMs for
# the backward. Logits live in HBM (chunk M for real vocab). Still batch-invariant:
# the GEMM has fixed K-order and the CE stats are one-row-per-block.

GTM, GTN, GTK = 128, 128, 64   # GEMM tiles (MMA output rows ≤ 128)
CTM = 64                        # rows per block for CE stat/grad kernels

@ct.kernel(occupancy=2)
def _gemm(A, B, C, KB: ct.Constant[int],
          TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C = A @ B, BF16-bit inputs (bitcast), f32 accumulate. Fixed K-order → invariant."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    ct.store(C, index=(m, n), tile=acc)


@ct.kernel(occupancy=2)
def _ce_stats(logits, labels, logprob, lse, V_BLOCKS: ct.Constant[int]):
    """Materialized logits (M,V) f32 → logprob, lse. One block owns CTM rows."""
    mb = ct.bid(0)
    lbl = ct.load(labels, index=(mb, 0), shape=(CTM, 1))
    m_run = ct.full((CTM, 1), -1e38, ct.float32)
    s_run = ct.zeros((CTM, 1), ct.float32)
    tgt   = ct.zeros((CTM, 1), ct.float32)
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTM, TV))   # f32, no bitcast
        m_new = ct.maximum(m_run, ct.max(lg, axis=-1, keepdims=True))
        s_run = s_run * ct.exp(m_run - m_new) + ct.sum(
            ct.exp(lg - ct.broadcast_to(m_new, (CTM, TV))), axis=-1, keepdims=True)
        m_run = m_new
        v_idx = ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0)
        mask  = ct.equal(ct.broadcast_to(lbl,   (CTM, TV)),
                         ct.broadcast_to(v_idx, (CTM, TV)))
        tgt   = tgt + ct.sum(ct.where(mask, lg, 0.0), axis=-1, keepdims=True)
    lse_v = m_run + ct.log(s_run)
    ct.store(lse,     index=(mb, 0), tile=lse_v)
    ct.store(logprob, index=(mb, 0), tile=tgt - lse_v)


@ct.kernel(occupancy=2)
def _argmax_id(logits, out_id, V_BLOCKS: ct.Constant[int]):
    """Greedy next-token: argmax_v logits[m,v] per row → out_id (M,1) int32. Streams vocab in TV-tiles
    (fixed order). Lets device decode pick the next token WITHOUT a (M,V) logits DtoH — feed out_id back
    into _ce_stats to get that token's logprob (bitwise-equal to training's CE on the same token).
    Tie-break = lowest index (matches np.argmax). Indices kept in f32 (V<2^24 → exact)."""
    mb = ct.bid(0)
    BIG = 1e30
    m_run = ct.full((CTM, 1), -1e38, ct.float32)    # running max logit
    a_run = ct.zeros((CTM, 1), ct.float32)          # running argmax index (as f32)
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTM, TV))               # f32
        tile_max = ct.max(lg, axis=-1, keepdims=True)                       # (CTM,1)
        v_idx = ct.astype(ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0), ct.float32)  # (1,TV)
        is_max = ct.equal(lg, ct.broadcast_to(tile_max, (CTM, TV)))
        tile_arg = ct.min(ct.where(is_max, ct.broadcast_to(v_idx, (CTM, TV)),
                                   ct.full((CTM, TV), BIG, ct.float32)), axis=-1, keepdims=True)   # lowest idx at tile max
        take = ct.greater(tile_max, m_run)                                  # strict → keep earliest tile on ties
        a_run = ct.where(take, tile_arg, a_run)
        m_run = ct.maximum(m_run, tile_max)
    ct.store(out_id, index=(mb, 0), tile=ct.astype(a_run, ct.int32))


@ct.kernel(occupancy=2)
def _sample_id(logits, seed_buf, out_id, V_BLOCKS: ct.Constant[int], inv_T: ct.Constant[float]):
    """Temperature SAMPLING via Gumbel-max: out_id[m] = argmax_v(logits[m,v]/T + g[m,v]) with
    g = -log(-log(u)), u from a counter-hash of (row, vocab-id) ^ seed — cuda-tile has no RNG, so
    this reuses the SR-grad dither recipe (coord-keyed ⇒ batch-invariant + deterministic given the
    seed sequence). The seed lives in DEVICE memory (incremented in-graph by fused._inc1) so the
    kernel is CUDA-graph replayable. Gumbel-max == multinomial softmax(logits/T) sampling, up to
    the hash's uniformity. Grid (M//CTM,)."""
    mb = ct.bid(0)
    sd = ct.astype(ct.reshape(ct.load(seed_buf, index=(0, 0), shape=(1, 1)), ()), ct.uint32)
    BIG = 1e30
    m_run = ct.full((CTM, 1), -1e38, ct.float32)
    a_run = ct.zeros((CTM, 1), ct.float32)
    rows = ct.broadcast_to(ct.reshape(ct.arange(CTM, dtype=ct.uint32), (CTM, 1)), (CTM, TV)) \
        + ct.astype(mb, ct.uint32) * CTM
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTM, TV)) * inv_T
        cols = ct.astype(ct.broadcast_to(ct.reshape(ct.arange(TV, dtype=ct.int32), (1, TV)),
                                         (CTM, TV)) + vb * TV, ct.uint32)
        h = ((rows * 73856093) ^ (cols * 19349663) ^ sd) * 0x9E3779B1       # murmur-style mix
        h = h ^ (h >> 16); h = h * 0x85EBCA6B; h = h ^ (h >> 13)
        u = (ct.astype(h >> 8, ct.float32) + 0.5) * (1.0 / 16777216.0)      # uniform in (0,1)
        s = lg + (0.0 - ct.log(0.0 - ct.log(u)))                            # + Gumbel(0,1)
        tile_max = ct.max(s, axis=-1, keepdims=True)
        v_idx = ct.astype(ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0), ct.float32)
        is_max = ct.equal(s, ct.broadcast_to(tile_max, (CTM, TV)))
        tile_arg = ct.min(ct.where(is_max, ct.broadcast_to(v_idx, (CTM, TV)),
                                   ct.full((CTM, TV), BIG, ct.float32)), axis=-1, keepdims=True)
        take = ct.greater(tile_max, m_run)
        a_run = ct.where(take, tile_arg, a_run)
        m_run = ct.maximum(m_run, tile_max)
    ct.store(out_id, index=(mb, 0), tile=ct.astype(a_run, ct.int32))


@ct.kernel(occupancy=2)
def _ce_grad(logits, lse, labels, adv, glogit,
             V_BLOCKS: ct.Constant[int], inv_M: ct.Constant[float]):
    """glogit[m,v] = inv_M·adv[m]·(softmax(logit) - onehot). Output BF16 bits (uint16)."""
    mb = ct.bid(0)
    tlse = ct.load(lse,    index=(mb, 0), shape=(CTM, 1))
    tlbl = ct.load(labels, index=(mb, 0), shape=(CTM, 1))
    tadv = ct.load(adv,    index=(mb, 0), shape=(CTM, 1))
    scl  = tadv * inv_M
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTM, TV))
        p  = ct.exp(lg - ct.broadcast_to(tlse, (CTM, TV)))
        v_idx = ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0)
        oneh  = ct.where(ct.equal(ct.broadcast_to(tlbl,  (CTM, TV)),
                                  ct.broadcast_to(v_idx, (CTM, TV))), 1.0, 0.0)
        g = ct.broadcast_to(scl, (CTM, TV)) * (p - oneh)
        ct.store(glogit, index=(mb, vb), tile=ct.bitcast(ct.astype(g, ct.bfloat16), ct.uint16))


# ── host helpers ────────────────────────────────────────────────────────────

def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


class _GpuArray:
    def __init__(self, arr: np.ndarray):
        self._shape, self._dtype, self._nbytes = arr.shape, arr.dtype, arr.nbytes
        err, self._ptr = cdrv.cuMemAlloc(arr.nbytes)
        if err.value:
            raise RuntimeError(f"cuMemAlloc: {err}")
        cdrv.cuMemcpyHtoD(self._ptr, arr, arr.nbytes)
        self.__cuda_array_interface__ = {
            "shape": arr.shape, "typestr": arr.dtype.str,
            "data": (int(self._ptr), False), "version": 3,
        }
    def to_numpy(self):
        out = np.empty(self._shape, self._dtype)
        cdrv.cuMemcpyDtoH(out, self._ptr, self._nbytes)
        return out
    def free(self): cdrv.cuMemFree(self._ptr)
    @classmethod
    def zeros(cls, shape, dtype): return cls(np.zeros(shape, dtype))


def fused_logprob(hidden: np.ndarray, w_head: np.ndarray, labels: np.ndarray,
                  stream_int: int, return_lse: bool = False):
    """
    hidden: (M, H) float32,  w_head: (H, V) float32,  labels: (M,) int.
    Returns logprob: (M,) float32 = log_softmax(hidden@w_head)[label].
    If return_lse: returns (logprob, lse) — lse is needed for the backward.
    Requires M % TM == 0 and V % TV == 0 (pad vocab for real models).
    """
    M, H = hidden.shape
    H2, V = w_head.shape
    assert H == H2 and M % TM == 0 and V % TV == 0
    V_BLOCKS = V // TV

    gh  = _GpuArray(f32_to_bf16_bits(hidden))
    gw  = _GpuArray(f32_to_bf16_bits(w_head))
    gl  = _GpuArray(labels.astype(np.int32).reshape(M, 1))
    gp  = _GpuArray.zeros((M, 1), np.float32)
    gls = _GpuArray.zeros((M, 1), np.float32)

    ct.launch(stream_int, (M // TM, 1, 1), _fused_logprob, (gh, gw, gl, gp, gls, H, V_BLOCKS))
    cudart.cudaStreamSynchronize(stream_int)

    lp  = gp.to_numpy().reshape(M)
    lse = gls.to_numpy().reshape(M)
    for g in (gh, gw, gl, gp, gls): g.free()
    return (lp, lse) if return_lse else lp


def fused_backward(hidden: np.ndarray, w_head: np.ndarray, labels: np.ndarray,
                   advantage: np.ndarray, lse: np.ndarray, stream_int: int):
    """
    Gradient of L = -(1/M) Σ_m adv[m] logprob[m]  w.r.t. hidden and W_head.
    Streams vocab → never materializes (M,V) glogit. Needs lse from forward.
    Returns (dhidden (M,H), dwhead (H,V)) float32.

    (For plain SFT cross-entropy, pass advantage = -ones(M): then
     glogit = (1/M)(onehot - p), the standard CE gradient.)
    """
    M, H = hidden.shape
    H2, V = w_head.shape
    assert H == H2 and M % TM == 0 and V % TV == 0
    V_BLOCKS, M_BLOCKS = V // TV, M // TM
    inv_M = 1.0 / M

    gh  = _GpuArray(f32_to_bf16_bits(hidden))
    gw  = _GpuArray(f32_to_bf16_bits(w_head))
    gl  = _GpuArray(labels.astype(np.int32).reshape(M, 1))
    gls = _GpuArray(lse.astype(np.float32).reshape(M, 1))
    ga  = _GpuArray(advantage.astype(np.float32).reshape(M, 1))
    gdh = _GpuArray.zeros((M, H), np.float32)
    gdw = _GpuArray.zeros((H, V), np.float32)

    ct.launch(stream_int, (M // TM, 1, 1), _bwd_dhidden,
              (gh, gw, gl, gls, ga, gdh, H, V_BLOCKS, inv_M))
    ct.launch(stream_int, (V // TV, H // TH, 1), _bwd_dwhead,
              (gh, gw, gl, gls, ga, gdw, H, M_BLOCKS, inv_M))
    cudart.cudaStreamSynchronize(stream_int)

    dh = gdh.to_numpy()
    dw = gdw.to_numpy()
    for g in (gh, gw, gl, gls, ga, gdh, gdw): g.free()
    return dh, dw


def cross_entropy(hidden, w_head, labels, stream_int) -> float:
    """SFT cross-entropy = -mean_m logprob[m]."""
    return float(-fused_logprob(hidden, w_head, labels, stream_int).mean())


def linear_ce(hidden, w_head, labels, stream_int, advantage=None):
    """
    FAST GEMM-based fused linear cross-entropy (Liger-style).
    hidden (M,H) f32, w_head (H,V) f32, labels (M,) int.
    Returns logprob (M,). If advantage (M,) given, also returns (dhidden (M,H), dW (H,V)).
    Materializes logits (M,V) in HBM — chunk M for the full Qwen3 vocab.
    """
    M, H = hidden.shape
    H2, V = w_head.shape
    assert H == H2
    assert M % GTM == 0 and M % CTM == 0 and M % GTK == 0
    assert V % GTN == 0 and V % TV == 0 and V % GTK == 0
    assert H % GTK == 0 and H % GTN == 0 and H % GTM == 0

    gh   = _GpuArray(f32_to_bf16_bits(hidden))
    gw   = _GpuArray(f32_to_bf16_bits(w_head))
    glab = _GpuArray(labels.astype(np.int32).reshape(M, 1))
    gLg  = _GpuArray.zeros((M, V), np.float32)        # materialized logits
    gp   = _GpuArray.zeros((M, 1), np.float32)
    gls  = _GpuArray.zeros((M, 1), np.float32)

    # logits = hidden @ w_head ; then logprob, lse
    ct.launch(stream_int, (M // GTM, V // GTN, 1), _gemm, (gh, gw, gLg, H // GTK, GTM, GTN, GTK))
    ct.launch(stream_int, (M // CTM, 1, 1), _ce_stats, (gLg, glab, gp, gls, V // TV))

    if advantage is None:
        cudart.cudaStreamSynchronize(stream_int)
        lp = gp.to_numpy().reshape(M)
        for g in (gh, gw, glab, gLg, gp, gls): g.free()
        return lp

    # backward: glogit (bf16) → dhidden = glogit @ Wᵀ, dW = hiddenᵀ @ glogit
    ga  = _GpuArray(advantage.astype(np.float32).reshape(M, 1))
    gG  = _GpuArray.zeros((M, V), np.uint16)                              # glogit bf16 bits
    gwT = _GpuArray(f32_to_bf16_bits(np.ascontiguousarray(w_head.T)))     # (V, H)
    ghT = _GpuArray(f32_to_bf16_bits(np.ascontiguousarray(hidden.T)))     # (H, M)
    gdh = _GpuArray.zeros((M, H), np.float32)
    gdw = _GpuArray.zeros((H, V), np.float32)

    ct.launch(stream_int, (M // CTM, 1, 1), _ce_grad, (gLg, gls, glab, ga, gG, V // TV, 1.0 / M))
    ct.launch(stream_int, (M // GTM, H // GTN, 1), _gemm, (gG, gwT, gdh, V // GTK, GTM, GTN, GTK))
    ct.launch(stream_int, (H // GTM, V // GTN, 1), _gemm, (ghT, gG, gdw, M // GTK, GTM, GTN, GTK))
    cudart.cudaStreamSynchronize(stream_int)

    lp = gp.to_numpy().reshape(M); dh = gdh.to_numpy(); dw = gdw.to_numpy()
    for g in (gh, gw, glab, gLg, gp, gls, ga, gG, gwT, ghT, gdh, gdw): g.free()
    return lp, dh, dw

# ── decode-grid variants: CTMb rows per block (Constant) ─────────────────────
# At decode Md=128 the CTM=64 vocab-streaming kernels launch only TWO blocks — 60 SMs idle on a
# 39 MB read (V=151936). These variants take the row-tile as a Constant so the decode engine runs
# CTMb=8 (16 blocks). Each ROW's reduction is its own lane and keeps the exact vb=0..V_BLOCKS-1
# online order ⇒ per-row results are BITWISE-identical to the CTM=64 training kernels (ratio=1
# safe — verified end-to-end in test_resident_moe_decode.py).

@ct.kernel(occupancy=2)
def _ce_stats_b(logits, labels, logprob, lse, V_BLOCKS: ct.Constant[int], CTMb: ct.Constant[int]):
    """_ce_stats with a Constant row-tile. Grid (M//CTMb,)."""
    mb = ct.bid(0)
    lbl = ct.load(labels, index=(mb, 0), shape=(CTMb, 1))
    m_run = ct.full((CTMb, 1), -1e38, ct.float32)
    s_run = ct.zeros((CTMb, 1), ct.float32)
    tgt   = ct.zeros((CTMb, 1), ct.float32)
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTMb, TV))
        m_new = ct.maximum(m_run, ct.max(lg, axis=-1, keepdims=True))
        s_run = s_run * ct.exp(m_run - m_new) + ct.sum(
            ct.exp(lg - ct.broadcast_to(m_new, (CTMb, TV))), axis=-1, keepdims=True)
        m_run = m_new
        v_idx = ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0)
        mask  = ct.equal(ct.broadcast_to(lbl,   (CTMb, TV)),
                         ct.broadcast_to(v_idx, (CTMb, TV)))
        tgt   = tgt + ct.sum(ct.where(mask, lg, 0.0), axis=-1, keepdims=True)
    lse_v = m_run + ct.log(s_run)
    ct.store(lse,     index=(mb, 0), tile=lse_v)
    ct.store(logprob, index=(mb, 0), tile=tgt - lse_v)


@ct.kernel(occupancy=2)
def _argmax_id_b(logits, out_id, V_BLOCKS: ct.Constant[int], CTMb: ct.Constant[int]):
    """_argmax_id with a Constant row-tile. Grid (M//CTMb,)."""
    mb = ct.bid(0)
    BIG = 1e30
    m_run = ct.full((CTMb, 1), -1e38, ct.float32)
    a_run = ct.zeros((CTMb, 1), ct.float32)
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTMb, TV))
        tile_max = ct.max(lg, axis=-1, keepdims=True)
        v_idx = ct.astype(ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0), ct.float32)
        is_max = ct.equal(lg, ct.broadcast_to(tile_max, (CTMb, TV)))
        tile_arg = ct.min(ct.where(is_max, ct.broadcast_to(v_idx, (CTMb, TV)),
                                   ct.full((CTMb, TV), BIG, ct.float32)), axis=-1, keepdims=True)
        take = ct.greater(tile_max, m_run)
        a_run = ct.where(take, tile_arg, a_run)
        m_run = ct.maximum(m_run, tile_max)
    ct.store(out_id, index=(mb, 0), tile=ct.astype(a_run, ct.int32))


@ct.kernel(occupancy=2)
def _ce_grad_b(logits, lse, labels, adv, glogit,
               V_BLOCKS: ct.Constant[int], inv_M: ct.Constant[float], CTMb: ct.Constant[int]):
    """_ce_grad with a Constant row-tile (training boundary: CTM=64 → 16 blocks at M=1024,
    ncu DRAM 17%/SM 5% — pure underfill on a ~1 GB stream; CTMb=4 → 256 blocks). Pure
    elementwise per row (no streamed reduce — the CTMb≤2 bit-flip hazard doesn't apply);
    grads only, lp untouched. Grid (M//CTMb,)."""
    mb = ct.bid(0)
    tlse = ct.load(lse,    index=(mb, 0), shape=(CTMb, 1))
    tlbl = ct.load(labels, index=(mb, 0), shape=(CTMb, 1))
    tadv = ct.load(adv,    index=(mb, 0), shape=(CTMb, 1))
    scl  = tadv * inv_M
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTMb, TV))
        p  = ct.exp(lg - ct.broadcast_to(tlse, (CTMb, TV)))
        v_idx = ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0)
        oneh  = ct.where(ct.equal(ct.broadcast_to(tlbl,  (CTMb, TV)),
                                  ct.broadcast_to(v_idx, (CTMb, TV))), 1.0, 0.0)
        g = ct.broadcast_to(scl, (CTMb, TV)) * (p - oneh)
        ct.store(glogit, index=(mb, vb), tile=ct.bitcast(ct.astype(g, ct.bfloat16), ct.uint16))


# ── DECODE-MEGAKERNEL fused pick+CE (2026-06-11): ONE pass over the (M,V) logits ──
# The decode boundary streamed the 78 MB logits TWICE (pick kernel, then _ce_stats_b on the
# picked id). These fuse both into one stream. Bitwise notes: the greedy target logit is the
# running max itself (lp = m_run − lse, the same f32 _ce_stats_b's masked-sum tgt extracts);
# the sampled target is captured at each take-tile by an exact index-match sum (no arithmetic
# on it). The lse stream is copied VERBATIM from _ce_stats_b but it IS a recompile → probed
# BITWISE vs the two-kernel path (_probe_decode_attn.py) before use.

@ct.kernel(occupancy=2)
def _argmax_ce_b(logits, out_id, logprob, lse, V_BLOCKS: ct.Constant[int], CTMb: ct.Constant[int]):
    """Greedy pick + that token's logprob in ONE logits pass (== _argmax_id_b → _ce_stats_b).
    Grid (M//CTMb,)."""
    mb = ct.bid(0)
    BIG = 1e30
    m_run = ct.full((CTMb, 1), -1e38, ct.float32)
    s_run = ct.zeros((CTMb, 1), ct.float32)
    a_run = ct.zeros((CTMb, 1), ct.float32)
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTMb, TV))
        tile_max = ct.max(lg, axis=-1, keepdims=True)
        v_idx = ct.astype(ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0), ct.float32)
        is_max = ct.equal(lg, ct.broadcast_to(tile_max, (CTMb, TV)))
        tile_arg = ct.min(ct.where(is_max, ct.broadcast_to(v_idx, (CTMb, TV)),
                                   ct.full((CTMb, TV), BIG, ct.float32)), axis=-1, keepdims=True)
        take = ct.greater(tile_max, m_run)
        a_run = ct.where(take, tile_arg, a_run)
        m_new = ct.maximum(m_run, tile_max)
        s_run = s_run * ct.exp(m_run - m_new) + ct.sum(
            ct.exp(lg - ct.broadcast_to(m_new, (CTMb, TV))), axis=-1, keepdims=True)
        m_run = m_new
    lse_v = m_run + ct.log(s_run)
    ct.store(out_id,  index=(mb, 0), tile=ct.astype(a_run, ct.int32))
    ct.store(lse,     index=(mb, 0), tile=lse_v)
    ct.store(logprob, index=(mb, 0), tile=m_run - lse_v)   # greedy target logit == the running max


@ct.kernel(occupancy=2)
def _sample_ce_b(logits, seed_buf, out_id, logprob, lse, V_BLOCKS: ct.Constant[int],
                 inv_T: ct.Constant[float], CTMb: ct.Constant[int]):
    """Gumbel-max sample + that token's logprob in ONE logits pass (== _sample_id_b →
    _ce_stats_b). The chosen RAW logit is captured at each take-tile by exact index match.
    Grid (M//CTMb,)."""
    mb = ct.bid(0)
    sd = ct.astype(ct.reshape(ct.load(seed_buf, index=(0, 0), shape=(1, 1)), ()), ct.uint32)
    BIG = 1e30
    ms_run = ct.full((CTMb, 1), -1e38, ct.float32)   # running max of the GUMBEL score
    a_run  = ct.zeros((CTMb, 1), ct.float32)
    cl_run = ct.zeros((CTMb, 1), ct.float32)         # raw logit of the current pick
    m_run  = ct.full((CTMb, 1), -1e38, ct.float32)   # CE stream (raw logits)
    s_run  = ct.zeros((CTMb, 1), ct.float32)
    rows = ct.broadcast_to(ct.reshape(ct.arange(CTMb, dtype=ct.uint32), (CTMb, 1)), (CTMb, TV)) \
        + ct.astype(mb, ct.uint32) * CTMb
    for vb in range(V_BLOCKS):
        lgr = ct.load(logits, index=(mb, vb), shape=(CTMb, TV))
        lg = lgr * inv_T
        cols = ct.astype(ct.broadcast_to(ct.reshape(ct.arange(TV, dtype=ct.int32), (1, TV)),
                                         (CTMb, TV)) + vb * TV, ct.uint32)
        h = ((rows * 73856093) ^ (cols * 19349663) ^ sd) * 0x9E3779B1
        h = h ^ (h >> 16); h = h * 0x85EBCA6B; h = h ^ (h >> 13)
        u = (ct.astype(h >> 8, ct.float32) + 0.5) * (1.0 / 16777216.0)
        s = lg + (0.0 - ct.log(0.0 - ct.log(u)))
        tile_max = ct.max(s, axis=-1, keepdims=True)
        v_idx = ct.astype(ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0), ct.float32)
        is_max = ct.equal(s, ct.broadcast_to(tile_max, (CTMb, TV)))
        tile_arg = ct.min(ct.where(is_max, ct.broadcast_to(v_idx, (CTMb, TV)),
                                   ct.full((CTMb, TV), BIG, ct.float32)), axis=-1, keepdims=True)
        take = ct.greater(tile_max, ms_run)
        a_run = ct.where(take, tile_arg, a_run)
        tile_lg = ct.sum(ct.where(ct.equal(ct.broadcast_to(v_idx, (CTMb, TV)),
                                           ct.broadcast_to(tile_arg, (CTMb, TV))), lgr, 0.0),
                         axis=-1, keepdims=True)
        cl_run = ct.where(take, tile_lg, cl_run)
        ms_run = ct.maximum(ms_run, tile_max)
        m_new = ct.maximum(m_run, ct.max(lgr, axis=-1, keepdims=True))
        s_run = s_run * ct.exp(m_run - m_new) + ct.sum(
            ct.exp(lgr - ct.broadcast_to(m_new, (CTMb, TV))), axis=-1, keepdims=True)
        m_run = m_new
    lse_v = m_run + ct.log(s_run)
    ct.store(out_id,  index=(mb, 0), tile=ct.astype(a_run, ct.int32))
    ct.store(lse,     index=(mb, 0), tile=lse_v)
    ct.store(logprob, index=(mb, 0), tile=cl_run - lse_v)


@ct.kernel(occupancy=2)
def _sample_id_b(logits, seed_buf, out_id, V_BLOCKS: ct.Constant[int], inv_T: ct.Constant[float],
                 CTMb: ct.Constant[int]):
    """_sample_id with a Constant row-tile. SAME coord hash (global row id = mb*CTMb + lane) =>
    the same (row, v, seed) draws the same Gumbel regardless of CTMb. Grid (M//CTMb,)."""
    mb = ct.bid(0)
    sd = ct.astype(ct.reshape(ct.load(seed_buf, index=(0, 0), shape=(1, 1)), ()), ct.uint32)
    BIG = 1e30
    m_run = ct.full((CTMb, 1), -1e38, ct.float32)
    a_run = ct.zeros((CTMb, 1), ct.float32)
    rows = ct.broadcast_to(ct.reshape(ct.arange(CTMb, dtype=ct.uint32), (CTMb, 1)), (CTMb, TV))         + ct.astype(mb, ct.uint32) * CTMb
    for vb in range(V_BLOCKS):
        lg = ct.load(logits, index=(mb, vb), shape=(CTMb, TV)) * inv_T
        cols = ct.astype(ct.broadcast_to(ct.reshape(ct.arange(TV, dtype=ct.int32), (1, TV)),
                                         (CTMb, TV)) + vb * TV, ct.uint32)
        h = ((rows * 73856093) ^ (cols * 19349663) ^ sd) * 0x9E3779B1
        h = h ^ (h >> 16); h = h * 0x85EBCA6B; h = h ^ (h >> 13)
        u = (ct.astype(h >> 8, ct.float32) + 0.5) * (1.0 / 16777216.0)
        s = lg + (0.0 - ct.log(0.0 - ct.log(u)))
        tile_max = ct.max(s, axis=-1, keepdims=True)
        v_idx = ct.astype(ct.expand_dims(ct.arange(TV, dtype=ct.int32) + vb * TV, 0), ct.float32)
        is_max = ct.equal(s, ct.broadcast_to(tile_max, (CTMb, TV)))
        tile_arg = ct.min(ct.where(is_max, ct.broadcast_to(v_idx, (CTMb, TV)),
                                   ct.full((CTMb, TV), BIG, ct.float32)), axis=-1, keepdims=True)
        take = ct.greater(tile_max, m_run)
        a_run = ct.where(take, tile_arg, a_run)
        m_run = ct.maximum(m_run, tile_max)
    ct.store(out_id, index=(mb, 0), tile=ct.astype(a_run, ct.int32))

