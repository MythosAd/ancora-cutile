"""
ancora/kernels/attention.py — Flash Attention forward, BF16, causal, GQA

Architecture: Qwen3-0.6B (head_dim=128, 16 Q heads, 8 KV heads, G=2)
Reference: gau-nernst/learn-cuda attention_v5 (94% SOL on RTX 5090 sm_120a)

Tile sizes for sm_120a + head_dim=128 (real Qwen3-0.6B; hardcoded — single model target):
  BQ=BKV=64, D=128  →  SMEM ≈ 48 KB  (at the 48 KB no-opt-in limit; gau-nernst 64×64×128 = 94% SOL)

Memory layout (flat 2D, strides computed by host):
  Q:   (B × Hq  × NQB,  D)   NQB  = Sq  // BQ
  K,V: (B × Hkv × NKVB, D)   NKVB = Skv // BKV
  O:   (B × Hq  × NQB,  D)

Grid: (NQB, B × Hq, 1)
  bid(0) = q block index
  bid(1) = hb = batch * Hq + q_head
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # sets CUDA_PATH

# ── tile constants ──────────────────────────────────────────────────────────
# 64×64 is optimal on sm_120 (confirmed: BQ=128 dropped 40→13 TFLOPS due to
# lower occupancy). Matches NVIDIA CUDA-Tile paper recommendation for RTX 50.
BQ  = 64   # query block size
BKV = 64   # key/value block size
D   = 128  # head dimension (Qwen3-0.6B real head_dim — hardcoded, single model target)


# ── forward kernel ──────────────────────────────────────────────────────────

# occupancy=2 found via ct.tune.exhaustive_search at S=2048 (best of 42 configs;
# occ 2-3 optimal, >=4 worse, num_ctas=2/clusters catastrophic). +12% vs auto.
@ct.kernel(occupancy=2)
def _attn_fwd(Q, K, V, O, L_out,
              NQB:   ct.Constant[int],    # Sq  // BQ
              NKVB:  ct.Constant[int],    # Skv // BKV
              Hq:    ct.Constant[int],    # num Q heads
              Hkv:   ct.Constant[int],    # num KV heads  (Hq // G for GQA)
              scale: ct.Constant[float]):
    """
    Online softmax Flash Attention, causal, GQA.

    For each (q_block, batch×q_head) block:
      1. Load Q tile once → registers
      2. For each KV block (past + diagonal only):
         a. Load K, V tiles
         b. S = Q @ K^T × scale
         c. Causal mask (set future to -inf)
         d. Online softmax update: m, l, O_acc
      3. Normalize O_acc / l → store
    """
    q_blk = ct.bid(0)
    hb    = ct.bid(1)   # hb = batch_id * Hq + q_head_id

    # GQA mapping: kv_head = q_head // G
    q_head  = hb % Hq
    batch   = hb // Hq
    kv_head = q_head * Hkv // Hq   # = q_head // G
    kv_hb   = batch * Hkv + kv_head

    q_row = hb * NQB + q_blk

    # ── Load Q tile once. Inputs are BF16-backed (uint16 bits) → bitcast (no value
    #    conversion). Native BF16 halves load bandwidth vs float32 (+27% at plateau). ──
    tQ = ct.bitcast(ct.load(Q, index=(q_row, 0), shape=(BQ, D)), ct.bfloat16)

    # ── Online softmax state, working in log2 units (exp2 = hardware ex2.approx).
    #    Fold softmax_scale * log2(e) into the scores once. exp(x)==exp2(x*log2e). ──
    LOG2E    = 1.4426950408889634
    qk_scale = scale * LOG2E
    NEG_INF  = -1e38

    m     = ct.full((BQ, 1), NEG_INF, ct.float32)
    l     = ct.zeros((BQ, 1),         ct.float32)
    O_acc = ct.zeros((BQ, D),         ct.float32)

    # Diagonal causal mask is the SAME lower-triangular pattern (i >= j) for every
    # diagonal block (since BQ == BKV) → precompute once, no per-iteration arange.
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)   # (BQ, BKV) lower-triangular

    # CAUSAL: only KV blocks 0..q_blk matter (future blocks contribute 0 → skip them).
    # Reduction order kv=0,1,...,q_blk is fixed → batch-invariant.
    for kv in range(q_blk + 1):
        kv_row = kv_hb * NKVB + kv

        # latency=10: hint heavy DRAM traffic → compiler prefetches next iter's K/V
        # aggressively, filling the softmax bubble. +13% at S=2048 (swept via experiment).
        tK = ct.bitcast(ct.load(K, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(V, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)

        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale

        # Only the diagonal block (kv == q_blk) needs masking; past blocks are full.
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)

        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))

        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
            ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new

    # ── Final normalisation + save logsumexp (base-2) for the backward ──
    result = O_acc / ct.broadcast_to(l, (BQ, D))
    ct.store(O, index=(q_row, 0), tile=result)
    # L = m + log2(l) in base-2 units. backward forms P = exp2(S_b2 - L) (= softmax).
    ct.store(L_out, index=(q_row, 0), tile=m + ct.log2(l))


# ── prefix-shared forward (GRPO / Prefix Grouper): shared prompt prefix, G completions ─────────
# GRPO makes G completions per prompt that share the prompt PREFIX. Naively the prefix is encoded G×.
# Here the prefix self-attention is computed ONCE (run _attn_fwd on the prompt), and each completion's
# SUFFIX query attends over the SHARED prefix KV (Kp/Vp, one copy) then its own suffix KV (Ks/Vs).
#
# BITWISE-equal to running standard _attn_fwd on the concatenated [prefix, suffix_i]: identical KV
# tiles, identical fixed kv order (prefix 0..NKVBp-1 then suffix 0..sq_blk), identical online-softmax
# math and diagonal mask. Combined with seq-len invariance (prefix output independent of suffix), the
# split reconstructs the full attention exactly ⇒ TRAINING-equivalent (Prefix Grouper, arXiv 2506.05433)
# ⇒ preserves ratio=1. Saves the (G-1)× redundant prefix self-attention + prefix KV storage.
@ct.kernel(occupancy=2)
def _attn_fwd_prefix(Qs, Kp, Vp, Ks, Vs, O, L_out,
                     NQBs:  ct.Constant[int],    # suffix query blocks  (Sc // BQ)
                     NKVBp: ct.Constant[int],    # prefix  KV blocks    (Sp // BKV)
                     NKVBs: ct.Constant[int],    # suffix  KV blocks    (Sc // BKV)
                     Hq:    ct.Constant[int],
                     Hkv:   ct.Constant[int],
                     scale: ct.Constant[float]):
    """Suffix query attends over SHARED prefix (Kp/Vp) then its own suffix (Ks/Vs, causal).
    Qs,O,Ks,Vs head-major per completion: (G*Hq*Sc | G*Hkv*Sc, D); Kp,Vp shared: (Hkv*Sp, D).
    Grid (NQBs, G*Hq), bid(1) = comp*Hq + q_head."""
    sq_blk = ct.bid(0)
    ghb    = ct.bid(1)
    q_head = ghb % Hq
    comp   = ghb // Hq
    kv_head = q_head * Hkv // Hq

    q_row = ghb * NQBs + sq_blk
    tQ = ct.bitcast(ct.load(Qs, index=(q_row, 0), shape=(BQ, D)), ct.bfloat16)

    LOG2E = 1.4426950408889634
    qk_scale = scale * LOG2E; NEG_INF = -1e38
    m = ct.full((BQ, 1), NEG_INF, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    # SHARED PREFIX blocks — every prefix position < every suffix position → full (no mask).
    for kvp in range(NKVBp):
        kp_row = kv_head * NKVBp + kvp
        tK = ct.bitcast(ct.load(Kp, index=(kp_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(Vp, index=(kp_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new

    # OWN SUFFIX blocks (per-completion) — causal within the suffix (mask the diagonal only).
    for kvs in range(sq_blk + 1):
        ks_row = (comp * Hkv + kv_head) * NKVBs + kvs
        tK = ct.bitcast(ct.load(Ks, index=(ks_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(Vs, index=(ks_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kvs == sq_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new

    result = O_acc / ct.broadcast_to(l, (BQ, D))
    ct.store(O, index=(q_row, 0), tile=result)
    ct.store(L_out, index=(q_row, 0), tile=m + ct.log2(l))


# ── token-major forward (TE strided-layout: read native layout, NO transposes) ──
# The head-major _attn_fwd needs 3 tok→head transposes (Q,K,V) + 1 head→tok (O) — pure
# memory-bound "other" traffic. TE reads the native (strided) layout directly; this kernel
# reads Q,K,V token-major (M, H*D) via (row-block, head) indexing and writes O token-major →
# the 4 transposes vanish. RoPE stays a SEPARATE token-major kernel (_rope_fwd_tok): folding
# it inline forces a 2× D-split MMA (32-deep, half-efficient) since cuda-tile can't concat the
# rotated halves back to a 64-wide tile — measured 58→33 TFLOPS, a net loss. One 64-wide MMA
# here keeps attention fast. Same causal/GQA/online-softmax, fixed kv order → batch-invariant.

@ct.kernel(occupancy=2)
def _attn_fwd_tok(Q, K, V, O, L_out,
                  NQB: ct.Constant[int], NKVB: ct.Constant[int],
                  Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float]):
    """Token-major flash attention (Q,K PRE-rotated). Q:(M,Hq*D), K,V:(M,Hkv*D) bf16 bits;
    O:(M,Hq*D) bf16 bits. Grid (NQB, B*Hq), bid(1)=batch*Hq+q_head."""
    q_blk = ct.bid(0); hb = ct.bid(1)
    q_head = hb % Hq; batch = hb // Hq
    kv_head = q_head * Hkv // Hq
    qr = batch * NQB + q_blk                 # token-major row-block (BQ rows for this batch)

    LOG2E = 1.4426950408889634
    qk_scale = scale * LOG2E; NEG_INF = -1e38

    tQ = ct.bitcast(ct.load(Q, index=(qr, q_head), shape=(BQ, D)), ct.bfloat16)   # one 64-wide load
    m = ct.full((BQ, 1), NEG_INF, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    for kv in range(q_blk + 1):
        kr = batch * NKVB + kv
        tK = ct.bitcast(ct.load(K, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(V, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new

    res = O_acc / ct.broadcast_to(l, (BQ, D))
    ct.store(O, index=(qr, q_head), tile=ct.bitcast(ct.astype(res, ct.bfloat16), ct.uint16))  # token-major bf16 bits
    ct.store(L_out, index=(hb * NQB + q_blk, 0), tile=m + ct.log2(l))


# ── FP8 attention (SageAttention-style) — Q,K,V,P in FP8 to lift attn's tensor peak 80→184 ──
# sm_120 throttles BF16+FP32-acc to 1/4 of FP8 (gemm_mfu_ceiling), so attention at ~58 TF (BF16)
# could ~2× in FP8 IF accuracy holds. Quantization chosen so the scales factor cleanly OUT of the
# matmuls (no scale inside a contraction):
#   • Q,K per-ROW (per-token) e4m3 scale → S_true[i,j] = mma(q8,k8ᵀ)[i,j]·sq[i]·sk[j]
#   • V per-COLUMN (per-d) e4m3 scale    → (P@V)[i,d] = mma(p8,v8)[i,d]·sv[d]   (sv factors out of Σ_j)
#   • P (softmax probs ∈[0,1]) quantized DIRECTLY to e4m3 (FP8-friendly: big weights kept, tiny ones ~0)
# E4M3_MAX=448. ct.mma takes FP8 inputs directly (f32 acc). PROTOTYPE — measure accuracy vs the BF16
# kernel before trusting it for training (batch-invariance also needs same precision both paths).
E4M3_MAX = 448.0

@ct.kernel(occupancy=2)
def _attn_fwd_tok_fp8(Q, K, V, O, L_out,
                      NQB: ct.Constant[int], NKVB: ct.Constant[int],
                      Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float]):
    """FP8 token-major flash attention (Q,K pre-rotated). Same layout/output as _attn_fwd_tok."""
    q_blk = ct.bid(0); hb = ct.bid(1)
    q_head = hb % Hq; batch = hb // Hq
    kv_head = q_head * Hkv // Hq
    qr = batch * NQB + q_blk
    LOG2E = 1.4426950408889634
    qk_scale = scale * LOG2E; NEG_INF = -1e38

    qf = ct.astype(ct.bitcast(ct.load(Q, index=(qr, q_head), shape=(BQ, D)), ct.bfloat16), ct.float32)
    sq = ct.max(ct.maximum(qf, 0.0 - qf), axis=-1, keepdims=True) * (1.0 / E4M3_MAX)   # (BQ,1) per-row
    q8 = ct.astype(qf / ct.broadcast_to(sq, (BQ, D)), ct.float8_e4m3fn)

    m = ct.full((BQ, 1), NEG_INF, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    for kv in range(q_blk + 1):
        kr = batch * NKVB + kv
        kf = ct.astype(ct.bitcast(ct.load(K, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16), ct.float32)
        sk = ct.max(ct.maximum(kf, 0.0 - kf), axis=-1, keepdims=True) * (1.0 / E4M3_MAX)   # (BKV,1) per-row
        k8 = ct.astype(kf / ct.broadcast_to(sk, (BKV, D)), ct.float8_e4m3fn)
        vf = ct.astype(ct.bitcast(ct.load(V, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16), ct.float32)
        sv = ct.max(ct.maximum(vf, 0.0 - vf), axis=0, keepdims=True) * (1.0 / E4M3_MAX)      # (1,D) per-column
        v8 = ct.astype(vf / ct.broadcast_to(sv, (BKV, D)), ct.float8_e4m3fn)

        # S_true = (q8@k8ᵀ)·sq·skᵀ·qk_scale  (sq over cols, sk over rows)
        S = ct.mma(q8, ct.transpose(k8), ct.zeros((BQ, BKV), ct.float32))
        S = S * ct.broadcast_to(sq, (BQ, BKV)) * ct.broadcast_to(ct.transpose(sk), (BQ, BKV)) * qk_scale
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))         # ∈ [0,1]
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        p8    = ct.astype(P, ct.float8_e4m3fn)                          # direct (P FP8-friendly)
        # (P@V)·sv  (sv per-column factors out)
        Ob    = ct.mma(p8, v8, ct.zeros((BQ, D), ct.float32)) * ct.broadcast_to(sv, (BQ, D))
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + Ob
        m = m_new

    res = O_acc / ct.broadcast_to(l, (BQ, D))
    ct.store(O, index=(qr, q_head), tile=ct.bitcast(ct.astype(res, ct.bfloat16), ct.uint16))
    ct.store(L_out, index=(hb * NQB + q_blk, 0), tile=m + ct.log2(l))


# Variant: fuse the o-proj input MXFP8 quant into attention's output store — attention writes
# FP8 + E8M0 directly (per-32 block: head's D=128 → 4 quant blocks via a (BQ,4,32) reshape),
# killing the separate _quant_mxfp8 kernel + the bf16 O round-trip. The o_proj reads these.
NB2 = D // 32   # quant blocks per head (=4 for D=128)

@ct.kernel(occupancy=2)
def _attn_fwd_tok_q(Q, K, V, O_fp8, O_scale, L_out,
                    NQB: ct.Constant[int], NKVB: ct.Constant[int],
                    Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float]):
    """Token-major flash attention with FUSED MXFP8 output quant. Q,K,V pre-rotated token-major.
    O_fp8:(M,Hq*D) u8, O_scale:(M,Hq*D//32) u8 (E8M0). Grid (NQB, B*Hq)."""
    q_blk = ct.bid(0); hb = ct.bid(1)
    q_head = hb % Hq; batch = hb // Hq
    kv_head = q_head * Hkv // Hq
    qr = batch * NQB + q_blk
    LOG2E = 1.4426950408889634
    qk_scale = scale * LOG2E; NEG_INF = -1e38

    tQ = ct.bitcast(ct.load(Q, index=(qr, q_head), shape=(BQ, D)), ct.bfloat16)
    m = ct.full((BQ, 1), NEG_INF, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    for kv in range(q_blk + 1):
        kr = batch * NKVB + kv
        tK = ct.bitcast(ct.load(K, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(V, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new

    res = O_acc / ct.broadcast_to(l, (BQ, D))                       # (BQ, D)
    r3 = ct.reshape(res, (BQ, NB2, 32))                             # per-32 blocks
    amax = ct.max(ct.maximum(r3, 0.0 - r3), axis=-1, keepdims=True) # (BQ, NB2, 1)
    ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
    byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((BQ, NB2, 1), 0, ct.uint32))
    sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
    fp8 = ct.bitcast(ct.reshape(ct.astype(r3 / ct.broadcast_to(sc, (BQ, NB2, 32)), ct.float8_e4m3fn), (BQ, D)), ct.uint8)
    ct.store(O_fp8,   index=(qr, q_head), tile=fp8)                          # (BQ, D) u8 at col-block q_head
    ct.store(O_scale, index=(qr, q_head), tile=ct.astype(ct.reshape(byte, (BQ, NB2)), ct.uint8))  # (BQ, NB2)
    ct.store(L_out, index=(hb * NQB + q_blk, 0), tile=m + ct.log2(l))


# ── backward: two-kernel split (Triton/Megatron-deterministic, no atomics) ───
# Kernel A (dQ): Q-parallel, loop kv in fixed order 0..q_blk → batch-invariant.
# Kernel B (dK,dV): KV-parallel, loop g over the G shared heads then q in fixed
#   order → batch-invariant. Both recompute S,P from saved L (no atomic dQ).
# Math (scale = 1/√d, applied at the end): P = exp2(S_b2 - L); dP = dO@Vᵀ;
#   D = rowsum(dO∘O); dS = P∘(dP - D); dQ = scale·dS@K; dK = scale·dSᵀ@Q; dV = Pᵀ@dO.

@ct.kernel
def _attn_bwd_dq(Q, K, V, dO, L, Dlt, dQ,
                 NQB: ct.Constant[int], NKVB: ct.Constant[int],
                 Hq: ct.Constant[int], Hkv: ct.Constant[int],
                 scale: ct.Constant[float]):
    """dQ[i] = scale · Σ_{kv≤i} dS[i,kv] @ K[kv].  Grid (NQB, B*Hq)."""
    q_blk, hb = ct.bid(0), ct.bid(1)
    q_head = hb % Hq
    batch  = hb // Hq
    kv_hb  = batch * Hkv + (q_head * Hkv // Hq)
    qr = hb * NQB + q_blk

    LOG2E = 1.4426950408889634
    qk = scale * LOG2E
    NEG = -1e38

    tQ   = ct.bitcast(ct.load(Q,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tdO  = ct.bitcast(ct.load(dO, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
    tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))

    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    acc = ct.zeros((BQ, D), ct.float32)
    for kv in range(q_blk + 1):
        kvr = kv_hb * NKVB + kv
        # K loaded TWICE: a loaded tile used BOTH transposed (for S) and not (for dS@K)
        # crashes the tile compiler. Two distinct loads avoid it (same data, cheap).
        tKt = ct.bitcast(ct.load(K, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)  # for S
        tK  = ct.bitcast(ct.load(K, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)  # for dS@K
        tV  = ct.bitcast(ct.load(V, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)
        S  = ct.mma(tQ, ct.transpose(tKt), ct.zeros((BQ, BKV), ct.float32)) * qk
        if kv == q_blk:
            S = ct.where(tri, S, NEG)
        P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
        dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
        dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
        acc = ct.mma(ct.astype(dS, ct.bfloat16), tK, acc)   # dS @ K
    ct.store(dQ, index=(qr, 0), tile=acc * scale)


@ct.kernel
def _attn_bwd_dkdv(Q, K, V, dO, L, Dlt, dK, dV,
                   NQB: ct.Constant[int], NKVB: ct.Constant[int],
                   Hq: ct.Constant[int], Hkv: ct.Constant[int], G: ct.Constant[int],
                   scale: ct.Constant[float]):
    """dK[j],dV[j] = Σ_g Σ_{i≥j} (...).  Grid (NKVB, B*Hkv). Sum over G shared heads."""
    kv_blk, kvhb = ct.bid(0), ct.bid(1)
    batch   = kvhb // Hkv
    kv_head = kvhb % Hkv
    kvr = kvhb * NKVB + kv_blk

    LOG2E = 1.4426950408889634
    qk = scale * LOG2E
    NEG = -1e38

    tK = ct.bitcast(ct.load(K, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)
    tV = ct.bitcast(ct.load(V, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)

    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    acc_dK = ct.zeros((BKV, D), ct.float32)
    acc_dV = ct.zeros((BKV, D), ct.float32)
    for g in range(G):                       # sum the G query-heads sharing this KV head
        hb = batch * Hq + kv_head * G + g
        for i in range(NQB):                  # causal: only q-blocks i >= kv_blk attend
            if i >= kv_blk:
                qr  = hb * NQB + i
                tQ   = ct.bitcast(ct.load(Q,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                tdO  = ct.bitcast(ct.load(dO, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
                tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))
                S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk
                if i == kv_blk:
                    S = ct.where(tri, S, NEG)
                P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
                Pb = ct.astype(P, ct.bfloat16)
                acc_dV = ct.mma(ct.transpose(Pb), tdO, acc_dV)        # Pᵀ @ dO
                dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
                dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
                acc_dK = ct.mma(ct.transpose(ct.astype(dS, ct.bfloat16)), tQ, acc_dK)  # dSᵀ @ Q
    ct.store(dK, index=(kvr, 0), tile=acc_dK * scale)
    ct.store(dV, index=(kvr, 0), tile=acc_dV)


# ── prefix-shared backward (GRPO) ───────────────────────────────────────────
# Decomposition (all fixed-order ⇒ batch-invariant; the prefix-grad group reduction is sequential,
# NO atomics — the key correctness/invariance requirement):
#   dQ_suffix  : new _attn_bwd_dq_prefix    — loops shared prefix kv (no mask) then own suffix (causal).
#                BITWISE == standard _attn_bwd_dq on [prefix,suffix_i] (same kv tiles/order).
#   dK/dV_suffix: REUSE _attn_bwd_dkdv on the suffix (B=G completions) with the full saved L — suffix
#                keys are attended only by suffix queries (causal), so this is exact + bitwise.
#   dQ_prefix  : REUSE _attn_bwd_dq on the prompt with dO_prefix (= Σ_i dO_full_i[prefix], the shared
#                prefix output's grad). dK/dV_prefix_self: REUSE _attn_bwd_dkdv on the prompt.
#   dK/dV_prefix_cross: new _attn_bwd_dkdv_prefix — every prefix key is attended by ALL G·GG·NQBs
#                suffix queries (no mask). dK/dV_prefix = self + cross (deterministic add).

@ct.kernel
def _attn_bwd_dq_prefix(Qs, Kp, Vp, Ks, Vs, dOs, L, Dlt, dQs,
                        NQBs: ct.Constant[int], NKVBp: ct.Constant[int], NKVBs: ct.Constant[int],
                        Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float]):
    """Suffix dQ[i] = scale·(Σ_{prefix kv} dS@Kp + Σ_{suffix kv≤i} dS@Ks). Grid (NQBs, G*Hq)."""
    sq_blk, ghb = ct.bid(0), ct.bid(1)
    q_head = ghb % Hq; comp = ghb // Hq
    kv_head = q_head * Hkv // Hq
    qr = ghb * NQBs + sq_blk
    LOG2E = 1.4426950408889634; qk = scale * LOG2E; NEG = -1e38
    tQ   = ct.bitcast(ct.load(Qs,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tdO  = ct.bitcast(ct.load(dOs, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
    tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)
    acc = ct.zeros((BQ, D), ct.float32)
    for kvp in range(NKVBp):                                # shared prefix — no mask
        kpr = kv_head * NKVBp + kvp
        tKt = ct.bitcast(ct.load(Kp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
        tK  = ct.bitcast(ct.load(Kp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
        tV  = ct.bitcast(ct.load(Vp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
        S  = ct.mma(tQ, ct.transpose(tKt), ct.zeros((BQ, BKV), ct.float32)) * qk
        P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
        dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
        dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
        acc = ct.mma(ct.astype(dS, ct.bfloat16), tK, acc)
    for kvs in range(sq_blk + 1):                           # own suffix — causal
        ksr = (comp * Hkv + kv_head) * NKVBs + kvs
        tKt = ct.bitcast(ct.load(Ks, index=(ksr, 0), shape=(BKV, D)), ct.bfloat16)
        tK  = ct.bitcast(ct.load(Ks, index=(ksr, 0), shape=(BKV, D)), ct.bfloat16)
        tV  = ct.bitcast(ct.load(Vs, index=(ksr, 0), shape=(BKV, D)), ct.bfloat16)
        S  = ct.mma(tQ, ct.transpose(tKt), ct.zeros((BQ, BKV), ct.float32)) * qk
        if kvs == sq_blk:
            S = ct.where(tri, S, NEG)
        P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
        dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
        dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
        acc = ct.mma(ct.astype(dS, ct.bfloat16), tK, acc)
    ct.store(dQs, index=(qr, 0), tile=acc * scale)


@ct.kernel
def _attn_bwd_dkdv_prefix(Qs, Kp, Vp, dOs, L, Dlt, dKp, dVp,
                          NQBs: ct.Constant[int], NKVBp: ct.Constant[int],
                          Hq: ct.Constant[int], Hkv: ct.Constant[int],
                          NCOMP: ct.Constant[int], GG: ct.Constant[int], scale: ct.Constant[float]):
    """Prefix CROSS dK/dV[j] = Σ_comp Σ_g Σ_{all suffix q-blocks} (no mask — every suffix query attends
    every prefix key). Fixed comp→g→block order ⇒ batch-invariant. Grid (NKVBp, Hkv). GG = Hq//Hkv."""
    kvp_blk, kv_head = ct.bid(0), ct.bid(1)
    kpr = kv_head * NKVBp + kvp_blk
    LOG2E = 1.4426950408889634; qk = scale * LOG2E
    tK = ct.bitcast(ct.load(Kp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
    tV = ct.bitcast(ct.load(Vp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
    acc_dK = ct.zeros((BKV, D), ct.float32)
    acc_dV = ct.zeros((BKV, D), ct.float32)
    for comp in range(NCOMP):
        for g in range(GG):
            ghb = comp * Hq + kv_head * GG + g
            for i in range(NQBs):
                qr   = ghb * NQBs + i
                tQ   = ct.bitcast(ct.load(Qs,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                tdO  = ct.bitcast(ct.load(dOs, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
                tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))
                S  = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk
                P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
                Pb = ct.astype(P, ct.bfloat16)
                acc_dV = ct.mma(ct.transpose(Pb), tdO, acc_dV)
                dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
                dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
                acc_dK = ct.mma(ct.transpose(ct.astype(dS, ct.bfloat16)), tQ, acc_dK)
    ct.store(dKp, index=(kpr, 0), tile=acc_dK * scale)
    ct.store(dVp, index=(kpr, 0), tile=acc_dV)


# ── sliding-window (LOCAL) attention — fwd + bwd ─────────────────────────────
# Gemma-3/MAI local layers: query i attends to the W most-recent keys (i-W, i] (causal AND
# i-j < W). W is a multiple of BKV (win_blocks = W//BKV). Separate kernels (not a flag on
# _attn_fwd) so the ~10 existing direct launchers stay untouched.
#
# COMPUTE-OPTIMIZED (O(window), not O(q_blk)): a query block q_blk only touches the win_blocks+1
# blocks [q_blk-win_blocks, q_blk]. So loop a STATIC range(win_blocks+1) (small, e.g. 9 for W=512)
# computing kv = q_blk-win_blocks+w, guarded by `if kv>=0` (drops negative blocks for early q_blk).
# For long S this skips the whole O(q_blk) tail instead of masking it. Static unroll → the `if kv>=0`
# conditional carry compiles (like dkdv); the two edge masks are compile-time `if w==…`:
#   w==win_blocks → diagonal (causal tri);  w==0 → window lower edge (keep key-col>query-row, b>a).
# m_init=-1e9 (≫ the -1e38 mask) so a fully-masked edge row gives exp2(-1e38+1e9)=0 — no spurious mass.
# Position t sees a FIXED window regardless of total S → seq-len/batch invariant (RL ratio=1).

@ct.kernel(occupancy=2)
def _attn_fwd_win(Q, K, V, O, L_out,
                  NQB: ct.Constant[int], NKVB: ct.Constant[int],
                  Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float],
                  win_blocks: ct.Constant[int]):
    """Sliding-window causal flash attention. Same layout/output as _attn_fwd."""
    q_blk = ct.bid(0); hb = ct.bid(1)
    q_head = hb % Hq; batch = hb // Hq
    kv_head = q_head * Hkv // Hq; kv_hb = batch * Hkv + kv_head
    q_row = hb * NQB + q_blk
    tQ = ct.bitcast(ct.load(Q, index=(q_row, 0), shape=(BQ, D)), ct.bfloat16)

    LOG2E = 1.4426950408889634; qk_scale = scale * LOG2E; NEG_INF = -1e38
    M_INIT = -1e9                    # ≫ the -1e38 mask → fully-masked edge rows give exp2≈0, no spurious mass
    m = ct.full((BQ, 1), M_INIT, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)   # causal: query-row >= key-col
    win = ct.greater(jj, ii)         # window lower edge: key-col > query-row

    for w in range(win_blocks + 1):              # only the in-window blocks (STATIC, win_blocks+1)
        kv = q_blk - win_blocks + w
        if kv >= 0:                              # drop negative blocks (early query blocks)
            kv_row = kv_hb * NKVB + kv
            tK = ct.bitcast(ct.load(K, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            tV = ct.bitcast(ct.load(V, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
            if w == 0:
                S = ct.where(win, S, NEG_INF)    # window lower edge (b>a); skipped when kv_lo<0
            if w == win_blocks:
                S = ct.where(tri, S, NEG_INF)    # diagonal: causal
            m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
            alpha = ct.exp2(m - m_new)
            P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
            l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
            O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
                ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
            m = m_new
    ct.store(O, index=(q_row, 0), tile=O_acc / ct.broadcast_to(l, (BQ, D)))
    ct.store(L_out, index=(q_row, 0), tile=m + ct.log2(l))


@ct.kernel
def _attn_bwd_dq_win(Q, K, V, dO, L, Dlt, dQ,
                     NQB: ct.Constant[int], NKVB: ct.Constant[int],
                     Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float],
                     win_blocks: ct.Constant[int]):
    """dQ for sliding-window attention. Grid (NQB, B*Hq)."""
    q_blk, hb = ct.bid(0), ct.bid(1)
    q_head = hb % Hq; batch = hb // Hq
    kv_hb = batch * Hkv + (q_head * Hkv // Hq)
    qr = hb * NQB + q_blk
    LOG2E = 1.4426950408889634; qk = scale * LOG2E; NEG = -1e38

    tQ   = ct.bitcast(ct.load(Q,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tdO  = ct.bitcast(ct.load(dO, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
    tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj); win = ct.greater(jj, ii)

    acc = ct.zeros((BQ, D), ct.float32)
    for w in range(win_blocks + 1):              # only the in-window blocks (O(window))
        kv = q_blk - win_blocks + w
        if kv >= 0:
            kvr = kv_hb * NKVB + kv
            tKt = ct.bitcast(ct.load(K, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)
            tK  = ct.bitcast(ct.load(K, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)
            tV  = ct.bitcast(ct.load(V, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)
            S  = ct.mma(tQ, ct.transpose(tKt), ct.zeros((BQ, BKV), ct.float32)) * qk
            if w == 0:
                S = ct.where(win, S, NEG)
            if w == win_blocks:
                S = ct.where(tri, S, NEG)
            P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))   # masked entries: S=-1e38 → P=0 → dS=0
            dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
            dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
            acc = ct.mma(ct.astype(dS, ct.bfloat16), tK, acc)
    ct.store(dQ, index=(qr, 0), tile=acc * scale)


@ct.kernel
def _attn_bwd_dkdv_win(Q, K, V, dO, L, Dlt, dK, dV,
                       NQB: ct.Constant[int], NKVB: ct.Constant[int],
                       Hq: ct.Constant[int], Hkv: ct.Constant[int], G: ct.Constant[int],
                       scale: ct.Constant[float], win_blocks: ct.Constant[int]):
    """dK,dV for sliding-window attention. Grid (NKVB, B*Hkv). A kv-block contributes to query
    block i only if kv_blk <= i <= kv_blk+win_blocks (causal AND within window)."""
    kv_blk, kvhb = ct.bid(0), ct.bid(1)
    batch = kvhb // Hkv; kv_head = kvhb % Hkv
    kvr = kvhb * NKVB + kv_blk
    LOG2E = 1.4426950408889634; qk = scale * LOG2E; NEG = -1e38

    tK = ct.bitcast(ct.load(K, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)
    tV = ct.bitcast(ct.load(V, index=(kvr, 0), shape=(BKV, D)), ct.bfloat16)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj); win = ct.greater(jj, ii)

    acc_dK = ct.zeros((BKV, D), ct.float32); acc_dV = ct.zeros((BKV, D), ct.float32)
    for g in range(G):
        hb = batch * Hq + kv_head * G + g
        for w in range(win_blocks + 1):          # only query blocks within window of this kv block
            i = kv_blk + w                        # causal (i>=kv_blk) AND within window (i<=kv_blk+W)
            if i < NQB:                           # don't run past the sequence end
                qr = hb * NQB + i
                tQ   = ct.bitcast(ct.load(Q,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                tdO  = ct.bitcast(ct.load(dO, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
                tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))
                S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk
                if w == 0:
                    S = ct.where(tri, S, NEG)                # i == kv_blk → diagonal causal
                if w == win_blocks:
                    S = ct.where(win, S, NEG)                # i == kv_blk+W → window lower edge (KV side)
                P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
                Pb = ct.astype(P, ct.bfloat16)
                acc_dV = ct.mma(ct.transpose(Pb), tdO, acc_dV)
                dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
                dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
                acc_dK = ct.mma(ct.transpose(ct.astype(dS, ct.bfloat16)), tQ, acc_dK)
    ct.store(dK, index=(kvr, 0), tile=acc_dK * scale)
    ct.store(dV, index=(kvr, 0), tile=acc_dV)


# ── WINDOWED prefix-shared attention (GRPO on LOCAL sliding-window layers) ──────────────────
# Local layers (RoPE + window W): a suffix query at global pos Sp+t sees keys (Sp+t-W, Sp+t] —
# the window can SPAN the prefix/suffix boundary, so the suffix query reads the LAST ≤W prompt
# keys (shared) + its own in-window suffix keys. Same trick as _attn_fwd_prefix, but the prefix
# loop is bounded by the window instead of running all NKVBp blocks.
#
# BITWISE equivalence with _attn_fwd_win on [prefix, suffix_i]: with BQ==BKV the suffix query
# block sq_blk is GLOBAL block NKVBp+sq_blk, and the naive kernel's loop index w (kv = q_blk -
# win_blocks + w) maps to the SAME w here — loop 1 takes the kv < NKVBp part (prefix tiles),
# loop 2 the kv ≥ NKVBp part (suffix tiles), in the same ascending order with masks at the same
# w (w==0 window lower edge, w==win_blocks causal diagonal — the diagonal is always in loop 2
# since prefix keys are strictly past). Same M_INIT=-1e9 as _attn_fwd_win. Compound runtime
# guards `(kv >= 0) and (kv < NB)` verified exact in tests/kernels/_probe_prefix_win.py.

@ct.kernel(occupancy=2)
def _attn_fwd_prefix_win(Qs, Kp, Vp, Ks, Vs, O, L_out,
                         NQBs:  ct.Constant[int],   # suffix query blocks (Sc // BQ)
                         NKVBp: ct.Constant[int],   # prefix  KV blocks   (Sp // BKV)
                         NKVBs: ct.Constant[int],   # suffix  KV blocks   (Sc // BKV)
                         Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float],
                         win_blocks: ct.Constant[int]):
    """Suffix query, sliding-window over [shared prefix; own suffix]. Layouts == _attn_fwd_prefix.
    Grid (NQBs, G*Hq)."""
    sq_blk = ct.bid(0); ghb = ct.bid(1)
    q_head = ghb % Hq; comp = ghb // Hq
    kv_head = q_head * Hkv // Hq
    q_row = ghb * NQBs + sq_blk
    tQ = ct.bitcast(ct.load(Qs, index=(q_row, 0), shape=(BQ, D)), ct.bfloat16)

    LOG2E = 1.4426950408889634; qk_scale = scale * LOG2E; NEG_INF = -1e38
    M_INIT = -1e9                                   # == _attn_fwd_win (bitwise)
    m = ct.full((BQ, 1), M_INIT, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj); win = ct.greater(jj, ii)

    # loop 1 — PREFIX tiles inside the window (kv = NKVBp+sq_blk-win_blocks+w < NKVBp), no causal
    for w in range(win_blocks + 1):
        kvp = NKVBp + sq_blk - win_blocks + w
        if (kvp >= 0) and (kvp < NKVBp):
            kp_row = kv_head * NKVBp + kvp
            tK = ct.bitcast(ct.load(Kp, index=(kp_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            tV = ct.bitcast(ct.load(Vp, index=(kp_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
            if w == 0:
                S = ct.where(win, S, NEG_INF)       # window lower edge falls in the prefix
            m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
            alpha = ct.exp2(m - m_new)
            P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
            l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
            O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
                ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
            m = m_new
    # loop 2 — OWN SUFFIX tiles inside the window (same w → same masks as _attn_fwd_win)
    for w in range(win_blocks + 1):
        kvs = sq_blk - win_blocks + w
        if kvs >= 0:
            ks_row = (comp * Hkv + kv_head) * NKVBs + kvs
            tK = ct.bitcast(ct.load(Ks, index=(ks_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            tV = ct.bitcast(ct.load(Vs, index=(ks_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
            if w == 0:
                S = ct.where(win, S, NEG_INF)       # window lower edge inside the suffix
            if w == win_blocks:
                S = ct.where(tri, S, NEG_INF)       # causal diagonal (kvs == sq_blk)
            m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
            alpha = ct.exp2(m - m_new)
            P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
            l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
            O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
                ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
            m = m_new
    ct.store(O, index=(q_row, 0), tile=O_acc / ct.broadcast_to(l, (BQ, D)))
    ct.store(L_out, index=(q_row, 0), tile=m + ct.log2(l))


@ct.kernel
def _attn_bwd_dq_prefix_win(Qs, Kp, Vp, Ks, Vs, dOs, L, Dlt, dQs,
                            NQBs: ct.Constant[int], NKVBp: ct.Constant[int], NKVBs: ct.Constant[int],
                            Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float],
                            win_blocks: ct.Constant[int]):
    """Suffix dQ for the windowed prefix attention — the two in-window loops of the fwd, dq body
    of _attn_bwd_dq_win. Grid (NQBs, G*Hq)."""
    sq_blk, ghb = ct.bid(0), ct.bid(1)
    q_head = ghb % Hq; comp = ghb // Hq
    kv_head = q_head * Hkv // Hq
    qr = ghb * NQBs + sq_blk
    LOG2E = 1.4426950408889634; qk = scale * LOG2E; NEG = -1e38
    tQ   = ct.bitcast(ct.load(Qs,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tdO  = ct.bitcast(ct.load(dOs, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
    tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
    tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj); win = ct.greater(jj, ii)
    acc = ct.zeros((BQ, D), ct.float32)
    for w in range(win_blocks + 1):                 # PREFIX tiles in window
        kvp = NKVBp + sq_blk - win_blocks + w
        if (kvp >= 0) and (kvp < NKVBp):
            kpr = kv_head * NKVBp + kvp
            tKt = ct.bitcast(ct.load(Kp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
            tK  = ct.bitcast(ct.load(Kp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
            tV  = ct.bitcast(ct.load(Vp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
            S  = ct.mma(tQ, ct.transpose(tKt), ct.zeros((BQ, BKV), ct.float32)) * qk
            if w == 0:
                S = ct.where(win, S, NEG)
            P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
            dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
            dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
            acc = ct.mma(ct.astype(dS, ct.bfloat16), tK, acc)
    for w in range(win_blocks + 1):                 # OWN SUFFIX tiles in window
        kvs = sq_blk - win_blocks + w
        if kvs >= 0:
            ksr = (comp * Hkv + kv_head) * NKVBs + kvs
            tKt = ct.bitcast(ct.load(Ks, index=(ksr, 0), shape=(BKV, D)), ct.bfloat16)
            tK  = ct.bitcast(ct.load(Ks, index=(ksr, 0), shape=(BKV, D)), ct.bfloat16)
            tV  = ct.bitcast(ct.load(Vs, index=(ksr, 0), shape=(BKV, D)), ct.bfloat16)
            S  = ct.mma(tQ, ct.transpose(tKt), ct.zeros((BQ, BKV), ct.float32)) * qk
            if w == 0:
                S = ct.where(win, S, NEG)
            if w == win_blocks:
                S = ct.where(tri, S, NEG)
            P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
            dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
            dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
            acc = ct.mma(ct.astype(dS, ct.bfloat16), tK, acc)
    ct.store(dQs, index=(qr, 0), tile=acc * scale)


@ct.kernel
def _attn_bwd_dkdv_prefix_win(Qs, Kp, Vp, dOs, L, Dlt, dKp, dVp,
                              NQBs: ct.Constant[int], NKVBp: ct.Constant[int],
                              Hq: ct.Constant[int], Hkv: ct.Constant[int],
                              NCOMP: ct.Constant[int], GG: ct.Constant[int], scale: ct.Constant[float],
                              win_blocks: ct.Constant[int]):
    """Prefix CROSS dK/dV under a window: prefix key block kvp is attended only by suffix query
    blocks sq = kvp+w-NKVBp for w ≤ win_blocks (global qg = kvp+w). w==win_blocks carries the
    KV-side window lower-edge mask (== _attn_bwd_dkdv_win); the causal diagonal can't occur
    (prefix keys are strictly past every suffix query). Fixed comp→g→w order ⇒ batch-invariant.
    Grid (NKVBp, Hkv)."""
    kvp_blk, kv_head = ct.bid(0), ct.bid(1)
    kpr = kv_head * NKVBp + kvp_blk
    LOG2E = 1.4426950408889634; qk = scale * LOG2E; NEG = -1e38
    tK = ct.bitcast(ct.load(Kp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
    tV = ct.bitcast(ct.load(Vp, index=(kpr, 0), shape=(BKV, D)), ct.bfloat16)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    win = ct.greater(jj, ii)
    acc_dK = ct.zeros((BKV, D), ct.float32); acc_dV = ct.zeros((BKV, D), ct.float32)
    for comp in range(NCOMP):
        for g in range(GG):
            ghb = comp * Hq + kv_head * GG + g
            for w in range(win_blocks + 1):
                sq = kvp_blk + w - NKVBp            # suffix query block at global distance w
                if (sq >= 0) and (sq < NQBs):
                    qr   = ghb * NQBs + sq
                    tQ   = ct.bitcast(ct.load(Qs,  index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                    tdO  = ct.bitcast(ct.load(dOs, index=(qr, 0), shape=(BQ, D)), ct.bfloat16)
                    tL   = ct.load(L,   index=(qr, 0), shape=(BQ, 1))
                    tDlt = ct.load(Dlt, index=(qr, 0), shape=(BQ, 1))
                    S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk
                    if w == win_blocks:
                        S = ct.where(win, S, NEG)   # window lower edge (KV side)
                    P  = ct.exp2(S - ct.broadcast_to(tL, (BQ, BKV)))
                    Pb = ct.astype(P, ct.bfloat16)
                    acc_dV = ct.mma(ct.transpose(Pb), tdO, acc_dV)
                    dP = ct.mma(tdO, ct.transpose(tV), ct.zeros((BQ, BKV), ct.float32))
                    dS = P * (dP - ct.broadcast_to(tDlt, (BQ, BKV)))
                    acc_dK = ct.mma(ct.transpose(ct.astype(dS, ct.bfloat16)), tQ, acc_dK)
    ct.store(dKp, index=(kpr, 0), tile=acc_dK * scale)
    ct.store(dVp, index=(kpr, 0), tile=acc_dV)


# ── decode attention (KV-cache, one query/sequence) ─────────────────────────
# FlashInfer/vLLM-style decode: the frontier token's query attends to the whole
# cached K/V (prompt + previously generated). Q is 1 row per (seq, head) → no MMA,
# just dot-products + online softmax over the cache (memory-bound, latency path).
# Loop the FULL preallocated cache (NKVB Constant) and mask positions ≥ S_cur, which
# is a RUNTIME scalar (the cache grows every step → must not recompile). Masked
# positions get exp2(-inf)=0, so they don't contribute.

@ct.kernel
def _attn_decode(Q, Kc, Vc, O,
                 NKVB: ct.Constant[int],     # maxS // BKV  (cache blocks per (seq,kv-head))
                 Hq: ct.Constant[int], Hkv: ct.Constant[int],
                 scale: ct.Constant[float],
                 S_cur: int):                # runtime: valid cache length
    """O[b,qh] = softmax(q·Kcacheᵀ·scale) · Vcache over the first S_cur positions.
    Grid (B, Hq). Q:(B*Hq, D); Kc,Vc:(B*Hkv*maxS, D) flat, block-strided by maxS."""
    b, qh = ct.bid(0), ct.bid(1)
    kvh   = qh * Hkv // Hq
    qrow  = b * Hq + qh
    cbase = (b * Hkv + kvh) * NKVB           # cache base in BLOCK units

    tq = ct.astype(ct.bitcast(ct.load(Q, index=(qrow, 0), shape=(1, D)), ct.bfloat16), ct.float32)
    LOG2E = 1.4426950408889634
    qk  = scale * LOG2E
    NEG = -1e38
    m   = ct.full((1, 1), NEG, ct.float32)
    l   = ct.zeros((1, 1), ct.float32)
    acc = ct.zeros((1, D), ct.float32)
    jj  = ct.expand_dims(ct.arange(BKV, dtype=ct.int32), -1)    # (BKV,1) within-block offset

    for kv in range(NKVB):
        Kt = ct.astype(ct.bitcast(ct.load(Kc, index=(cbase + kv, 0), shape=(BKV, D)), ct.bfloat16), ct.float32)
        s  = ct.sum(ct.broadcast_to(tq, (BKV, D)) * Kt, axis=-1, keepdims=True) * qk   # (BKV,1) q·k
        pos = jj + kv * BKV                                       # (BKV,1) absolute position
        s  = ct.where(ct.less(pos, ct.full((BKV, 1), S_cur, ct.int32)), s, NEG)        # mask ≥ S_cur
        m_new = ct.maximum(m, ct.max(s, axis=0, keepdims=True))   # (1,1)
        alpha = ct.exp2(m - m_new)
        p  = ct.exp2(s - ct.broadcast_to(m_new, (BKV, 1)))        # (BKV,1)
        l  = alpha * l + ct.sum(p, axis=0, keepdims=True)
        Vt = ct.astype(ct.bitcast(ct.load(Vc, index=(cbase + kv, 0), shape=(BKV, D)), ct.bfloat16), ct.float32)
        acc = ct.broadcast_to(alpha, (1, D)) * acc + ct.sum(ct.broadcast_to(p, (BKV, D)) * Vt, axis=0, keepdims=True)
        m  = m_new
    ct.store(O, index=(qrow, 0), tile=acc / ct.broadcast_to(l, (1, D)))


# ── UNIFIED decode (rollout==training): reuse the PREFILL reduction ──────────
# _attn_decode above uses ct.sum dot-products (CUDA-core) → a DIFFERENT reduction order
# than _attn_fwd's ct.mma (tensor-core) → ~0.2% off → π_train≠π_infer. A *single-row* MMA
# (1,D)×(D,BKV) is ALSO not bitwise: the tensor-core lowering for M=1 differs from row r of
# an M=BQ MMA by ~1 ULP (tested). So this kernel runs the FULL BQ-query block, IDENTICAL to
# _attn_fwd, with the active block index q_blk passed at RUNTIME — the frontier query's row
# (pmod) is then computed inside the same BQ-row MMA as training ⇒ BITWISE-identical
# (matmul rows are independent, so the other BQ-1 rows can be garbage). q_blk runtime →
# data-dependent range(q_blk+1), same trick _attn_fwd uses with bid(0). Cost: BQ× the
# attention work per step (only row pmod is kept) — correctness-first; optimize later.

@ct.kernel(occupancy=2)
def _attn_decode_blk(Q, Kc, Vc, O,
                     NKVB: ct.Constant[int], Hq: ct.Constant[int], Hkv: ct.Constant[int],
                     scale: ct.Constant[float], q_blk: int):
    """Full BQ-query block at runtime block q_blk over the cache; MMA reduction IDENTICAL to
    _attn_fwd → the frontier query's row is bitwise-equal to prefill. Grid (B*Hq,).
    Q,O: (B*Hq*BQ, D) — frontier query at in-block row pmod=pos%BQ; other rows are unused."""
    hb = ct.bid(0)                              # batch*Hq + q_head
    q_head  = hb % Hq
    batch   = hb // Hq
    kv_head = q_head * Hkv // Hq
    kv_hb   = batch * Hkv + kv_head
    tQ = ct.bitcast(ct.load(Q, index=(hb, 0), shape=(BQ, D)), ct.bfloat16)

    LOG2E = 1.4426950408889634
    qk_scale = scale * LOG2E
    NEG_INF = -1e38
    m     = ct.full((BQ, 1), NEG_INF, ct.float32)
    l     = ct.zeros((BQ, 1),         ct.float32)
    O_acc = ct.zeros((BQ, D),         ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    for kv in range(q_blk + 1):
        kv_row = kv_hb * NKVB + kv
        tK = ct.bitcast(ct.load(Kc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(Vc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
            ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new
    ct.store(O, index=(hb, 0), tile=O_acc / ct.broadcast_to(l, (BQ, D)))


@ct.kernel(occupancy=2)
def _attn_decode_blk_win(Q, Kc, Vc, O,
                         NKVB: ct.Constant[int], Hq: ct.Constant[int], Hkv: ct.Constant[int],
                         scale: ct.Constant[float], win_blocks: ct.Constant[int], q_blk: int):
    """Sliding-window decode: the _attn_decode_blk trick (full BQ-block at runtime q_blk, frontier
    row pmod) with _attn_fwd_win's loop structure — only the win_blocks+1 in-window cache blocks,
    SAME masks (w==0 window edge / w==win_blocks causal diagonal) and M_INIT=-1e9 ⇒ the frontier
    row is BITWISE-equal to the windowed prefill row at pos. O(window) per token regardless of
    context length (the MoE local layers' decode win). Grid (B*Hq,)."""
    hb = ct.bid(0)
    q_head  = hb % Hq
    batch   = hb // Hq
    kv_hb   = batch * Hkv + (q_head * Hkv // Hq)
    tQ = ct.bitcast(ct.load(Q, index=(hb, 0), shape=(BQ, D)), ct.bfloat16)

    LOG2E = 1.4426950408889634; qk_scale = scale * LOG2E; NEG_INF = -1e38
    M_INIT = -1e9                                   # == _attn_fwd_win (bitwise)
    m = ct.full((BQ, 1), M_INIT, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj); win = ct.greater(jj, ii)

    for w in range(win_blocks + 1):
        kv = q_blk - win_blocks + w
        if kv >= 0:
            kv_row = kv_hb * NKVB + kv
            tK = ct.bitcast(ct.load(Kc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            tV = ct.bitcast(ct.load(Vc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
            if w == 0:
                S = ct.where(win, S, NEG_INF)       # window lower edge
            if w == win_blocks:
                S = ct.where(tri, S, NEG_INF)       # causal diagonal (kv == q_blk)
            m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
            alpha = ct.exp2(m - m_new)
            P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
            l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
            O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
                ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
            m = m_new
    ct.store(O, index=(hb, 0), tile=O_acc / ct.broadcast_to(l, (BQ, D)))


@ct.kernel
def _scatter_blk(src, dst, BQ_: ct.Constant[int]):
    """Place the decode frontier query into a BQ-block at in-block row pmod, on-device.
    src (R, D) → dst (R*BQ_, D) at row r*BQ_ (dst data ptr PRE-OFFSET by pmod rows on host, like
    _append_kv) → absolute row r*BQ_+pmod. Other block rows keep their (harmless) stale values.
    Feeds _attn_decode_blk so the frontier attention is bitwise-equal to prefill. Grid (R,)."""
    r = ct.bid(0)
    ct.store(dst, index=(r * BQ_, 0), tile=ct.load(src, index=(r, 0), shape=(1, D)))


@ct.kernel
def _gather_blk(src, dst, BQ_: ct.Constant[int]):
    """Inverse of _scatter_blk: pull the frontier row pmod out of each BQ-block. src (R*BQ_, D)
    PRE-OFFSET by pmod rows on host → dst (R, D)[r] = src[r*BQ_+pmod]. f32 (attention O). Grid (R,)."""
    r = ct.bid(0)
    ct.store(dst, index=(r, 0), tile=ct.load(src, index=(r * BQ_, 0), shape=(1, D)))


@ct.kernel
def _append_kv(Knew, Vnew, Kc, Vc,
               MAXS: ct.Constant[int], Hkv: ct.Constant[int]):
    """Write the new token's K,V (one row per (seq,kv-head)) into the sequence-major
    cache. Knew,Vnew:(B*Hkv, D) uint16; Kc,Vc:(B*Hkv*MAXS, D) uint16 views ALREADY
    OFFSET by `pos` rows on the host (data ptr += pos*D). Grid (B, Hkv). Pure bit-copy.

    The host pointer-offset (instead of a runtime `pos` arg) keeps the in-kernel index
    `bid`-only — clean, and sidesteps any runtime-scalar-in-store-index question.
    DETERMINISM CAVEAT (the real gotcha): the per-token staging upload MUST be on the
    SAME stream as this append (cuMemcpyHtoDAsync(.., si)). Uploading on the default
    stream while appending on `si` is a cross-stream race → the cache is intermittently
    corrupt (~15% wrong, flaky across runs), which no amount of cudaDeviceSynchronize
    fixes. Same lesson as the to_numpy() stream pitfall in CLAUDE.md."""
    b, kvh = ct.bid(0), ct.bid(1)
    src = b * Hkv + kvh
    dst = src * MAXS
    ct.store(Kc, index=(dst, 0), tile=ct.load(Knew, index=(src, 0), shape=(1, D)))
    ct.store(Vc, index=(dst, 0), tile=ct.load(Vnew, index=(src, 0), shape=(1, D)))


# ── DEVICE-POSITION decode kernels (CUDA-graph replayable) ───────────────────
# A captured graph freezes host at_pos pointers and runtime args (q_blk) — these variants read
# `pos` from a (1,1) i32 device buffer instead (scalar store-index / bit-ops / data-dependent loop
# bounds all probed OK, tests/kernels/_probe_devpos.py). BQ=BKV=64: block = pos>>6, in-block row =
# pos&63. CMASK/SMASK support the local-layer RING cache (CROWS = pow2-blocks x BKV rows, mask =
# CROWS-1 / blocks-1); the full cache passes an all-ones mask so `&` is the identity. Ring safety:
# the window reads only the last win_blocks+1 logical blocks <= ring blocks, and the current
# block's not-yet-written rows are causally masked (exp2(-inf)=0) — identical VALUES to the full
# cache => bitwise-equal attention.

@ct.kernel
def _append_kv_p(Knew, Vnew, Kc, Vc, gpos,
                 CROWS: ct.Constant[int], Hkv: ct.Constant[int], CMASK: ct.Constant[int]):
    """Cache append at device pos: row = pos & CMASK within each (seq,head)'s CROWS rows."""
    b, kvh = ct.bid(0), ct.bid(1)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    src = b * Hkv + kvh
    dst = src * CROWS + (pos & CMASK)
    ct.store(Kc, index=(dst, 0), tile=ct.load(Knew, index=(src, 0), shape=(1, D)))
    ct.store(Vc, index=(dst, 0), tile=ct.load(Vnew, index=(src, 0), shape=(1, D)))


@ct.kernel
def _scatter_blk_p(src, dst, gpos, BQ_: ct.Constant[int]):
    """_scatter_blk with the in-block row pmod = pos & (BQ_-1) from the device. Grid (R,)."""
    r = ct.bid(0)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    ct.store(dst, index=(r * BQ_ + (pos & (BQ_ - 1)), 0), tile=ct.load(src, index=(r, 0), shape=(1, D)))


@ct.kernel
def _gather_blk_p(src, dst, gpos, BQ_: ct.Constant[int]):
    """_gather_blk with device pmod. Grid (R,)."""
    r = ct.bid(0)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    ct.store(dst, index=(r, 0), tile=ct.load(src, index=(r * BQ_ + (pos & (BQ_ - 1)), 0), shape=(1, D)))


@ct.kernel(occupancy=2)
def _attn_decode_blk_pd(Q, Kc, Vc, O, gpos,
                        NKVB: ct.Constant[int], Hq: ct.Constant[int], Hkv: ct.Constant[int],
                        scale: ct.Constant[float]):
    """_attn_decode_blk with q_blk = (device pos) >> 6 — full-causal decode over the cache,
    frontier row bitwise == _attn_fwd row at pos. Grid (B*Hq,)."""
    hb = ct.bid(0)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    q_blk = pos >> 6
    q_head  = hb % Hq
    batch   = hb // Hq
    kv_hb   = batch * Hkv + (q_head * Hkv // Hq)
    tQ = ct.bitcast(ct.load(Q, index=(hb, 0), shape=(BQ, D)), ct.bfloat16)
    LOG2E = 1.4426950408889634; qk_scale = scale * LOG2E; NEG_INF = -1e38
    m = ct.full((BQ, 1), NEG_INF, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)
    for kv in range(q_blk + 1):
        kv_row = kv_hb * NKVB + kv
        tK = ct.bitcast(ct.load(Kc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(Vc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
            ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new
    ct.store(O, index=(hb, 0), tile=O_acc / ct.broadcast_to(l, (BQ, D)))


@ct.kernel(occupancy=2)
def _attn_decode_blk_win_pd(Q, Kc, Vc, O, gpos,
                            NRB: ct.Constant[int], Hq: ct.Constant[int], Hkv: ct.Constant[int],
                            scale: ct.Constant[float], win_blocks: ct.Constant[int],
                            SMASK: ct.Constant[int]):
    """_attn_decode_blk_win with device pos + RING cache: logical block kv maps to ring slot
    kv & SMASK within each (seq,head)'s NRB blocks (full cache: SMASK all-ones, NRB = maxS//BKV).
    Frontier row bitwise == _attn_fwd_win row at pos. Grid (B*Hq,)."""
    hb = ct.bid(0)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    q_blk = pos >> 6
    q_head  = hb % Hq
    batch   = hb // Hq
    kv_hb   = batch * Hkv + (q_head * Hkv // Hq)
    tQ = ct.bitcast(ct.load(Q, index=(hb, 0), shape=(BQ, D)), ct.bfloat16)
    LOG2E = 1.4426950408889634; qk_scale = scale * LOG2E; NEG_INF = -1e38
    M_INIT = -1e9                                   # == _attn_fwd_win (bitwise)
    m = ct.full((BQ, 1), M_INIT, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj); win = ct.greater(jj, ii)
    for w in range(win_blocks + 1):
        kv = q_blk - win_blocks + w
        if kv >= 0:
            kv_row = kv_hb * NRB + (kv & SMASK)
            tK = ct.bitcast(ct.load(Kc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            tV = ct.bitcast(ct.load(Vc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
            if w == 0:
                S = ct.where(win, S, NEG_INF)
            if w == win_blocks:
                S = ct.where(tri, S, NEG_INF)
            m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
            alpha = ct.exp2(m - m_new)
            P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
            l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
            O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(
                ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
            m = m_new
    ct.store(O, index=(hb, 0), tile=O_acc / ct.broadcast_to(l, (BQ, D)))


# ── GQA-PAIRED decode attention (decode megakernel, 2026-06-11) ───────────────
# Decode attention is KV-cache-BANDWIDTH-bound (~800 GB/s) and the GQA group (Hq/Hkv = 2)
# makes the single-head kernels read each K/V tile TWICE. These process both q heads of a
# kv group per block (grid (B*Hkv,)): ONE K/V load, two independent online-softmax states —
# halves the dominant traffic. Per-head math is the verbatim single-head body on the same
# tiles, but it IS a recompile → must be probed BITWISE vs the single-head kernels before
# use (_probe_decode_attn.py; the 1-ULP FMA-contraction precedent).

@ct.kernel(occupancy=2)
def _attn_decode_blk_pd2(Q, Kc, Vc, O, gpos,
                         NKVB: ct.Constant[int], Hq: ct.Constant[int], Hkv: ct.Constant[int],
                         scale: ct.Constant[float]):
    """_attn_decode_blk_pd for BOTH q heads of one kv group (GQA group must be 2).
    Grid (B*Hkv,)."""
    kb = ct.bid(0)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    q_blk = pos >> 6
    kvh   = kb % Hkv
    batch = kb // Hkv
    kv_hb = batch * Hkv + kvh
    hb0 = batch * Hq + kvh * 2
    hb1 = hb0 + 1
    tQ0 = ct.bitcast(ct.load(Q, index=(hb0, 0), shape=(BQ, D)), ct.bfloat16)
    tQ1 = ct.bitcast(ct.load(Q, index=(hb1, 0), shape=(BQ, D)), ct.bfloat16)
    LOG2E = 1.4426950408889634; qk_scale = scale * LOG2E; NEG_INF = -1e38
    m0 = ct.full((BQ, 1), NEG_INF, ct.float32); l0 = ct.zeros((BQ, 1), ct.float32)
    m1 = ct.full((BQ, 1), NEG_INF, ct.float32); l1 = ct.zeros((BQ, 1), ct.float32)
    O0 = ct.zeros((BQ, D), ct.float32); O1 = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)
    for kv in range(q_blk + 1):
        kv_row = kv_hb * NKVB + kv
        tK = ct.bitcast(ct.load(Kc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(Vc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
        S0 = ct.mma(tQ0, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        S1 = ct.mma(tQ1, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kv == q_blk:
            S0 = ct.where(tri, S0, NEG_INF)
            S1 = ct.where(tri, S1, NEG_INF)
        m0n = ct.maximum(m0, ct.max(S0, axis=-1, keepdims=True))
        a0  = ct.exp2(m0 - m0n)
        P0  = ct.exp2(S0 - ct.broadcast_to(m0n, (BQ, BKV)))
        l0  = a0 * l0 + ct.sum(P0, axis=-1, keepdims=True)
        O0  = ct.broadcast_to(a0, (BQ, D)) * O0 + ct.mma(
            ct.astype(P0, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m0 = m0n
        m1n = ct.maximum(m1, ct.max(S1, axis=-1, keepdims=True))
        a1  = ct.exp2(m1 - m1n)
        P1  = ct.exp2(S1 - ct.broadcast_to(m1n, (BQ, BKV)))
        l1  = a1 * l1 + ct.sum(P1, axis=-1, keepdims=True)
        O1  = ct.broadcast_to(a1, (BQ, D)) * O1 + ct.mma(
            ct.astype(P1, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m1 = m1n
    ct.store(O, index=(hb0, 0), tile=O0 / ct.broadcast_to(l0, (BQ, D)))
    ct.store(O, index=(hb1, 0), tile=O1 / ct.broadcast_to(l1, (BQ, D)))


@ct.kernel(occupancy=2)
def _attn_decode_blk_win_pd2(Q, Kc, Vc, O, gpos,
                             NRB: ct.Constant[int], Hq: ct.Constant[int], Hkv: ct.Constant[int],
                             scale: ct.Constant[float], win_blocks: ct.Constant[int],
                             SMASK: ct.Constant[int]):
    """_attn_decode_blk_win_pd for both q heads of one kv group (windowed + ring cache).
    Grid (B*Hkv,)."""
    kb = ct.bid(0)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    q_blk = pos >> 6
    kvh   = kb % Hkv
    batch = kb // Hkv
    kv_hb = batch * Hkv + kvh
    hb0 = batch * Hq + kvh * 2
    hb1 = hb0 + 1
    tQ0 = ct.bitcast(ct.load(Q, index=(hb0, 0), shape=(BQ, D)), ct.bfloat16)
    tQ1 = ct.bitcast(ct.load(Q, index=(hb1, 0), shape=(BQ, D)), ct.bfloat16)
    LOG2E = 1.4426950408889634; qk_scale = scale * LOG2E; NEG_INF = -1e38
    M_INIT = -1e9                                   # == _attn_fwd_win (bitwise)
    m0 = ct.full((BQ, 1), M_INIT, ct.float32); l0 = ct.zeros((BQ, 1), ct.float32)
    m1 = ct.full((BQ, 1), M_INIT, ct.float32); l1 = ct.zeros((BQ, 1), ct.float32)
    O0 = ct.zeros((BQ, D), ct.float32); O1 = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj); win = ct.greater(jj, ii)
    for w in range(win_blocks + 1):
        kv = q_blk - win_blocks + w
        if kv >= 0:
            kv_row = kv_hb * NRB + (kv & SMASK)
            tK = ct.bitcast(ct.load(Kc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            tV = ct.bitcast(ct.load(Vc, index=(kv_row, 0), shape=(BKV, D), latency=10), ct.bfloat16)
            S0 = ct.mma(tQ0, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
            S1 = ct.mma(tQ1, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
            if w == 0:
                S0 = ct.where(win, S0, NEG_INF)
                S1 = ct.where(win, S1, NEG_INF)
            if w == win_blocks:
                S0 = ct.where(tri, S0, NEG_INF)
                S1 = ct.where(tri, S1, NEG_INF)
            m0n = ct.maximum(m0, ct.max(S0, axis=-1, keepdims=True))
            a0  = ct.exp2(m0 - m0n)
            P0  = ct.exp2(S0 - ct.broadcast_to(m0n, (BQ, BKV)))
            l0  = a0 * l0 + ct.sum(P0, axis=-1, keepdims=True)
            O0  = ct.broadcast_to(a0, (BQ, D)) * O0 + ct.mma(
                ct.astype(P0, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
            m0 = m0n
            m1n = ct.maximum(m1, ct.max(S1, axis=-1, keepdims=True))
            a1  = ct.exp2(m1 - m1n)
            P1  = ct.exp2(S1 - ct.broadcast_to(m1n, (BQ, BKV)))
            l1  = a1 * l1 + ct.sum(P1, axis=-1, keepdims=True)
            O1  = ct.broadcast_to(a1, (BQ, D)) * O1 + ct.mma(
                ct.astype(P1, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
            m1 = m1n
    ct.store(O, index=(hb0, 0), tile=O0 / ct.broadcast_to(l0, (BQ, D)))
    ct.store(O, index=(hb1, 0), tile=O1 / ct.broadcast_to(l1, (BQ, D)))


# ── host helpers ────────────────────────────────────────────────────────────

class _GpuArray:
    def __init__(self, arr: np.ndarray):
        self._shape  = arr.shape
        self._dtype  = arr.dtype
        self._nbytes = arr.nbytes
        err, self._ptr = cdrv.cuMemAlloc(arr.nbytes)
        if err.value:
            raise RuntimeError(f"cuMemAlloc: {err}")
        cdrv.cuMemcpyHtoD(self._ptr, arr, arr.nbytes)
        self.__cuda_array_interface__ = {
            "shape": arr.shape, "typestr": arr.dtype.str,
            "data": (int(self._ptr), False), "version": 3,
        }

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, self._dtype)
        cdrv.cuMemcpyDtoH(out, self._ptr, self._nbytes)
        return out

    def free(self): cdrv.cuMemFree(self._ptr)

    @classmethod
    def from_numpy(cls, arr): return cls(arr)

    @classmethod
    def zeros(cls, shape, dtype=np.float32): return cls(np.zeros(shape, dtype))


def _f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 → bf16 bits as uint16 (truncate low 16 bits). numpy has no bf16,
    so we carry BF16 as its uint16 bit pattern and bitcast inside the kernel."""
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _prep_qkv(x: np.ndarray, num_blocks: int, block_size: int) -> np.ndarray:
    """
    Reshape (B, H, S, D) → flat (B*H*S, D) uint16 (BF16 bits) for ct.load tiling.
    Consecutive `block_size` rows form one tile; ct.load(index=(row,0), shape=(bs,D)).
    """
    B, H, S, d = x.shape
    assert S == num_blocks * block_size
    return _f32_to_bf16_bits(x.reshape(B * H * num_blocks * block_size, d))


# ── public API ──────────────────────────────────────────────────────────────

def flash_attn_forward(
    Q: np.ndarray,   # (B, Hq,  Sq,  D) float32
    K: np.ndarray,   # (B, Hkv, Skv, D) float32
    V: np.ndarray,   # (B, Hkv, Skv, D) float32
    stream_int: int,
    causal: bool = True,
    return_lse: bool = False,
    window: int = 0,   # >0: sliding-window causal (local layer); 0: full causal (global)
):
    """
    Runs ct.mma-based Flash Attention on sm_120a. Returns O: (B, Hq, Sq, D) float32.
    If return_lse: returns (O, L) where L: (B, Hq, Sq) base-2 logsumexp (for backward).
    window>0 → LOCAL sliding-window attention (query i sees keys (i-window, i]); must be %64.

    Requirements: Sq,Skv % 64 == 0, D == 128, Hq % Hkv == 0 (GQA).
    """
    B, Hq, Sq, d = Q.shape
    _, Hkv, Skv, _ = K.shape
    assert d == D, f"head_dim must be {D}, got {d}"
    assert Sq  % BQ  == 0 and Skv % BKV == 0 and Hq % Hkv == 0

    NQB, NKVB = Sq // BQ, Skv // BKV
    scale = float(1.0 / math.sqrt(D))

    gQ = _GpuArray(_prep_qkv(Q, NQB, BQ))
    gK = _GpuArray(_prep_qkv(K, NKVB, BKV))
    gV = _GpuArray(_prep_qkv(V, NKVB, BKV))
    gO = _GpuArray(np.zeros((B * Hq * Sq, D), np.float32))
    gL = _GpuArray(np.zeros((B * Hq * Sq, 1), np.float32))

    if window and window > 0:
        assert window % BKV == 0, f"window {window} must be a multiple of {BKV}"
        ct.launch(stream_int, (NQB, B * Hq, 1), _attn_fwd_win,
                  (gQ, gK, gV, gO, gL, NQB, NKVB, Hq, Hkv, scale, window // BKV))
    else:
        ct.launch(stream_int, (NQB, B * Hq, 1), _attn_fwd,
                  (gQ, gK, gV, gO, gL, NQB, NKVB, Hq, Hkv, scale))
    cudart.cudaStreamSynchronize(stream_int)

    O = gO.to_numpy().reshape(B, Hq, Sq, D)
    L = gL.to_numpy().reshape(B, Hq, Sq)
    for g in (gQ, gK, gV, gO, gL): g.free()
    return (O, L) if return_lse else O


def flash_attn_forward_prefix(pq, pk, pv, sq, sk, sv, stream_int, window: int = 0):
    """Prefix-shared forward (GRPO). pq/pk/pv: prompt PREFIX (Hq|Hkv, Sp, D) f32 (ALREADY RoPE'd at
    positions 0..Sp-1). sq/sk/sv: per-completion SUFFIX (G, Hq|Hkv, Sc, D) f32 (RoPE'd at Sp..Sp+Sc-1).
    Runs _attn_fwd once on the prompt (shared) + _attn_fwd_prefix on the G suffixes. Returns
    (Op (Hq,Sp,D), Os (G,Hq,Sc,D), Lp (Hq,Sp), Ls (G,Hq,Sc)) — bitwise-equal to standard attn on
    each concatenated [prefix, suffix_i] (test_attn_prefix.py).
    window>0 → LOCAL sliding-window layer (MoE local; query i sees keys (i-window, i], %64):
    prompt self via _attn_fwd_win, suffixes via _attn_fwd_prefix_win — bitwise vs _attn_fwd_win
    on each [prefix, suffix_i] (test_attn_prefix_win.py)."""
    Hq, Sp, d = pq.shape; Hkv = pk.shape[0]; G, _, Sc, _ = sq.shape
    assert d == D and Sp % BKV == 0 and Sc % BQ == 0
    NQBs, NKVBp, NKVBs = Sc // BQ, Sp // BKV, Sc // BKV
    scale = float(1.0 / math.sqrt(D))
    Op, Lp = flash_attn_forward(pq[None], pk[None], pv[None], stream_int, return_lse=True, window=window)
    gQs = _GpuArray(_f32_to_bf16_bits(sq.reshape(G * Hq * Sc, D)))
    gKp = _GpuArray(_f32_to_bf16_bits(pk.reshape(Hkv * Sp, D))); gVp = _GpuArray(_f32_to_bf16_bits(pv.reshape(Hkv * Sp, D)))
    gKs = _GpuArray(_f32_to_bf16_bits(sk.reshape(G * Hkv * Sc, D))); gVs = _GpuArray(_f32_to_bf16_bits(sv.reshape(G * Hkv * Sc, D)))
    gO = _GpuArray(np.zeros((G * Hq * Sc, D), np.float32)); gL = _GpuArray(np.zeros((G * Hq * Sc, 1), np.float32))
    if window and window > 0:
        assert window % BKV == 0, f"window {window} must be a multiple of {BKV}"
        ct.launch(stream_int, (NQBs, G * Hq, 1), _attn_fwd_prefix_win,
                  (gQs, gKp, gVp, gKs, gVs, gO, gL, NQBs, NKVBp, NKVBs, Hq, Hkv, scale, window // BKV))
    else:
        ct.launch(stream_int, (NQBs, G * Hq, 1), _attn_fwd_prefix,
                  (gQs, gKp, gVp, gKs, gVs, gO, gL, NQBs, NKVBp, NKVBs, Hq, Hkv, scale))
    cudart.cudaStreamSynchronize(stream_int)
    Os = gO.to_numpy().reshape(G, Hq, Sc, D); Ls = gL.to_numpy().reshape(G, Hq, Sc)
    for g in (gQs, gKp, gVp, gKs, gVs, gO, gL): g.free()
    return Op[0], Os, Lp[0], Ls


def flash_attn_backward_prefix(pq, pk, pv, sq, sk, sv, Op, Os, Lp, Ls, dOp, dOs, stream_int, window: int = 0):
    """Prefix-shared backward (GRPO). prompt PREFIX pq/pk/pv (Hq|Hkv,Sp,D), Op (Hq,Sp,D), Lp (Hq,Sp),
    dOp (Hq,Sp,D)=Σ_i dO_i[prefix] (shared prompt-output grad); suffix sq/sk/sv (G,Hq|Hkv,Sc,D), Os/Ls/
    dOs per-completion. Returns (dQp,dKp,dVp, dQs,dKs,dVs). The prompt KV grad is self + Σ_G cross
    (fixed-order) → training-equivalent (test_attn_prefix_bwd.py: suffix grads bitwise vs standard).
    window>0 → LOCAL sliding-window layer: windowed prefix-cross/dq kernels + windowed self parts
    (must equal the forward's window)."""
    Hq, Sp, d = pq.shape; Hkv = pk.shape[0]; G, _, Sc, _ = sq.shape; GG = Hq // Hkv
    NQBs, NKVBp, NKVBs = Sc // BQ, Sp // BKV, Sc // BKV; scale = float(1.0 / math.sqrt(D))
    Ds = (Os * dOs).sum(-1)                                              # suffix Delta (full O)
    gQs = _GpuArray(_f32_to_bf16_bits(sq.reshape(G * Hq * Sc, D)))
    gKp = _GpuArray(_f32_to_bf16_bits(pk.reshape(Hkv * Sp, D))); gVp = _GpuArray(_f32_to_bf16_bits(pv.reshape(Hkv * Sp, D)))
    gKs = _GpuArray(_f32_to_bf16_bits(sk.reshape(G * Hkv * Sc, D))); gVs = _GpuArray(_f32_to_bf16_bits(sv.reshape(G * Hkv * Sc, D)))
    gdOs = _GpuArray(_f32_to_bf16_bits(dOs.reshape(G * Hq * Sc, D)))
    gLs = _GpuArray(Ls.reshape(G * Hq * Sc, 1).astype(np.float32)); gDs = _GpuArray(Ds.reshape(G * Hq * Sc, 1).astype(np.float32))
    gdQs = _GpuArray(np.zeros((G * Hq * Sc, D), np.float32))
    gdKpc = _GpuArray(np.zeros((Hkv * Sp, D), np.float32)); gdVpc = _GpuArray(np.zeros((Hkv * Sp, D), np.float32))
    if window and window > 0:
        assert window % BKV == 0, f"window {window} must be a multiple of {BKV}"
        wb = window // BKV
        ct.launch(stream_int, (NQBs, G * Hq, 1), _attn_bwd_dq_prefix_win,
                  (gQs, gKp, gVp, gKs, gVs, gdOs, gLs, gDs, gdQs, NQBs, NKVBp, NKVBs, Hq, Hkv, scale, wb))
        ct.launch(stream_int, (NKVBp, Hkv, 1), _attn_bwd_dkdv_prefix_win,
                  (gQs, gKp, gVp, gdOs, gLs, gDs, gdKpc, gdVpc, NQBs, NKVBp, Hq, Hkv, G, GG, scale, wb))
    else:
        ct.launch(stream_int, (NQBs, G * Hq, 1), _attn_bwd_dq_prefix,
                  (gQs, gKp, gVp, gKs, gVs, gdOs, gLs, gDs, gdQs, NQBs, NKVBp, NKVBs, Hq, Hkv, scale))
        ct.launch(stream_int, (NKVBp, Hkv, 1), _attn_bwd_dkdv_prefix,
                  (gQs, gKp, gVp, gdOs, gLs, gDs, gdKpc, gdVpc, NQBs, NKVBp, Hq, Hkv, G, GG, scale))
    cudart.cudaStreamSynchronize(stream_int)
    dQs = gdQs.to_numpy().reshape(G, Hq, Sc, D)
    dKpc = gdKpc.to_numpy().reshape(Hkv, Sp, D); dVpc = gdVpc.to_numpy().reshape(Hkv, Sp, D)
    for gg in (gQs, gKp, gVp, gKs, gVs, gdOs, gLs, gDs, gdQs, gdKpc, gdVpc): gg.free()
    # suffix dK/dV: suffix keys are attended only by (in-window) suffix queries → standard dkdv, B=G
    _, dKs, dVs = flash_attn_backward(sq, sk, sv, Os, dOs, Ls, stream_int, window=window)
    # dQ_prompt + dK/dV_prompt_SELF (prompt self-attn) with the shared prompt-output grad dOp
    dQp, dKps, dVps = flash_attn_backward(pq[None], pk[None], pv[None], Op[None], dOp[None], Lp[None],
                                          stream_int, window=window)
    return dQp[0], dKps[0] + dKpc, dVps[0] + dVpc, dQs, dKs, dVs       # prompt grad = self + cross


def flash_attn_backward(Q, K, V, O, dO, L, stream_int, window: int = 0):
    """
    Q,K,V: (B,Hq/Hkv,S,D) f32;  O,dO: (B,Hq,Sq,D) f32;  L: (B,Hq,Sq) base-2 lse (from fwd).
    Returns (dQ (B,Hq,Sq,D), dK (B,Hkv,Skv,D), dV (B,Hkv,Skv,D)) float32.
    Two batch-invariant kernels (no atomics). Delta = rowsum(O∘dO) computed on host.
    window>0 → matches the LOCAL sliding-window forward (must equal the fwd window).
    """
    B, Hq, Sq, d = Q.shape
    _, Hkv, Skv, _ = K.shape
    G = Hq // Hkv
    NQB, NKVB = Sq // BQ, Skv // BKV
    scale = float(1.0 / math.sqrt(D))

    Dlt = (O * dO).sum(-1)   # (B, Hq, Sq)  softmax-Jacobian correction

    gQ  = _GpuArray(_prep_qkv(Q,  NQB,  BQ))
    gK  = _GpuArray(_prep_qkv(K,  NKVB, BKV))
    gV  = _GpuArray(_prep_qkv(V,  NKVB, BKV))
    gdO = _GpuArray(_prep_qkv(dO, NQB,  BQ))
    gL  = _GpuArray(L.reshape(B * Hq * Sq, 1).astype(np.float32))
    gD  = _GpuArray(Dlt.reshape(B * Hq * Sq, 1).astype(np.float32))
    gdQ = _GpuArray(np.zeros((B * Hq  * Sq,  D), np.float32))
    gdK = _GpuArray(np.zeros((B * Hkv * Skv, D), np.float32))
    gdV = _GpuArray(np.zeros((B * Hkv * Skv, D), np.float32))

    if window and window > 0:
        assert window % BKV == 0, f"window {window} must be a multiple of {BKV}"
        wb = window // BKV
        ct.launch(stream_int, (NQB, B * Hq, 1), _attn_bwd_dq_win,
                  (gQ, gK, gV, gdO, gL, gD, gdQ, NQB, NKVB, Hq, Hkv, scale, wb))
        ct.launch(stream_int, (NKVB, B * Hkv, 1), _attn_bwd_dkdv_win,
                  (gQ, gK, gV, gdO, gL, gD, gdK, gdV, NQB, NKVB, Hq, Hkv, G, scale, wb))
    else:
        ct.launch(stream_int, (NQB, B * Hq, 1), _attn_bwd_dq,
                  (gQ, gK, gV, gdO, gL, gD, gdQ, NQB, NKVB, Hq, Hkv, scale))
        ct.launch(stream_int, (NKVB, B * Hkv, 1), _attn_bwd_dkdv,
                  (gQ, gK, gV, gdO, gL, gD, gdK, gdV, NQB, NKVB, Hq, Hkv, G, scale))
    cudart.cudaStreamSynchronize(stream_int)

    dQ = gdQ.to_numpy().reshape(B, Hq,  Sq,  D)
    dK = gdK.to_numpy().reshape(B, Hkv, Skv, D)
    dV = gdV.to_numpy().reshape(B, Hkv, Skv, D)
    for g in (gQ, gK, gV, gdO, gL, gD, gdQ, gdK, gdV): g.free()
    return dQ, dK, dV
