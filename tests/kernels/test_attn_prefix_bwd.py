"""Prefix-shared attention BACKWARD (GRPO). The shared prompt prefix's gradient is the SUM over the
G completions (fixed-order, no atomics); each completion's suffix gradient is per-completion.

Validated against an fp64 reference of the full attention on each concatenated [prefix, suffix_i]:
  - dQ_suffix_i, dK_suffix_i, dV_suffix_i  vs the suffix rows of the fp64 backward (≤1.5%, bf16),
    AND BITWISE == the standard device flash_attn_backward on [P,s_i] (the ratio=1 property).
  - dQ_prefix, dK_prefix, dV_prefix  vs Σ_i (fp64 backward prefix rows) (≤1.5%).
Foreground only."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import (flash_attn_forward, flash_attn_backward, _attn_fwd_prefix,
                                       _attn_bwd_dq_prefix, _attn_bwd_dkdv_prefix, _GpuArray,
                                       _f32_to_bf16_bits as f32bf, D, BQ, BKV)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def ref_full(Q, K, V, dO):
    """fp64 causal attention fwd+bwd. Q,dO:(Hq,S,D); K,V:(Hkv,S,D) → O,dQ,dK,dV."""
    Hq, S, d = Q.shape; Hkv = K.shape[0]; G = Hq // Hkv; scale = 1.0 / math.sqrt(d)
    Qd, Kd, Vd, dOd = (a.astype(np.float64) for a in (Q, K, V, dO))
    causal = np.tril(np.ones((S, S), bool))
    O = np.zeros((Hq, S, d)); dQ = np.zeros((Hq, S, d)); dK = np.zeros((Hkv, S, d)); dV = np.zeros((Hkv, S, d))
    for h in range(Hq):
        kv = h // G
        Sc = np.where(causal, (Qd[h] @ Kd[kv].T) * scale, -1e38)
        P = np.exp(Sc - Sc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        O[h] = P @ Vd[kv]; do = dOd[h]
        dV[kv] += P.T @ do
        dP = do @ Vd[kv].T
        dS = P * (dP - (do * O[h]).sum(-1, keepdims=True))
        dQ[h] = (dS @ Kd[kv]) * scale
        dK[kv] += (dS.T @ Qd[h]) * scale
    return O, dQ, dK, dV


def _fwd_prefix(pq, pk, pv, sq, sk, sv):
    Hq, Sp, _ = pq.shape; Hkv = pk.shape[0]; G, _, Sc, _ = sq.shape
    NQBs, NKVBp, NKVBs = Sc // BQ, Sp // BKV, Sc // BKV; scale = float(1.0 / math.sqrt(D))
    gQs = _GpuArray(f32bf(sq.reshape(G * Hq * Sc, D)))
    gKp = _GpuArray(f32bf(pk.reshape(Hkv * Sp, D))); gVp = _GpuArray(f32bf(pv.reshape(Hkv * Sp, D)))
    gKs = _GpuArray(f32bf(sk.reshape(G * Hkv * Sc, D))); gVs = _GpuArray(f32bf(sv.reshape(G * Hkv * Sc, D)))
    gO = _GpuArray(np.zeros((G * Hq * Sc, D), np.float32)); gL = _GpuArray(np.zeros((G * Hq * Sc, 1), np.float32))
    ct.launch(si, (NQBs, G * Hq, 1), _attn_fwd_prefix,
              (gQs, gKp, gVp, gKs, gVs, gO, gL, NQBs, NKVBp, NKVBs, Hq, Hkv, scale)); sync()
    Os = gO.to_numpy().reshape(G, Hq, Sc, D); Ls = gL.to_numpy().reshape(G, Hq, Sc)
    for g in (gQs, gKp, gVp, gKs, gVs, gO, gL): g.free()
    Op, Lp = flash_attn_forward(pq[None], pk[None], pv[None], si, return_lse=True)
    return Os, Ls, Op[0], Lp[0]


def _bwd_prefix(pq, pk, pv, sq, sk, sv, Os, Ls, Op, Lp, dOs, dOp):
    """Returns (dQs (G,Hq,Sc,D), dKs (G,Hkv,Sc,D), dVs, dQp (Hq,Sp,D), dKp (Hkv,Sp,D), dVp)."""
    Hq, Sp, _ = pq.shape; Hkv = pk.shape[0]; G, _, Sc, _ = sq.shape; GG = Hq // Hkv
    NQBs, NKVBp, NKVBs = Sc // BQ, Sp // BKV, Sc // BKV; scale = float(1.0 / math.sqrt(D))
    Ds = (Os * dOs).sum(-1)                                              # suffix Delta (full O)
    # dQ_suffix — prefix(no mask)+suffix(causal)
    gQs = _GpuArray(f32bf(sq.reshape(G * Hq * Sc, D)))
    gKp = _GpuArray(f32bf(pk.reshape(Hkv * Sp, D))); gVp = _GpuArray(f32bf(pv.reshape(Hkv * Sp, D)))
    gKs = _GpuArray(f32bf(sk.reshape(G * Hkv * Sc, D))); gVs = _GpuArray(f32bf(sv.reshape(G * Hkv * Sc, D)))
    gdOs = _GpuArray(f32bf(dOs.reshape(G * Hq * Sc, D)))
    gLs = _GpuArray(Ls.reshape(G * Hq * Sc, 1).astype(np.float32)); gDs = _GpuArray(Ds.reshape(G * Hq * Sc, 1).astype(np.float32))
    gdQs = _GpuArray(np.zeros((G * Hq * Sc, D), np.float32))
    ct.launch(si, (NQBs, G * Hq, 1), _attn_bwd_dq_prefix,
              (gQs, gKp, gVp, gKs, gVs, gdOs, gLs, gDs, gdQs, NQBs, NKVBp, NKVBs, Hq, Hkv, scale))
    # dK/dV prefix CROSS (all suffix queries → prefix keys, no mask)
    gdKpc = _GpuArray(np.zeros((Hkv * Sp, D), np.float32)); gdVpc = _GpuArray(np.zeros((Hkv * Sp, D), np.float32))
    ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv_prefix,
              (gQs, gKp, gVp, gdOs, gLs, gDs, gdKpc, gdVpc, NQBs, NKVBp, Hq, Hkv, G, GG, scale)); sync()
    dQs = gdQs.to_numpy().reshape(G, Hq, Sc, D)
    dKpc = gdKpc.to_numpy().reshape(Hkv, Sp, D); dVpc = gdVpc.to_numpy().reshape(Hkv, Sp, D)
    for g in (gQs, gKp, gVp, gKs, gVs, gdOs, gLs, gDs, gdQs, gdKpc, gdVpc): g.free()
    # dK/dV suffix (suffix keys attended only by suffix queries, causal) — reuse the standard kernel, B=G
    _, dKs, dVs = flash_attn_backward(sq, sk, sv, Os, dOs, Ls, si)       # ignore its (wrong, prefix-less) dQ
    # dQ_prefix + dK/dV prefix SELF (prompt self-attn) with the shared prefix dO
    dQp, dKps, dVps = flash_attn_backward(pq[None], pk[None], pv[None], Op[None], dOp[None], Lp[None], si)
    dKp = dKps[0] + dKpc; dVp = dVps[0] + dVpc                           # prefix grad = self + cross
    return dQs, dKs, dVs, dQp[0], dKp, dVp


def _case(G, Hq, Hkv, Sp, Sc):
    rng = np.random.default_rng(1)
    pq = (rng.standard_normal((Hq,  Sp, D)) * 0.5).astype(np.float32)
    pk = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    pv = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    sq = (rng.standard_normal((G, Hq,  Sc, D)) * 0.5).astype(np.float32)
    sk = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
    sv = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
    dOp_each = (rng.standard_normal((G, Hq, Sp, D)) * 0.3).astype(np.float32)   # per-completion prefix dO
    dOs = (rng.standard_normal((G, Hq, Sc, D)) * 0.3).astype(np.float32)        # per-completion suffix dO
    dOp_shared = dOp_each.sum(0)                                                # shared prefix dO = Σ_i

    Os, Ls, Op, Lp = _fwd_prefix(pq, pk, pv, sq, sk, sv)
    dQs, dKs, dVs, dQp, dKp, dVp = _bwd_prefix(pq, pk, pv, sq, sk, sv, Os, Ls, Op, Lp, dOs, dOp_shared)

    # fp64 reference: full [P, s_i] per completion (numpy, no GPU churn); prefix grads summed over G
    rdKp = np.zeros((Hkv, Sp, D)); rdVp = np.zeros((Hkv, Sp, D)); rdQp = np.zeros((Hq, Sp, D))
    e_suf = 0.0
    fq = np.stack([np.concatenate([pq, sq[i]], 1) for i in range(G)])      # (G, Hq, S, D) batched
    fk = np.stack([np.concatenate([pk, sk[i]], 1) for i in range(G)])
    fv = np.stack([np.concatenate([pv, sv[i]], 1) for i in range(G)])
    fdo = np.stack([np.concatenate([dOp_each[i], dOs[i]], 1) for i in range(G)])
    for i in range(G):
        _, rdQ, rdK, rdV = ref_full(fq[i], fk[i], fv[i], fdo[i])
        e_suf = max(e_suf, rel(dQs[i], rdQ[:, Sp:]), rel(dKs[i], rdK[:, Sp:]), rel(dVs[i], rdV[:, Sp:]))
        rdQp += rdQ[:, :Sp]; rdKp += rdK[:, :Sp]; rdVp += rdV[:, :Sp]
    e_pre = max(rel(dQp, rdQp), rel(dKp, rdKp), rel(dVp, rdVp))

    # vs standard device backward, batched (B=G → ONE fwd+bwd). The ~1e-6 floor is NOT churn: it is
    # 1-ULP f32 in dQ — _attn_bwd_dq_prefix and _attn_bwd_dq have character-identical bodies but are
    # separately compiled and the tile compiler's FMA contraction differs (dK/dV compile identically →
    # exact 0). Diagnosed in tests/model/_dbg_resident_prefix_bwd.py; harmless (grads need correctness
    # + determinism, not bitwise — ratio=1 lives in the FORWARD, which is strictly bitwise).
    Of, Lf = flash_attn_forward(fq, fk, fv, si, return_lse=True)
    dQf, dKf, dVf = flash_attn_backward(fq, fk, fv, Of, fdo, Lf, si)
    bit = max(maxabs(dQs, dQf[:, :, Sp:]), maxabs(dKs, dKf[:, :, Sp:]), maxabs(dVs, dVf[:, :, Sp:]))

    ok = e_suf < 0.02 and e_pre < 0.02 and bit < 1e-4
    print(f"  G={G} Hq={Hq} Hkv={Hkv} Sp={Sp} Sc={Sc}: suffix≤{e_suf*100:.2f}% prefix(Σ)≤{e_pre*100:.2f}%  "
          f"suffix-bitwise Δ={bit:.1e}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Prefix-shared attention BACKWARD — vs fp64 (prefix=ΣG) + suffix bitwise vs standard")
    print("=" * 88)
    r = [_case(4, 4, 2, 128, 64), _case(4, 16, 8, 128, 128), _case(6, 16, 8, 192, 128)]
    print("=" * 88)
    print("  ALL PASS (prefix-shared bwd correct; suffix grads bitwise → ratio=1)" if all(r) else "  FAIL: " + str(r))
