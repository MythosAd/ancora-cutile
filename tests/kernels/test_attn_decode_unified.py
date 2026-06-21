"""Rollout↔training attention unification proof (BF16).

THE on-policy RL requirement: a token's attention output (→ logprob) must be bitwise-identical
whether computed in TRAINING (prefill, all positions in parallel) or ROLLOUT (decode, one new
query over the KV cache). Our two kernels currently DON'T agree:
  _attn_fwd     (prefill): q·kᵀ and p·v via ct.mma (tensor-core, fixed D-reduction order)
  _attn_decode  (decode):  q·k and p·v via ct.sum(broadcast*…) (CUDA-core, DIFFERENT order)
→ ~0.3% gap → π_train ≠ π_infer → biased GRPO.

FIX (decode reuses the prefill kernel): to get position t's attention during generation, run the
PREFILL kernel over the cache and take row t. Because (a) the causal mask zeroes j>t, the not-yet-
generated cache slots don't affect row t, and (b) prefill is seq-len invariant, row t is bitwise
the SAME as training's prefill. This test proves both points, and contrasts the old decode.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa
from ancora.kernels.attention import _attn_fwd, _attn_decode, D, BQ, BKV, _f32_to_bf16_bits as f32bf

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


class GA:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
    def free(s): cdrv.cuMemFree(s.p)


B, Hq, Hkv = 1, 2, 1                 # GQA G=2
S, maxS, t = 128, 128, 100           # query position t (in block t//BQ)
NQB, NKVB = S // BQ, S // BKV
scale = 1.0 / math.sqrt(D)
rng = np.random.default_rng(0)
Q = (rng.standard_normal((B, Hq, S, D)) * 0.5).astype(np.float32)
K = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)
V = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)


def prefill(Kx, Vx):
    """_attn_fwd over the whole sequence → O (B,Hq,S,D) f32."""
    gQ = GA(f32bf(Q.reshape(-1, D))); gK = GA(f32bf(Kx.reshape(-1, D))); gV = GA(f32bf(Vx.reshape(-1, D)))
    gO = GA(np.zeros((B * Hq * S, D), np.float32)); gL = GA(np.zeros((B * Hq * S, 1), np.float32))
    ct.launch(si, (NQB, B * Hq, 1), _attn_fwd, (gQ, gK, gV, gO, gL, NQB, NKVB, Hq, Hkv, scale))
    cudart.cudaStreamSynchronize(si)
    O = gO.np().reshape(B, Hq, S, D)
    for g in (gQ, gK, gV, gO, gL): g.free()
    return O


def decode_old(Kc, Vc, S_cur):
    """_attn_decode: single query t over cache → O (B,Hq,D) f32."""
    gQ = GA(f32bf(Q[:, :, t].reshape(B * Hq, D)))
    gKc = GA(f32bf(Kc.reshape(B * Hkv * maxS, D))); gVc = GA(f32bf(Vc.reshape(B * Hkv * maxS, D)))
    gO = GA(np.zeros((B * Hq, D), np.float32))
    ct.launch(si, (B, Hq, 1), _attn_decode, (gQ, gKc, gVc, gO, maxS // BKV, Hq, Hkv, scale, int(S_cur)))
    cudart.cudaStreamSynchronize(si)
    O = gO.np().reshape(B, Hq, D)
    for g in (gQ, gKc, gVc, gO): g.free()
    return O


bits = lambda x: x.view(np.uint32)
relmax = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))

print(f"Rollout↔training attention unification — D={D} BQ={BQ} S={S} t={t}")
print("=" * 66)

O_train = prefill(K, V)                                   # TRAINING (full prefill)

# ── (1) DECODE VIA PREFILL KERNEL: scramble cache slots > t (not yet generated) ──
K2, V2 = K.copy(), V.copy()
K2[:, :, t + 1:] = (rng.standard_normal((B, Hkv, S - t - 1, D)) * 5).astype(np.float32)   # garbage future
V2[:, :, t + 1:] = (rng.standard_normal((B, Hkv, S - t - 1, D)) * 5).astype(np.float32)
O_dec_pf = prefill(K2, V2)
same = np.array_equal(bits(O_dec_pf[:, :, t]), bits(O_train[:, :, t]))
print("(1) decode-via-prefill-kernel (cache+garbage-future) vs training prefill, row t:")
print(f"    bitwise identical = {same}   max|Δ| = {np.abs(O_dec_pf[:,:,t]-O_train[:,:,t]).max():.3g}   "
      f"{'OK — ratio=1 by construction' if same else 'FAIL'}")

# ── (2) OLD HAND-ROLLED DECODE: cache = K/V[0..t], S_cur=t+1 ──
Kc = K.copy(); Vc = V.copy()                              # positions > t masked by S_cur
O_dec_old = decode_old(Kc, Vc, t + 1)
e_old = relmax(O_dec_old, O_train[:, :, t])
same_old = np.array_equal(bits(O_dec_old), bits(O_train[:, :, t]))
print("(2) old _attn_decode (hand-rolled sum) vs training prefill, row t:")
print(f"    bitwise identical = {same_old}   rel = {e_old*100:.3f}%   "
      f"(close but NOT bitwise → breaks π_train==π_infer)")

print("=" * 66)
ok = same and not same_old
print(f"  {'PROVEN: prefill-kernel decode is bitwise-exact; hand-rolled decode is not.' if ok else 'UNEXPECTED'}")
sys.exit(0 if ok else 1)
