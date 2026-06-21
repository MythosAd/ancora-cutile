"""Decode attention (KV-cache, 1 query/sequence) vs numpy softmax. Exercises the
runtime S_cur mask at block-aligned and non-aligned cache lengths. Keep."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

import ancora.env
from ancora.kernels.attention import _attn_decode, _append_kv, D, BKV, _f32_to_bf16_bits as f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
    def free(s): cdrv.cuMemFree(s.p)
    def at_pos(s, pos):   # view offset by `pos` rows (data ptr += pos*D elems) — folds pos
        v = type("V", (), {})(); itemsize = np.dtype(s.dt).itemsize
        v.__cuda_array_interface__ = {"shape": s.sh, "typestr": np.dtype(s.dt).str,
                                      "data": (int(s.p) + pos * D * itemsize, False), "version": 3}
        return v


def ref(q, Kc, Vc, S_cur, Hq, Hkv):
    B = q.shape[0]; G = Hq // Hkv; scale = 1.0 / math.sqrt(D)
    o = np.zeros((B, Hq, D), np.float64)
    for b in range(B):
        for qh in range(Hq):
            kvh = qh // G
            K = Kc[b, kvh, :S_cur].astype(np.float64); V = Vc[b, kvh, :S_cur].astype(np.float64)
            sc = (q[b, qh].astype(np.float64) @ K.T) * scale
            p = np.exp(sc - sc.max()); p /= p.sum()
            o[b, qh] = p @ V
    return o


def test():
    print("--- decode attention vs numpy softmax ---")
    B, Hq, Hkv, maxS = 4, 16, 8, 128
    rng = np.random.default_rng(0); ok = True
    qf = (rng.standard_normal((B, Hq, D)) * 0.5).astype(np.float32)
    Kc = (rng.standard_normal((B, Hkv, maxS, D)) * 0.5).astype(np.float32)
    Vc = (rng.standard_normal((B, Hkv, maxS, D)) * 0.5).astype(np.float32)
    gQ  = GA(f32bf(qf.reshape(B * Hq, D)))
    gKc = GA(f32bf(Kc.reshape(B * Hkv * maxS, D)))
    gVc = GA(f32bf(Vc.reshape(B * Hkv * maxS, D)))
    scale = 1.0 / math.sqrt(D); NKVB = maxS // BKV
    for S_cur in [64, 40, 100, 128]:
        gO = GA(np.zeros((B * Hq, D), np.float32))
        ct.launch(si, (B, Hq, 1), _attn_decode, (gQ, gKc, gVc, gO, NKVB, Hq, Hkv, scale, int(S_cur)))
        cudart.cudaStreamSynchronize(si)
        o = gO.np().reshape(B, Hq, D)
        o_r = ref(qf, Kc, Vc, S_cur, Hq, Hkv)
        e = np.abs(o - o_r).max() / (np.abs(o_r).max() + 1e-9)
        good = e < 0.02; ok &= good
        print(f"  S_cur={S_cur:3d} (blocks≤{NKVB}): rel {e*100:.2f}%  {'OK' if good else 'FAIL'}")
        gO.free()
    for g in (gQ, gKc, gVc): g.free()
    return ok


def test_cache_build():
    """Build the cache one position at a time via _append_kv, then read with _attn_decode
    → validates the full cache write→read path (the rollout-engine core)."""
    print("--- cache build (append per position) + decode read ---")
    B, Hq, Hkv, maxS, S = 4, 16, 8, 128, 70
    rng = np.random.default_rng(1)
    q = (rng.standard_normal((B, Hq, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)
    Vv = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)
    Kc = GA(np.zeros((B * Hkv * maxS, D), np.uint16)); Vc = GA(np.zeros((B * Hkv * maxS, D), np.uint16))
    staging = []; hostbufs = []   # keep both device + host buffers alive for async copies
    for pos in range(S):
        kh = f32bf(K[:, :, pos].reshape(B * Hkv, D)); vh = f32bf(Vv[:, :, pos].reshape(B * Hkv, D))
        hostbufs += [kh, vh]
        Kn = GA(np.zeros((B * Hkv, D), np.uint16)); Vn = GA(np.zeros((B * Hkv, D), np.uint16))
        staging += [Kn, Vn]
        cdrv.cuMemcpyHtoDAsync(Kn.p, kh, kh.nbytes, si)   # upload ON si → ordered w/ the append
        cdrv.cuMemcpyHtoDAsync(Vn.p, vh, vh.nbytes, si)
        ct.launch(si, (B, Hkv, 1), _append_kv, (Kn, Vn, Kc.at_pos(pos), Vc.at_pos(pos), maxS, Hkv))
    cudart.cudaStreamSynchronize(si)
    gQ = GA(f32bf(q.reshape(B * Hq, D))); gO = GA(np.zeros((B * Hq, D), np.float32))
    ct.launch(si, (B, Hq, 1), _attn_decode, (gQ, Kc, Vc, gO, maxS // BKV, Hq, Hkv, 1.0 / math.sqrt(D), int(S)))
    cudart.cudaDeviceSynchronize()
    o = gO.np().reshape(B, Hq, D); o_r = ref(q, K, Vv, S, Hq, Hkv)
    e = np.abs(o - o_r).max() / (np.abs(o_r).max() + 1e-9)
    print(f"  appended {S} positions, decode: rel {e*100:.2f}%  {'OK' if e < 0.02 else 'FAIL'}")
    for g in (Kc, Vc, gQ, gO, *staging): g.free()
    return e < 0.02


if __name__ == "__main__":
    print(f"decode attention — D={D} BKV={BKV}")
    print("=" * 50)
    ok = test()
    bc = test_cache_build()
    print("=" * 50)
    print(f"  {'PASS' if ok and bc else 'FAIL'}")
