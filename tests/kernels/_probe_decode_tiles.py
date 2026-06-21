"""Probe the decode-megakernel tile choices: (1) is _gemm_bf16 BITWISE-invariant to TN (and to
packing several N-segments into one launch)?  (2) vocab-stream kernel DCTM sweep (per-row math is
CTM-independent by construction — measure only).  Times at the REAL decode shapes (M=128)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.kernels.fused import _gemm_bf16
from ancora.kernels.loss import _ce_stats_b, _argmax_id_b, TV, f32_to_bf16_bits, _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

REPS = 200

def bench(fn):
    fn(); sync()
    t0 = time.perf_counter()
    for _ in range(REPS): fn()
    sync()
    return (time.perf_counter() - t0) / REPS * 1e6  # µs


def gemm_probe():
    rng = np.random.default_rng(0)
    M, K = 128, 1024
    for N in (1024, 2048):
        A = _GpuArray(f32_to_bf16_bits(rng.standard_normal((M, K), np.float32)))
        B = _GpuArray(f32_to_bf16_bits(rng.standard_normal((K, N), np.float32)))
        C = _GpuArray(np.zeros((M, N), np.float32))  # f32-out probe via bf16 store? use uint16 out
        C = _GpuArray(np.zeros((M, N), np.uint16))
        ref = None
        print(f"  _gemm_bf16 M={M} N={N} K={K}:")
        for TN in (128, 64, 32, 16):
            us = bench(lambda: ct.launch(si, (M // 128, N // TN, 1), _gemm_bf16,
                                         (A, B, C, K // 64, 128, TN, 64)))
            out = C.to_numpy()
            if ref is None:
                ref = out; tag = "(ref)"
            else:
                tag = "BITWISE==TN128" if np.array_equal(out, ref) else "*** DIFFERS ***"
            print(f"    TN={TN:4d}  blocks={N//TN:4d}  {us:7.1f} us  {tag}")

    # packed qkv: one (K, 4096) weight vs three separate launches — compare segment bits
    qd = kd = None
    qd, kd = 2048, 1024
    A = _GpuArray(f32_to_bf16_bits(rng.standard_normal((M, K), np.float32)))
    Wq = f32_to_bf16_bits(rng.standard_normal((K, qd), np.float32))
    Wk = f32_to_bf16_bits(rng.standard_normal((K, kd), np.float32))
    Wv = f32_to_bf16_bits(rng.standard_normal((K, kd), np.float32))
    gWq, gWk, gWv = _GpuArray(Wq), _GpuArray(Wk), _GpuArray(Wv)
    gWp = _GpuArray(np.concatenate([Wq, Wk, Wv], 1))
    Cq, Ck, Cv = (_GpuArray(np.zeros((M, n), np.uint16)) for n in (qd, kd, kd))
    Cp = _GpuArray(np.zeros((M, qd + 2 * kd), np.uint16))
    def sep():
        ct.launch(si, (1, qd // 128, 1), _gemm_bf16, (A, gWq, Cq, K // 64, 128, 128, 64))
        ct.launch(si, (1, kd // 128, 1), _gemm_bf16, (A, gWk, Ck, K // 64, 128, 128, 64))
        ct.launch(si, (1, kd // 128, 1), _gemm_bf16, (A, gWv, Cv, K // 64, 128, 128, 64))
    us_sep = bench(sep)
    Np = qd + 2 * kd
    for TN in (64, 32):
        us_p = bench(lambda: ct.launch(si, (1, Np // TN, 1), _gemm_bf16, (A, gWp, Cp, K // 64, 128, TN, 64)))
        out = Cp.to_numpy()
        okq = np.array_equal(out[:, :qd], Cq.to_numpy())
        okk = np.array_equal(out[:, qd:qd + kd], Ck.to_numpy())
        okv = np.array_equal(out[:, qd + kd:], Cv.to_numpy())
        print(f"  qkv: separate(3 launches,TN=128) {us_sep:6.1f} us | packed TN={TN} ({Np//TN} blk) "
              f"{us_p:6.1f} us  {'BITWISE' if okq and okk and okv else '*** DIFFERS ***'}")


def vocab_probe():
    rng = np.random.default_rng(1)
    M, V = 128, 151936
    lg = _GpuArray(rng.standard_normal((M, V), np.float32) * 4)
    lab = _GpuArray(rng.integers(0, V, (M, 1)).astype(np.int32))
    lp = _GpuArray(np.zeros((M, 1), np.float32)); lse = _GpuArray(np.zeros((M, 1), np.float32))
    oid = _GpuArray(np.zeros((M, 1), np.int32))
    ref_lp = ref_id = None
    print(f"  vocab stream (M={M}, V={V}, 78 MB):")
    for CTMb in (8, 4, 2, 1):
        us_ce = bench(lambda: ct.launch(si, (M // CTMb, 1, 1), _ce_stats_b, (lg, lab, lp, lse, V // TV, CTMb)))
        us_am = bench(lambda: ct.launch(si, (M // CTMb, 1, 1), _argmax_id_b, (lg, oid, V // TV, CTMb)))
        lpv, idv = lp.to_numpy(), oid.to_numpy()
        if ref_lp is None:
            ref_lp, ref_id, tag = lpv, idv, "(ref)"
        else:
            tag = "BITWISE" if np.array_equal(lpv, ref_lp) and np.array_equal(idv, ref_id) else "*** DIFFERS ***"
        print(f"    CTMb={CTMb}  blocks={M//CTMb:3d}  ce {us_ce:6.1f} us  argmax {us_am:6.1f} us  {tag}")


if __name__ == "__main__":
    gemm_probe()
    vocab_probe()
