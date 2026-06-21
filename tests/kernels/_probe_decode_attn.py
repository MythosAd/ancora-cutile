"""Probe the RISKY decode-megakernel recompiles for BITWISE equality vs the existing kernels:
  1. _attn_decode_blk_pd2 / _attn_decode_blk_win_pd2 (GQA-paired, halved KV traffic) vs the
     single-head kernels — the 1-ULP FMA-contraction precedent makes this the must-check.
  2. _argmax_ce_b / _sample_ce_b (ONE logits pass) vs the two-kernel pick→_ce_stats_b path.
  3. _gemm_nt_f32 TN sweep (vocab GEMM, mma-safe expectation).
Timings at the real decode shapes."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.kernels.attention import (_attn_decode_blk_pd, _attn_decode_blk_pd2,
                                       _attn_decode_blk_win_pd, _attn_decode_blk_win_pd2, BQ, BKV, D)
from ancora.kernels.fused import _gemm_nt_f32
from ancora.kernels.loss import (_ce_stats_b, _argmax_id_b, _sample_id_b, _argmax_ce_b,
                                  _sample_ce_b, TV, f32_to_bf16_bits, _GpuArray)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

REPS = 100
def bench(fn):
    fn(); sync()
    t0 = time.perf_counter()
    for _ in range(REPS): fn()
    sync()
    return (time.perf_counter() - t0) / REPS * 1e6

rng = np.random.default_rng(0)
G = lambda a: _GpuArray(np.ascontiguousarray(a))
BF = lambda *s: G(f32_to_bf16_bits(rng.standard_normal(s, np.float32)))


def attn_probe():
    B, Hq, Hkv = 32, 16, 8
    scale = 1.0 / np.sqrt(D)
    pos = 516
    gpos = G(np.array([[pos]], np.int32))
    # full-cache (global) layout: NKVB blocks per (seq, kv-head)
    NKVB = 1024 // BKV
    Q = BF(B * Hq * BQ, D)
    Kc, Vc = BF(B * Hkv * NKVB * BKV, D), BF(B * Hkv * NKVB * BKV, D)
    O1 = G(np.zeros((B * Hq * BQ, D), np.float32))
    O2 = G(np.zeros((B * Hq * BQ, D), np.float32))
    u1 = bench(lambda: ct.launch(si, (B * Hq, 1, 1), _attn_decode_blk_pd,
                                 (Q, Kc, Vc, O1, gpos, NKVB, Hq, Hkv, scale)))
    u2 = bench(lambda: ct.launch(si, (B * Hkv, 1, 1), _attn_decode_blk_pd2,
                                 (Q, Kc, Vc, O2, gpos, NKVB, Hq, Hkv, scale)))
    ok = np.array_equal(O1.to_numpy(), O2.to_numpy())
    print(f"  attn global  : single {u1:6.1f} us → paired {u2:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")
    # windowed + ring (local): NRB ring blocks, win_blocks=8 (window 512)
    WB = 8; NRB = 16; SMASK = NRB - 1
    Kr, Vr = BF(B * Hkv * NRB * BKV, D), BF(B * Hkv * NRB * BKV, D)
    u1 = bench(lambda: ct.launch(si, (B * Hq, 1, 1), _attn_decode_blk_win_pd,
                                 (Q, Kr, Vr, O1, gpos, NRB, Hq, Hkv, scale, WB, SMASK)))
    u2 = bench(lambda: ct.launch(si, (B * Hkv, 1, 1), _attn_decode_blk_win_pd2,
                                 (Q, Kr, Vr, O2, gpos, NRB, Hq, Hkv, scale, WB, SMASK)))
    ok = np.array_equal(O1.to_numpy(), O2.to_numpy())
    print(f"  attn local   : single {u1:6.1f} us → paired {u2:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")


def pickce_probe():
    M, V, CTMb = 128, 151936, 4
    lg = G((rng.standard_normal((M, V), np.float32) * 4).astype(np.float32))
    gseed = G(np.array([[7]], np.int32))
    oid1, oid2 = G(np.zeros((M, 1), np.int32)), G(np.zeros((M, 1), np.int32))
    lp1, lp2 = G(np.zeros((M, 1), np.float32)), G(np.zeros((M, 1), np.float32))
    ls1, ls2 = G(np.zeros((M, 1), np.float32)), G(np.zeros((M, 1), np.float32))
    def sep_g():
        ct.launch(si, (M // CTMb, 1, 1), _argmax_id_b, (lg, oid1, V // TV, CTMb))
        ct.launch(si, (M // CTMb, 1, 1), _ce_stats_b, (lg, oid1, lp1, ls1, V // TV, CTMb))
    def fus_g():
        ct.launch(si, (M // CTMb, 1, 1), _argmax_ce_b, (lg, oid2, lp2, ls2, V // TV, CTMb))
    u1, u2 = bench(sep_g), bench(fus_g)
    ok = (np.array_equal(oid1.to_numpy(), oid2.to_numpy()) and np.array_equal(lp1.to_numpy(), lp2.to_numpy())
          and np.array_equal(ls1.to_numpy(), ls2.to_numpy()))
    print(f"  argmax+ce    : 2-pass {u1:6.1f} us → 1-pass {u2:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")
    def sep_s():
        ct.launch(si, (M // CTMb, 1, 1), _sample_id_b, (lg, gseed, oid1, V // TV, 1.0, CTMb))
        ct.launch(si, (M // CTMb, 1, 1), _ce_stats_b, (lg, oid1, lp1, ls1, V // TV, CTMb))
    def fus_s():
        ct.launch(si, (M // CTMb, 1, 1), _sample_ce_b, (lg, gseed, oid2, lp2, ls2, V // TV, 1.0, CTMb))
    u1, u2 = bench(sep_s), bench(fus_s)
    ok = (np.array_equal(oid1.to_numpy(), oid2.to_numpy()) and np.array_equal(lp1.to_numpy(), lp2.to_numpy())
          and np.array_equal(ls1.to_numpy(), ls2.to_numpy()))
    print(f"  sample+ce    : 2-pass {u1:6.1f} us → 1-pass {u2:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")


def vocab_gemm_probe():
    M, K, V = 128, 1024, 151936
    A, B = BF(M, K), BF(V, K)
    C1, C2 = G(np.zeros((M, V), np.float32)), G(np.zeros((M, V), np.float32))
    u1 = bench(lambda: ct.launch(si, (1, V // 128, 1), _gemm_nt_f32, (A, B, C1, K // 64, 128, 128, 64)))
    ref = C1.to_numpy()
    for TN in (64, 32):
        u2 = bench(lambda: ct.launch(si, (1, V // TN, 1), _gemm_nt_f32, (A, B, C2, K // 64, 128, TN, 64)))
        ok = np.array_equal(C2.to_numpy(), ref)
        print(f"  vocab nt_f32 : TN=128 {u1:6.1f} us → TN={TN} {u2:6.1f} us  "
              f"{'BITWISE' if ok else '*** DIFFERS ***'}")


if __name__ == "__main__":
    attn_probe()
    pickce_probe()
    vocab_gemm_probe()
