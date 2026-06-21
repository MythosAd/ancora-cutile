"""Probe the decode-megakernel FUSED kernels for BITWISE equality vs the separate training
path (the ratio=1 requirement) + timing at the real decode shapes (Md=128, H=Ie=1024, E=16).
  1. _gemm_bf16_res  == _gemm_bf16 + _residual_add_rf32
  2. _gemm_af32_res  == _cast_bf16 + _gemm_bf16 + _residual_add_rf32
  3. _ggemm_gus      == _ggemm(gate) + _ggemm(up) + _swiglu_g
     _ggemm_b(TN=64) == _ggemm(TN=128)
     _combine_rf32   == _combine_bf16 + _residual_add_rf32
  4. _rmsnorm_{stats,apply}_f32_b TMb sweep == TM=32 kernels"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.kernels.fused import (_gemm_bf16, _gemm_bf16_res, _gemm_af32_res, _cast_bf16,
                                  _residual_add_rf32, RTM, RTN)
from ancora.kernels.moe import (_ggemm, _ggemm_b, _ggemm_gus, _swiglu_g, _combine_bf16,
                                _combine_rf32, TM as MTM, TN as MTN, TK as MTK, SM, SN)
from ancora.kernels.norm import (_rmsnorm_stats_f32, _rmsnorm_apply_f32, _rmsnorm_stats_f32_b,
                                 _rmsnorm_apply_f32_b, TM as NTM, TH)
from ancora.kernels.loss import f32_to_bf16_bits, _GpuArray

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
    return (time.perf_counter() - t0) / REPS * 1e6

rng = np.random.default_rng(0)
G = lambda a: _GpuArray(np.ascontiguousarray(a))
BF = lambda *s: G(f32_to_bf16_bits(rng.standard_normal(s, np.float32)))


def gemm_res_probe():
    M, K, N = 128, 2048, 1024
    A16, Af, B = BF(M, K), G(rng.standard_normal((M, K), np.float32)), BF(K, N)
    R = G(rng.standard_normal((M, N), np.float32))
    Csep, Cfus = G(np.zeros((M, N), np.float32)), G(np.zeros((M, N), np.float32))
    Ctmp, A16c = G(np.zeros((M, N), np.uint16)), G(np.zeros((M, K), np.uint16))
    TN = 32
    # 1) bf16-A + residual
    def sep():
        ct.launch(si, (1, N // TN, 1), _gemm_bf16, (A16, B, Ctmp, K // 64, 128, TN, 64))
        ct.launch(si, (M // RTM, N // RTN, 1), _residual_add_rf32, (R, Ctmp, Csep))
    def fus():
        ct.launch(si, (1, N // TN, 1), _gemm_bf16_res, (A16, B, R, Cfus, K // 64, 128, TN, 64))
    us, uf = bench(sep), bench(fus)
    ok = np.array_equal(Csep.to_numpy(), Cfus.to_numpy())
    print(f"  _gemm_bf16_res   : sep {us:6.1f} us → fused {uf:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")
    # 2) f32-A cast + GEMM + residual
    def sep2():
        ct.launch(si, (M // RTM, K // RTN, 1), _cast_bf16, (Af, A16c))
        ct.launch(si, (1, N // TN, 1), _gemm_bf16, (A16c, B, Ctmp, K // 64, 128, TN, 64))
        ct.launch(si, (M // RTM, N // RTN, 1), _residual_add_rf32, (R, Ctmp, Csep))
    def fus2():
        ct.launch(si, (1, N // TN, 1), _gemm_af32_res, (Af, B, R, Cfus, K // 64, 128, TN, 64))
    us, uf = bench(sep2), bench(fus2)
    ok = np.array_equal(Csep.to_numpy(), Cfus.to_numpy())
    print(f"  _gemm_af32_res   : sep {us:6.1f} us → fused {uf:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")


def moe_probe():
    H = Ie = 1024; E = 16; M, k = 128, 2
    Rt = 18; R = Rt * MTM                                  # fixed decode grid incl. padding tiles
    tile_e = rng.integers(0, E, Rt).astype(np.int32); tile_e[-2:] = E   # 2 padding tiles
    Xg = BF(R, H); Wg, Wu = BF(E * H, Ie), BF(E * H, Ie); Wd = BF(E * Ie, H)
    gte = G(tile_e.reshape(-1, 1))
    Gg, Ug = G(np.zeros((R, Ie), np.float32)), G(np.zeros((R, Ie), np.float32))
    Asep, Afus = G(np.zeros((R, Ie), np.uint16)), G(np.zeros((R, Ie), np.uint16))
    def sep():
        ct.launch(si, (Rt, Ie // MTN, 1), _ggemm, (Xg, Wg, gte, Gg, H // MTK, E))
        ct.launch(si, (Rt, Ie // MTN, 1), _ggemm, (Xg, Wu, gte, Ug, H // MTK, E))
        ct.launch(si, (R // SM, Ie // SN, 1), _swiglu_g, (Gg, Ug, Asep))
    def fus():
        ct.launch(si, (Rt, Ie // 32, 1), _ggemm_gus, (Xg, Wg, Wu, gte, Afus, H // MTK, E, 32))
    us, uf = bench(sep), bench(fus)
    ok = np.array_equal(Asep.to_numpy(), Afus.to_numpy())
    print(f"  _ggemm_gus       : sep {us:6.1f} us → fused {uf:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")
    for TNb in (64, 16):
        uf2 = bench(lambda: ct.launch(si, (Rt, Ie // TNb, 1), _ggemm_gus,
                                      (Xg, Wg, Wu, gte, Afus, H // MTK, E, TNb)))
        ok2 = np.array_equal(Asep.to_numpy(), Afus.to_numpy())
        print(f"  _ggemm_gus TN{TNb:<3d}: {uf2:6.1f} us  {'BITWISE' if ok2 else '*** DIFFERS ***'}")
    # down GEMM TN re-tile
    Ysep, Yfus = G(np.zeros((R, H), np.float32)), G(np.zeros((R, H), np.float32))
    us = bench(lambda: ct.launch(si, (Rt, H // MTN, 1), _ggemm, (Asep, Wd, gte, Ysep, Ie // MTK, E)))
    uf = bench(lambda: ct.launch(si, (Rt, H // 64, 1), _ggemm_b, (Asep, Wd, gte, Yfus, Ie // MTK, E, 64)))
    ok = np.array_equal(Ysep.to_numpy(), Yfus.to_numpy())
    print(f"  _ggemm_b TN64    : 128 {us:6.1f} us →  64   {uf:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")
    # combine + residual
    slots = rng.integers(0, R, (M, k)).astype(np.int32)
    gates = rng.random((R, 1)).astype(np.float32)
    gsl, gga = G(slots), G(gates)
    Res = G(rng.standard_normal((M, H), np.float32))
    Otmp = G(np.zeros((M, H), np.uint16))
    Osep, Ofus = G(np.zeros((M, H), np.float32)), G(np.zeros((M, H), np.float32))
    def sep3():
        ct.launch(si, (M, 1, 1), _combine_bf16, (gsl, gga, Ysep, Otmp, k, H))
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (Res, Otmp, Osep))
    def fus3():
        ct.launch(si, (M, 1, 1), _combine_rf32, (gsl, gga, Ysep, Res, Ofus, k, H))
    us, uf = bench(sep3), bench(fus3)
    ok = np.array_equal(Osep.to_numpy(), Ofus.to_numpy())
    print(f"  _combine_rf32    : sep {us:6.1f} us → fused {uf:6.1f} us  {'BITWISE' if ok else '*** DIFFERS ***'}")


def norm_probe():
    M, H = 128, 1024
    x = G(rng.standard_normal((M, H), np.float32))
    w = BF(1, H)
    rs, rb = G(np.zeros((M, 1), np.float32)), G(np.zeros((M, 1), np.float32))
    ys, yb = G(np.zeros((M, H), np.uint16)), G(np.zeros((M, H), np.uint16))
    us_s = bench(lambda: ct.launch(si, (M // NTM, 1, 1), _rmsnorm_stats_f32, (x, rs, H // TH, 1.0 / H, 1e-6)))
    us_a = bench(lambda: ct.launch(si, (M // NTM, 1, 1), _rmsnorm_apply_f32, (x, w, rs, ys, H // TH)))
    ref_r, ref_y = rs.to_numpy(), ys.to_numpy()
    print(f"  norm f32 TM=32   : stats {us_s:5.1f} us  apply {us_a:5.1f} us  (ref, {M//NTM} blk)")
    for TMb in (16, 8, 4):
        ub_s = bench(lambda: ct.launch(si, (M // TMb, 1, 1), _rmsnorm_stats_f32_b, (x, rb, H // TH, 1.0 / H, 1e-6, TMb)))
        ub_a = bench(lambda: ct.launch(si, (M // TMb, 1, 1), _rmsnorm_apply_f32_b, (x, w, rb, yb, H // TH, TMb)))
        ok = np.array_equal(rb.to_numpy(), ref_r) and np.array_equal(yb.to_numpy(), ref_y)
        print(f"  norm f32 TMb={TMb:2d}  : stats {ub_s:5.1f} us  apply {ub_a:5.1f} us  ({M//TMb} blk)  "
              f"{'BITWISE' if ok else '*** DIFFERS ***'}")


if __name__ == "__main__":
    gemm_res_probe()
    moe_probe()
    norm_probe()
