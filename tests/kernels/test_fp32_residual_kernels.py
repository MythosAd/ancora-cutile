"""FP32-residual kernel variants — correctness vs numpy fp64.

Validates the kernels behind the device-resident FP32 residual stream (kills the
late-layer massive-activation bf16 drift, [[resident-layer]]):
  _rmsnorm_stats_f32 / _rmsnorm_apply_f32   (input_ln/post_ln read fp32 residual)
  _rmsnorm_bwd_dx_f32 / _rmsnorm_dw_part_f32 (backward re-reads the same fp32 residual)
  _residual_add_rf32                          (fp32 residual + bf16 branch -> fp32)

The reference is computed from the FULL-precision fp32 x (the whole point: the
residual is NOT pre-rounded to bf16). One channel is set to 6912 to mimic the Qwen3
massive activation. Keep this — re-run after any cuda-tile upgrade."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.kernels.norm import (_GpuArray, _rmsnorm_stats_f32, _rmsnorm_apply_f32,
                                  _rmsnorm_bwd_dx_f32, _rmsnorm_dw_part_f32, _rmsnorm_dw_reduce,
                                  f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32,
                                  TM, TH, TD, PART)
from ancora.kernels.fused import _residual_add_rf32, RTM, RTN

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))
_bf = lambda x: bf32(f32bf(x))


def _mk_x(M, H, rng):
    x = (rng.standard_normal((M, H)) * 0.8).astype(np.float32)
    x[:, 7] = 6912.0                      # massive-activation channel (real Qwen3 magnitude)
    return x


def test_fwd():
    print("--- fwd: _rmsnorm_stats_f32 + _rmsnorm_apply_f32 (vs fp64 from fp32 x) ---")
    eps = 1e-6; ok = True; rng = np.random.default_rng(0)
    for (M, H) in [(128, 1024), (512, 1024)]:
        x = _mk_x(M, H, rng)
        w = (1.0 + rng.standard_normal(H) * 0.2).astype(np.float32)
        gx = _GpuArray(x); gw = _GpuArray(f32bf(w.reshape(1, H)))
        gr = _GpuArray.zeros((M, 1), np.float32); gy = _GpuArray.zeros((M, H), np.uint16)
        ct.launch(si, (M // TM, 1, 1), _rmsnorm_stats_f32, (gx, gr, H // TH, 1.0 / H, eps))
        ct.launch(si, (M // TM, 1, 1), _rmsnorm_apply_f32, (gx, gw, gr, gy, H // TH))
        cudart.cudaStreamSynchronize(si)
        rstd, y = gr.to_numpy(), bf32(gy.to_numpy())
        xd, wd = x.astype(np.float64), _bf(w).astype(np.float64)
        rstd_r = 1.0 / np.sqrt((xd * xd).mean(-1, keepdims=True) + eps)
        y_r = xd * rstd_r * wd
        rr, ry = rel(rstd, rstd_r), rel(_bf(y), y_r)
        o = rr < 0.005 and ry < 0.03; ok &= o
        for g in (gx, gw, gr, gy): g.free()
        print(f"  M={M:4d} H={H}: rstd={rr*100:.3f}% y={ry*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_bwd():
    print("--- bwd: _rmsnorm_bwd_dx_f32 + _rmsnorm_dw_part_f32 (vs fp64 from fp32 x) ---")
    eps = 1e-6; ok = True; rng = np.random.default_rng(1)
    for (M, H) in [(128, 1024), (512, 1024)]:
        x = _mk_x(M, H, rng)
        w = (1.0 + rng.standard_normal(H) * 0.2).astype(np.float32)
        dy = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)
        rstd = (1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)).astype(np.float32)
        gx = _GpuArray(x); gw = _GpuArray(f32bf(w.reshape(1, H))); gdy = _GpuArray(f32bf(dy))
        gr = _GpuArray(rstd); gdx = _GpuArray.zeros((M, H), np.uint16)
        gpart = _GpuArray.zeros((PART, H), np.float32); gdw = _GpuArray.zeros((1, H), np.float32)
        MB = M // TM; BPP = (MB + PART - 1) // PART
        ct.launch(si, (M // TM, 1, 1), _rmsnorm_bwd_dx_f32, (gx, gw, gdy, gr, gdx, H // TH, 1.0 / H))
        ct.launch(si, (H // TD, PART, 1), _rmsnorm_dw_part_f32, (gx, gdy, gr, gpart, MB, BPP))
        ct.launch(si, (H // TD, 1, 1), _rmsnorm_dw_reduce, (gpart, gdw))
        cudart.cudaStreamSynchronize(si)
        dx, dw = bf32(gdx.to_numpy()), gdw.to_numpy().reshape(H)
        xd, wd, dyd = x.astype(np.float64), _bf(w).astype(np.float64), _bf(dy).astype(np.float64)
        rd = rstd.astype(np.float64); dyw = dyd * wd
        c = (dyw * xd).sum(-1, keepdims=True)
        dx_r = rd * dyw - rd ** 3 * xd * c / H
        dw_r = (dyd * xd * rd).sum(0)
        rdx, rdw = rel(_bf(dx), dx_r), rel(dw, dw_r)
        o = rdx < 0.04 and rdw < 0.03; ok &= o
        for g in (gx, gw, gdy, gr, gdx, gpart, gdw): g.free()
        print(f"  M={M:4d} H={H}: dx={rdx*100:.2f}% dw={rdw*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_residual_add():
    print("--- _residual_add_rf32: f32 residual + bf16 branch -> f32 (vs fp64) ---")
    rng = np.random.default_rng(2); ok = True
    for (M, H) in [(128, 1024), (256, 1024)]:
        a = _mk_x(M, H, rng)                          # fp32 residual (carries 6912)
        b = (rng.standard_normal((M, H)) * 0.3).astype(np.float32)  # bf16 branch (GEMM out)
        ga = _GpuArray(a); gb = _GpuArray(f32bf(b)); go = _GpuArray.zeros((M, H), np.float32)
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (ga, gb, go))
        cudart.cudaStreamSynchronize(si)
        out = go.to_numpy()
        out_r = a.astype(np.float64) + _bf(b).astype(np.float64)   # exact f32 add of f32 + bf16(b)
        r = rel(out, out_r)
        o = r < 1e-3; ok &= o                          # near-exact: only b was rounded (an input)
        for g in (ga, gb, go): g.free()
        print(f"  M={M:4d} H={H}: out={r*100:.4f}%  {'OK' if o else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"FP32-residual kernels — TM={TM} TH={TH} TD={TD} PART={PART} RTM={RTM} RTN={RTN}")
    print("=" * 64)
    a = test_fwd(); b = test_bwd(); c = test_residual_add()
    print("=" * 64)
    print(f"  fwd {'PASS' if a else 'FAIL'} | bwd {'PASS' if b else 'FAIL'} | "
          f"residual-add {'PASS' if c else 'FAIL'}")
    sys.exit(0 if (a and b and c) else 1)
