"""
ancora/kernels/norm.py — RMSNorm forward + backward (BF16 I/O, FP32 compute)

Used by Qwen3-0.6B: input_layernorm, post_attention_layernorm (over hidden=1024),
and q_norm / k_norm (QK-Norm over head_dim=128). Same kernels, H passed in.

Reference implementations studied
---------------------------------
  PyTorch nn.RMSNorm / HF LlamaRMSNorm / Qwen3RMSNorm:
      upcast fp32 -> x * rsqrt(mean(x^2,-1) + eps) -> * weight -> downcast.
      (Qwen3 uses plain `weight`, NOT Gemma's `1 + weight`.)
  vLLM csrc/layernorm_kernels.cu `rms_norm_kernel`:
      ONE thread-block per token (row); block-level reduce of sum(x^2) in fp32.
  Unsloth `rms_layernorm` (Triton):
      one program per row; saves r = rstd to a buffer for the backward.
  Megatron-LM / Apex FusedRMSNorm (csrc/layer_norm_cuda_kernel.cu):
      grad_input  = rstd*(dy*w - x*rstd^2 * mean_d(dy*w*x))   <- our _rmsnorm_bwd_dx
      grad_weight = TWO-PASS partial reduction: cuComputePartGradGammaBeta builds
                    part_grad(part_size, H), then cuComputeGradGammaBeta reduces
                    -> grad(H).  We mirror this in _rmsnorm_dw_part/_dw_reduce.

Batch invariance (RL train/rollout logprob match — rule #5 "reductions on one core")
-----------------------------------------------------------------------------------
  RMSNorm forward reduces ONLY within a row (over H); each block owns whole rows,
  no cross-token reduction -> forward is inherently batch-invariant.
  Backward dW reduces over tokens with a FIXED tiling (PART compile-time partials,
  contiguous chunks) -> deterministic, no atomics.

Precision / layout
------------------
  x, w, y, dy, dx : uint16 BF16 bits (bitcast in-kernel; matches the rest of ANCORA).
  rstd            : float32, shape (M,1)  — saved by fwd, consumed by bwd.
  dW (param grad) : float32, shape (1,H)  — fp32 accumulation (FSDP reduce-scatters
                    weight grads in fp32 for the optimizer).
  All math in float32 (upcast on load, downcast on store) — same as vLLM/Unsloth.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # sets CUDA_PATH before any cuda.* import

# ── tile sizes ───────────────────────────────────────────────────────────────
# H is CHUNKED into TH-wide tiles (NOT loaded whole): a (TM,1024) tile compiles
# pathologically slowly in cuda-tile 1.4.0 (fwd barely finished, dx hung >150 s),
# while <=256-wide tiles compile fast (as in linear/attention). The per-row sum(x^2)
# still accumulates on ONE block over the short H//TH loop -> batch-invariant.
TM   = 32    # tokens (rows) per block
TH   = 64    # H-chunk width for fwd / dx  (divides hidden=1024 AND head_dim=128 for QK-Norm)
TD   = 64    # feature-slice width for the dW reduction (must divide head_dim=128 too)
PART = 16    # dW partial-reduction groups (Megatron/Apex two-pass; parallelism)


# ── forward ──────────────────────────────────────────────────────────────────

# Two kernels (stats + apply). A SINGLE kernel with the reduction loop followed by
# a second loop that re-loads x silently miscompiles in cuda-tile 1.4.0: exactly one
# iteration of the second unrolled loop reads garbage (chunk 5 of 8 → 300% error,
# while rstd was correct). dx has the same two-loop shape yet works, so the trigger
# is subtle — splitting into two single-loop kernels sidesteps it entirely. Cost:
# one extra launch + an (M,1) rstd round-trip through HBM (negligible). x is read
# twice from DRAM, which RMSNorm needs anyway (reduce, then normalize).

@ct.kernel
def _rmsnorm_stats(x, rstd_out, HB: ct.Constant[int],
                   inv_H: ct.Constant[float], eps: ct.Constant[float]):
    """rstd[i] = 1/sqrt(mean_d(x[i,:]^2) + eps). One block per TM-row tile; the
    sum(x^2) reduction stays on the block (batch-invariant).  Grid: (M // TM,)."""
    rb = ct.bid(0)
    ss = ct.zeros((TM, 1), ct.float32)
    for h in range(HB):
        xf = ct.astype(ct.bitcast(ct.load(x, index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        ss = ss + ct.sum(xf * xf, axis=-1, keepdims=True)
    ct.store(rstd_out, index=(rb, 0), tile=ct.rsqrt(ss * inv_H + eps))


@ct.kernel
def _rmsnorm_apply(x, w, rstd, y, HB: ct.Constant[int]):
    """y[i,d] = x[i,d] * rstd[i] * w[d]. Pure elementwise.  Grid: (M // TM,)."""
    rb = ct.bid(0)
    r  = ct.load(rstd, index=(rb, 0), shape=(TM, 1))
    for h in range(HB):
        xf = ct.astype(ct.bitcast(ct.load(x, index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        wf = ct.astype(ct.bitcast(ct.load(w, index=(0, h),  shape=(1,  TH)), ct.bfloat16), ct.float32)
        yv = xf * ct.broadcast_to(r, (TM, TH)) * ct.broadcast_to(wf, (TM, TH))
        ct.store(y, index=(rb, h), tile=ct.bitcast(ct.astype(yv, ct.bfloat16), ct.uint16))


# ── FP32-RESIDUAL variants (read x as native float32, not bf16 bits) ──────────
# The device-resident residual stream is FP32 (input_ln / post_ln read it directly),
# so the massive activation (~6912 in Qwen3) is not coarsely rounded across 28 layers
# (the late-layer bf16 drift, [[resident-layer]]). Only the `x` LOAD differs from the
# bf16 kernels above — gain w / output y stay bf16. Used by ResidentLayerTrain for the
# two hidden-state norms; q_norm/k_norm keep the bf16 path (they read head activations).

@ct.kernel
def _rmsnorm_stats_f32(x, rstd_out, HB: ct.Constant[int],
                       inv_H: ct.Constant[float], eps: ct.Constant[float]):
    """rstd[i] = 1/sqrt(mean_d(x[i,:]^2)+eps); x is FP32. Grid (M//TM,)."""
    rb = ct.bid(0)
    ss = ct.zeros((TM, 1), ct.float32)
    for h in range(HB):
        xf = ct.load(x, index=(rb, h), shape=(TM, TH))
        ss = ss + ct.sum(xf * xf, axis=-1, keepdims=True)
    ct.store(rstd_out, index=(rb, 0), tile=ct.rsqrt(ss * inv_H + eps))


@ct.kernel
def _rmsnorm_apply_f32(x, w, rstd, y, HB: ct.Constant[int]):
    """y[i,d] = x[i,d]*rstd[i]*w[d]; x FP32 in, y BF16 bits out (the GEMM input). Grid (M//TM,)."""
    rb = ct.bid(0)
    r  = ct.load(rstd, index=(rb, 0), shape=(TM, 1))
    for h in range(HB):
        xf = ct.load(x, index=(rb, h), shape=(TM, TH))
        wf = ct.astype(ct.bitcast(ct.load(w, index=(0, h), shape=(1, TH)), ct.bfloat16), ct.float32)
        yv = xf * ct.broadcast_to(r, (TM, TH)) * ct.broadcast_to(wf, (TM, TH))
        ct.store(y, index=(rb, h), tile=ct.bitcast(ct.astype(yv, ct.bfloat16), ct.uint16))


# ── launch-time row-tile variants (decode megakernel, 2026-06-11) ─────────────
# At Md=128 the fixed TM=32 gives a 4-block grid (60 SMs idle, latency-bound ~38µs for a
# 0.5 MB pass). TMb is a launch arg so decode can fill the machine (TMb=8 → 16 blocks).
# ⚠ Row-tile changes CAN flip bits in cuda-tile reductions (the vocab-stream kernels differ
# at CTMb≤2) — _probe_decode_tiles.py must show BITWISE vs the TM=32 kernel before use.

@ct.kernel
def _rmsnorm_stats_f32_b(x, rstd_out, HB: ct.Constant[int],
                         inv_H: ct.Constant[float], eps: ct.Constant[float],
                         TMb: ct.Constant[int]):
    """_rmsnorm_stats_f32 with a launch-time row tile. Grid (M//TMb,)."""
    rb = ct.bid(0)
    ss = ct.zeros((TMb, 1), ct.float32)
    for h in range(HB):
        xf = ct.load(x, index=(rb, h), shape=(TMb, TH))
        ss = ss + ct.sum(xf * xf, axis=-1, keepdims=True)
    ct.store(rstd_out, index=(rb, 0), tile=ct.rsqrt(ss * inv_H + eps))


@ct.kernel
def _rmsnorm_apply_f32_b(x, w, rstd, y, HB: ct.Constant[int], TMb: ct.Constant[int]):
    """_rmsnorm_apply_f32 with a launch-time row tile. Grid (M//TMb,)."""
    rb = ct.bid(0)
    r  = ct.load(rstd, index=(rb, 0), shape=(TMb, 1))
    for h in range(HB):
        xf = ct.load(x, index=(rb, h), shape=(TMb, TH))
        wf = ct.astype(ct.bitcast(ct.load(w, index=(0, h), shape=(1, TH)), ct.bfloat16), ct.float32)
        yv = xf * ct.broadcast_to(r, (TMb, TH)) * ct.broadcast_to(wf, (TMb, TH))
        ct.store(y, index=(rb, h), tile=ct.bitcast(ct.astype(yv, ct.bfloat16), ct.uint16))


QB = 32   # MXFP8 scale block (= quant.B); RMSNorm-apply fused-quant chunk


@ct.kernel
def _rmsnorm_apply_q(x, w, rstd, fp8_out, scale_out, KB: ct.Constant[int]):
    """FUSED RMSNorm-apply + MXFP8 quant (CODA epilogue fusion): y=x*rstd*w, then per-32
    E8M0 quant → fp8_out(M,H) u8 + scale_out(M,H//32) u8 — a direct _fwd_mxfp8 input with
    NO separate _quant_mxfp8 launch and NO bf16 round-trip (saves ~21% of MXFP8 proj time;
    the projections then run at the ~93%-of-bf16-peak GEMM-only ceiling). Grid (M//TM,),
    chunk = one 32-wide quant block. Quant math identical to quant._quant_mxfp8."""
    rb = ct.bid(0)
    r  = ct.load(rstd, index=(rb, 0), shape=(TM, 1))
    for kb in range(KB):
        xf = ct.astype(ct.bitcast(ct.load(x, index=(rb, kb), shape=(TM, QB)), ct.bfloat16), ct.float32)
        wf = ct.astype(ct.bitcast(ct.load(w, index=(0, kb),  shape=(1,  QB)), ct.bfloat16), ct.float32)
        yv = xf * ct.broadcast_to(r, (TM, QB)) * ct.broadcast_to(wf, (TM, QB))
        amax = ct.max(ct.maximum(yv, 0.0 - yv), axis=-1, keepdims=True)
        ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
        byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((TM, 1), 0, ct.uint32))
        sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
        fp8 = ct.bitcast(ct.astype(yv / ct.broadcast_to(sc, (TM, QB)), ct.float8_e4m3fn), ct.uint8)
        ct.store(fp8_out,   index=(rb, kb), tile=fp8)
        ct.store(scale_out, index=(rb, kb), tile=ct.astype(byte, ct.uint8))


@ct.kernel
def _rmsnorm_apply_q_f32(x, w, rstd, fp8_out, scale_out, KB: ct.Constant[int]):
    """FP32-residual variant of _rmsnorm_apply_q: x (the residual) is native f32. Used by the
    rollout ResidentLayer so input_ln/post_ln read the fp32 residual stream (no bf16 rounding of
    the ~6912 massive activation). Output is the same fp8(M,H)+E8M0(M,H//32) MXFP8 GEMM input."""
    rb = ct.bid(0)
    r  = ct.load(rstd, index=(rb, 0), shape=(TM, 1))
    for kb in range(KB):
        xf = ct.load(x, index=(rb, kb), shape=(TM, QB))
        wf = ct.astype(ct.bitcast(ct.load(w, index=(0, kb), shape=(1, QB)), ct.bfloat16), ct.float32)
        yv = xf * ct.broadcast_to(r, (TM, QB)) * ct.broadcast_to(wf, (TM, QB))
        amax = ct.max(ct.maximum(yv, 0.0 - yv), axis=-1, keepdims=True)
        ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
        byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((TM, 1), 0, ct.uint32))
        sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
        fp8 = ct.bitcast(ct.astype(yv / ct.broadcast_to(sc, (TM, QB)), ct.float8_e4m3fn), ct.uint8)
        ct.store(fp8_out,   index=(rb, kb), tile=fp8)
        ct.store(scale_out, index=(rb, kb), tile=ct.astype(byte, ct.uint8))


# ── backward: dX (per-row, trivially batch-invariant) ────────────────────────

@ct.kernel
def _rmsnorm_bwd_dx(x, w, dy, rstd, dx,
                    HB: ct.Constant[int], inv_H: ct.Constant[float]):
    """
    dx[i,e] = rstd[i]*dy[i,e]*w[e]  -  rstd[i]^3 * x[i,e] * c[i]/H,
        c[i] = sum_d dy[i,d]*w[d]*x[i,d].
    Same formula as Megatron/Apex FusedRMSNorm grad_input. Pass 1 reduces c[i]
    (on one block), pass 2 writes dx. H-chunked (same compile reason as fwd).
    Grid: (M // TM,).
    """
    rb = ct.bid(0)
    r  = ct.load(rstd, index=(rb, 0), shape=(TM, 1))         # (TM,1) fp32

    c = ct.zeros((TM, 1), ct.float32)
    for h in range(HB):
        xf  = ct.astype(ct.bitcast(ct.load(x,  index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        dyf = ct.astype(ct.bitcast(ct.load(dy, index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        wf  = ct.astype(ct.bitcast(ct.load(w,  index=(0, h),  shape=(1,  TH)), ct.bfloat16), ct.float32)
        c = c + ct.sum((dyf * ct.broadcast_to(wf, (TM, TH))) * xf, axis=-1, keepdims=True)
    coef = r * r * r * c * inv_H                             # (TM,1) rstd^3 * c/H

    for h in range(HB):
        xf  = ct.astype(ct.bitcast(ct.load(x,  index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        dyf = ct.astype(ct.bitcast(ct.load(dy, index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        wf  = ct.astype(ct.bitcast(ct.load(w,  index=(0, h),  shape=(1,  TH)), ct.bfloat16), ct.float32)
        dxv = ct.broadcast_to(r, (TM, TH)) * dyf * ct.broadcast_to(wf, (TM, TH)) \
              - ct.broadcast_to(coef, (TM, TH)) * xf
        ct.store(dx, index=(rb, h), tile=ct.bitcast(ct.astype(dxv, ct.bfloat16), ct.uint16))


@ct.kernel
def _rmsnorm_bwd_dx_f32(x, w, dy, rstd, dx,
                        HB: ct.Constant[int], inv_H: ct.Constant[float]):
    """Same as _rmsnorm_bwd_dx but x (the residual) is native FP32. dy/w bf16, dx bf16-out.
    (dx is a gradient → still bf16; only the residual x read changes.)  Grid (M//TM,)."""
    rb = ct.bid(0)
    r  = ct.load(rstd, index=(rb, 0), shape=(TM, 1))
    c = ct.zeros((TM, 1), ct.float32)
    for h in range(HB):
        xf  = ct.load(x, index=(rb, h), shape=(TM, TH))
        dyf = ct.astype(ct.bitcast(ct.load(dy, index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        wf  = ct.astype(ct.bitcast(ct.load(w,  index=(0, h),  shape=(1,  TH)), ct.bfloat16), ct.float32)
        c = c + ct.sum((dyf * ct.broadcast_to(wf, (TM, TH))) * xf, axis=-1, keepdims=True)
    coef = r * r * r * c * inv_H
    for h in range(HB):
        xf  = ct.load(x, index=(rb, h), shape=(TM, TH))
        dyf = ct.astype(ct.bitcast(ct.load(dy, index=(rb, h), shape=(TM, TH)), ct.bfloat16), ct.float32)
        wf  = ct.astype(ct.bitcast(ct.load(w,  index=(0, h),  shape=(1,  TH)), ct.bfloat16), ct.float32)
        dxv = ct.broadcast_to(r, (TM, TH)) * dyf * ct.broadcast_to(wf, (TM, TH)) \
              - ct.broadcast_to(coef, (TM, TH)) * xf
        ct.store(dx, index=(rb, h), tile=ct.bitcast(ct.astype(dxv, ct.bfloat16), ct.uint16))


# ── backward: dW — Megatron/Apex two-pass partial reduction ──────────────────
# Apex FusedRMSNorm splits dW over PART partials (cuComputePartGradGammaBeta ->
# cuComputeGradGammaBeta). Pass 1: PART*(H/TD) blocks, each sums its contiguous
# token-block chunk -> part_dw (PART, H). Pass 2: sum the PART partials -> dw (H,).
# Why not one block looping all tokens: at M=4096/TM=32 that unrolls 128x -> the
# compiler hangs; and 8 blocks underfill 60 SMs. PART is a fixed compile-time
# constant -> reduction order deterministic for given M (batch-invariant tiling).

@ct.kernel
def _rmsnorm_dw_part(x, dy, rstd, part_dw,
                     M_BLOCKS: ct.Constant[int], BPP: ct.Constant[int]):
    """part_dw[p, d] = sum over token-blocks [p*BPP, p*BPP+BPP) of sum_i dy*x*rstd.
    Grid: (H // TD, PART)."""
    db, p = ct.bid(0), ct.bid(1)
    acc = ct.zeros((1, TD), ct.float32)
    for j in range(BPP):
        mb = p * BPP + j
        if mb < M_BLOCKS:
            xf  = ct.astype(ct.bitcast(ct.load(x,  index=(mb, db), shape=(TM, TD)), ct.bfloat16), ct.float32)
            dyf = ct.astype(ct.bitcast(ct.load(dy, index=(mb, db), shape=(TM, TD)), ct.bfloat16), ct.float32)
            r   = ct.load(rstd, index=(mb, 0), shape=(TM, 1))
            acc = acc + ct.sum((dyf * xf) * ct.broadcast_to(r, (TM, TD)), axis=0, keepdims=True)
    ct.store(part_dw, index=(p, db), tile=acc)


@ct.kernel
def _rmsnorm_dw_part_f32(x, dy, rstd, part_dw,
                         M_BLOCKS: ct.Constant[int], BPP: ct.Constant[int]):
    """Same as _rmsnorm_dw_part but x (the residual) is native FP32. dy bf16. Grid (H//TD, PART)."""
    db, p = ct.bid(0), ct.bid(1)
    acc = ct.zeros((1, TD), ct.float32)
    for j in range(BPP):
        mb = p * BPP + j
        if mb < M_BLOCKS:
            xf  = ct.load(x, index=(mb, db), shape=(TM, TD))
            dyf = ct.astype(ct.bitcast(ct.load(dy, index=(mb, db), shape=(TM, TD)), ct.bfloat16), ct.float32)
            r   = ct.load(rstd, index=(mb, 0), shape=(TM, 1))
            acc = acc + ct.sum((dyf * xf) * ct.broadcast_to(r, (TM, TD)), axis=0, keepdims=True)
    ct.store(part_dw, index=(p, db), tile=acc)


@ct.kernel
def _rmsnorm_dw_reduce(part_dw, dw):
    """dw[d] = sum_{p<PART} part_dw[p, d].  Grid: (H // TD,)."""
    db = ct.bid(0)
    acc = ct.zeros((1, TD), ct.float32)
    for p in range(PART):
        acc = acc + ct.load(part_dw, index=(p, db), shape=(1, TD))
    ct.store(dw, index=(0, db), tile=acc)


@ct.kernel
def _rmsnorm_dw_reduce_acc(part_dw, dw):
    """dw[d] += sum_p part_dw[p, d] — GRADIENT-ACCUMULATION variant (micro-batch ≥1 adds onto
    the previous micro-batches' gain grad in place; f32 add). Grid: (H // TD,)."""
    db = ct.bid(0)
    acc = ct.load(dw, index=(0, db), shape=(1, TD))
    for p in range(PART):
        acc = acc + ct.load(part_dw, index=(p, db), shape=(1, TD))
    ct.store(dw, index=(0, db), tile=acc)


# ── host helpers ─────────────────────────────────────────────────────────────

class _GpuArray:
    """Minimal device array exposing __cuda_array_interface__ for ct.launch."""
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


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits (uint16), round-to-nearest-even (matches HW store)."""
    u = x.astype(np.float32).view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_bits_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


# ── public API ───────────────────────────────────────────────────────────────

def rmsnorm_forward(x_f32: np.ndarray, w_f32: np.ndarray, stream_int: int,
                    eps: float = 1e-6):
    """
    x_f32: (M, H) float32, w_f32: (H,) float32. Returns (y (M,H) f32, rstd (M,1) f32).
    Inputs are rounded to BF16 (the storage precision) before the kernel.
    """
    M, H = x_f32.shape
    assert M % TM == 0 and H % TH == 0 and w_f32.shape == (H,)
    gx = _GpuArray(f32_to_bf16_bits(x_f32))
    gw = _GpuArray(f32_to_bf16_bits(w_f32.reshape(1, H)))
    gy = _GpuArray.zeros((M, H), np.uint16)
    gr = _GpuArray.zeros((M, 1), np.float32)

    ct.launch(stream_int, (M // TM, 1, 1), _rmsnorm_stats, (gx, gr, H // TH, 1.0 / H, float(eps)))
    ct.launch(stream_int, (M // TM, 1, 1), _rmsnorm_apply, (gx, gw, gr, gy, H // TH))
    cudart.cudaStreamSynchronize(stream_int)

    y    = bf16_bits_to_f32(gy.to_numpy())
    rstd = gr.to_numpy()
    for g in (gx, gw, gy, gr): g.free()
    return y, rstd


def rmsnorm_backward(x_f32, w_f32, dy_f32, rstd, stream_int):
    """
    x_f32,dy_f32: (M,H) f32; w_f32: (H,) f32; rstd: (M,1) f32 (from forward).
    Returns (dx (M,H) f32, dw (H,) f32). dX in BF16-precision, dW in fp32.
    """
    M, H = x_f32.shape
    assert M % TM == 0 and H % TH == 0 and H % TD == 0
    M_BLOCKS = M // TM
    BPP = (M_BLOCKS + PART - 1) // PART
    gx  = _GpuArray(f32_to_bf16_bits(x_f32))
    gw  = _GpuArray(f32_to_bf16_bits(w_f32.reshape(1, H)))
    gdy = _GpuArray(f32_to_bf16_bits(dy_f32))
    gr  = _GpuArray(rstd.astype(np.float32))
    gdx = _GpuArray.zeros((M, H), np.uint16)
    gpart = _GpuArray.zeros((PART, H), np.float32)
    gdw = _GpuArray.zeros((1, H), np.float32)

    ct.launch(stream_int, (M // TM, 1, 1), _rmsnorm_bwd_dx, (gx, gw, gdy, gr, gdx, H // TH, 1.0 / H))
    ct.launch(stream_int, (H // TD, PART, 1), _rmsnorm_dw_part, (gx, gdy, gr, gpart, M_BLOCKS, BPP))
    ct.launch(stream_int, (H // TD, 1, 1), _rmsnorm_dw_reduce, (gpart, gdw))
    cudart.cudaStreamSynchronize(stream_int)

    dx = bf16_bits_to_f32(gdx.to_numpy())
    dw = gdw.to_numpy().reshape(H)
    for g in (gx, gw, gdy, gr, gdx, gpart, gdw): g.free()
    return dx, dw
