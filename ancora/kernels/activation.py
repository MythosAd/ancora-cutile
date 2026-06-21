"""
ancora/kernels/activation.py — SwiGLU forward + backward (BF16 I/O, FP32 compute)

Qwen3 MLP: down_proj( silu(gate_proj(x)) * up_proj(x) ).  This kernel does the
elementwise  out = silu(gate) * up  fusion in the middle.

  silu(g) = g * sigmoid(g) = g * 0.5 * (1 + tanh(0.5 g))   (tanh form — ct.tanh exists,
            avoids a divide; numerically identical to g/(1+exp(-g))).

Reference: PyTorch F.silu / LlamaMLP / Qwen3MLP (act_fn = SiLU, gate*up).
vLLM/SGLang fuse silu+mul into one kernel (`silu_and_mul`) exactly like this.

Backward (out = silu(g) * u):
  d_gate = d_out * u * silu'(g),  silu'(g) = sig(g) * (1 + g*(1 - sig(g)))
  d_up   = d_out * silu(g)
where sig(g) = 0.5*(1 + tanh(0.5 g)).

Batch invariance: pure elementwise (no cross-token reduction) → trivially invariant.
Grid: (M // TM, I // TI).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # sets CUDA_PATH

TM = 64    # tokens per block
TI = 128   # intermediate columns per block


@ct.kernel
def _swiglu_fwd(gate, up, out):
    """out = silu(gate) * up.  Grid: (M // TM, I // TI)."""
    m, n = ct.bid(0), ct.bid(1)
    g = ct.astype(ct.bitcast(ct.load(gate, index=(m, n), shape=(TM, TI)), ct.bfloat16), ct.float32)
    u = ct.astype(ct.bitcast(ct.load(up,   index=(m, n), shape=(TM, TI)), ct.bfloat16), ct.float32)
    sig  = 0.5 + 0.5 * ct.tanh(0.5 * g)        # sigmoid(g)
    silu = g * sig
    y = silu * u
    ct.store(out, index=(m, n), tile=ct.bitcast(ct.astype(y, ct.bfloat16), ct.uint16))


QB = 32      # MXFP8 scale block (= quant.B)
TMQ = 64     # rows per block for the fused swiglu-quant


@ct.kernel
def _swiglu_fwd_q(gate, up, fp8_out, scale_out, KB: ct.Constant[int]):
    """FUSED SwiGLU + MXFP8 quant (CODA epilogue fusion): out=silu(gate)*up, then per-32
    E8M0 quant → fp8_out(M,I) u8 + scale_out(M,I//32) u8 — a direct down_proj _fwd_mxfp8
    input with NO separate quant launch / bf16 round-trip. Grid (M//TMQ,), chunk = one
    32-wide quant block. Quant math identical to quant._quant_mxfp8."""
    m = ct.bid(0)
    for kb in range(KB):
        g = ct.astype(ct.bitcast(ct.load(gate, index=(m, kb), shape=(TMQ, QB)), ct.bfloat16), ct.float32)
        u = ct.astype(ct.bitcast(ct.load(up,   index=(m, kb), shape=(TMQ, QB)), ct.bfloat16), ct.float32)
        sig = 0.5 + 0.5 * ct.tanh(0.5 * g)
        yv  = (g * sig) * u
        amax = ct.max(ct.maximum(yv, 0.0 - yv), axis=-1, keepdims=True)
        ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
        byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((TMQ, 1), 0, ct.uint32))
        sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
        fp8 = ct.bitcast(ct.astype(yv / ct.broadcast_to(sc, (TMQ, QB)), ct.float8_e4m3fn), ct.uint8)
        ct.store(fp8_out,   index=(m, kb), tile=fp8)
        ct.store(scale_out, index=(m, kb), tile=ct.astype(byte, ct.uint8))


@ct.kernel
def _swiglu_bwd(gate, up, dout, dgate, dup):
    """d_gate = dout*u*silu'(g);  d_up = dout*silu(g).  Grid: (M // TM, I // TI)."""
    m, n = ct.bid(0), ct.bid(1)
    g  = ct.astype(ct.bitcast(ct.load(gate, index=(m, n), shape=(TM, TI)), ct.bfloat16), ct.float32)
    u  = ct.astype(ct.bitcast(ct.load(up,   index=(m, n), shape=(TM, TI)), ct.bfloat16), ct.float32)
    do = ct.astype(ct.bitcast(ct.load(dout, index=(m, n), shape=(TM, TI)), ct.bfloat16), ct.float32)
    sig  = 0.5 + 0.5 * ct.tanh(0.5 * g)
    silu = g * sig
    dsilu = sig * (1.0 + g * (1.0 - sig))      # d silu / d g
    dg = do * u * dsilu
    du = do * silu
    ct.store(dgate, index=(m, n), tile=ct.bitcast(ct.astype(dg, ct.bfloat16), ct.uint16))
    ct.store(dup,   index=(m, n), tile=ct.bitcast(ct.astype(du, ct.bfloat16), ct.uint16))


# ── host helpers ─────────────────────────────────────────────────────────────

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


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_bits_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def swiglu_forward(gate_f32: np.ndarray, up_f32: np.ndarray, stream_int: int):
    """gate,up: (M, I) f32 → out (M, I) f32. Inputs rounded to BF16."""
    M, I = gate_f32.shape
    assert M % TM == 0 and I % TI == 0 and up_f32.shape == (M, I)
    gg = _GpuArray(f32_to_bf16_bits(gate_f32)); gu = _GpuArray(f32_to_bf16_bits(up_f32))
    go = _GpuArray.zeros((M, I), np.uint16)
    ct.launch(stream_int, (M // TM, I // TI, 1), _swiglu_fwd, (gg, gu, go))
    cudart.cudaStreamSynchronize(stream_int)
    out = bf16_bits_to_f32(go.to_numpy())
    for g in (gg, gu, go): g.free()
    return out


def swiglu_backward(gate_f32, up_f32, dout_f32, stream_int):
    """→ (dgate, dup) both (M,I) f32."""
    M, I = gate_f32.shape
    gg = _GpuArray(f32_to_bf16_bits(gate_f32)); gu = _GpuArray(f32_to_bf16_bits(up_f32))
    gd = _GpuArray(f32_to_bf16_bits(dout_f32))
    gdg = _GpuArray.zeros((M, I), np.uint16); gdu = _GpuArray.zeros((M, I), np.uint16)
    ct.launch(stream_int, (M // TM, I // TI, 1), _swiglu_bwd, (gg, gu, gd, gdg, gdu))
    cudart.cudaStreamSynchronize(stream_int)
    dgate = bf16_bits_to_f32(gdg.to_numpy()); dup = bf16_bits_to_f32(gdu.to_numpy())
    for g in (gg, gu, gd, gdg, gdu): g.free()
    return dgate, dup
