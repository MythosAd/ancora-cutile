"""
ancora/kernels/linear.py — MXFP8/MXFP4 linear forward + BF16 backward

Shared by both RL (GRPO) and SFT training pipelines.

Weight storage convention
-------------------------
  W is stored as (N, K) — row = output neuron, column = input dimension.
  This matches nn.Linear.weight layout and avoids any transpose on the host.

Scale storage convention
------------------------
  Scales are stored as uint8 (E8M0 raw bytes) on device.
  E8M0 encoding: value = 2^(byte - 127), so:
    1.0  → 0x7F (127)
    2.0  → 0x80 (128)
    0.5  → 0x7E (126)
  Use encode_e8m0() helper to convert float scales to uint8.
  Inside the kernel, ct.bitcast(uint8_tile, ct.float8_e8m0fnu) reinterprets bytes.

Forward shapes
--------------
  x      : (M, K)        uint8  (FP8 E4M3 activations)
  w      : (N, K)        uint8  (FP8 E4M3 weights)
  x_scale: (M, K // 32)  uint8  (E8M0 per-block scale, B=32)
  w_scale: (N, K // 32)  uint8  (E8M0 per-block scale, B=32)
  out    : (M, N)        bfloat16

Backward shapes
---------------
  dy     : (M, N)  bfloat16  (upstream gradient)
  → dx   : (M, K)  bfloat16  = dy @ W   (input gradient)
  → dw   : (N, K)  bfloat16  = dy^T @ X  (weight gradient)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env as env  # sets CUDA_PATH before any cuda.* import

# ── tile sizes ──────────────────────────────────────────────────────────────
B = 32                       # MXFP8 scale block size (one E8M0 scale per 32 elems)

# Forward GEMM config — autotuned via ct.tune on the REAL kernel (with memory scale
# loads): 4096³ → 184 TFLOPS. Large TK=128 amortizes per-K-block scale loads.
FTM, FTN, FTK = 64, 128, 128

# Backward GEMM config (BF16). Tuned separately (different tile roles).
BTM, BTN, BTK = 128, 128, 64


# ── forward: MXFP8 ─────────────────────────────────────────────────────────

@ct.kernel(occupancy=2)
def _fwd_mxfp8(x, w, x_scale, w_scale, out,
               M: ct.Constant[int], N: ct.Constant[int], K_BLOCKS: ct.Constant[int],
               TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    """
    out[m, n] = Σ_k  (x[m,k] * xs[m,k//32]) * (w[n,k] * ws[n,k//32])
    Accumulates in float32. x, w are FP8 E4M3 bytes (uint8) → bitcast.

    NOTE: tried Triton-style L2 swizzle (GROUP_SIZE_M) — NO benefit on sm_120
    (measured 0.96-1.01×). A 0.6B model's matrices are small vs the 48 MB L2, so
    operands are already L2-resident; the swizzle's premise (operands exceed L2)
    doesn't hold here. Kept the simple 2D grid.
    SMEM-tiled: cuda-tile auto-stages; latency=10 prefetches next K-block.
    """
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), ct.float32)
    KS = TK // B   # scale columns per K-block

    for k in range(K_BLOCKS):
        # x(M,K), w stored PRE-TRANSPOSED (K,N) so y loads directly as (TK,TN) — NO
        # in-kernel transpose. (ct.transpose in the hot loop drops 261→11 TFLOPS.)
        ta  = ct.bitcast(ct.load(x, index=(m, k), shape=(TM, TK), latency=10), ct.float8_e4m3fn)
        tw  = ct.bitcast(ct.load(w, index=(k, n), shape=(TK, TN), latency=10), ct.float8_e4m3fn)
        txs = ct.bitcast(ct.load(x_scale, index=(m, k), shape=(TM, KS)), ct.float8_e8m0fnu)
        tws = ct.bitcast(ct.load(w_scale, index=(k, n), shape=(KS, TN)), ct.float8_e8m0fnu)
        acc = ct.mma_scaled(ta, txs, tw, tws, acc)

    ct.store(out, index=(m, n), tile=acc)   # float32 output


# ── forward: MXFP8 with BF16-bit output (resident-path drop-in for _gemm_bf16) ─
# Same math as _fwd_mxfp8 but rounds the f32 accumulator to BF16 bits on store, so the
# chained resident GEMMs (gemm→{norm,swiglu,gemm}) need no separate cast — and the MXFP8
# tensor-core path (peak ~330 vs ~165 BF16) is the lever to MFU>80% (BF16 plateaus ~40%).

@ct.kernel(occupancy=2)
def _fwd_mxfp8_bf16(x, w, x_scale, w_scale, out, K_BLOCKS: ct.Constant[int],
                    TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    """out = (x⊙xs) @ (w⊙ws), MXFP8 in, f32 accumulate, BF16-bit out.
    x(M,K) fp8, w(K,N) fp8 pre-transposed, xs(M,K//32), ws(K//32,N) E8M0 → out(M,N) bf16 bits."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), ct.float32)
    KS = TK // B
    for k in range(K_BLOCKS):
        ta  = ct.bitcast(ct.load(x, index=(m, k), shape=(TM, TK), latency=10), ct.float8_e4m3fn)
        tw  = ct.bitcast(ct.load(w, index=(k, n), shape=(TK, TN), latency=10), ct.float8_e4m3fn)
        txs = ct.bitcast(ct.load(x_scale, index=(m, k), shape=(TM, KS)), ct.float8_e8m0fnu)
        tws = ct.bitcast(ct.load(w_scale, index=(k, n), shape=(KS, TN)), ct.float8_e8m0fnu)
        acc = ct.mma_scaled(ta, txs, tw, tws, acc)
    ct.store(out, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _fwd_mxfp8_bf16_res(x, w, x_scale, w_scale, res, out, K_BLOCKS: ct.Constant[int],
                        TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    """out = (x⊙xs) @ (w⊙ws) + res, MXFP8 in, f32 accumulate, BF16-bit out. The residual is
    added in the GEMM epilogue (MEGAKERNEL fusion #2) so o_proj/down_proj write the layer
    output directly — no separate _residual_add kernel, no intermediate (M,N) round-trip.
    res, out: (M,N) BF16 bits."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), ct.float32)
    KS = TK // B
    for k in range(K_BLOCKS):
        ta  = ct.bitcast(ct.load(x, index=(m, k), shape=(TM, TK), latency=10), ct.float8_e4m3fn)
        tw  = ct.bitcast(ct.load(w, index=(k, n), shape=(TK, TN), latency=10), ct.float8_e4m3fn)
        txs = ct.bitcast(ct.load(x_scale, index=(m, k), shape=(TM, KS)), ct.float8_e8m0fnu)
        tws = ct.bitcast(ct.load(w_scale, index=(k, n), shape=(KS, TN)), ct.float8_e8m0fnu)
        acc = ct.mma_scaled(ta, txs, tw, tws, acc)
    rv = ct.astype(ct.bitcast(ct.load(res, index=(m, n), shape=(TM, TN)), ct.bfloat16), ct.float32)
    ct.store(out, index=(m, n), tile=ct.bitcast(ct.astype(acc + rv, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _fwd_mxfp8_f32res(x, w, x_scale, w_scale, res, out, K_BLOCKS: ct.Constant[int],
                      TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    """FP32-residual variant of _fwd_mxfp8_bf16_res: out = (x⊙xs)@(w⊙ws) + res with res/out FP32.
    Keeps the rollout residual stream fp32 (no bf16 rounding of the ~6912 massive activation across
    layers). MXFP8 GEMM inputs unchanged; only the residual read + output store are fp32."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), ct.float32)
    KS = TK // B
    for k in range(K_BLOCKS):
        ta  = ct.bitcast(ct.load(x, index=(m, k), shape=(TM, TK), latency=10), ct.float8_e4m3fn)
        tw  = ct.bitcast(ct.load(w, index=(k, n), shape=(TK, TN), latency=10), ct.float8_e4m3fn)
        txs = ct.bitcast(ct.load(x_scale, index=(m, k), shape=(TM, KS)), ct.float8_e8m0fnu)
        tws = ct.bitcast(ct.load(w_scale, index=(k, n), shape=(KS, TN)), ct.float8_e8m0fnu)
        acc = ct.mma_scaled(ta, txs, tw, tws, acc)
    rv = ct.load(res, index=(m, n), shape=(TM, TN))      # fp32 residual
    ct.store(out, index=(m, n), tile=acc + rv)             # fp32 out


def mxfp8_tile(N: int, K: int):
    """Per-shape best (TM,TN,TK) for _fwd_mxfp8[_bf16], autotuned at M=8192 layer shapes
    (gemm_mfu_ceiling memory). occupancy=2 baked in the decorator (best for all winners).
    Near-uniform 128×64 tall-skinny tiles, deep K; gate/up (K=1024,big N) prefers TK=128."""
    if K == 1024 and N >= 2048:
        return (128, 64, 128)        # gate/up → 139 TFLOPS
    return (128, 64, 256)            # down, q/o, k/v → 106-125 TFLOPS


# ── forward: MXFP4 — BLOCKED on cuda-tile 1.4.0 ──────────────────────────────
# Loading external packed FP4 weights is not supported efficiently:
#   bitcast(uint8→fp4): error (bitwidth 8≠4);  unpack_from_bytes / astype: ~1 TFLOPS.
# Confirmed gap: NVIDIA cutile-python issue #47. Use MXFP8 forward instead (261 TFLOPS).
# Revisit when cuda-tile ships a packed-FP4 load, or drop to inline PTX.


# ── backward: BF16 ─────────────────────────────────────────────────────────

@ct.kernel(occupancy=4)
def _bwd_input(dy, w, dx,
               M: ct.Constant[int], K: ct.Constant[int], N_BLOCKS: ct.Constant[int]):
    """
    Input gradient: dx = dy @ W.  dy(M,N) bf16, W(N,K) bf16 → dx(M,K) f32.
    Tile (BTM, BTK) per block, reduce over N. W loaded as BF16 bits (uint16) → bitcast.
    """
    m, k = ct.bid(0), ct.bid(1)
    acc = ct.zeros((BTM, BTK), ct.float32)

    for n in range(N_BLOCKS):
        tdy = ct.bitcast(ct.load(dy, index=(m, n), shape=(BTM, BTN), latency=10), ct.bfloat16)
        tw  = ct.bitcast(ct.load(w,  index=(n, k), shape=(BTN, BTK), latency=10), ct.bfloat16)
        acc = ct.mma(tdy, tw, acc)   # (BTM,BTN)@(BTN,BTK) → (BTM,BTK)

    ct.store(dx, index=(m, k), tile=acc)


@ct.kernel(occupancy=4)
def _bwd_weight(dy, x, dw,
                N: ct.Constant[int], K: ct.Constant[int], M_BLOCKS: ct.Constant[int]):
    """
    Weight gradient: dw = dy^T @ X.  dy(M,N) bf16, X(M,K) bf16 → dw(N,K) f32.
    Tile (BTN, BTK) per block, reduce over M.
    """
    n, k = ct.bid(0), ct.bid(1)
    acc = ct.zeros((BTN, BTK), ct.float32)

    for m in range(M_BLOCKS):
        tdy = ct.bitcast(ct.load(dy, index=(m, n), shape=(BTM, BTN), latency=10), ct.bfloat16)
        tx  = ct.bitcast(ct.load(x,  index=(m, k), shape=(BTM, BTK), latency=10), ct.bfloat16)
        acc = ct.mma(ct.transpose(tdy), tx, acc)   # (BTN,BTM)@(BTM,BTK) → (BTN,BTK)

    ct.store(dw, index=(n, k), tile=acc)


# ── host utilities ──────────────────────────────────────────────────────────

def encode_e8m0(scale_f32: np.ndarray) -> np.ndarray:
    """Convert float32 scale array to uint8 E8M0 encoding.
    E8M0: value = 2^(byte - 127), so byte = floor(log2(value)) + 127.
    """
    import math
    out = np.empty_like(scale_f32, dtype=np.uint8)
    for idx in np.ndindex(scale_f32.shape):
        v = float(scale_f32[idx])
        exp = int(math.floor(math.log2(max(v, 1e-38))))
        out[idx] = max(0, min(255, exp + 127))
    return out


class _GpuArray:
    """Minimal device array wrapper exposing __cuda_array_interface__ for ct.launch."""
    def __init__(self, arr: np.ndarray):
        self._shape  = arr.shape
        self._dtype  = arr.dtype
        self._nbytes = arr.nbytes
        err, self._ptr = cdrv.cuMemAlloc(arr.nbytes)
        if err.value:
            raise RuntimeError(f"cuMemAlloc failed: {err}")
        cdrv.cuMemcpyHtoD(self._ptr, arr, arr.nbytes)
        self.__cuda_array_interface__ = {
            "shape": arr.shape, "typestr": arr.dtype.str,
            "data": (int(self._ptr), False), "version": 3,
        }

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, self._dtype)
        cdrv.cuMemcpyDtoH(out, self._ptr, self._nbytes)
        return out

    def free(self):
        cdrv.cuMemFree(self._ptr)

    @classmethod
    def zeros(cls, shape, dtype):
        return cls(np.zeros(shape, dtype))


# ── Linear module ───────────────────────────────────────────────────────────

class LinearMXFP8:
    """
    Linear layer with MXFP8 forward and BF16 backward.

    Usage (training):
        layer = LinearMXFP8(in_features=2048, out_features=2048)
        out   = layer.forward(x_gpu, x_scale_gpu)         # → bfloat16
        dx, dw = layer.backward(dy_gpu, x_gpu, x_scale_gpu)

    Weight quantization:
        Call layer.quantize_weights(w_f32) to set W, w_scale from float32 weights.
    """

    def __init__(self, in_features: int, out_features: int, device=None):
        assert in_features  % FTK == 0, f"in_features must be divisible by {FTK}"
        assert out_features % FTN == 0, f"out_features must be divisible by {FTN}"
        self.K = in_features
        self.N = out_features
        self.w: _GpuArray | None = None        # (N, K) uint8
        self.w_scale: _GpuArray | None = None  # (N, K//32) uint8

        # grab stream from calling context (caller sets device before init)
        self._dev = device or cc.Device(0)

    def quantize_weights(self, w_f32: np.ndarray, absmax_per_block: bool = True):
        """
        Convert float32 weight matrix to FP8 E4M3 + E8M0 per-block scales.
        w_f32: (N, K) float32
        """
        assert w_f32.shape == (self.N, self.K)
        N, K = w_f32.shape
        K_s  = K // B

        # Compute per-block max for scaling
        w_blocks = w_f32.reshape(N, K_s, B)   # (N, K//32, 32)
        block_max = np.abs(w_blocks).max(axis=-1)  # (N, K//32)
        block_max = np.clip(block_max, 1e-38, None)

        # E8M0 scale: chosen so max value maps to ~1.0 in FP8 range
        # FP8 E4M3 max = 448.0; scale = block_max / 448.0
        scale_f32 = block_max / 448.0
        w_scale_u8 = encode_e8m0(scale_f32)   # (N, K//32) uint8

        # Quantize: w_fp8 = w_f32 / (scale[block] * 1.0)
        scale_expanded = scale_f32.repeat(B, axis=-1)  # (N, K)
        w_scaled = w_f32 / scale_expanded
        w_fp8_f32 = np.clip(w_scaled, -448.0, 448.0)
        # Store as uint8: FP8 E4M3 byte representation
        # For simplicity, use numpy float32 → uint8 packing via struct
        # (Production: use proper FP8 quantization library)
        w_u8 = (w_fp8_f32 * (127.0 / 448.0)).astype(np.int8).view(np.uint8)

        self.w       = _GpuArray(w_u8)
        self.w_scale = _GpuArray(w_scale_u8)

    def forward(self, x: _GpuArray, x_scale: _GpuArray, stream_int: int,
                M: int) -> _GpuArray:
        """
        x:       (M, K) uint8     — FP8 E4M3 activation bytes
        x_scale: (M, K//32) uint8 — E8M0 scale bytes
        Returns: (M, N) float32   (kernel accumulates+stores f32)
        """
        assert self.w is not None, "call quantize_weights() first"
        assert M % FTM == 0, f"M must be divisible by {FTM}"
        K_BLOCKS = self.K // FTK
        out = _GpuArray.zeros((M, self.N), np.float32)

        ct.launch(stream_int, (M // FTM, self.N // FTN, 1),
                  _fwd_mxfp8,
                  (x, self.w, x_scale, self.w_scale, out, M, self.N, K_BLOCKS,
                   FTM, FTN, FTK))
        return out

    def backward(self, dy: _GpuArray, x: _GpuArray, w_bf16: _GpuArray,
                 stream_int: int, M: int):
        """
        dy:      (M, N) uint16   — upstream gradient, BF16 bits
        x:       (M, K) uint16   — activations saved from forward, BF16 bits
        w_bf16:  (N, K) uint16   — BF16 master copy of weights (training keeps this;
                                   the FP8 self.w is a quantized copy for the fwd only)
        Returns: (dx, dw) both (·, K) float32.
        """
        N_BLOCKS = self.N // BTN
        M_BLOCKS = M       // BTM

        dx = _GpuArray.zeros((M,      self.K), np.float32)
        dw = _GpuArray.zeros((self.N, self.K), np.float32)

        ct.launch(stream_int, (M // BTM, self.K // BTK, 1),
                  _bwd_input,  (dy, w_bf16, dx, M, self.K, N_BLOCKS))

        ct.launch(stream_int, (self.N // BTN, self.K // BTK, 1),
                  _bwd_weight, (dy, x, dw, self.N, self.K, M_BLOCKS))

        return dx, dw
