"""
ancora/kernels/rope.py — Rotary Position Embedding (RoPE) forward + backward

Qwen3-0.6B: rope_theta = 1e6, head_dim = 128, applied to Q and K AFTER QK-Norm.

Reference implementations studied
---------------------------------
  HF transformers Qwen3/Llama `apply_rotary_pos_emb` (rotate_half style):
      q_embed = q*cos + rotate_half(q)*sin, rotate_half(x)=cat(-x[d/2:], x[:d/2]).
      cos/sin are width-D (the half is DUPLICATED across the two halves).
  vLLM csrc/pos_encoding_kernels.cu `rotary_embedding_kernel`, is_neox_style=True:
      splits x into [0:d/2] (x1) and [d/2:d] (x2); precomputed cos_sin_cache indexed
      by position. (Qwen3 = NEOX/rotate-half, NOT GPT-J interleaved.)
  SGLang / FlashInfer: same NEOX rotate-half over a precomputed cos/sin cache.

Because cos/sin are duplicated, both halves share one width-(D/2) cos `c` and sin `s`:
      y1 = x1·c − x2·s
      y2 = x2·c + x1·s          (rotation by +angle)
Backward (rotation by −angle, J = [[c,−s],[s,c]] → dx = Jᵀ dy):
      dx1 =  dy1·c + dy2·s
      dx2 = −dy1·s + dy2·c

Batch invariance: RoPE is elementwise per (token, head, dim-pair) — no cross-token
reduction → trivially batch-invariant (rollout and training rotate identically).

Layout / precision
------------------
  x, y, dy, dx : uint16 BF16 bits, flat (B*H*S, D), row = (b*H+h)*S + pos.
  cos, sin     : float32, shape (S, D/2)  (the half; built on host from positions).
  Grid: (S // RTM, B*H);  bid(0)=position block, bid(1)=batch*head.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # sets CUDA_PATH

RTM = 32   # sequence positions per block


@ct.kernel
def _rope_fwd(x, cos, sin, y,
              NSB: ct.Constant[int],   # S // RTM (position-blocks per head)
              Dh:  ct.Constant[int]):  # D // 2
    """y1 = x1·c − x2·s ; y2 = x2·c + x1·s.  cos/sin shared across all heads."""
    pb, bh = ct.bid(0), ct.bid(1)
    rb = bh * NSB + pb                    # row-block of this (head, position-block)

    x1 = ct.astype(ct.bitcast(ct.load(x, index=(rb, 0), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    x2 = ct.astype(ct.bitcast(ct.load(x, index=(rb, 1), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    c  = ct.load(cos, index=(pb, 0), shape=(RTM, Dh))
    s  = ct.load(sin, index=(pb, 0), shape=(RTM, Dh))

    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    ct.store(y, index=(rb, 0), tile=ct.bitcast(ct.astype(y1, ct.bfloat16), ct.uint16))
    ct.store(y, index=(rb, 1), tile=ct.bitcast(ct.astype(y2, ct.bfloat16), ct.uint16))


@ct.kernel
def _rope_fwd_dec(x, cos_p, sin_p, y, Dh: ct.Constant[int]):
    """Single-position RoPE for DECODE: every row is at the SAME position pos, so cos_p,sin_p are
    (1,Dh) = cos/sin[pos] broadcast over rows. Identical fp32 op to _rope_fwd → the frontier token's
    RoPE'd Q/K is BITWISE-equal to training's prefill at pos. Grid (R//RTM,). x,y:(R, 2*Dh) head-major."""
    rb = ct.bid(0)
    x1 = ct.astype(ct.bitcast(ct.load(x, index=(rb, 0), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    x2 = ct.astype(ct.bitcast(ct.load(x, index=(rb, 1), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    c  = ct.broadcast_to(ct.load(cos_p, index=(0, 0), shape=(1, Dh)), (RTM, Dh))
    s  = ct.broadcast_to(ct.load(sin_p, index=(0, 0), shape=(1, Dh)), (RTM, Dh))
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    ct.store(y, index=(rb, 0), tile=ct.bitcast(ct.astype(y1, ct.bfloat16), ct.uint16))
    ct.store(y, index=(rb, 1), tile=ct.bitcast(ct.astype(y2, ct.bfloat16), ct.uint16))


@ct.kernel
def _rope_fwd_dec_p(x, cosT, sinT, y, gpos, Dh: ct.Constant[int]):
    """DEVICE-POSITION _rope_fwd_dec: the position comes from a (1,1) i32 device buffer instead of
    host-offset cos/sin pointers - CUDA-graph replayable (a graph would freeze an at_pos pointer).
    cosT/sinT are the FULL (maxS, Dh) tables; row `pos` is selected in-kernel (data-dependent load,
    probed OK in _probe_devpos.py). Identical fp32 math to _rope_fwd_dec/_rope_fwd => bitwise.
    Grid (R//RTM,)."""
    rb = ct.bid(0)
    pos = ct.reshape(ct.load(gpos, index=(0, 0), shape=(1, 1)), ())
    x1 = ct.astype(ct.bitcast(ct.load(x, index=(rb, 0), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    x2 = ct.astype(ct.bitcast(ct.load(x, index=(rb, 1), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    c  = ct.broadcast_to(ct.load(cosT, index=(pos, 0), shape=(1, Dh)), (RTM, Dh))
    s  = ct.broadcast_to(ct.load(sinT, index=(pos, 0), shape=(1, Dh)), (RTM, Dh))
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    ct.store(y, index=(rb, 0), tile=ct.bitcast(ct.astype(y1, ct.bfloat16), ct.uint16))
    ct.store(y, index=(rb, 1), tile=ct.bitcast(ct.astype(y2, ct.bfloat16), ct.uint16))


@ct.kernel
def _rope_bwd(dy, cos, sin, dx,
              NSB: ct.Constant[int], Dh: ct.Constant[int]):
    """dx1 = dy1·c + dy2·s ; dx2 = −dy1·s + dy2·c  (rotate by −angle)."""
    pb, bh = ct.bid(0), ct.bid(1)
    rb = bh * NSB + pb

    d1 = ct.astype(ct.bitcast(ct.load(dy, index=(rb, 0), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    d2 = ct.astype(ct.bitcast(ct.load(dy, index=(rb, 1), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    c  = ct.load(cos, index=(pb, 0), shape=(RTM, Dh))
    s  = ct.load(sin, index=(pb, 0), shape=(RTM, Dh))

    x1 = d1 * c + d2 * s
    x2 = d2 * c - d1 * s
    ct.store(dx, index=(rb, 0), tile=ct.bitcast(ct.astype(x1, ct.bfloat16), ct.uint16))
    ct.store(dx, index=(rb, 1), tile=ct.bitcast(ct.astype(x2, ct.bfloat16), ct.uint16))


@ct.kernel
def _rope_fwd_tok(x, cos, sin, y,
                  NSB: ct.Constant[int],   # S // RTM  (position-blocks per sequence)
                  Dh:  ct.Constant[int]):  # D // 2
    """Token-major RoPE: x,y are (M, H*D); rotate each head's D in place. Pairs with the
    token-major attention (_attn_fwd_tok) so NO tok→head transpose is needed. Grid (M//RTM, H):
    bid(0)=row-block over M (positions cross batches; pos-block = mb % NSB), bid(1)=head."""
    mb, h = ct.bid(0), ct.bid(1)
    pb = mb % NSB                          # position-block within the sequence (cos/sin row)
    x1 = ct.astype(ct.bitcast(ct.load(x, index=(mb, h * 2),     shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    x2 = ct.astype(ct.bitcast(ct.load(x, index=(mb, h * 2 + 1), shape=(RTM, Dh)), ct.bfloat16), ct.float32)
    c  = ct.load(cos, index=(pb, 0), shape=(RTM, Dh))
    s  = ct.load(sin, index=(pb, 0), shape=(RTM, Dh))
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    ct.store(y, index=(mb, h * 2),     tile=ct.bitcast(ct.astype(y1, ct.bfloat16), ct.uint16))
    ct.store(y, index=(mb, h * 2 + 1), tile=ct.bitcast(ct.astype(y2, ct.bfloat16), ct.uint16))


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
    u = u + 0x7FFF + ((u >> 16) & 1)         # round-to-nearest-even
    return (u >> 16).astype(np.uint16)


def bf16_bits_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def build_cos_sin(S: int, D: int, base: float = 1e6, dtype=np.float32):
    """Qwen3/Llama rotate-half cos/sin cache, shape (S, D/2) each.
    inv_freq[j] = base^(-2j/D); angle[s,j] = s·inv_freq[j]."""
    Dh = D // 2
    inv_freq = base ** (-(np.arange(Dh, dtype=np.float64) * 2.0 / D))   # (D/2,)
    pos = np.arange(S, dtype=np.float64)[:, None]                        # (S,1)
    ang = pos * inv_freq[None, :]                                        # (S, D/2)
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


# ── public API ───────────────────────────────────────────────────────────────

def rope_forward(x_f32: np.ndarray, stream_int: int, base: float = 1e6):
    """x_f32: (B, H, S, D) float32 → y (B,H,S,D) float32. Inputs rounded to BF16."""
    Bb, H, S, D = x_f32.shape
    assert S % RTM == 0 and D % 2 == 0
    cos, sin = build_cos_sin(S, D, base)
    flat = x_f32.reshape(Bb * H * S, D)
    gx = _GpuArray(f32_to_bf16_bits(flat))
    gc = _GpuArray(cos); gs = _GpuArray(sin)
    gy = _GpuArray.zeros((Bb * H * S, D), np.uint16)

    ct.launch(stream_int, (S // RTM, Bb * H, 1), _rope_fwd, (gx, gc, gs, gy, S // RTM, D // 2))
    cudart.cudaStreamSynchronize(stream_int)

    y = bf16_bits_to_f32(gy.to_numpy()).reshape(Bb, H, S, D)
    for g in (gx, gc, gs, gy): g.free()
    return y


def rope_backward(dy_f32: np.ndarray, stream_int: int, base: float = 1e6):
    """dy_f32: (B,H,S,D) f32 → dx (B,H,S,D) f32."""
    Bb, H, S, D = dy_f32.shape
    assert S % RTM == 0 and D % 2 == 0
    cos, sin = build_cos_sin(S, D, base)
    flat = dy_f32.reshape(Bb * H * S, D)
    gdy = _GpuArray(f32_to_bf16_bits(flat))
    gc = _GpuArray(cos); gs = _GpuArray(sin)
    gdx = _GpuArray.zeros((Bb * H * S, D), np.uint16)

    ct.launch(stream_int, (S // RTM, Bb * H, 1), _rope_bwd, (gdy, gc, gs, gdx, S // RTM, D // 2))
    cudart.cudaStreamSynchronize(stream_int)

    dx = bf16_bits_to_f32(gdx.to_numpy()).reshape(Bb, H, S, D)
    for g in (gdy, gc, gs, gdx): g.free()
    return dx
