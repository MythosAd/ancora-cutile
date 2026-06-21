"""_attn_decode_mma (unified decode) vs _attn_fwd (prefill) — BITWISE.

The frontier token's attention computed by the single-query decode must equal training's
prefill row exactly (else π_train≠π_infer). _attn_decode_mma mirrors _attn_fwd's reduction
(MMA q·kᵀ / p·v, exp2 online softmax, fixed kv order). This checks bitwise equality across
positions: block-aligned, mid-block, and across multiple blocks."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa
from ancora.kernels.attention import _attn_fwd, _attn_decode_blk, D, BQ, BKV, _f32_to_bf16_bits as f32bf

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


class GA:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
    def free(s): cdrv.cuMemFree(s.p)


B, Hq, Hkv, S = 1, 2, 1, 256
NQB, NKVB = S // BQ, S // BKV
scale = 1.0 / math.sqrt(D)
rng = np.random.default_rng(0)
Q = (rng.standard_normal((B, Hq, S, D)) * 0.5).astype(np.float32)
K = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)
V = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)

# prefill (training): O for all positions
gQ = GA(f32bf(Q.reshape(-1, D))); gK = GA(f32bf(K.reshape(-1, D))); gV = GA(f32bf(V.reshape(-1, D)))
gO = GA(np.zeros((B * Hq * S, D), np.float32)); gL = GA(np.zeros((B * Hq * S, 1), np.float32))
ct.launch(si, (NQB, B * Hq, 1), _attn_fwd, (gQ, gK, gV, gO, gL, NQB, NKVB, Hq, Hkv, scale))
cudart.cudaStreamSynchronize(si)
O_train = gO.np().reshape(B, Hq, S, D)

# cache = the same sequence K/V (decode reads blocks 0..q_blk; >pos masked anyway)
gKc = GA(f32bf(K.reshape(B * Hkv * S, D))); gVc = GA(f32bf(V.reshape(B * Hkv * S, D)))
bits = lambda x: x.view(np.uint32)

print(f"_attn_decode_blk vs _attn_fwd (bitwise) — D={D} BQ={BQ} S={S}")
print("(frontier query at row pmod; OTHER block rows are GARBAGE → must not affect row pmod)")
print("=" * 60)
grng = np.random.default_rng(99)
ok = True
for pos in [0, 33, 63, 64, 100, 127, 128, 200, 255]:
    q_blk, pmod = pos // BQ, pos % BQ
    blk = (grng.standard_normal((B, Hq, BQ, D)) * 3).astype(np.float32)   # garbage block
    blk[:, :, pmod] = Q[:, :, pos]                                        # real frontier query at pmod
    gQd = GA(f32bf(blk.reshape(B * Hq * BQ, D)))
    gOd = GA(np.zeros((B * Hq * BQ, D), np.float32))
    ct.launch(si, (B * Hq, 1, 1), _attn_decode_blk, (gQd, gKc, gVc, gOd, NKVB, Hq, Hkv, scale, int(q_blk)))
    cudart.cudaStreamSynchronize(si)
    Od = gOd.np().reshape(B, Hq, BQ, D)[:, :, pmod]                       # extract frontier row
    same = np.array_equal(bits(Od), bits(O_train[:, :, pos]))
    mx = np.abs(Od - O_train[:, :, pos]).max()
    ok &= same
    print(f"  pos={pos:3d} (blk {q_blk}, row {pmod:2d}): bitwise={same}  max|Δ|={mx:.3g}  {'OK' if same else 'FAIL'}")
    gQd.free(); gOd.free()

print("=" * 60)
print(f"  {'PASS — unified decode is bitwise-exact vs prefill (ratio=1 attention)' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
