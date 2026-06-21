"""Device-resident decode plumbing — bitwise vs prefill. Validates the NEW kernels that let a
device decode reuse training's exact reduction:
  (1) _rope_fwd_dec (single-position RoPE)  vs _rope_fwd row at pos
  (2) _scatter_blk → _attn_decode_blk → _gather_blk (device KV-cache decode) vs _attn_fwd row at pos
Both must be BITWISE (max|Δ|=0) — that's the ratio=1 requirement. These are the device pieces;
the full ResidentDecodeLayer just chains them with the existing _gemm_bf16/_rms kernels."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa
from ancora.kernels.attention import (_attn_fwd, _attn_decode_blk, _scatter_blk, _gather_blk,
                                       D, BQ, BKV, _f32_to_bf16_bits as f32bf)
from ancora.kernels.rope import _rope_fwd, _rope_fwd_dec, build_cos_sin, RTM

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
bits = lambda x: x.view(np.uint32)


class GA:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
    def at(s, rows):   # ptr += rows*D elements (host pointer-offset view)
        v = type("V", (), {})(); it = np.dtype(s.dt).itemsize
        v.__cuda_array_interface__ = {"shape": s.sh, "typestr": np.dtype(s.dt).str,
                                      "data": (int(s.p) + rows * D * it, False), "version": 3}
        return v
    def free(s): cdrv.cuMemFree(s.p)


def test_rope_dec():
    print("--- (1) _rope_fwd_dec vs _rope_fwd at pos (bitwise) ---")
    R, S, Dh, pos = 64, 256, D // 2, 100               # R rows (head-major), one position
    rng = np.random.default_rng(0)
    x = f32bf((rng.standard_normal((R, S, D)) * 0.5).astype(np.float32))   # (R, S, D) head-major
    cos, sin = build_cos_sin(S, D, 1e6)
    # full-seq prefill rope → take pos
    gx = GA(x.reshape(R * S, D)); gc = GA(cos); gsn = GA(sin); gy = GA(np.zeros((R * S, D), np.uint16))
    ct.launch(si, (S // RTM, R, 1), _rope_fwd, (gx, gc, gsn, gy, S // RTM, Dh))
    cudart.cudaStreamSynchronize(si)
    y_pre = gy.np().reshape(R, S, D)[:, pos]            # (R, D) bf16 bits
    # single-position decode rope at pos
    gxd = GA(x[:, pos].reshape(R, D)); gcp = GA(cos[pos:pos + 1]); gsp = GA(sin[pos:pos + 1])
    gyd = GA(np.zeros((R, D), np.uint16))
    ct.launch(si, (R // RTM, 1, 1), _rope_fwd_dec, (gxd, gcp, gsp, gyd, Dh))
    cudart.cudaStreamSynchronize(si)
    y_dec = gyd.np().reshape(R, D)
    same = np.array_equal(y_pre, y_dec)
    print(f"    bitwise={same}  {'OK' if same else 'FAIL'}")
    for g in (gx, gc, gsn, gy, gxd, gcp, gsp, gyd): g.free()
    return same


def test_decode_attn_plumbing():
    print("--- (2) device decode attn (scatter→attn_decode_blk→gather) vs _attn_fwd at pos (bitwise) ---")
    B, Hq, Hkv, S = 1, 2, 1, 256
    maxS, scale = S, 1.0 / math.sqrt(D)
    NQB, NKVB = S // BQ, S // BKV
    rng = np.random.default_rng(1)
    Q = (rng.standard_normal((B, Hq, S, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((B, Hkv, S, D)) * 0.5).astype(np.float32)
    # prefill attention reference
    gQ = GA(f32bf(Q.reshape(-1, D))); gK = GA(f32bf(K.reshape(-1, D))); gV = GA(f32bf(V.reshape(-1, D)))
    gO = GA(np.zeros((B * Hq * S, D), np.float32)); gL = GA(np.zeros((B * Hq * S, 1), np.float32))
    ct.launch(si, (NQB, B * Hq, 1), _attn_fwd, (gQ, gK, gV, gO, gL, NQB, NKVB, Hq, Hkv, scale))
    cudart.cudaStreamSynchronize(si)
    O_train = gO.np().reshape(B, Hq, S, D)

    Kc = GA(f32bf(K.reshape(B * Hkv * maxS, D))); Vc = GA(f32bf(V.reshape(B * Hkv * maxS, D)))   # device cache
    ok = True
    for pos in [50, 64, 100, 200, 255]:
        q_blk, pmod = pos // BQ, pos % BQ
        blockQ = GA(np.zeros((B * Hq * BQ, D), np.uint16))                    # garbage block
        gQpos = GA(f32bf(Q[:, :, pos].reshape(B * Hq, D)))
        ct.launch(si, (B * Hq, 1, 1), _scatter_blk, (gQpos, blockQ.at(pmod), BQ))   # frontier → row pmod
        blockO = GA(np.zeros((B * Hq * BQ, D), np.float32))
        ct.launch(si, (B * Hq, 1, 1), _attn_decode_blk, (blockQ, Kc, Vc, blockO, NKVB, Hq, Hkv, scale, int(q_blk)))
        O_dec = GA(np.zeros((B * Hq, D), np.float32))
        ct.launch(si, (B * Hq, 1, 1), _gather_blk, (blockO.at(pmod), O_dec, BQ))    # pull row pmod
        cudart.cudaStreamSynchronize(si)
        same = np.array_equal(bits(O_dec.np()), bits(O_train[:, :, pos].reshape(B * Hq, D)))
        ok &= same
        print(f"    pos={pos:3d}: bitwise={same}  {'OK' if same else 'FAIL'}")
        for g in (blockQ, gQpos, blockO, O_dec): g.free()
    for g in (gQ, gK, gV, gO, gL, Kc, Vc): g.free()
    return ok


if __name__ == "__main__":
    print(f"Device decode plumbing — D={D} BQ={BQ}"); print("=" * 66)
    a = test_rope_dec(); b = test_decode_attn_plumbing()
    print("=" * 66)
    print(f"  {'PASS — device decode kernels bitwise == prefill' if a and b else 'FAIL'}")
    sys.exit(0 if a and b else 1)
