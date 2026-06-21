"""token↔head-major transpose kernels (device-resident attention plumbing): vs numpy
transpose + roundtrip identity. Keep."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.fused import _tok_to_head, _head_to_tok, TT

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
f32bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)

class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o


def test():
    B, S, Hh, Dh = 2, 128, 16, 64
    M, NSB = B * S, S // TT
    rng = np.random.default_rng(0)
    tok = rng.standard_normal((M, Hh * Dh)).astype(np.float32)
    head_ref = tok.reshape(B, S, Hh, Dh).transpose(0, 2, 1, 3).reshape(B * Hh * S, Dh)

    gtok = GA(f32bf(tok)); ghead = GA(np.zeros((B * Hh * S, Dh), np.uint16))
    ct.launch(si, (B * Hh, NSB, 1), _tok_to_head, (gtok, ghead, Hh, NSB))
    cudart.cudaStreamSynchronize(si)
    e1 = np.abs(bf32(ghead.np()) - bf32(f32bf(head_ref))).max()
    print(f"  tok→head vs numpy transpose: max diff {e1:.3e}  {'OK' if e1 == 0 else 'FAIL'}")

    gtok2 = GA(np.zeros((M, Hh * Dh), np.uint16))
    ct.launch(si, (B * Hh, NSB, 1), _head_to_tok, (ghead, gtok2, Hh, NSB))
    cudart.cudaStreamSynchronize(si)
    e2 = np.abs(bf32(gtok2.np()) - bf32(f32bf(tok))).max()
    print(f"  head→tok roundtrip identity: max diff {e2:.3e}  {'OK' if e2 == 0 else 'FAIL'}")
    return e1 == 0 and e2 == 0


if __name__ == "__main__":
    print(f"token↔head transpose (TT={TT})"); print("=" * 50)
    ok = test(); print("=" * 50); print(f"  {'PASS' if ok else 'FAIL'}")
