"""③ gather/scatter cleanup probe (cuda-tile 1.5.0): tile-batched ct.gather/ct.scatter vs the
production one-block-per-row kernels.

  embed gather: fused._embed_gather (grid (R,), scalar row index, HB-chunk loop of (1,128))
                vs ct.gather with a (TM,1) id tile → (TM,128) gathered tile, grid (R/TM, HB).
  onehot set  : fused._onehot_set (grid (M,), one (1,1) store per block)
                vs ct.scatter of a (TM,1) value at (rows, ids), grid (M/TM,).

Both are PURE DATA MOVEMENT → the variants must be BITWISE == production (they feed the
ratio=1 forward / the bitwise dW path; adopt only if bitwise AND faster)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.fused import _embed_gather, _onehot_set
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

GTM = 64      # rows per block in the gather variants


@ct.kernel
def _embed_gather_g(ids, embed, out, HB: ct.Constant[int]):
    """ct.gather version: (GTM,1) id tile × (1,128) column arange → (GTM,128) gathered tile.
    Grid (R//GTM, HB)."""
    rb, hb = ct.bid(0), ct.bid(1)
    ind0 = ct.load(ids, index=(rb, 0), shape=(GTM, 1))
    ind1 = ct.reshape(ct.arange(128, dtype=ct.int32), (1, 128)) + hb * 128
    t = ct.gather(embed, (ind0, ind1))
    ct.store(out, index=(rb, hb), tile=ct.astype(ct.bitcast(t, ct.bfloat16), ct.float32))


@ct.kernel
def _onehot_set_s(ids, oh, GTM_: ct.Constant[int]):
    """ct.scatter version: oh[rb*GTM+i, ids[i]] = 0x3F80 for a whole row tile. Grid (M//GTM,)."""
    rb = ct.bid(0)
    rows = ct.reshape(ct.arange(GTM_, dtype=ct.int32), (GTM_, 1)) + rb * GTM_
    cols = ct.load(ids, index=(rb, 0), shape=(GTM_, 1))
    ct.scatter(oh, (rows, cols), ct.full((GTM_, 1), 0x3F80, ct.uint16))


def tim(fn, reps=100):
    fn(); sync()
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps): fn()
        sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
    return best


def main():
    V, H, HB = 151936, 1024, 8
    rng = np.random.default_rng(0)
    emb = _GpuArray(_f32bf(rng.standard_normal((V, H)).astype(np.float32) * 0.02))
    ok = True

    print("-- embed gather (out (R,1024) f32) --")
    for R in (128, 2048):
        ids = _GpuArray(rng.integers(0, V, (R, 1)).astype(np.int32))
        o_p, o_g = _GpuArray.zeros((R, H), np.float32), _GpuArray.zeros((R, H), np.float32)
        ct.launch(si, (R,), _embed_gather, (ids, emb, o_p, HB))
        ct.launch(si, (R // GTM, HB), _embed_gather_g, (ids, emb, o_g, HB)); sync()
        nbad = int((o_p.to_numpy() != o_g.to_numpy()).sum()); ok &= nbad == 0
        t_p = tim(lambda: ct.launch(si, (R,), _embed_gather, (ids, emb, o_p, HB)))
        t_g = tim(lambda: ct.launch(si, (R // GTM, HB), _embed_gather_g, (ids, emb, o_g, HB)))
        print(f"  R={R:5d}: prod {t_p:6.1f} us | ct.gather {t_g:6.1f} us ({t_p/t_g:.2f}x)  bits {'OK' if nbad==0 else nbad}")
        for g in (ids, o_p, o_g): g.free()

    print("-- onehot set (oh (M,V) u16, memset by caller) --")
    for M in (1024, 2048):
        ids = _GpuArray(rng.integers(0, V, (M, 1)).astype(np.int32))
        oh_p, oh_s = _GpuArray.zeros((M, V), np.uint16), _GpuArray.zeros((M, V), np.uint16)
        ct.launch(si, (M,), _onehot_set, (ids, oh_p))
        ct.launch(si, (M // GTM,), _onehot_set_s, (ids, oh_s, GTM)); sync()
        nbad = int((oh_p.to_numpy() != oh_s.to_numpy()).sum()); ok &= nbad == 0
        t_p = tim(lambda: ct.launch(si, (M,), _onehot_set, (ids, oh_p)))
        t_s = tim(lambda: ct.launch(si, (M // GTM,), _onehot_set_s, (ids, oh_s, GTM)))
        print(f"  M={M:5d}: prod {t_p:6.1f} us | ct.scatter {t_s:6.1f} us ({t_p/t_s:.2f}x)  bits {'OK' if nbad==0 else nbad}")
        for g in (ids, oh_p, oh_s): g.free()

    emb.free()
    print("  " + ("PASS (variants bitwise == production)" if ok else "FAIL"))


if __name__ == "__main__":
    print(f"gather/scatter cleanup probe (cuda-tile {ct.__version__})")
    print("=" * 72)
    main()
