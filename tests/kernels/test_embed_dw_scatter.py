"""_embed_dw_scatter vs the (MC,V) onehot-GEMM input-embed dW it replaces.

Gates:
  (A) numpy reference: out_base + onehotᵀ@dy(f32)  → rel error ≤ regroup-ulp
  (B) OLD kernel pair (_onehot_set + _gemm_dW_acc) on identical inputs → rel ≤ ulp
      (same value set, different f32 association — NOT bitwise, that's expected)
  (C) repeat-determinism of the scatter path → BITWISE Δ=0 (stable sort ⇒ fixed order)
  (D) duplicate-heavy stress (vocab=64 → giant groups) — same gates
  (E) timing at the real boundary shape (M=2048, V=151936, H=1024)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.fused import (_embed_dw_scatter, build_id_groups, _onehot_set,
                                  _gemm_dW_acc)
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
_DWT, _DWN = 64, 128


def run_case(M, V, H, seed, timing=False):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, V, M).astype(np.int32)
    dyb = _f32bf(rng.standard_normal((M, H)).astype(np.float32))
    base = rng.standard_normal((V, H)).astype(np.float32) * 0.1

    gdy = _GpuArray(dyb)
    srt, st, cn, gi = build_id_groups(ids)
    gsrt, gst, gcnt, ggid = _GpuArray(srt), _GpuArray(st), _GpuArray(cn), _GpuArray(gi)

    # NEW: scatter path (out preloaded with base — the += semantics)
    g_new = _GpuArray(base.copy())
    ct.launch(si, (M, H // 128), _embed_dw_scatter, (gcnt, gst, ggid, gsrt, gdy, g_new, 128))
    sync(); out_new = g_new.to_numpy()

    # OLD: onehot GEMM path (single chunk MC=M — chunking only regroups the same sum)
    g_old = _GpuArray(base.copy())
    goh = _GpuArray.zeros((M, V), np.uint16)
    gid_col = _GpuArray(ids.reshape(M, 1))
    def old_fn():
        cdrv.cuMemsetD8Async(goh._ptr, 0, goh._nbytes, si)
        ct.launch(si, (M, 1, 1), _onehot_set, (gid_col, goh))
        ct.launch(si, (V // _DWT, H // _DWN, 1), _gemm_dW_acc,
                  (goh, gdy, g_old, M // _DWT, _DWT, _DWN, _DWT))
    old_fn(); sync(); out_old = g_old.to_numpy()

    # numpy reference (f32 accumulation, ascending row order per id)
    ref = base.copy()
    dyf = b2f(dyb)
    np.add.at(ref, ids, dyf.astype(np.float64).astype(np.float32))  # close-enough ref
    scale = np.abs(ref).max() + 1e-30
    rel_np  = float(np.abs(out_new - ref).max() / scale)
    rel_old = float(np.abs(out_new - out_old).max() / scale)

    # repeat-determinism (bitwise)
    g_new2 = _GpuArray(base.copy())
    ct.launch(si, (M, H // 128), _embed_dw_scatter, (gcnt, gst, ggid, gsrt, gdy, g_new2, 128))
    sync()
    det = int((g_new2.to_numpy() != out_new).sum())

    line = f"  M={M} V={V:6d}: vs-numpy rel {rel_np:.1e} | vs-onehot-GEMM rel {rel_old:.1e} | repeat-det {'Δ=0' if det==0 else 'FAIL'}"
    if timing:
        def tim(fn, reps=30):
            fn(); sync()
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                for _ in range(reps): fn()
                sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
            return best
        t_new = tim(lambda: ct.launch(si, (M, H // 128), _embed_dw_scatter, (gcnt, gst, ggid, gsrt, gdy, g_new, 128)))
        t_old = tim(old_fn)
        line += f" | old {t_old/1e3:6.2f} ms → new {t_new/1e3:6.3f} ms ({t_old/t_new:.0f}x)"
    print(line)
    for g in (gdy, gsrt, gst, gcnt, ggid, g_new, g_new2, g_old, goh, gid_col): g.free()
    return rel_np < 5e-5 and rel_old < 5e-5 and det == 0


if __name__ == "__main__":
    print("input-embed dW: deterministic sorted-scatter vs onehot GEMM")
    print("=" * 100)
    ok = run_case(512, 2048, 1024, seed=0)                     # small, moderate duplicates
    ok &= run_case(512, 64, 1024, seed=1)                      # duplicate-HEAVY (giant groups)
    ok &= run_case(2048, 151936, 1024, seed=2, timing=True)    # real boundary shape + timing
    print("  " + ("PASS" if ok else "FAIL"))
