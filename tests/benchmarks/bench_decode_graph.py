"""FEASIBILITY PROBE — fold the ~420 launches/decode-step (28 layers × ~15 kernels) into ONE CUDA-graph.

For a FIXED position the decode layer-stack's pointers (cache at_pos, cos/sin row, q_blk) are fixed, so
the chain captures into a valid graph (the bench_resident_graph pattern). In RL every rollout revisits
pos 0,1,2,… so a graph-per-pos cache means each step becomes 1 graph replay instead of ~420 host launches.

This probe checks the make-or-break questions BEFORE building the cache:
  (1) does capturing ResidentDecodeLayer-stack.forward(pos) + replay reproduce the direct result BITWISE?
  (2) how much faster is replay than direct launch at the decode size (Md=128)?

Run:  python tests/benchmarks/bench_decode_graph.py [NL] [Bp]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident import _DBuf
from ancora.model.resident_decode import ResidentDecodeLayer, MGEMM

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden
ubits = lambda a: np.ascontiguousarray(a, np.float32).view(np.uint32)


def stack_fwd(layers, gx, pos, stream_int):
    x = gx
    for l in layers:
        x = l.forward(x, pos, stream_int)
    return x


def main(NL=4, Bp=8):
    Md, maxS = MGEMM, 128
    print(f"Decode CUDA-graph probe — {NL}-layer stack, Bp={Bp}, Md={Md}")
    print("=" * 80)
    rng = np.random.default_rng(0)
    layers = [ResidentDecodeLayer(cfg, TransformerLayer(cfg, seed=i).w, Bp, maxS) for i in range(NL)]
    gin = _DBuf.zeros((Md, H), np.float32)
    x0 = (rng.standard_normal((Bp, H)) * 0.5).astype(np.float32)

    POS = 40   # a representative mid-sequence position
    # warm up JIT + fill cache for positions 0..POS (so attention has real cache to read)
    for p in range(POS + 1):
        cdrv.cuMemcpyHtoDAsync(gin.ptr, np.ascontiguousarray(x0, np.float32), Bp * H * 4, si)
        out = stack_fwd(layers, gin, p, si)
    cudart.cudaStreamSynchronize(si)
    ref = out.to_numpy().copy()                       # direct result at pos=POS

    # ── (1) capture the stack at pos=POS, replay, compare bitwise ──
    try:
        gb = dev.create_graph_builder(); gb.begin_building()
        gout = stack_fwd(layers, gin, POS, int(gb.__cuda_stream__()[1]))
        gb.end_building(); graph = gb.complete()
    except Exception as e:
        print(f"  (1) CAPTURE FAILED: {type(e).__name__}: {e}")
        print("=" * 80); print("  STOP — decode stack is not graph-capturable as-is.")
        return False
    cdrv.cuMemcpyHtoDAsync(gin.ptr, np.ascontiguousarray(x0, np.float32), Bp * H * 4, si)
    graph.launch(so); so.sync()
    rep = layers[-1].gout.to_numpy().copy()
    same = np.array_equal(ubits(ref), ubits(rep))
    print(f"  (1) graph replay == direct (pos={POS}): {'BITWISE IDENTICAL' if same else 'DIFFER'}  "
          f"max|Δ|={np.abs(ref-rep).max():.3g}  {'OK' if same else 'FAIL'}")

    # ── (2) timing: direct launch vs graph replay ──
    def tus(fn, it=50, wm=10):
        for _ in range(wm): fn()
        so.sync(); t = time.perf_counter()
        for _ in range(it): fn()
        so.sync(); return (time.perf_counter() - t) / it * 1e6
    td = tus(lambda: stack_fwd(layers, gin, POS, si))
    tg = tus(lambda: graph.launch(so))
    print(f"  (2) direct {td:8.1f} µs/step  |  graph replay {tg:8.1f} µs/step  |  {td/tg:.2f}× fewer host launches "
          f"({(td-tg)/1e3:.2f} ms/step saved, ×NL/{NL}-scaled → full-28L est {(td-tg)/NL*28/1e3:.1f} ms/step)")

    print("=" * 80)
    print(f"  {'VIABLE — graph-per-pos cache would fold the 420 launches (build the cache next)' if same else 'FAIL — not bitwise, do NOT use'}")
    return same


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    Bp = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sys.exit(0 if main(NL, Bp) else 1)
