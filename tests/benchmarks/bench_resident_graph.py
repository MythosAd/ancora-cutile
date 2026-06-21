"""CUDA-graph capture of the device-resident MULTI-LAYER forward (the bitwise-safe MFU step).

The device-resident forward is a chain of ~15 cuda.tile launches per layer on persistent buffers.
Each launch carries host overhead; at training/decode sizes that overhead is a real fraction of the
wall time. Capturing the whole NL-layer chain into ONE CUDA graph and replaying it collapses N launches
to one — and because it's the SAME kernels in the SAME order on the SAME buffers, replay is BITWISE
identical to direct launch (zero numerical change → ratio=1 preserved). This is the megakernel's host
-overhead win without the PTX (the cross-operator compute overlap still needs a real megakernel).

  (1) graph replay == direct forward, BITWISE.
  (2) host overhead: direct (NL×~15 launches) vs graph (1 launch), across sizes.

Run:  python tests/benchmarks/bench_resident_graph.py [n_layers] [S]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident import _DBuf
from ancora.model.resident_train import ResidentLayerTrain

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden
ubits = lambda a: np.ascontiguousarray(a, np.float32).view(np.uint32)


def fwd_chain(layers, gx, stream_int):
    """Run the NL-layer device-resident forward; return the last layer's gout buffer."""
    x = gx
    for l in layers:
        x = l.forward(x, stream_int)
    return x


def main(NL=4, S=128):
    B = 1; M = B * S
    print(f"CUDA-graph capture — device-resident {NL}-layer forward   B={B} S={S} M={M} H={H}")
    print("=" * 78)
    rng = np.random.default_rng(0)
    layers = [ResidentLayerTrain(cfg, TransformerLayer(cfg, seed=i).w, B, S, sr_grad=False) for i in range(NL)]
    x0 = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)
    gin = _DBuf(np.ascontiguousarray(x0, np.float32))

    # warm up (JIT-compile every kernel before capture — capture cannot JIT)
    for _ in range(3):
        out = fwd_chain(layers, gin, si)
    cudart.cudaStreamSynchronize(si)
    ref = out.to_numpy().copy()

    # ── (1) capture + bitwise replay ──
    gb = dev.create_graph_builder(); gb.begin_building()
    fwd_chain(layers, gin, int(gb.__cuda_stream__()[1]))
    gb.end_building(); graph = gb.complete()
    graph.launch(so); so.sync()
    rep = layers[-1].gout.to_numpy().copy()
    same = np.array_equal(ubits(ref), ubits(rep))
    print(f"  (1) graph replay == direct forward: {'bitwise IDENTICAL' if same else 'DIFFER'}  "
          f"max|Δ|={np.abs(ref-rep).max():.3g}  {'OK' if same else 'FAIL'}")

    # ── (2) host-overhead timing ──
    def tus(fn, it=50, wm=10):
        for _ in range(wm): fn()
        so.sync(); t = time.perf_counter()
        for _ in range(it): fn()
        so.sync(); return (time.perf_counter() - t) / it * 1e6
    td = tus(lambda: fwd_chain(layers, gin, si))
    tg = tus(lambda: graph.launch(so))
    print(f"  (2) direct {td:8.1f} µs  |  graph {tg:8.1f} µs  |  {td/tg:.2f}× less host overhead  "
          f"({td-tg:.0f} µs/forward saved)")

    print("=" * 78)
    print(f"  {'PASS — multi-layer forward is CUDA-graph-capturable & bitwise (host-overhead MFU win)' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    sys.exit(0 if main(NL, S) else 1)
