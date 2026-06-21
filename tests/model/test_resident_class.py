"""ResidentLayer (ancora/model/resident.py) — the formalized device-resident layer. Validates:
  (1) forward numerics vs host TransformerLayer.forward (MXFP8 + both megakernel fusions),
  (2) buffer reuse across calls (deterministic), and
  (3) the whole schedule is CUDA-graph-capturable (the persistent-megakernel base).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.model.resident import ResidentLayer, _DBuf, _f32bf

bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])

cfg = Qwen3Config(); B, S = 1, 256
H = cfg.hidden; M = B * S
layer = TransformerLayer(cfg, seed=0)
rng = np.random.default_rng(1)
x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
ref = layer.forward(x, si)

rl = ResidentLayer(cfg, layer.w, B, S)
gx = _DBuf(np.ascontiguousarray(x.reshape(M, H), np.float32))   # fp32 residual stream

if __name__ == "__main__":
    print(f"ResidentLayer  B={B} S={S} H={H}"); print("=" * 60)

    # (1) correctness
    out = rl.forward(gx, si); cudart.cudaStreamSynchronize(si)
    devv = out.to_numpy().reshape(B, S, H)                       # gout is fp32 (residual stream)
    e = rel(devv, ref)
    print(f"  (1) forward vs host TransformerLayer.forward: {e*100:.2f}%  {'OK' if e < 0.06 else 'FAIL'}")

    # (2) buffer reuse: a second call must reproduce bitwise-identical output
    out2 = rl.forward(gx, si); cudart.cudaStreamSynchronize(si)
    same = np.array_equal(out.to_numpy(), out2.to_numpy())
    print(f"  (2) buffer reuse deterministic (2nd call == 1st): {'OK' if same else 'FAIL'}")

    # (3) CUDA-graph capture of the full layer schedule
    try:
        for _ in range(3): rl.forward(gx, si)
        stream.sync()
        gb = dev.create_graph_builder(); gb.begin_building()
        rl.forward(gx, int(gb.__cuda_stream__()[1]))
        gb.end_building(); graph = gb.complete()
        graph.launch(stream); stream.sync()
        gv = rl.gout.to_numpy().reshape(B, S, H)                  # gout is fp32
        eg = rel(gv, ref)
        print(f"  (3) CUDA-graph replay vs host: {eg*100:.2f}%  {'OK' if eg < 0.06 else 'FAIL'}")
        def tg(fn, it=50, wm=10):
            for _ in range(wm): fn()
            stream.sync(); t = time.perf_counter()
            for _ in range(it): fn()
            stream.sync(); return (time.perf_counter() - t) / it * 1e6
        td = tg(lambda: rl.forward(gx, si)); tgr = tg(lambda: graph.launch(stream))
        print(f"      direct {td:.0f} µs | graph {tgr:.0f} µs | {td/tgr:.2f}x less host overhead")
    except Exception as ex:
        import traceback; traceback.print_exc(); print(f"  (3) GRAPH FAILED: {ex}")
    print("=" * 60)
