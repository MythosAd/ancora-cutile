"""ResidentLayer.forward_cutlass — the 7 projections on CUTLASS MXFP8 (1.2-1.4× each), the rest
in cuda-tile. Validates: (1) numerics vs host TransformerLayer.forward, (2) end-to-end layer
speedup vs the all-cuda-tile forward() at training size. Needs cutlass_mxfp8.dll."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc
from cuda.bindings import runtime as cudart
from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.model.resident import ResidentLayer, _DBuf, _f32bf

bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden


def numeric():
    B, S = 1, 256; M = B * S
    layer = TransformerLayer(cfg, seed=0); rng = np.random.default_rng(1)
    x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
    ref = layer.forward(x, si)
    rl = ResidentLayer(cfg, layer.w, B, S, use_cutlass=True)
    gx = _DBuf(np.ascontiguousarray(x.reshape(M, H), np.float32))   # fp32 residual stream
    out = rl.forward_cutlass(gx, si); cudart.cudaStreamSynchronize(si)
    o1 = out.to_numpy().copy()                                       # gout is fp32
    e = rel(o1.reshape(B, S, H), ref)
    print(f"  (1) forward_cutlass vs host: {e*100:.2f}%  {'OK' if e < 0.07 else 'FAIL'}")
    # determinism (HARD requirement for on-policy RL): same input twice → bitwise identical
    out2 = rl.forward_cutlass(gx, si); cudart.cudaStreamSynchronize(si)
    det = np.array_equal(o1, out2.to_numpy())
    print(f"  (1b) CUTLASS-hybrid deterministic (2nd call bitwise == 1st): {'OK' if det else 'FAIL — breaks batch-invariance!'}")
    return e < 0.07 and det


def perf(B=4, S=2048):
    M = B * S
    layer = TransformerLayer(cfg, seed=0)
    rl = ResidentLayer(cfg, layer.w, B, S, use_cutlass=True)
    rng = np.random.default_rng(2)
    gx = _DBuf(np.ascontiguousarray((rng.standard_normal((M, H)) * 0.5).astype(np.float32)))  # fp32 residual
    def wall(fn, it=30, wm=10):
        for _ in range(wm): fn()
        stream.sync(); t = time.perf_counter()
        for _ in range(it): fn()
        stream.sync(); return (time.perf_counter() - t) / it * 1e3
    t_ct = wall(lambda: rl.forward(gx, si))
    t_cu = wall(lambda: rl.forward_cutlass(gx, si))
    print(f"  (2) M={M}: cuda-tile {t_ct*1000:.0f} µs | CUTLASS-hybrid {t_cu*1000:.0f} µs | {t_ct/t_cu:.2f}x")


if __name__ == "__main__":
    print("ResidentLayer CUTLASS-hybrid forward"); print("=" * 60)
    ok = numeric()
    perf()
    print("=" * 60); print(f"  {'PASS' if ok else 'FAIL'}")
