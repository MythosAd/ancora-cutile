"""Per-operator GPU-time breakdown of the device-resident layer forward (admin-free, CUDA events).

ncu needs admin GPU-counter access (ERR_NVGPUCTRPERM); this needs none. It wraps ct.launch to record
a start/end CUDA event around every kernel, runs the layer forward, then reports per-kernel GPU time
grouped by kernel name — showing which hand-written cuda.tile op dominates (the optimization target).
Profiles BOTH the BF16 and MXFP8 forwards at a compute-representative size.

Run:  python tests/hardware/_profile_forward.py [S]
"""
import sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident import _DBuf
from ancora.model.resident_train import ResidentLayerTrain

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden

_real_launch = ct.launch
_rec = []                         # (name, ev_start, ev_end)
_capture = False

def _timed_launch(stream, grid, kernel, args):
    if not _capture:
        return _real_launch(stream, grid, kernel, args)
    pf = getattr(kernel, "_pyfunc", None)            # @ct.kernel wraps the fn; _pyfunc is the original
    name = getattr(pf, "__name__", None) or str(kernel)[:28]
    _, e0 = cudart.cudaEventCreate(); _, e1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(e0, stream)
    r = _real_launch(stream, grid, kernel, args)
    cudart.cudaEventRecord(e1, stream)
    _rec.append((name, e0, e1))
    return r

ct.launch = _timed_launch         # patch the module attr → ResidentLayerTrain's ct.launch uses it


def profile(mxfp8, S, iters=20):
    global _capture
    B = 1; M = B * S
    layer = ResidentLayerTrain(cfg, TransformerLayer(cfg, seed=0).w, B, S, sr_grad=False, mxfp8=mxfp8)
    gx = _DBuf((np.random.default_rng(0).standard_normal((M, H)) * 0.5).astype(np.float32))
    for _ in range(3): layer.forward(gx, si)          # warm up (JIT) — not captured
    cudart.cudaStreamSynchronize(si)
    _rec.clear(); _capture = True
    for _ in range(iters): layer.forward(gx, si)
    cudart.cudaStreamSynchronize(si); _capture = False

    by = collections.defaultdict(lambda: [0.0, 0])    # name -> [total_ms, count]
    for name, e0, e1 in _rec:
        _, ms = cudart.cudaEventElapsedTime(e0, e1)
        by[name][0] += ms; by[name][1] += 1
        cudart.cudaEventDestroy(e0); cudart.cudaEventDestroy(e1)
    total = sum(v[0] for v in by.values())
    print(f"\n  {'MXFP8' if mxfp8 else 'BF16 '} forward   M={M}   per-layer total {total/iters*1000:.1f} µs   "
          f"({len(by)} distinct kernels, {sum(v[1] for v in by.values())//iters} launches/layer)")
    print(f"    {'kernel':<26}{'µs/layer':>10}{'% fwd':>8}{'n/layer':>9}")
    for name, (tms, cnt) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"    {name:<26}{tms/iters*1000:>10.1f}{tms/total*100:>7.1f}%{cnt//iters:>9}")
    return total / iters * 1000


def main(S=512):
    print(f"Device-resident layer forward — per-operator GPU-time breakdown   B=1 S={S} H={H}")
    print("=" * 78)
    profile(mxfp8=False, S=S)
    profile(mxfp8=True, S=S)
    print("=" * 78)


if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    main(S)
