"""Device-resident + CUDA-graph PoC (de-risk the perf path). Runs the Qwen3 layer's 7
projection GEMMs (the ~78%-of-forward compute) three ways and times END-TO-END wall-clock
(includes host overhead — the real limiter):
  (1) host-glue   — linear_bf16 per GEMM (alloc/upload/launch/sync/download/free), = current model
  (2) device-resident — persistent buffers, direct ct.launch chain + one sync
  (3) graph        — capture the (2) chain in a CUDA graph, replay
Shows how much device-residency removes (alloc/upload/download) and how much the graph removes
(per-launch overhead)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.loss import _gemm, GTM, GTN, GTK, _GpuArray
from ancora.model.qwen3_layer import linear_bf16

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])
_bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

M, H, I = 4096, 1024, 3072
# 7 projections: (in_buf_key, K, N). q,k,v,gate,up read H; o reads qd=H; down reads I.
PROJS = [("h", H, H), ("h", H, H // 2), ("h", H, H // 2),      # q, k, v
         ("o", H, H), ("h", H, I), ("h", H, I), ("a", I, H)]   # o, gate, up, down
rng = np.random.default_rng(0)
acts = {"h": rng.standard_normal((M, H)).astype(np.float32),
        "o": rng.standard_normal((M, H)).astype(np.float32),
        "a": rng.standard_normal((M, I)).astype(np.float32)}
ws = [(rng.standard_normal((K, N)) * 0.05).astype(np.float32) for _, K, N in PROJS]


def wall_ms(fn, it=30, warm=10):
    for _ in range(warm): fn()
    stream.sync(); t = time.perf_counter()
    for _ in range(it): fn()
    return (time.perf_counter() - t) / it * 1000


# (1) host-glue: linear_bf16 allocs/uploads/downloads each call (current model)
def host_glue():
    for (akey, K, N), w in zip(PROJS, ws):
        linear_bf16(acts[akey], w, si)

# (2)/(3) device-resident: persistent buffers
gact = {k: _GpuArray(_bf(v)) for k, v in acts.items()}
gw = [_GpuArray(_bf(w)) for w in ws]
gout = [_GpuArray.zeros((M, N), np.float32) for _, K, N in PROJS]
def chain(s):
    for j, (akey, K, N) in enumerate(PROJS):
        ct.launch(s, (M // GTM, N // GTN, 1), _gemm, (gact[akey], gw[j], gout[j], K // GTK, GTM, GTN, GTK))
def device_resident():
    chain(si); stream.sync()

# graph capture
gb = dev.create_graph_builder(); gb.begin_building()
chain(int(gb.__cuda_stream__()[1])); gb.end_building(); graph = gb.complete()
def graph_replay():
    graph.launch(stream); stream.sync()


if __name__ == "__main__":
    print(f"Device-resident + CUDA-graph PoC — 7 projection GEMMs, M={M}")
    print("=" * 64)
    t1 = wall_ms(host_glue); t2 = wall_ms(device_resident); t3 = wall_ms(graph_replay)
    print(f"  (1) host-glue (current model)   {t1*1000:7.0f} µs")
    print(f"  (2) device-resident chain       {t2*1000:7.0f} µs   ({t1/t2:.1f}x vs host-glue)")
    print(f"  (3) device-resident + graph     {t3*1000:7.0f} µs   ({t1/t3:.1f}x vs host-glue, {t2/t3:.1f}x vs direct)")
    print("=" * 64)
    print(f"  device-residency removes alloc/upload/download; graph removes per-launch overhead")
