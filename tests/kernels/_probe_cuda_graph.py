"""DECISIVE PROBE: can ct.launch (cuda-tile) be captured into a cuda.core CUDA graph and
replayed? If yes, the megakernel becomes 'capture the forward in a graph' — eliminating the
per-kernel host-glue overhead that limits everything. Verifies correctness + times
graph-replay vs direct-launch for a multi-kernel chain."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])

class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o

@ct.kernel
def add1(x, out):
    ct.store(out, index=(0, 0), tile=ct.load(x, index=(0, 0), shape=(64, 64)) + 1.0)

x = np.zeros((64, 64), np.float32)
gx = GA(x); g1 = GA(x); g2 = GA(x); g3 = GA(x)
NK = 8   # chain length

def direct():
    bufs = [gx, g1, g2, g3, g1, g2, g3, g1, g2]
    for i in range(NK):
        ct.launch(si, (1, 1, 1), add1, (bufs[i], bufs[i + 1]))

# --- capture into a graph ---
try:
    gb = dev.create_graph_builder()
    gb.begin_building()
    gs = int(gb.__cuda_stream__()[1])
    bufs = [gx, g1, g2, g3, g1, g2, g3, g1, g2]
    for i in range(NK):
        ct.launch(gs, (1, 1, 1), add1, (bufs[i], bufs[i + 1]))
    gb.end_building()
    graph = gb.complete()
    print("graph built OK; Graph methods:", [n for n in dir(graph) if not n.startswith('_')][:12])

    # replay
    graph.launch(stream); stream.sync()
    # correctness: g2 should be gx + NK (8 chained +1)
    val = g2.np()[0, 0]
    print(f"  after replay g2[0,0] = {val}  (expect {float(NK)})  {'OK' if abs(val - NK) < 1e-3 else 'FAIL'}")

    # timing: direct (NK launches + sync) vs graph replay (1 launch + sync)
    def time_ms(fn, it=100):
        for _ in range(10): fn()
        stream.sync(); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
        cudart.cudaEventRecord(t0, si)
        for _ in range(it): fn()
        cudart.cudaEventRecord(t1, si); cudart.cudaEventSynchronize(t1)
        return cudart.cudaEventElapsedTime(t0, t1)[1] / it
    md = time_ms(lambda: (direct(), stream.sync()))
    mg = time_ms(lambda: (graph.launch(stream), stream.sync()))
    print(f"  {NK}-kernel chain: direct {md*1000:.1f} µs | graph replay {mg*1000:.1f} µs | {md/mg:.1f}x less host overhead")
except Exception as ex:
    import traceback; traceback.print_exc()
    print(f"GRAPH CAPTURE FAILED: {type(ex).__name__}: {str(ex)[:160]}")
