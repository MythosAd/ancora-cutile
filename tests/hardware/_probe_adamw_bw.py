"""AdamW sweep bandwidth probe — is the optimizer step at the HBM wall or below it?

The real-size step measures ~47.6ms for ~16.5GB of AdamW traffic (30 B/param: read
g/m/v/p32 = 16B, write m/v/p32/p16 = 14B) ≈ 350GB/s = 39% of the 896GB/s peak.
Isolate the `_adamw` kernel on ONE fat buffer (64M params) and sweep tile shapes +
occupancy to find the kernel's intrinsic BW. Elementwise → every variant is
bitwise-identical per element (no reduction), so a re-tile is a SAFE optimization."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.optim.adamw import _adamw
from ancora.kernels.loss import _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

NP_ = 64 * 1024 * 1024          # 64M params (≈ 4 expert stacks); 1.1GB total state
BYTES_PER = 30                   # r: g,m,v,p32 (16) + w: m,v,p32 (12) + p16 (2)


def _mk(shape_rc):
    R, Cw = shape_rc
    rng = np.random.default_rng(0)
    g = _GpuArray(rng.standard_normal((R, Cw)).astype(np.float32) * 1e-3)
    m = _GpuArray.zeros((R, Cw), np.float32); v = _GpuArray.zeros((R, Cw), np.float32)
    p = _GpuArray(rng.standard_normal((R, Cw)).astype(np.float32))
    p16 = _GpuArray.zeros((R, Cw), np.uint16)
    return g, m, v, p, p16


@ct.kernel(occupancy=2)
def _adamw_o2(grad, m, v, p32, p16, OTM: ct.Constant[int], CW: ct.Constant[int],
              beta1: float, beta2: float, eps: float, lr: float, wd: float,
              ibc1: float, ibc2: float):
    """_adamw with occupancy=2 + parametric tile width (probe only)."""
    r = ct.bid(0)
    g  = ct.load(grad, index=(r, 0), shape=(OTM, CW))
    tm = ct.load(m,    index=(r, 0), shape=(OTM, CW))
    tv = ct.load(v,    index=(r, 0), shape=(OTM, CW))
    tp = ct.load(p32,  index=(r, 0), shape=(OTM, CW))
    mn = beta1 * tm + (1.0 - beta1) * g
    vn = beta2 * tv + (1.0 - beta2) * g * g
    upd = (mn * ibc1) / (ct.sqrt(vn * ibc2) + eps) + wd * tp
    pn  = tp - lr * upd
    ct.store(m,   index=(r, 0), tile=mn)
    ct.store(v,   index=(r, 0), tile=vn)
    ct.store(p32, index=(r, 0), tile=pn)
    ct.store(p16, index=(r, 0), tile=ct.bitcast(ct.astype(pn, ct.bfloat16), ct.uint16))


def bench(tag, launch, reps=20):
    launch(); sync()
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps): launch()
        sync(); best = min(best, (time.perf_counter() - t) / reps)
    gbs = NP_ * BYTES_PER / best / 1e9
    print(f"  {tag:44s} {best*1e6:8.0f} us   {gbs:6.0f} GB/s  ({gbs/896*100:4.1f}% peak)")
    return best


if __name__ == "__main__":
    args = (0.9, 0.95, 1e-8, 1e-3, 0.01, 1.0, 1.0)
    print(f"AdamW sweep BW probe — {NP_//2**20}M params, {NP_*BYTES_PER/2**30:.1f} GiB traffic/step")
    print("=" * 78)

    # production shape: (R,64), OTM=128 (what _pick_otm gives for big weights)
    g, m, v, p, p16 = _mk((NP_ // 64, 64))
    bench("production _adamw  (R,64)  OTM=128", lambda: ct.launch(
        si, (NP_ // 64 // 128, 1, 1), _adamw, (g, m, v, p, p16, 128, *[float(a) for a in args])))
    bench("occupancy=2        (R,64)  OTM=128", lambda: ct.launch(
        si, (NP_ // 64 // 128, 1, 1), _adamw_o2, (g, m, v, p, p16, 128, 64, *[float(a) for a in args])))
    for otm in (256, 64, 32):
        bench(f"occupancy=2        (R,64)  OTM={otm}", lambda o=otm: ct.launch(
            si, (NP_ // 64 // o, 1, 1), _adamw_o2, (g, m, v, p, p16, o, 64, *[float(a) for a in args])))
    for gg in (g, m, v, p, p16): gg.free()

    # wider rows (same elementwise math, reshaped view of the same params)
    for Cw, otm in ((256, 32), (256, 64), (512, 16), (1024, 8)):
        bufs = _mk((NP_ // Cw, Cw))
        bench(f"occupancy=2        (R,{Cw}) OTM={otm}", lambda b=bufs, o=otm, c=Cw: ct.launch(
            si, (NP_ // c // o, 1, 1), _adamw_o2, (*b, o, c, *[float(a) for a in args])))
        for gg in bufs: gg.free()
