"""WHY is the M=1024 training step only ~8% MFU? Admin-free per-kernel GPU-time breakdown of a full
ResidentModel step (forward + loss_backward + AdamW), wrapping ct.launch with CUDA events. Reveals:
  • GPU-busy (Σ kernel times) vs WALL → the host-idle fraction (boundary syncs / to_numpy / host RMSNorm)
  • per-kernel time → which kernels (LM-head GEMM? layer GEMMs? CE?) dominate
So we can tell if 8% is host-round-trip-bound, small-M-GEMM-bound, or both — NOT guess.

Run:  python tests/hardware/_profile_train_step.py [B] [S]
"""
import sys, os, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident_model import ResidentModel

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden

_real = ct.launch
_rec = []; _cap = False
def _timed(stream, grid, kernel, args):
    if not _cap:
        return _real(stream, grid, kernel, args)
    pf = getattr(kernel, "_pyfunc", None)
    name = getattr(pf, "__name__", None) or str(kernel)[:26]
    _, e0 = cudart.cudaEventCreate(); _, e1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(e0, stream); r = _real(stream, grid, kernel, args); cudart.cudaEventRecord(e1, stream)
    _rec.append((name, e0, e1)); return r
ct.launch = _timed


def main(B=4, S=256):
    global _cap
    NL, V = 28, 151936; M = B * S
    print(f"Training-step profile — Qwen3-0.6B NL={NL} V={V}  B={B} S={S} M={M}")
    print("=" * 84)
    rng = np.random.default_rng(0)
    w = {"layers": [TransformerLayer(cfg, seed=i).w for i in range(NL)],
         "embed": (rng.standard_normal((V, H)) * 0.02).astype(np.float32),
         "final_norm": (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)}
    m = ResidentModel(cfg, w, B, S, vocab=V)
    ids = rng.integers(0, V, (B, S)).astype(np.int64)
    lab = rng.integers(0, V, (B, S)).astype(np.int64)
    adv = rng.standard_normal(M).astype(np.float32)
    def step():
        h = m.forward(ids, si); m.loss_backward(h, lab, si, advantage=adv); m.step(si)
    for _ in range(2): step()
    cudart.cudaStreamSynchronize(si)

    # wall time of the 3 phases (host+device)
    def wall(fn, it=5):
        cudart.cudaStreamSynchronize(si); t = time.perf_counter()
        for _ in range(it): fn()
        cudart.cudaStreamSynchronize(si); return (time.perf_counter() - t) / it
    h = [None]
    t_fwd  = wall(lambda: h.__setitem__(0, m.forward(ids, si)))
    t_bwd  = wall(lambda: m.loss_backward(h[0], lab, si, advantage=adv))
    t_step = wall(lambda: m.step(si))
    wall_total = t_fwd + t_bwd + t_step

    # one captured step → per-kernel GPU time
    _rec.clear(); _cap = True
    step()
    cudart.cudaStreamSynchronize(si); _cap = False
    by = collections.defaultdict(lambda: [0.0, 0])
    for nm, e0, e1 in _rec:
        _, ms = cudart.cudaEventElapsedTime(e0, e1); by[nm][0] += ms; by[nm][1] += 1
        cudart.cudaEventDestroy(e0); cudart.cudaEventDestroy(e1)
    gpu_busy = sum(v[0] for v in by.values()) / 1e3   # s
    nlaunch = sum(v[1] for v in by.values())

    print(f"  WALL: fwd {t_fwd*1e3:.0f}  bwd {t_bwd*1e3:.0f}  step {t_step*1e3:.0f}  = {wall_total*1e3:.0f} ms/step")
    print(f"  GPU-busy (Σ {nlaunch} kernel launches): {gpu_busy*1e3:.0f} ms   "
          f"→ host-idle (syncs/to_numpy/host-RMSNorm) = {(wall_total-gpu_busy)*1e3:.0f} ms "
          f"({(wall_total-gpu_busy)/wall_total*100:.0f}% of WALL)")
    print(f"  GPU-busy MFU-equiv: {6*(NL*(H*2048+2*H*1024+2048*H+2*H*3072+3072*H)+V*H)*M/(gpu_busy*80e12)*100:.0f}% "
          f"(if host-idle were 0)")
    print("-" * 84)
    print(f"    {'kernel':<24}{'ms/step':>9}{'% GPU':>7}{'n':>6}")
    for nm, (tms, cnt) in sorted(by.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"    {nm:<24}{tms:>9.1f}{tms/(gpu_busy*1e3)*100:>6.1f}%{cnt:>6}")
    m.free()
    print("=" * 84)


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 4, int(a[2]) if len(a) > 2 else 256)
