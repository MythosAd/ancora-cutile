"""Per-kernel CUDA-event profile of ONE real-size MoE decode token step (direct launches).
Patches cuda.tile.launch (+ the moe_dispatch raw-CUDA wrappers) to bracket every launch with
events, runs a few steady-state steps at pos≈P, and aggregates GPU time by kernel.
MUST run in a fresh process with nothing else on the GPU (WDDM paging, CLAUDE.md)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ctile
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host
from ancora.model.resident_moe_decode import ResidentMoEDecodeModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

# ── event-bracketing patch ────────────────────────────────────────────────────
_orig_launch = ctile.launch
_records = []          # (name, grid, ev_start, ev_end)
_armed = False

def _mkev():
    return cudart.cudaEventCreate()[1]

def _patched(stream, grid, kernel, args):
    if not _armed:
        return _orig_launch(stream, grid, kernel, args)
    s, e = _mkev(), _mkev()
    cudart.cudaEventRecord(s, stream)
    _orig_launch(stream, grid, kernel, args)
    cudart.cudaEventRecord(e, stream)
    name = getattr(kernel, "__name__", None) or getattr(getattr(kernel, "fn", None), "__name__", str(kernel))
    _records.append((name, grid, s, e))

ctile.launch = _patched

import ancora.kernels.moe_dispatch as md
for fname in ("router_gate", "build_layout_dev"):
    fn = getattr(md, fname)
    def _wrap(fn=fn, fname=fname):
        def w(*a):
            if not _armed:
                return fn(*a)
            s, e = _mkev(), _mkev()
            cudart.cudaEventRecord(s, a[-1])      # stream is the last arg
            r = fn(*a)
            cudart.cudaEventRecord(e, a[-1])
            _records.append((fname, (0, 0, 0), s, e))
            return r
        return w
    setattr(md, fname, _wrap())


def main():
    global _armed
    cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
    Bp, P, maxS = 32, 512, 1024
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, 128)
    train = ResidentMoEModel(cfg, w, 1, 128, device_route=True)
    eng = ResidentMoEDecodeModel(train, Bp=Bp, maxS=maxS, si=si)
    rng = np.random.default_rng(0)
    prompts = rng.integers(0, cfg.vocab, size=(Bp, P)).astype(np.int64)
    eng.generate(prompts, 4, si)                       # warm/JIT + advance the caches past P

    # steady-state: profile NSTEP direct token steps at pos ≈ P (caches stay warm; we just
    # keep decoding the same engine — pos keeps advancing, well past the prompt)
    NSTEP = 8
    eng._put_ids(np.zeros(Bp, np.int64), si)
    cudart.cudaStreamSynchronize(si)
    _armed = True
    t0 = time.perf_counter()
    for _ in range(NSTEP):
        eng._token_step(si, False, 1.0)
    cudart.cudaStreamSynchronize(si)
    wall = (time.perf_counter() - t0) / NSTEP
    _armed = False

    # aggregate
    agg = {}
    for name, grid, s, e in _records:
        ms = cudart.cudaEventElapsedTime(s, e)[1]
        blocks = grid[0] * grid[1] * grid[2] if grid != (0, 0, 0) else 0
        k = (name, blocks)
        if k not in agg:
            agg[k] = [0, 0.0]
        agg[k][0] += 1
        agg[k][1] += ms
    tot = sum(v[1] for v in agg.values()) / NSTEP
    print(f"steady-state direct step: wall {wall*1e3:.2f} ms | sum of kernel GPU times {tot:.2f} ms "
          f"| launches/step {len(_records)//NSTEP}")
    print(f"{'kernel':38s} {'blocks':>7s} {'n/step':>6s} {'us/call':>8s} {'ms/step':>8s} {'%':>5s}")
    for (name, blocks), (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        per = ms / NSTEP
        print(f"{name:38s} {blocks:7d} {n//NSTEP:6d} {ms/n*1e3:8.1f} {per:8.3f} {per/tot*100:5.1f}")


if __name__ == "__main__":
    main()
