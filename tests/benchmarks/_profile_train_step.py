"""Per-kernel CUDA-event profile of ONE real-size MoE TRAINING step (fwd+bwd+AdamW) at M=1024
(the MFU-best batch). Same monkeypatch approach as _profile_decode_step.py. Fresh process only."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ctile
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401

_orig_launch = ctile.launch
_records, _armed, _phase = [], False, [""]

def _mkev(): return cudart.cudaEventCreate()[1]

def _patched(stream, grid, kernel, args):
    if not _armed:
        return _orig_launch(stream, grid, kernel, args)
    s, e = _mkev(), _mkev()
    cudart.cudaEventRecord(s, stream)
    _orig_launch(stream, grid, kernel, args)
    cudart.cudaEventRecord(e, stream)
    src = getattr(kernel, "_pyfunc", None) or getattr(kernel, "fn", None) or getattr(kernel, "func", None)
    name = getattr(src, "__name__", None) or getattr(kernel, "__name__", None) or repr(kernel)[-14:-1]
    _records.append((f"{_phase[0]}:{name}", grid, s, e))

ctile.launch = _patched
import ancora.kernels.moe_dispatch as md
for fname in ("router_gate", "build_layout_dev", "router_gate_bwd", "router_dW", "router_dh"):
    if not hasattr(md, fname): continue
    fn = getattr(md, fname)
    def _wrap(fn=fn, fname=fname):
        def w(*a):
            if not _armed: return fn(*a)
            s, e = _mkev(), _mkev()
            cudart.cudaEventRecord(s, a[-1]); r = fn(*a); cudart.cudaEventRecord(e, a[-1])
            _records.append((f"{_phase[0]}:{fname}", (0, 0, 0), s, e))
            return r
        return w
    setattr(md, fname, _wrap())

from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

def main():
    global _armed
    M = 1024
    cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    train = ResidentMoEModel(cfg, from_host(host, 1, M), 1, M, device_route=True)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab, size=(1, M)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, size=(M,)).astype(np.int64)
    for _ in range(2):
        train.forward(ids, si); train.loss_backward(None, labels, si)
        train.step(si, 1e-4); cudart.cudaStreamSynchronize(si)
    _armed = True
    NSTEP = 3
    t0 = time.perf_counter()
    for _ in range(NSTEP):
        _phase[0] = "F"; train.forward(ids, si)
        _phase[0] = "B"; train.loss_backward(None, labels, si)
        _phase[0] = "O"; train.step(si, 1e-4); cudart.cudaStreamSynchronize(si)
    wall = (time.perf_counter() - t0) / NSTEP
    _armed = False
    agg = {}
    for name, grid, s, e in _records:
        ms = cudart.cudaEventElapsedTime(s, e)[1]
        blocks = grid[0] * grid[1] * grid[2] if grid != (0, 0, 0) else 0
        k = (name, blocks)
        if k not in agg: agg[k] = [0, 0.0]
        agg[k][0] += 1; agg[k][1] += ms
    tot = sum(v[1] for v in agg.values()) / NSTEP
    print(f"M={M} step wall {wall*1e3:.1f} ms | kernel GPU sum {tot:.1f} ms | launches/step {len(_records)//NSTEP}")
    print(f"{'phase:kernel':42s} {'blocks':>7s} {'n':>4s} {'us/call':>8s} {'ms/step':>8s} {'%':>5s}")
    for (name, blocks), (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:34]:
        per = ms / NSTEP
        print(f"{name:42s} {blocks:7d} {n//NSTEP:4d} {ms/n*1e3:8.1f} {per:8.3f} {per/tot*100:5.1f}")

if __name__ == "__main__":
    main()
