"""Per-operator GPU-time breakdown of the device-resident MoE layer (fwd+bwd), admin-free (CUDA
events). Wraps BOTH ct.launch (cuda.tile kernels: attention, norms, expert GEMMs, swiglu, combine,
weight grads) AND moe_dispatch._launch (the raw-CUDA router kernels: gate, build_layout, gate_bwd,
dW, dh). Reports which operator dominates the MoE forward and backward — the optimization target.

Run:  python tests/hardware/_profile_moe.py [S] [global|local]
"""
import sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart, driver as cdrv

from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, _bf
from ancora.model.resident_moe import ResidentMoELayer
from ancora.model.resident import _DBuf, _f32bf
import ancora.kernels.moe_dispatch as md

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

_rec = []; _capture = False
_real_ct = ct.launch
_real_md = md._launch


def _name_ct(kernel):
    pf = getattr(kernel, "_pyfunc", None)
    return getattr(pf, "__name__", None) or str(kernel)[:28]


def _ev(stream, fn):
    _, e0 = cudart.cudaEventCreate(); _, e1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(e0, stream); r = fn(); cudart.cudaEventRecord(e1, stream)
    return r, e0, e1


def _ct_launch(stream, grid, kernel, args):
    if not _capture: return _real_ct(stream, grid, kernel, args)
    r, e0, e1 = _ev(stream, lambda: _real_ct(stream, grid, kernel, args))
    _rec.append((_name_ct(kernel), e0, e1)); return r


def _md_launch(fn, grid, block, vals, typs, si_, smem=0):
    if not _capture: return _real_md(fn, grid, block, vals, typs, si_, smem)
    nm = _md_names.get(int(fn) if not hasattr(fn, "value") else int(fn.value), "router_cuda")
    r, e0, e1 = _ev(si_, lambda: _real_md(fn, grid, block, vals, typs, si_, smem))
    _rec.append((nm, e0, e1)); return r


ct.launch = _ct_launch
md._launch = _md_launch
_md_names = {}


def _report(tag, iters):
    by = collections.defaultdict(lambda: [0.0, 0])
    for name, e0, e1 in _rec:
        _, ms = cudart.cudaEventElapsedTime(e0, e1); by[name][0] += ms; by[name][1] += 1
        cudart.cudaEventDestroy(e0); cudart.cudaEventDestroy(e1)
    total = sum(v[0] for v in by.values())
    print(f"\n  {tag}   total {total/iters*1000:.1f} µs   ({len(by)} kernels, {sum(v[1] for v in by.values())//iters} launches)")
    print(f"    {'kernel':<26}{'µs':>9}{'%':>7}{'n':>5}")
    for name, (tms, cnt) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"    {name:<26}{tms/iters*1000:>9.1f}{tms/total*100:>6.1f}%{cnt//iters:>5}")
    _rec.clear()
    return total / iters * 1000


def main(S=1024, is_global=False):
    global _capture
    cfg = MoEConfig(); B = 2; M = B * S; H = cfg.hidden
    host = MoEDecoderLayer(cfg, is_global=is_global, ffn_dense=False, seed=7)
    res = ResidentMoELayer(cfg, host.attn, host.ffn.w, B, S, is_global=is_global, window=cfg.window, device_route=True)
    md._ensure(); _md_names.update({int(v): k.decode() for k, v in md._FN.items()})
    rng = np.random.default_rng(2)
    gx = _DBuf((rng.standard_normal((M, H)) * 0.5).astype(np.float32))
    gd = _DBuf(_f32bf((rng.standard_normal((M, H)) * 0.3).astype(np.float32)))
    for _ in range(4): res.forward(gx, si); res.backward(gd, si)     # warm up (JIT)
    cudart.cudaStreamSynchronize(si)

    tag = "global+MoE" if is_global else "local+MoE"
    print(f"Device-resident {tag} layer (device_route)  B={B} S={S} M={M} H={H}")
    print("=" * 64)
    it = 20
    _capture = True
    for _ in range(it): res.forward(gx, si)
    cudart.cudaStreamSynchronize(si); _capture = False
    tf = _report(f"{tag} FORWARD ", it)
    _capture = True
    for _ in range(it): res.forward(gx, si); res.backward(gd, si)
    cudart.cudaStreamSynchronize(si); _capture = False
    # subtract forward to isolate backward
    _rec.clear()
    _capture = True
    for _ in range(it): res.backward(gd, si)
    cudart.cudaStreamSynchronize(si); _capture = False
    _report(f"{tag} BACKWARD", it)
    print("=" * 64)


if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    g = (len(sys.argv) > 2 and sys.argv[2] == "global")
    main(S, g)
