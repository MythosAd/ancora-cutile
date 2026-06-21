"""
ancora/kernels/moe_dispatch.py — device-resident MoE router (stage A gating + stage B dispatch),
the plumbing that makes GroupedMoEFFN.forward_resident need ZERO host round-trip.

Two plain-CUDA kernels (moe_dispatch.cu, NVRTC-compiled for sm_120a — no cub, no system headers,
no Windows NVRTC hang) launched on the resident stream:
  - moe_router_gate : gh2 (M,H bf16) + Wr (H,E fp32) → topi (M,k), topw (M,k), probs (M,E)  [FP32]
  - moe_build_layout: topi/topw → src_row/slot_gate/tile_expert/tok_slots/off_tiles (device), the
                      SAME stable-argsort grouping build_layout does on host (bitwise-verified,
                      tests/kernels/test_moe_dispatch.py), on a FIXED Rmax grid (padding → 0).

Both are per-row / deterministic ⇒ batch-invariant. The router weight Wr lives on device (FP32).
"""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from cuda.bindings import nvrtc, driver as cdrv
import ancora.env  # noqa: F401

_CU = os.path.join(os.path.dirname(__file__), "moe_dispatch.cu")
_FN = {}   # name -> CUfunction (compiled once, lazily)


def _ensure():
    if _FN: return
    src = open(_CU, "rb").read()
    opts = [b"--gpu-architecture=sm_120a", b"--std=c++17"]
    prog = nvrtc.nvrtcCreateProgram(src, b"moe_dispatch.cu", 0, [], [])[1]
    nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    lsz = nvrtc.nvrtcGetProgramLogSize(prog)[1]
    if lsz > 1:
        buf = b" " * lsz; nvrtc.nvrtcGetProgramLog(prog, buf)
        log = buf.decode(errors="replace")
        if "error" in log.lower(): raise RuntimeError("NVRTC moe_dispatch:\n" + log)
    csz = nvrtc.nvrtcGetCUBINSize(prog)[1]
    cb = b" " * csz; nvrtc.nvrtcGetCUBIN(prog, cb)
    mod = cdrv.cuModuleLoadData(np.char.array(cb))[1]
    for nm in (b"moe_router_gate", b"moe_build_layout", b"moe_router_gate_bwd",
               b"moe_router_dW_part", b"moe_router_dW_reduce", b"moe_router_dh"):
        _FN[nm] = cdrv.cuModuleGetFunction(mod, nm)[1]


ROUTER_DW_NSPL = 16   # M-splits for the router weight-grad (occupancy); Gpart is (NSPL,H,E)


_P = lambda b: int(b._ptr if hasattr(b, "_ptr") else b.ptr)


def _launch(fn, grid, block, vals, typs, si, smem=0):
    cfg = cdrv.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block
    cfg.sharedMemBytes = smem; cfg.hStream = cdrv.CUstream(si); cfg.numAttrs = 0
    err, = cdrv.cuLaunchKernelEx(cfg, fn, (vals, typs), 0)
    if err.value: raise RuntimeError(f"moe_dispatch launch: {err}")


def router_gate(h, Wr, topi, topw, probs, M, H, E, k, norm, si):
    """Device gating: h(M,H bf16 bits), Wr(H,E f32) → topi(M,k i32), topw(M,k f32), probs(M,E f32)."""
    assert E == 16, "moe_dispatch gate/router-bwd kernels hardcode RE=16 (this model family is E=16)"
    _ensure()
    p = lambda b: int(b._ptr if hasattr(b, "_ptr") else b.ptr)
    vals = (p(h), p(Wr), p(topi), p(topw), p(probs), M, H, E, k, int(norm))
    typs = (ctypes.c_void_p,) * 5 + (ctypes.c_int,) * 5
    # one WARP per token (coalesced h load) → 256 threads = 8 tokens/block
    _launch(_FN[b"moe_router_gate"], ((M * 32 + 255) // 256, 1, 1), (256, 1, 1), vals, typs, si)


def build_layout_dev(topi, topw, src_row, slot_gate, tile_expert, tok_slots, off_tiles,
                     M, k, E, TM, Rmax, Rtmax, si):
    """Device dispatch (single block): topi/topw → the 5 grouped-layout arrays (== host build_layout)."""
    _ensure()
    p = lambda b: int(b._ptr if hasattr(b, "_ptr") else b.ptr)
    vals = (p(topi), p(topw), p(src_row), p(slot_gate), p(tile_expert), p(tok_slots), p(off_tiles),
            M, k, E, TM, Rmax, Rtmax)
    typs = (ctypes.c_void_p,) * 7 + (ctypes.c_int,) * 6
    blk = max(32 * E, 256)   # ≥ E warps (one per expert for the stable scatter)
    _launch(_FN[b"moe_build_layout"], (1, 1, 1), (blk, 1, 1), vals, typs, si, smem=(2 * E + 1) * 4)


# ── router BACKWARD on device (gate-bwd → d_logits → the two router GEMMs) ──
def router_gate_bwd(dsg, tok_slots, topi, probs, dlogits, M, E, k, norm, si):
    """Gate backward: dsg (Rmax,) per-slot grad → dlogits (M,E) (mirrors host _gate_backward)."""
    _ensure(); P = _P
    vals = (P(dsg), P(tok_slots), P(topi), P(probs), P(dlogits), M, E, k, int(norm))
    typs = (ctypes.c_void_p,) * 5 + (ctypes.c_int,) * 4
    _launch(_FN[b"moe_router_gate_bwd"], ((M + 127) // 128, 1, 1), (128, 1, 1), vals, typs, si)


def router_dW(h, dlogits, Grouter, Gpart, M, H, E, si):
    """Router weight grad Grouter (H,E) = hᵀ @ dlogits, 2-pass split-M. Gpart is a (NSPL·H, E) scratch."""
    assert E == 16, "moe_router_dW_part hardcodes RE=16"
    _ensure(); P = _P; ns = ROUTER_DW_NSPL; blk = 128
    a = (P(h), P(dlogits), P(Gpart), M, H, E, ns)
    at = (ctypes.c_void_p,) * 3 + (ctypes.c_int,) * 4
    _launch(_FN[b"moe_router_dW_part"], ((H + blk - 1) // blk, ns, 1), (blk, 1, 1), a, at, si)
    b = (P(Gpart), P(Grouter), H * E, ns)
    bt = (ctypes.c_void_p,) * 2 + (ctypes.c_int,) * 2
    _launch(_FN[b"moe_router_dW_reduce"], ((H * E + 255) // 256, 1, 1), (256, 1, 1), b, bt, si)


def router_dh(dlogits, Wr, gdh2_e, gdh2, M, H, E, si):
    """Router→gh2 path + fuse the expert+router grad add: gdh2 = gdh2_e + dlogits @ Wrᵀ (warp/token)."""
    _ensure(); P = _P
    vals = (P(dlogits), P(Wr), P(gdh2_e), P(gdh2), M, H, E)
    typs = (ctypes.c_void_p,) * 4 + (ctypes.c_int,) * 3
    _launch(_FN[b"moe_router_dh"], ((M * 32 + 255) // 256, 1, 1), (256, 1, 1), vals, typs, si)
