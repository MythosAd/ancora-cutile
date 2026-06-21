"""De-risk the device-resident MoE dispatch: compile moe_dispatch.cu (plain CUDA, NVRTC) and verify
its output is BITWISE-identical to the host moe.build_layout (numpy stable-argsort grouping) on random
routing. This is the make-or-break for a fully-resident router (stage B). Foreground only."""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import nvrtc, driver as cdrv, runtime as cudart
import ancora.env

from ancora.kernels.moe import build_layout, TM
from ancora.kernels.loss import _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

CU = os.path.join(os.path.dirname(__file__), "..", "..", "ancora", "kernels", "moe_dispatch.cu")


def _compile(path, fname=b"moe_build_layout", arch=b"sm_120a"):
    """NVRTC-compile a plain-CUDA .cu (no system headers → no Windows hang) → CUfunction."""
    src = open(path, "rb").read()
    opts = [b"--gpu-architecture=" + arch, b"--std=c++17"]
    prog = nvrtc.nvrtcCreateProgram(src, b"moe_dispatch.cu", 0, [], [])[1]
    nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    lsz = nvrtc.nvrtcGetProgramLogSize(prog)[1]
    if lsz > 1:
        buf = b" " * lsz; nvrtc.nvrtcGetProgramLog(prog, buf)
        log = buf.decode(errors="replace")
        if "error" in log.lower(): raise RuntimeError("NVRTC:\n" + log)
    csz = nvrtc.nvrtcGetCUBINSize(prog)[1]
    cb = b" " * csz; nvrtc.nvrtcGetCUBIN(prog, cb)
    mod = cdrv.cuModuleLoadData(np.char.array(cb))[1]
    return cdrv.cuModuleGetFunction(mod, fname)[1]


FN = _compile(CU)


def _launch_layout(topi, topw, M, k, E):
    Rmax = M * k + E * TM; Rtmax = Rmax // TM
    d_topi = _GpuArray(topi.reshape(-1).astype(np.int32))
    d_topw = _GpuArray(topw.reshape(-1).astype(np.float32))
    d_src  = _GpuArray.zeros((Rmax,),  np.int32)
    d_gate = _GpuArray.zeros((Rmax,),  np.float32)
    d_tile = _GpuArray.zeros((Rtmax,), np.int32)
    d_toks = _GpuArray.zeros((M * k,), np.int32)
    d_off  = _GpuArray.zeros((E + 1,), np.int32)
    ptrs = [d_topi, d_topw, d_src, d_gate, d_tile, d_toks, d_off]
    vals = tuple(int(p._ptr) for p in ptrs) + (M, k, E, TM, Rmax, Rtmax)
    typs = (ctypes.c_void_p,) * len(ptrs) + (ctypes.c_int,) * 6
    smem = (2 * E + 1) * 4
    cfg = cdrv.CUlaunchConfig()
    cfg.gridDimX = cfg.gridDimY = cfg.gridDimZ = 1
    cfg.blockDimX = max(32 * E, 256); cfg.blockDimY = cfg.blockDimZ = 1   # ≥ E warps (one/expert)
    cfg.sharedMemBytes = smem; cfg.hStream = cdrv.CUstream(si); cfg.numAttrs = 0
    err, = cdrv.cuLaunchKernelEx(cfg, FN, (vals, typs), 0)
    if err.value: raise RuntimeError(f"launch: {err}")
    cudart.cudaStreamSynchronize(si)
    return dict(src=d_src.to_numpy(), gate=d_gate.to_numpy(), tile=d_tile.to_numpy(),
                toks=d_toks.to_numpy().reshape(M, k), off=d_off.to_numpy())


def _case(M, k=2, E=16, seed=0):
    rng = np.random.default_rng(seed)
    # random top-k routing: distinct experts per token (argsort of random scores), renormed weights
    scores = rng.standard_normal((M, E)).astype(np.float32)
    topi = np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int32)
    w = np.take_along_axis(np.exp(scores - scores.max(1, keepdims=True)), topi, 1)
    topw = (w / w.sum(1, keepdims=True)).astype(np.float32)

    host = build_layout(topi, topw, E)
    R, Rt = host["R"], host["Rt"]
    dev = _launch_layout(topi, topw, M, k, E)

    e_off  = np.array_equal(dev["off"], host["off_tiles"])
    e_tile = np.array_equal(dev["tile"][:Rt], host["tile_expert"])
    e_src  = np.array_equal(dev["src"][:R], host["src_row"])
    e_gate = np.array_equal(dev["gate"][:R].view(np.uint32), host["slot_gate"].view(np.uint32))
    e_toks = np.array_equal(dev["toks"], host["tok_slots"])
    # padding region (beyond R) must be zero so the fixed-Rmax grouped GEMM is harmless there
    pad_ok = (dev["src"][R:] == 0).all() and (dev["gate"][R:] == 0).all()
    ok = e_off and e_tile and e_src and e_gate and e_toks and pad_ok
    print(f"  M={M:5d} R={R:5d} Rt={Rt:3d}: off {e_off} tile {e_tile} src {e_src} gate {e_gate} "
          f"tok_slots {e_toks} pad0 {bool(pad_ok)}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("device moe_build_layout (NVRTC plain-CUDA) vs host build_layout — bitwise")
    print("=" * 84)
    r = [_case(M, seed=s) for s, M in enumerate([128, 256, 512, 1024, 2048])]
    print("=" * 84)
    print("  ALL PASS (device dispatch bitwise == host stable-argsort grouping)" if all(r)
          else "  FAIL: " + str(r))
