"""
Quick environment sanity check. Run after any toolkit/driver update.
Uses only NVIDIA official libraries (cuda.bindings, cuda.core, cuda.tile).
"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import grpo_rl.env  # must be first — sets CUDA_PATH

import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import runtime as cudart, driver as cdrv, nvrtc

def _chk(r):
    e = r[0] if isinstance(r, tuple) else r
    if hasattr(e, "value") and e.value != 0:
        raise RuntimeError(str(e))
    return r[1] if isinstance(r, tuple) and len(r) > 1 else None

def main():
    _chk(cudart.cudaFree(0))

    v = _chk(cudart.cudaRuntimeGetVersion())
    attrs = {}
    for attr in ["MultiProcessorCount", "GlobalMemoryBusWidth", "MemoryClockRate",
                 "L2CacheSize", "MaxSharedMemoryPerMultiprocessor", "CooperativeLaunch"]:
        val = _chk(cudart.cudaDeviceGetAttribute(
            getattr(cudart.cudaDeviceAttr, f"cudaDevAttr{attr}"), 0))
        attrs[attr] = val

    bw = attrs["MemoryClockRate"] * (attrs["GlobalMemoryBusWidth"] / 8) * 2 / 1e6

    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                        "--format=csv,noheader"], capture_output=True, text=True)

    print("Environment")
    print("=" * 52)
    print(f"  Python         : {sys.version.split()[0]}")
    print(f"  cuda-tile      : {ct.__version__}")
    print(f"  CUDA runtime   : {v//1000}.{(v%1000)//10}")
    print(f"  CUDA_PATH      : {grpo_rl.env.CUDA_ROOT}")
    print(f"  GPU            : {r.stdout.strip()}")
    print(f"  SM count       : {attrs['MultiProcessorCount']}")
    print(f"  L2 cache       : {attrs['L2CacheSize']//1024//1024} MB")
    print(f"  SMEM/SM (optin): {attrs['MaxSharedMemoryPerMultiprocessor']//1024} KB")
    print(f"  Bandwidth calc : {bw:.0f} GB/s")
    print(f"  CoopLaunch     : {attrs['CooperativeLaunch']}")

    # FP4 smoke test via nvrtc
    src = rb"""
extern "C" __global__ void smoke(float *D) {
    unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0,c1=0,c2=0,c3=0,d0,d1,d2,d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32.kind::f8f6f4 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n\t"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c0),"f"(c1),"f"(c2),"f"(c3));
    if(threadIdx.x==0) D[0]=d0;
}
"""
    try:
        opts = [b"--gpu-architecture=sm_120a", b"--std=c++17"] + grpo_rl.env.NVRTC_INCLUDES
        prog = _chk(nvrtc.nvrtcCreateProgram(src, b"smoke.cu", 0, [], []))
        _chk(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        print(f"\n  FP4 sm_120a smoke : OK")
    except Exception as e:
        print(f"\n  FP4 sm_120a smoke : FAIL — {str(e)[:100]}")

    print("=" * 52)

if __name__ == "__main__":
    main()
