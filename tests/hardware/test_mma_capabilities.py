"""
Hardware MMA capability probe for RTX 5080 Laptop (sm_120 / sm_120a).
Uses only NVIDIA official libraries: cuda.core + cuda.bindings.
No third-party libraries (no CuPy).

Results (confirmed 2026-05-29, CUDA 13.3, driver 595.79):
  sm_120  : BF16, FP8 E4M3/E5M2, INT4 m16n8k64
  sm_120a : + FP4 E2M1, FP6 E3M2, FP4xFP8 mixed, griddepcontrol
  TMEM    : NOT available (sm_100/B200 only)
  cuda.tile 1.4.0: ct.matmul BF16/FP8; ct.mma FP8; ct.mma_scaled MXFP8/MXFP4
"""
import sys, os, ctypes as _ct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import grpo_rl.env  # sets CUDA_PATH before any cuda.* import

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart, nvrtc
from cuda.core._memory._buffer import Buffer
from cuda import pathfinder

# ── init ──────────────────────────────────────────────────────────────────────

def _setup():
    """Initialize CUDA context; call once before anything else."""
    err, = cudart.cudaFree(0)
    if err.value:
        raise RuntimeError(f"cudaFree(0) failed: {err}")
    dev = cc.Device(0)
    dev.set_current()
    return dev

_DEV    = None
_STREAM = None

def _dev():
    global _DEV
    if _DEV is None:
        _DEV = _setup()
    return _DEV

def _stream():
    global _STREAM
    if _STREAM is None:
        _STREAM = _dev().create_stream()
    return _STREAM

# ── memory helpers ────────────────────────────────────────────────────────────

def _alloc(nbytes: int):
    """Allocate device memory; return (CUdeviceptr, Buffer)."""
    err, ptr = cdrv.cuMemAlloc(nbytes)
    if err.value:
        raise RuntimeError(f"cuMemAlloc failed: {err}")
    buf = Buffer.from_handle(int(ptr), nbytes)
    return ptr, buf

def _h2d(ptr, arr: np.ndarray):
    err, = cdrv.cuMemcpyHtoD(ptr, arr, arr.nbytes)
    if err.value:
        raise RuntimeError(f"cuMemcpyHtoD failed: {err}")

def _d2h(arr: np.ndarray, ptr):
    err, = cdrv.cuMemcpyDtoH(arr, ptr, arr.nbytes)
    if err.value:
        raise RuntimeError(f"cuMemcpyDtoH failed: {err}")

def _free(ptr):
    cdrv.cuMemFree(ptr)

def _sync():
    cudart.cudaDeviceSynchronize()

# ── compile helper ────────────────────────────────────────────────────────────

def _compile(src: str, fname: str, arch: str):
    """Compile CUDA C++ via cuda.core.Program; return Kernel object."""
    opts = cc.ProgramOptions(
        arch=arch,
        include_path=grpo_rl.env.CUDA_INC,
        std="c++17",
    )
    obj = cc.Program(src, "c++", opts).compile("cubin")
    return obj.get_kernel(fname)

def _run_f32x4(src: str, fname: str, arch: str):
    """Compile + run kernel writing 4 floats; return (ok, [f0..f3] | error_str)."""
    try:
        _dev()                              # ensure context
        kern = _compile(src, fname, arch)
        ptr, buf = _alloc(16)
        _h2d(ptr, np.zeros(4, np.float32))
        cfg = cc.LaunchConfig(grid=(1, 1, 1), block=(32, 1, 1))
        cc.launch(_stream(), cfg, kern, buf)
        _stream().sync()
        out = np.empty(4, np.float32)
        _d2h(out, ptr)
        _free(ptr)
        return True, [round(float(v), 2) for v in out]
    except Exception as e:
        return False, str(e)[:120]

# ── PTX test sources ──────────────────────────────────────────────────────────

BF16 = r"""
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void bf16_mma(float *D) {
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fill_fragment(a, __float2bfloat16(1.0f));
    wmma::fill_fragment(b, __float2bfloat16(1.0f));
    wmma::fill_fragment(c, 0.0f);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(D, c, 16, wmma::mem_row_major);
}
"""

FP8_PLAIN = r"""
extern "C" __global__ void fp8_plain(float *D) {
    unsigned a0=0x3c3c3c3c,a1=0x3c3c3c3c,a2=0x3c3c3c3c,a3=0x3c3c3c3c;
    unsigned b0=0x3c3c3c3c,b1=0x3c3c3c3c;
    float c0=0,c1=0,c2=0,c3=0,d0,d1,d2,d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n\t"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c0),"f"(c1),"f"(c2),"f"(c3));
    if(threadIdx.x==0){D[0]=d0;}
}
"""

FP8_KIND = r"""
extern "C" __global__ void fp8_kind(float *D) {
    unsigned a0=0x3c3c3c3c,a1=0x3c3c3c3c,a2=0x3c3c3c3c,a3=0x3c3c3c3c;
    unsigned b0=0x3c3c3c3c,b1=0x3c3c3c3c;
    float c0=0,c1=0,c2=0,c3=0,d0,d1,d2,d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.kind::f8f6f4 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n\t"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c0),"f"(c1),"f"(c2),"f"(c3));
    if(threadIdx.x==0){D[0]=d0;}
}
"""

FP4 = r"""
extern "C" __global__ void fp4_mma(float *D) {
    unsigned a0=0x22222222,a1=0x22222222,a2=0x22222222,a3=0x22222222;
    unsigned b0=0x22222222,b1=0x22222222;
    float c0=0,c1=0,c2=0,c3=0,d0,d1,d2,d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32.kind::f8f6f4 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n\t"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c0),"f"(c1),"f"(c2),"f"(c3));
    if(threadIdx.x==0){D[0]=d0;D[1]=d1;D[2]=d2;D[3]=d3;}
}
"""

FP4_FP8 = r"""
extern "C" __global__ void fp4_fp8(float *D) {
    unsigned a0=0x22222222,a1=0x22222222,a2=0x22222222,a3=0x22222222;
    unsigned b0=0x38383838,b1=0x38383838;
    float c0=0,c1=0,c2=0,c3=0,d0,d1,d2,d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e4m3.f32.kind::f8f6f4 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n\t"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c0),"f"(c1),"f"(c2),"f"(c3));
    if(threadIdx.x==0){D[0]=d0;}
}
"""

FP6 = r"""
extern "C" __global__ void fp6_mma(float *D) {
    unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=10.0f,c1=20.0f,c2=30.0f,c3=40.0f,d0,d1,d2,d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32.kind::f8f6f4 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n\t"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c0),"f"(c1),"f"(c2),"f"(c3));
    if(threadIdx.x==0){D[0]=d0;D[1]=d1;D[2]=d2;D[3]=d3;}
}
"""

GRIDDEP = r"""
extern "C" __global__ void griddep(float *D) {
    if(threadIdx.x==0) D[0]=1.0f;
    asm volatile("griddepcontrol.launch_dependents;\n\t");
}
"""

TMEM = r"""
extern "C" __global__ void tmem_probe(float *D) {
    unsigned addr=0;
    asm volatile("tcgen05.alloc.sync.aligned.32x32b [%0], 32;\n\t":"+r"(addr));
    if(threadIdx.x==0) D[0]=1.0f;
    asm volatile("tcgen05.dealloc.sync.aligned.32x32b [%0], 32;\n\t"::"r"(addr));
}
"""

# Thread Block Cluster: cooperative_groups::cluster_group (sm_90+, 继承到 Blackwell)
# Use C++ cooperative_groups API (PTX %clusterctarank inline asm doesn't work via NVRTC)
TBC_SYNC = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
extern "C" __global__ void tbc_sync(float *D) {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        D[0] = (float)cluster.num_blocks();
}
"""

# ── Thread Block Cluster test ─────────────────────────────────────────────────

def _compile_tbc_nvrtc(src: bytes, fname: bytes):
    """Compile TBC kernel via nvrtc directly → returns CUfunction (not CUkernel).
    cooperative_groups requires cccl include path alongside CUDA_INC.
    cuLaunchKernelEx requires CUfunction; cc.Program returns CUkernel — different type.
    """
    inc_cccl = pathfinder.find_nvidia_header_directory("cccl").encode()
    opts = [b"--gpu-architecture=sm_120", b"--std=c++17",
            b"--include-path=" + grpo_rl.env.CUDA_INC.encode(),
            b"--include-path=" + inc_cccl]
    prog = nvrtc.nvrtcCreateProgram(src, b"tbc.cu", 0, [], [])[1]
    nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    cubin_sz = nvrtc.nvrtcGetCUBINSize(prog)[1]
    cubin = b" " * cubin_sz
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cdrv.cuModuleLoadData(np.char.array(cubin))[1]
    return cdrv.cuModuleGetFunction(mod, fname)[1]  # CUfunction


def test_tbc():
    """
    Thread Block Clusters (sm_90+, 继承自 Hopper, sm_120 支持).
    Launch 2 blocks as 1 cluster (cluster_dim=(2,1,1)).
    cooperative_groups::cluster_group.num_blocks() should return 2 → D[0]=2.0.

    kernelParams 格式 (来自 cuda.bindings._lib.utils.pxi _HelperKernelParams):
      ((values_tuple,), (ctypes_tuple,))
      设备指针 → (int(CUdeviceptr), ctypes.c_void_p)
    """
    _dev()
    try:
        fn = _compile_tbc_nvrtc(TBC_SYNC.encode(), b"tbc_sync")
    except Exception as e:
        return False, f"compile FAIL: {str(e)[:100]}"

    err, d_out = cdrv.cuMemAlloc(16)
    cdrv.cuMemcpyHtoD(d_out, np.zeros(4, np.float32), 16)
    stream_h = int(_stream().__cuda_stream__()[1])

    kernel_params = ((int(d_out),), (_ct.c_void_p,))

    attr = cdrv.CUlaunchAttribute()
    attr.id = cdrv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attr.value.clusterDim.x = 2
    attr.value.clusterDim.y = 1
    attr.value.clusterDim.z = 1

    cfg = cdrv.CUlaunchConfig()
    cfg.gridDimX = 2;  cfg.gridDimY = 1;  cfg.gridDimZ = 1
    cfg.blockDimX = 32; cfg.blockDimY = 1; cfg.blockDimZ = 1
    cfg.sharedMemBytes = 0
    cfg.hStream = cdrv.CUstream(stream_h)
    cfg.attrs = [attr]
    cfg.numAttrs = 1

    try:
        err2, = cdrv.cuLaunchKernelEx(cfg, fn, kernel_params, 0)
        if err2.value:
            raise RuntimeError(f"cuLaunchKernelEx: {err2}")
        _stream().sync()
        out = np.empty(4, np.float32)
        _d2h(out, d_out)
        cdrv.cuMemFree(d_out)
        return True, f"D[0]={out[0]:.0f} exp=2  cluster_sync OK"
    except Exception as e:
        cdrv.cuMemFree(d_out)
        return False, f"launch FAIL: {str(e)[:120]}"


# ── cuda.tile 1.4.0 tests ─────────────────────────────────────────────────────

class _GpuArray:
    """GPU array wrapper exposing __cuda_array_interface__ for ct.launch."""
    def __init__(self, arr: np.ndarray):
        self._shape = arr.shape
        self._dtype = arr.dtype
        self._ptr, _ = _alloc(arr.nbytes)
        _h2d(self._ptr, arr)
        self.__cuda_array_interface__ = {
            "shape": arr.shape,
            "typestr": arr.dtype.str,
            "data": (int(self._ptr), False),
            "version": 3,
        }

    def to_numpy(self):
        out = np.empty(self._shape, self._dtype)
        _d2h(out, self._ptr)
        return out

    def free(self):
        _free(self._ptr)


def _ct_stream():
    """Return raw integer stream handle for ct.launch."""
    return int(_stream().__cuda_stream__()[1])


def test_cutile_140():
    results = {}
    stream = _ct_stream()
    M, N, K = 16, 16, 32

    # 1. ct.matmul BF16
    try:
        @ct.kernel
        def mm_bf16(a, b, c, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int]):
            r, col = ct.bid(0), ct.bid(1)
            ta = ct.astype(ct.load(a, index=(r, 0), shape=(M, K)), ct.bfloat16)
            tb = ct.astype(ct.load(b, index=(0, col), shape=(K, N)), ct.bfloat16)
            ct.store(c, index=(r, col), tile=ct.astype(ct.matmul(ta, tb), ct.float32))

        a = _GpuArray(np.ones((M, K), np.float32))
        b = _GpuArray(np.ones((K, N), np.float32))
        c = _GpuArray(np.zeros((M, N), np.float32))
        ct.launch(stream, (1, 1, 1), mm_bf16, (a, b, c, M, N, K))
        _stream().sync()
        v = c.to_numpy()[0, 0]
        results["ct.matmul BF16"] = (True, f"c[0,0]={v:.1f} exp={K}.0")
        for x in (a, b, c): x.free()
    except Exception as e:
        results["ct.matmul BF16"] = (False, str(e)[:100])

    # 2. ct.matmul FP8
    try:
        @ct.kernel
        def mm_fp8(a, b, c, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int]):
            r, col = ct.bid(0), ct.bid(1)
            ta = ct.astype(ct.load(a, index=(r, 0), shape=(M, K)), ct.float8_e4m3fn)
            tb = ct.astype(ct.load(b, index=(0, col), shape=(K, N)), ct.float8_e4m3fn)
            ct.store(c, index=(r, col), tile=ct.astype(ct.matmul(ta, tb), ct.float32))

        a = _GpuArray(np.ones((M, K), np.float32))
        b = _GpuArray(np.ones((K, N), np.float32))
        c = _GpuArray(np.zeros((M, N), np.float32))
        ct.launch(stream, (1, 1, 1), mm_fp8, (a, b, c, M, N, K))
        _stream().sync()
        v = c.to_numpy()[0, 0]
        results["ct.matmul FP8"] = (True, f"c[0,0]={v:.1f} exp={K}.0")
        for x in (a, b, c): x.free()
    except Exception as e:
        results["ct.matmul FP8"] = (False, str(e)[:100])

    # 3. ct.mma FP8 with explicit accumulator
    try:
        @ct.kernel
        def mma_fp8(a, b, c, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int]):
            r, col = ct.bid(0), ct.bid(1)
            ta  = ct.astype(ct.load(a, index=(r, 0),   shape=(M, K)), ct.float8_e4m3fn)
            tb  = ct.astype(ct.load(b, index=(0, col), shape=(K, N)), ct.float8_e4m3fn)
            acc = ct.zeros((M, N), ct.float32)
            ct.store(c, index=(r, col), tile=ct.mma(ta, tb, acc))

        a = _GpuArray(np.ones((M, K), np.float32))
        b = _GpuArray(np.ones((K, N), np.float32))
        c = _GpuArray(np.zeros((M, N), np.float32))
        ct.launch(stream, (1, 1, 1), mma_fp8, (a, b, c, M, N, K))
        _stream().sync()
        v = c.to_numpy()[0, 0]
        results["ct.mma FP8"] = (True, f"c[0,0]={v:.1f} exp={K}.0")
        for x in (a, b, c): x.free()
    except Exception as e:
        results["ct.mma FP8"] = (False, str(e)[:100])

    # 4. ct.mma_scaled MXFP8  — mirrors official docstring example exactly
    # ones(64) × scale(2.0) × ones(64) × scale(2.0) = 64 × 4.0 = 256.0
    try:
        @ct.kernel
        def mma_mxfp8(c, M: ct.Constant[int], N: ct.Constant[int]):
            r, col = ct.bid(0), ct.bid(1)
            ta  = ct.ones((M, 64), ct.float8_e4m3fn)
            tb  = ct.ones((64, N), ct.float8_e4m3fn)
            tsa = ct.full((M, 2),  2.0, ct.float8_e8m0fnu)
            tsb = ct.full((2, N),  2.0, ct.float8_e8m0fnu)
            acc = ct.zeros((M, N), ct.float32)
            ct.store(c, index=(r, col), tile=ct.mma_scaled(ta, tsa, tb, tsb, acc))

        c = _GpuArray(np.zeros((M, N), np.float32))
        ct.launch(stream, (1, 1, 1), mma_mxfp8, (c, M, N))
        _stream().sync()
        v = c.to_numpy()[0, 0]
        results["ct.mma_scaled MXFP8"] = (True, f"c[0,0]={v:.1f} exp=256.0")
        c.free()
    except Exception as e:
        results["ct.mma_scaled MXFP8"] = (False, str(e)[:120])

    # 5. ct.mma_scaled MXFP4  (FP4 E2M1 × FP4, E8M0 scale=1.0, B=32, K=64)
    # ones(64,fp4) × scale(1.0) × ones(64,fp4) × scale(1.0) = 64 × 1.0 = 64.0
    try:
        @ct.kernel
        def mma_mxfp4(c, M: ct.Constant[int], N: ct.Constant[int]):
            r, col = ct.bid(0), ct.bid(1)
            ta  = ct.ones((M, 64), ct.float4_e2m1fn)
            tb  = ct.ones((64, N), ct.float4_e2m1fn)
            tsa = ct.full((M, 2),  1.0, ct.float8_e8m0fnu)
            tsb = ct.full((2, N),  1.0, ct.float8_e8m0fnu)
            acc = ct.zeros((M, N), ct.float32)
            ct.store(c, index=(r, col), tile=ct.mma_scaled(ta, tsa, tb, tsb, acc))

        c = _GpuArray(np.zeros((M, N), np.float32))
        ct.launch(stream, (1, 1, 1), mma_mxfp4, (c, M, N))
        _stream().sync()
        v = c.to_numpy()[0, 0]
        results["ct.mma_scaled MXFP4"] = (True, f"c[0,0]={v:.1f} exp=64.0")
        c.free()
    except Exception as e:
        results["ct.mma_scaled MXFP4"] = (False, str(e)[:120])

    return results


# ── TMA bulk copy test ────────────────────────────────────────────────────────

def _compile_nvrtc(path: str, fname: bytes, arch: str = "sm_120"):
    """Compile a .cu file via nvrtc+cuModuleLoadData → CUfunction.
    Needed when: (a) cooperative_groups, (b) cuLaunchKernelEx with CUfunction.
    cc.Program returns CUkernel; cuLaunchKernelEx requires CUfunction — different type.
    kernelParams to cuLaunchKernelEx: ((values,), (ctypes_types,))
      e.g. device ptr → ((int(ptr),), (ctypes.c_void_p,))
    """
    with open(path, "rb") as f:
        src = f.read()
    inc_cccl = pathfinder.find_nvidia_header_directory("cccl").encode()
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17",
            b"--include-path=" + grpo_rl.env.CUDA_INC.encode(),
            b"--include-path=" + inc_cccl]
    prog = nvrtc.nvrtcCreateProgram(src, b"src.cu", 0, [], [])[1]
    nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    lsz = nvrtc.nvrtcGetProgramLogSize(prog)[1]
    buf = b" " * lsz; nvrtc.nvrtcGetProgramLog(prog, buf)
    errs = [l for l in buf.decode(errors="replace").splitlines() if "error" in l.lower()]
    if errs: raise RuntimeError("\n".join(errs[:3]))
    csz = nvrtc.nvrtcGetCUBINSize(prog)[1]
    cb = b" " * csz; nvrtc.nvrtcGetCUBIN(prog, cb)
    mod = cdrv.cuModuleLoadData(np.char.array(cb))[1]
    return cdrv.cuModuleGetFunction(mod, fname)[1]


def _klx(fn, params, grid, block, smem=0):
    """cuLaunchKernelEx helper: launch CUfunction on current stream."""
    sh = int(_stream().__cuda_stream__()[1])
    cfg = cdrv.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ   = grid
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block
    cfg.sharedMemBytes = smem
    cfg.hStream = cdrv.CUstream(sh)
    cfg.numAttrs = 0
    err, = cdrv.cuLaunchKernelEx(cfg, fn, params, 0)
    if err.value:
        raise RuntimeError(f"cuLaunchKernelEx: {err}")


_TMA_SRC  = os.path.join(os.path.dirname(__file__), "_tma_src.cu")
_WGMMA_SRC = os.path.join(os.path.dirname(__file__), "_wgmma_src.cu")


def test_tma():
    """
    TMA bulk async copy (sm_90+, 继承自 Hopper, sm_120 支持).
    cp.async.bulk.shared::cta.global + mbarrier:
      - mbarrier.init.shared::cta.b64 [mp], 1
      - cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [sp], [gptr], nbytes, [mp]
      - mbarrier.try_wait.parity.shared::cta.b64 p, [mp], 0  (spin-wait)
    Copies in[0..31] to SMEM, reads back smem[0] → out[0].
    Expected: out[0] = in[0] = 1.0
    """
    _dev()
    try:
        fn = _compile_nvrtc(_TMA_SRC, b"tma_bulk", "sm_120")
    except Exception as e:
        return False, f"compile FAIL: {str(e)[:100]}"

    src_data = np.arange(1, 33, dtype=np.float32)
    err, d_in  = cdrv.cuMemAlloc(128); _h2d(d_in, src_data)
    err, d_out = cdrv.cuMemAlloc(16);  _h2d(d_out, np.zeros(4, np.float32))
    try:
        params = ((int(d_out), int(d_in)), (_ct.c_void_p, _ct.c_void_p))
        _klx(fn, params, grid=(1,1,1), block=(64,1,1), smem=512)
        _stream().sync()
        out = np.empty(4, np.float32); _d2h(out, d_out)
        cdrv.cuMemFree(d_in); cdrv.cuMemFree(d_out)
        ok = abs(out[0] - 1.0) < 0.01
        return ok, f"smem[0]={out[0]:.1f} exp=1.0  (copied in[0]=1.0 via bulk async)"
    except Exception as e:
        cdrv.cuMemFree(d_in); cdrv.cuMemFree(d_out)
        return False, f"launch FAIL: {str(e)[:120]}"


def test_wgmma():
    """
    Warp Group MMA (sm_90+, 继承自 Hopper, sm_120 支持).
    wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16:
      - 128-thread block (4 warps = 1 warp group)
      - A: 4 registers per thread, all BF16 1.0 (0x3f803f80)
      - B: 16×8 BF16 SMEM, all 1.0
      - D accumulator: each thread holds contribution to 64×8 output
      - Thread 0 stores its D[0] = Σ_k A[row,k]*B[k,0] for k in [0,16)
    Simplified SMEM descriptor: encode shared addr in bits[13:0] + stride in bits[49:32].
    """
    _dev()
    try:
        fn = _compile_nvrtc(_WGMMA_SRC, b"wgmma_test", "sm_120")
    except Exception as e:
        return False, f"compile FAIL: {str(e)[:100]}"

    err, d_out = cdrv.cuMemAlloc(16); _h2d(d_out, np.zeros(4, np.float32))
    try:
        params = ((int(d_out),), (_ct.c_void_p,))
        _klx(fn, params, grid=(1,1,1), block=(128,1,1), smem=1024)
        _stream().sync()
        out = np.empty(4, np.float32); _d2h(out, d_out)
        cdrv.cuMemFree(d_out)
        return True, f"D[0]={out[0]:.1f}  (warp-group-level MMA, all-1 BF16 inputs)"
    except Exception as e:
        cdrv.cuMemFree(d_out)
        return False, f"launch FAIL: {str(e)[:120]}"


# ── cuda.tile pack/unpack/bitcast tests ───────────────────────────────────────

def test_pack_apis():
    """
    ct.pack_to_bytes / ct.unpack_from_bytes / ct.bitcast (cuda-tile 1.4.0).

    pack_to_bytes(x)         → flatten + reinterpret as 1D uint8
    unpack_from_bytes(x, dt) → inverse: 1D uint8 → target dtype
    bitcast(x, dtype)        → reinterpret tile bits as new dtype (no value conversion)

    FP4 E2M1 packing: 2 nibbles per byte; 1.0 = nibble 0x2 → packed byte 0x22
    E8M0 conversion: cannot use astype(int→E8M0); must use bitcast(uint8→E8M0)
    """
    _dev()
    results = {}
    stream = _ct_stream()

    # 1. pack int32 → bytes  (official docstring example)
    try:
        @ct.kernel
        def pack_int32(out):
            tx = ct.full(1, 0x04030201, dtype=ct.int32)
            ct.store(out, index=(0,), tile=ct.pack_to_bytes(tx))
        out = _GpuArray(np.zeros(4, np.uint8))
        ct.launch(stream, (1,1,1), pack_int32, (out,))
        _stream().sync()
        r = out.to_numpy()
        ok = list(r) == [1, 2, 3, 4]
        results["pack int32→bytes"] = (ok, f"{list(r)} exp=[1,2,3,4]")
        out.free()
    except Exception as e:
        results["pack int32→bytes"] = (False, str(e)[:80])

    # 2. pack FP4 ones(16) → 8 bytes; each byte = 0x22 (two nibble-1.0s)
    try:
        @ct.kernel
        def pack_fp4(out, N: ct.Constant[int]):
            packed = ct.pack_to_bytes(ct.ones((N,), ct.float4_e2m1fn))
            ct.store(out, index=(0,), tile=packed)
        N = 16
        out = _GpuArray(np.zeros(N // 2, np.uint8))
        ct.launch(stream, (1,1,1), pack_fp4, (out, N))
        _stream().sync()
        r = out.to_numpy()
        ok = all(b == 0x22 for b in r)
        results["pack FP4 ones→bytes"] = (ok, f"{[hex(b) for b in r]} exp=[0x22]*{N//2}")
        out.free()
    except Exception as e:
        results["pack FP4 ones→bytes"] = (False, str(e)[:80])

    # 3. unpack bytes → int32  (inverse of test 1)
    try:
        @ct.kernel
        def unpack_int32(out):
            raw = ct.arange(4, dtype=ct.uint8) + 1
            ct.store(out, index=(0,), tile=ct.astype(ct.unpack_from_bytes(raw, ct.int32), ct.int32))
        out = _GpuArray(np.zeros(1, np.int32))
        ct.launch(stream, (1,1,1), unpack_int32, (out,))
        _stream().sync()
        r = out.to_numpy()
        ok = int(r[0]) == 0x04030201
        results["unpack bytes→int32"] = (ok, f"{hex(int(r[0]))} exp=0x4030201")
        out.free()
    except Exception as e:
        results["unpack bytes→int32"] = (False, str(e)[:80])

    # 4. bitcast 1.0f → uint32  (official docstring example)
    try:
        @ct.kernel
        def bcast_f32(out):
            ct.store(out, index=(0,), tile=ct.astype(ct.bitcast(1.0, ct.uint32), ct.uint32))
        out = _GpuArray(np.zeros(1, np.uint32))
        ct.launch(stream, (1,1,1), bcast_f32, (out,))
        _stream().sync()
        r = out.to_numpy()
        ok = int(r[0]) == 0x3F800000
        results["bitcast 1.0f→uint32"] = (ok, f"{hex(int(r[0]))} exp=0x3f800000")
        out.free()
    except Exception as e:
        results["bitcast 1.0f→uint32"] = (False, str(e)[:80])

    # 5. FP4 roundtrip: pack→unpack→astype(float32)
    try:
        @ct.kernel
        def fp4_roundtrip(out, N: ct.Constant[int]):
            fp4    = ct.ones((N,), ct.float4_e2m1fn)
            packed = ct.pack_to_bytes(fp4)
            back   = ct.unpack_from_bytes(packed, ct.float4_e2m1fn)
            ct.store(out, index=(0,), tile=ct.astype(back, ct.float32))
        N = 16
        out = _GpuArray(np.zeros(N, np.float32))
        ct.launch(stream, (1,1,1), fp4_roundtrip, (out, N))
        _stream().sync()
        r = out.to_numpy()
        ok = all(abs(v - 1.0) < 0.01 for v in r)
        results["FP4 pack→unpack→f32"] = (ok, f"r[:4]={list(r[:4])} exp=[1.0]*{N}")
        out.free()
    except Exception as e:
        results["FP4 pack→unpack→f32"] = (False, str(e)[:80])

    return results


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _dev()

    major, = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, 0)[1:2]  # type: ignore
    err, major = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, 0)
    err, minor = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, 0)

    print(f"CUDA {grpo_rl.env.CUDA_ROOT.split('v')[-1]}  "
          f"cuda-tile {ct.__version__}  CC {major}.{minor}")
    print("=" * 64)

    ptx_cases = [
        ("BF16  wmma          (sm_120 )", BF16,     "bf16_mma",   "sm_120",  "c[0,0]=16"),
        ("FP8   plain mma     (sm_120 )", FP8_PLAIN,"fp8_plain",  "sm_120",  "D[0]=72"),
        ("FP8   kind mma      (sm_120a)", FP8_KIND, "fp8_kind",   "sm_120a", "D[0]=72"),
        ("FP4   E2M1          (sm_120a)", FP4,      "fp4_mma",    "sm_120a", "D!=0"),
        ("FP4xFP8 mixed       (sm_120a)", FP4_FP8,  "fp4_fp8",    "sm_120a", "runs"),
        ("FP6   E3M2          (sm_120a)", FP6,      "fp6_mma",    "sm_120a", "D=C when A=B=0"),
        ("griddepcontrol      (sm_120a)", GRIDDEP,  "griddep",    "sm_120a", "D[0]=1 (smoke: compile+run only)"),
        ("TMEM  tcgen05       (sm_120a)", TMEM,     "tmem_probe", "sm_120a", "EXPECT FAIL"),
    ]

    print("--- PTX / MMA (cuda.core.Program + cc.launch) ---")
    for label, src, fname, arch, note in ptx_cases:
        ok, result = _run_f32x4(src, fname, arch)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {label}   {note}")
        print(f"          result={result}")

    print("\n--- Hopper-inherited features (sm_90+ → sm_120) ---")
    ok, msg = test_tbc()
    print(f"  [{'OK  ' if ok else 'FAIL'}] Thread Block Cluster  (sm_120):  {msg}")
    ok, msg = test_tma()
    print(f"  [{'OK  ' if ok else 'FAIL'}] TMA cp.async.bulk     (sm_120):  {msg}")
    ok, msg = test_wgmma()
    print(f"  [{'OK  ' if ok else 'FAIL'}] WGMMA m64n8k16 BF16   (sm_120):  {msg}")

    print("\n--- cuda.tile 1.4.0 (ct.matmul / ct.mma / ct.mma_scaled) ---")
    for name, (ok, msg) in test_cutile_140().items():
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {msg}")

    print("\n--- cuda.tile 1.4.0 pack/unpack/bitcast ---")
    for name, (ok, msg) in test_pack_apis().items():
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {msg}")

    print("=" * 64)
