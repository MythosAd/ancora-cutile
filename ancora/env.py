"""
Environment bootstrap for cuda-python / cuda-tile on Windows.
Must be imported BEFORE any cuda.* module — sets CUDA_PATH so
cuda.pathfinder resolves DLLs and headers from the correct toolkit.
"""
import os, sys, ctypes

CUDA_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
CUDA_INC  = os.path.join(CUDA_ROOT, "include")

# Force-override any stale CUDA_PATH left by older toolkit installer
os.environ["CUDA_PATH"] = CUDA_ROOT
os.environ["CUDA_HOME"] = CUDA_ROOT

if sys.platform == "win32":
    for sub in ("bin\\x64", "bin"):
        d = os.path.join(CUDA_ROOT, sub)
        if os.path.isdir(d):
            os.add_dll_directory(d)
            ctypes.windll.kernel32.AddDllDirectory(d)

# Default compile targets
ARCH_120  = ("--gpu-architecture=sm_120",)
ARCH_120A = ("--gpu-architecture=sm_120a",)

# NVRTC include flags (used by raw kernel helpers)
NVRTC_INCLUDES = [f"--include-path={CUDA_INC}".encode()]
