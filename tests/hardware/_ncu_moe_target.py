"""ncu target: runs the device-resident MoE layer forward (device_route) N times so ncu can
--launch-skip past JIT/warmup and profile steady-state kernels. Usage:
  ncu --launch-skip K --launch-count C --section SpeedOfLight python tests/hardware/_ncu_moe_target.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, _bf
from ancora.model.resident_moe import ResidentMoELayer
from ancora.model.resident import _DBuf, _f32bf

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = MoEConfig(); B, S = 2, 1024; M = B * S; H = cfg.hidden
host = MoEDecoderLayer(cfg, is_global=False, ffn_dense=False, seed=7)
res = ResidentMoELayer(cfg, host.attn, host.ffn.w, B, S, is_global=False, window=cfg.window, device_route=True)
gx = _DBuf((np.random.default_rng(2).standard_normal((M, H)) * 0.5).astype(np.float32))
gd = _DBuf(_f32bf((np.random.default_rng(3).standard_normal((M, H)) * 0.3).astype(np.float32)))
mode = sys.argv[1] if len(sys.argv) > 1 else "fwd"
step = (lambda: res.forward(gx, si)) if mode == "fwd" else (lambda: (res.forward(gx, si), res.backward(gd, si)))
for _ in range(3): step()                          # warm up (JIT) — ncu --launch-skip past these
cudart.cudaStreamSynchronize(si)
for _ in range(12): step()                         # steady state
cudart.cudaStreamSynchronize(si)
