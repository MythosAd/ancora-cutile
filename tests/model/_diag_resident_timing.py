"""Where does the ResidentModel per-step time go? (2026-06-04: boundary is now DEVICE-resident +
TIED embed/LM-head with device AdamW, so the old host-AdamW/linear_ce drill-downs are gone — this
just times forward / loss_backward / step at method level. Was ~4.5s/step host-boundary → ~0.11s.)"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.resident_model import ResidentModel
from ancora.model.load_qwen3 import load_qwen3

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

NL = int(sys.argv[1]) if len(sys.argv) > 1 else 28
cfg = Qwen3Config(); V = 151936; B, S = 1, 128
W = load_qwen3(n_layers=NL)
rng = np.random.default_rng(1)
ids = rng.integers(0, V, (B, S)).astype(np.int64)
labels = rng.integers(0, V, B * S).astype(np.int64)
rm = ResidentModel(cfg, W, B, S, V)


def sync(): cudart.cudaStreamSynchronize(si)


# warm up (JIT)
h = rm.forward(ids, si); rm.loss_backward(h, labels, si); rm.step(si, lr=1e-9)

# method-level timing (avg of 3)
tf = tl = ts = 0.0
for _ in range(3):
    t = time.time(); h = rm.forward(ids, si); tf += time.time() - t
    t = time.time(); rm.loss_backward(h, labels, si); tl += time.time() - t
    t = time.time(); rm.step(si, lr=1e-9); ts += time.time() - t
tf, tl, ts = tf/3, tl/3, ts/3
print(f"NL={NL}  method-level (avg 3):")
print(f"  forward       {tf*1000:7.0f} ms   (device gather onehot@embed + {NL} layers + host final-norm)")
print(f"  loss_backward {tl*1000:7.0f} ms   (device tied-head logits/CE/grads + final-norm-bwd + {NL} layers-bwd)")
print(f"  step          {ts*1000:7.0f} ms   ({NL}× layer device-AdamW + embed device-AdamW + final-norm host)")
print(f"  TOTAL/step    {(tf+tl+ts)*1000:7.0f} ms   (was ~4500 ms with the host boundary)")

# drill into step(): layer AdamW vs the tied embed device-AdamW
t = time.time()
for l in rm.layers: l.step(si, lr=1e-9)
sync(); t_layers = time.time() - t
t = time.time(); rm.step(si, lr=1e-9); sync(); t_full = time.time() - t
print(f"  └ step split: {NL}× layer-AdamW {t_layers*1000:.0f} ms   full step (incl. embed device-AdamW) {t_full*1000:.0f} ms")
