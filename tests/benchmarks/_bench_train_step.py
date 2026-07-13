"""Real-size MoE TRAINING step bench: ResidentMoEModel SFT step (fwd → loss_backward → AdamW)
at NL=12 / V=151936 / window=512, swept over M = B·S to find the best batch and the step MFU.
Useful FLOPs = 6·P per token (fwd 2P + dgrad 2P + wgrad 2P) for projections/FFN/boundary-head
+ flash-attention fwd/bwd; the onehot input-gather GEMM and input-embed dW (2·M·V·H each) are
counted as OVERHEAD (a gather implemented as GEMM), reported separately.
Fresh process per M (WDDM rule).  Usage: _bench_train_step.py [M] [accN|acc] [mx] [fp8bwd] [muon]
(no arg → sweep incl. the accumulation ladder acc2/4/8 at the best batch M=2048 — accumulation
amortizes the fixed AdamW BW-sweep: MFU(N) = MFU_fb · t_fb·N/(t_fb·N + t_adamw), asymptote = the
fwd+bwd MFU itself)."""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 2:
    for m in (512, 1024, 2048, 4096):
        subprocess.run([sys.executable, __file__, str(m)])
    for acc in ("acc2", "acc4", "acc8"):    # accumulation MFU ladder @ best batch (4096-16384 tok/step)
        subprocess.run([sys.executable, __file__, "2048", acc])
    sys.exit(0)

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

M = int(sys.argv[1]); B, S = 1, M
_acc = next((a for a in sys.argv[2:] if a.startswith("acc")), None)
NACC = (int(_acc[3:]) if len(_acc) > 3 else 2) if _acc else 1   # accumulated micro-batches per step
MX = "mx" in sys.argv[2:]                            # MXFP8 forward GEMMs
FP8B = "fp8bwd" in sys.argv[2:]                      # FP8 E4M3 data-gradient (dgrad)
MUON = "muon" in sys.argv[2:]                        # Muon/AdamW hybrid (proj+experts → Muon NS)
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
H, V, W = cfg.hidden, cfg.vocab, cfg.window
Hq, Hkv, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
qd, kd, I, Ie, k = Hq * Dh, Hkv * Dh, cfg.dense_inter, cfg.expert_inter, cfg.top_k

try:
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, B, S)
    train = ResidentMoEModel(cfg, w, B, S, device_route=True, mxfp8=MX, fp8_bwd=FP8B,
                             optimizer=("muon" if MUON else "adamw"))
except Exception as e:
    print(f"  M={M:5d}: BUILD FAIL ({type(e).__name__}: {e})"); sys.exit(0)

rng = np.random.default_rng(0)
ids = rng.integers(0, V, size=(B, S)).astype(np.int64)
labels = rng.integers(0, V, size=(M,)).astype(np.int64)

def one_step():
    fs = bs = 0.0
    t0 = time.perf_counter()
    for i in range(NACC):                            # micro-batch i>0: grads ADD in place
        train.forward(ids, si)
        t1 = time.perf_counter(); fs += t1 - t0
        train.loss_backward(None, labels, si,
                            norm=(NACC * M if NACC > 1 else None), accumulate=(i > 0))
        t0 = time.perf_counter(); bs += t0 - t1
    train.step(si, lr=1e-4); sync()
    return fs, bs, time.perf_counter() - t0

try:
    one_step(); one_step()                       # warm (JIT + lazy buffers)
    REP = 3
    fs = bs = ss = 0.0
    for _ in range(REP):
        f, b, s = one_step(); fs += f; bs += b; ss += s
    fs, bs, ss = fs / REP, bs / REP, ss / REP
except Exception as e:
    print(f"  M={M:5d}: STEP FAIL ({type(e).__name__}: {e})"); sys.exit(0)

tot = fs + bs + ss
# ── useful FLOPs ──
nl_d = sum(1 for is_g, fd in host.sched if fd); nl_m = cfg.n_layers - nl_d
n_loc = sum(1 for is_g, fd in host.sched if not is_g); n_glob = cfg.n_layers - n_loc
P_attn = H * qd + 2 * H * kd + qd * H
P_layer = cfg.n_layers * P_attn + nl_d * 3 * H * I + nl_m * k * 3 * H * Ie
proj = 6 * M * P_layer                                   # fwd 2P + dgrad 2P + wgrad 2P
ctx_g, ctx_l = (S + 1) / 2, min((S + 1) / 2, W)
attn = 3.5 * 4 * M * Hq * Dh * (n_glob * ctx_g + n_loc * ctx_l)   # fwd + 2.5x bwd
head = 6 * M * V * H                                     # logits fwd + dhidden + head dW
useful = proj + attn + head
overhead = 0                                             # onehot gather GEMM → _embed_gather (2026-06-12)
                                                         # + input-embed dW → _embed_dw_scatter (2026-07-13):
                                                         # both former 2·M·V·H GEMMs are gone
useful, overhead = NACC * useful, NACC * overhead        # N micro-batches per optimizer step
free, total = cudart.cudaMemGetInfo()[1:]
toks = NACC * M
tag = (f"M={M}x{NACC}acc" if NACC > 1 else f"M={M:5d}") + (" MX" if MX else "") + (" FP8bwd" if FP8B else "") + (" muon" if MUON else "")
print(f"  {tag}: step {tot*1e3:7.1f} ms (fwd {fs*1e3:6.1f} | bwd {bs*1e3:6.1f} | adamw {ss*1e3:5.1f})"
      f"  useful {useful/tot/1e12:5.1f} TF = MFU {useful/tot/80e12*100:4.1f}%"
      f"  (+gather-GEMM ovh {overhead/tot/1e12:4.1f} TF)"
      f"  {toks/tot:5.0f} tok/s  VRAM {(total-free)/2**30:4.1f} GB")
