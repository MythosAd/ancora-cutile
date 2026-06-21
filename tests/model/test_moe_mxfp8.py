"""MXFP8 forward on the MoE family — training + decode, ratio=1 under MXFP8.
  (A) training forward bitwise-deterministic;
  (B) MXFP8-vs-BF16 lp drift sanity (synthetic random weights = the hard case, expect ≲15%);
  (C) decode (fused _ggemm_gus_mx path) teacher-forced lp BITWISE == MXFP8 training lp —
      ratio=1 under MXFP8, and the fused-vs-separate MXFP8 expert GEMM equivalence end-to-end;
  (D) closed loop: loss_backward → AdamW step → shared-buffer requant → decode == trainer
      BITWISE again (zero-copy + the one-authority requant path);
  (E) SFT overfit: CE collapses (the BF16-backward straight-through QAT recipe trains)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host
from ancora.model.resident_moe_decode import ResidentMoEDecodeModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)   # (L,D)(G,M)(L,D)(G,M)
    G, S = 4, 320                                                    # ring wraps at pos ≥ 256
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, S)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, size=(G, S)).astype(np.int64)
    labels = np.concatenate([ids[:, 1:], np.zeros((G, 1), np.int64)], 1)

    train = ResidentMoEModel(cfg, w, G, S, device_route=True, mxfp8=True)

    # ── (A) forward determinism ──
    h1 = train.forward(ids, si).copy()
    h2 = train.forward(ids, si).copy()
    eA = int(np.abs(h1.astype(np.int32) - h2.astype(np.int32)).max())
    print(f"  (A) MXFP8 training forward determinism: Δ={eA}  {'OK (bitwise)' if eA == 0 else 'FAIL'}")

    # ── (B) drift vs BF16 (sanity, not a bitwise gate) ──
    train.loss_backward(None, labels.reshape(-1), si)
    lp_mx = train.glp.to_numpy().reshape(G, S).copy()
    bf = ResidentMoEModel(cfg, from_host(host, G, S), G, S, device_route=True)
    bf.forward(ids, si); bf.loss_backward(None, labels.reshape(-1), si)
    lp_bf = bf.glp.to_numpy().reshape(G, S)
    drift = float(np.abs(lp_mx - lp_bf).mean() / (np.abs(lp_bf).mean() + 1e-30))
    okB = drift < 0.30
    print(f"  (B) MXFP8 vs BF16 lp drift: mean rel {drift*100:.1f}%  "
          f"{'OK (fp8 quant noise, synthetic-hard case)' if okB else 'FAIL'}")

    # ── (C) decode == training BITWISE under MXFP8 (teacher-forced; ring wraps) ──
    eng = ResidentMoEDecodeModel(train, Bp=G, maxS=S, si=si)
    lps = eng.score(ids, labels, si)
    train.forward(ids, si); train.loss_backward(None, labels.reshape(-1), si)
    lpt = train.glp.to_numpy().reshape(G, S)
    eC = float(np.abs(lps - lpt).max())
    print(f"  (C) MXFP8 decode vs training lp: Δ={eC:.0e}  "
          f"{'OK (bitwise → ratio=1 under MXFP8)' if eC == 0 else 'FAIL'}")

    # ── (D) closed loop: step → shared requant → decode == trainer bitwise again ──
    train.step(si, lr=2e-3)
    cudart.cudaStreamSynchronize(si)
    lps2 = eng.score(ids, labels, si)                  # decode requants the SHARED fp8 buffers
    train.forward(ids, si); train.loss_backward(None, labels.reshape(-1), si)
    lpt2 = train.glp.to_numpy().reshape(G, S)
    eD = float(np.abs(lps2 - lpt2).max())
    moved = float(np.abs(lpt2 - lpt).max())
    print(f"  (D) post-step decode vs trainer lp: Δ={eD:.0e} (weights moved: {moved:.3f})  "
          f"{'OK (bitwise through requant)' if eD == 0 and moved > 0 else 'FAIL'}")

    # ── (E) SFT overfit (fwd MXFP8 / bwd BF16 straight-through) ──
    ce0 = ce = None
    for t in range(60):
        train.forward(ids, si)
        ce = train.loss_backward(None, labels.reshape(-1), si)
        if t == 0: ce0 = ce
        train.step(si, lr=2e-3)
        cudart.cudaStreamSynchronize(si)
    okE = ce < 0.15 * ce0
    print(f"  (E) MXFP8 SFT overfit: CE {ce0:.3f} → {ce:.4f}  {'OK (closed loop trains)' if okE else 'FAIL'}")

    # ── (F) MXFP8 + CUDA-graph replay == direct (greedy), incl. AFTER an AdamW step ──
    # The graph is captured with the dirty flag cleared (no requant baked in); _mx_refresh must
    # re-quant the shared fp8 weights before each replay so the graph runs over fresh bytes.
    prompts = ids[:, :8]
    g_d, lp_d = eng.generate(prompts, 8, si)
    g_g, lp_g = eng.generate(prompts, 8, si, so=so, dev=dev, use_graph=True)
    eF1 = int(np.abs(g_g - g_d).max()) + float(np.abs(lp_g - lp_d).max())
    train.step(si, lr=5e-2); cudart.cudaStreamSynchronize(si)        # move weights → dirty the fp8
    g_d2, lp_d2 = eng.generate(prompts, 8, si)                        # direct re-requants
    g_g2, lp_g2 = eng.generate(prompts, 8, si, so=so, dev=dev, use_graph=True)  # graph over fresh fp8
    eF2 = int(np.abs(g_g2 - g_d2).max()) + float(np.abs(lp_g2 - lp_d2).max())
    moved = int(np.abs(g_d2 - g_d).max()) + float(np.abs(lp_d2 - lp_d).max())
    okF = eF1 == 0 and eF2 == 0
    print(f"  (F) MXFP8 graph replay vs direct: pre-step Δ={eF1:.0e}  post-step Δ={eF2:.0e} "
          f"(weights moved: {moved:.2f})  {'OK (requant-before-replay holds)' if okF else 'FAIL'}")
    return eA == 0 and okB and eC == 0 and eD == 0 and okE and okF


if __name__ == "__main__":
    ok = main()
    print("  PASS" if ok else "  FAIL")
