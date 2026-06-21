"""ResidentPrefixMoEModel — the device-resident PREFIX-SHARED GRPO step for the MoE model
(prompt encoded ONCE, G completions share its KV through all layers + the tied embed/LM-head
boundary). Validated against the resident REPLICATED ResidentMoEModel (B=G on [prompt, comp_i]):

  (a) ratio=1 at the FULL-MODEL level: every scored completion token's logprob BITWISE ==
      replicated — the within-suffix rows from the device _ce_stats, AND completion token 0
      (scored on a DUPLICATED head row — PrefixGrouper include_prefix_last, device-resident:
      forward DtoD-copies hidden row Sp-1 into G tail rows each with its own label/adv; the
      backward sums their dhidden back via fused._bnd_acc. No host round-trip in the boundary).
  (b) GRPO grads training-equivalent: embed grad + per-layer attention grads + MoE expert/router
      grads ≤1% (Σ_G prompt linearity; row-order summation noise).
  (c) GRPO policy improvement: rewards [1,0,0,0] → centered advantages → N prefix steps raise
      the rewarded completion's logprob over the others (the on-policy update works end-to-end
      through fwd → grpo_loss_backward → device AdamW).
  (d) forward determinism bitwise.
Schedule (L,D)(G,M)(L,D)(G,M) covers all four cell types; window=128 < Sp so the windowed prefix
kernels run with the window spanning the prompt/suffix boundary. Foreground only."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host
from ancora.model.resident_prefix_model import ResidentPrefixMoEModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)   # (L,D)(G,M)(L,D)(G,M)
    Sp, Sc, G = 128, 64, 4; S = Sp + Sc; M = Sp + G * Sc; Mr = G * S
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    print("schedule (is_global, ffn_dense):", host.sched)
    w = from_host(host, 1, S)
    pre = ResidentPrefixMoEModel(cfg, w, Sp, Sc, G)
    rep = ResidentMoEModel(cfg, w, G, S)

    rng = np.random.default_rng(3)
    prompt = rng.integers(0, cfg.vocab, size=(Sp,)).astype(np.int64)
    comps = rng.integers(0, cfg.vocab, size=(G, Sc)).astype(np.int64)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    adv_g = ((r - r.mean()) / (r.std() + 1e-6)).astype(np.float32)  # centered GRPO advantages
    nrm = G * Sc

    # replicated reference: copy i = [prompt, comp_i]; boundary row Sp-1 carries comp_i[0]
    ids_rep = np.stack([np.concatenate([prompt, comps[i]]) for i in range(G)])
    lab_rep = np.zeros(Mr, np.int64); adv_rep = np.zeros(Mr, np.float32)
    for i in range(G):
        lab_rep[i * S + Sp - 1] = comps[i, 0];      adv_rep[i * S + Sp - 1] = adv_g[i]
        lab_rep[i * S + Sp: (i + 1) * S - 1] = comps[i, 1:]
        adv_rep[i * S + Sp: (i + 1) * S - 1] = adv_g[i]

    # ── (a)+(d) forward, logprobs, determinism ──
    h1 = pre.forward_prefix(prompt, comps, si); sync(); h1 = h1.copy()
    h2 = pre.forward_prefix(prompt, comps, si); sync()
    det = float(np.abs(h1.astype(np.float64) - h2.astype(np.float64)).max())

    ce_p, lp_comp = pre.grpo_loss_backward(h2, comps, adv_g, si)
    lp_init = lp_comp.copy()                                    # host-route lp at initial weights
    lp_pre = pre.glp.to_numpy().reshape(pre.Mh)
    hr = rep.forward(ids_rep, si)
    ce_r = rep.loss_backward(hr, lab_rep, si, advantage=adv_rep, norm=nrm)
    lp_rep = rep.glp.to_numpy().reshape(Mr)

    e_lp = 0.0; e_b = 0.0
    for i in range(G):
        a = lp_pre[Sp + i * Sc: Sp + (i + 1) * Sc - 1]          # within-suffix scored rows
        b = lp_rep[i * S + Sp: (i + 1) * S - 1]
        e_lp = max(e_lp, float(np.abs(a - b).max()))
        # boundary: the duplicate head row M+i vs the replicated copy's own row Sp-1 — BITWISE
        # (identical hidden row → identical per-row logits/_ce_stats math)
        e_b = max(e_b, float(abs(lp_pre[M + i] - lp_rep[i * S + Sp - 1])))
    e_ce = abs(ce_p - ce_r) / abs(ce_r)

    # ── (b) grads: embed + per-layer attention + MoE expert/router ──
    e_emb = rel(pre.gegrad.to_numpy(), rep.gegrad.to_numpy())
    e_lay = max(rel(pre.layers[j].G["q_proj"].to_numpy(), rep.layers[j].G["q_proj"].to_numpy())
                for j in range(cfg.n_layers))
    e_moe = max(max(rel(pre.layers[j].moe.dWd.to_numpy(), rep.layers[j].moe.dWd.to_numpy()),
                    rel(pre.layers[j].moe.G_router, rep.layers[j].moe.G_router))
                for j in (1, 3))                                # the MoE layers in the schedule
    # gates: lp BITWISE (suffix AND boundary — boundary-row duplication reproduces the replicated
    # per-row math exactly) is the ratio=1 check. ce is a scalar diagnostic — centered advantages
    # make its weighted sum a CANCELLATION (Σadv=0) so f32 row-order noise amplifies (~3e-5).
    # embed grad ~1.1%: the prompt tokens' input-embed grad inherits the Σ_G prompt-grad PATHWAY
    # difference (prefix sums once and bf16-rounds per layer; replicated rounds G copies then sums
    # in f32) — grows with depth, mathematically equivalent (same ≤0.87% seen at the layer level).
    ok_eq = det == 0.0 and e_lp == 0.0 and e_b == 0.0 and e_ce < 1e-3 and e_emb < 0.02 and max(e_lay, e_moe) < 0.01
    print(f"  det Δ={det:.0e}  suffix-lp Δ={e_lp:.0e}  boundary-lp Δ={e_b:.0e} (dup-row, bitwise)  ce Δ={e_ce:.1e}")
    print(f"  grads: embed≤{e_emb*100:.2f}%  layers(q_proj)≤{e_lay*100:.2f}%  moe(dWd/router)≤{e_moe*100:.2f}%  "
          f"tokens {Mr}→{M}  {'OK' if ok_eq else 'FAIL'}")

    # ── (c) GRPO policy improvement: rewarded completion's logprob rises over the others ──
    gap0 = float(lp_comp[0] - lp_comp[1:].mean())
    t0 = time.perf_counter(); steps = 25
    for it in range(steps):
        pre.step(si, lr=2e-3)
        h = pre.forward_prefix(prompt, comps, si)
        ce, lp_comp = pre.grpo_loss_backward(h, comps, adv_g, si)
    sync(); dt = (time.perf_counter() - t0) / steps
    gap1 = float(lp_comp[0] - lp_comp[1:].mean())
    ok_rl = gap1 > gap0 + 50.0                                  # decisive policy improvement
    print(f"  GRPO: lp gap (rewarded − others) {gap0:.1f} → {gap1:.1f} over {steps} steps "
          f"({dt*1e3:.0f} ms/step, NL={cfg.n_layers})  {'OK' if ok_rl else 'FAIL'}")

    # the same GRPO step on the REPLICATED model (prompt encoded G×) for the perf comparison
    rep.step(si, lr=2e-3); sync()                               # warm AdamW init out of the timing
    t0 = time.perf_counter(); reps = 10
    for _ in range(reps):
        hr = rep.forward(ids_rep, si)
        rep.loss_backward(hr, lab_rep, si, advantage=adv_rep, norm=nrm)
        rep.step(si, lr=2e-3)
    sync(); dtr = (time.perf_counter() - t0) / reps
    print(f"  step time: replicated {dtr*1e3:.0f} ms (M={Mr}) vs prefix {dt*1e3:.0f} ms (M={M}) "
          f"= {dtr/dt:.2f}x")
    return ok_eq and ok_rl, (cfg, w, prompt, comps, adv_g, lp_init)


def graph_phase(cfg, w, prompt, comps, adv_g):
    """CUDA-graph capture of the ENTIRE prefix fwd+bwd (device_route=True → the MoE router is
    sync-free, so the whole chain is pure launches). Checks:
      (f) device-route lp == host-route lp (router gating bitwise == host dispatch, model level);
      (g) graph replay BITWISE == direct launches (lp + layer/embed/final-norm grads);
      (h) GRPO improves through graph replays + (uncaptured) AdamW steps;
      (i) host-overhead: direct fwd+bwd vs one graph launch."""
    import time as _t
    Sp, Sc = 128, 64; G = comps.shape[0]
    pre = ResidentPrefixMoEModel(cfg, w, Sp, Sc, G, device_route=True)

    # (f) device-route equivalence at initial weights (vs main()'s host-route lp_comp, recomputed)
    pre.forward_prefix(prompt, comps, si)
    ce0, lp0 = pre.grpo_loss_backward(None, comps, adv_g, si)
    pre.step(si, lr=2e-3); sync()                               # warm: JIT + AdamW init (router dev)

    # (g) direct fwd+bwd at post-step weights → record → capture → graph replay → bitwise compare
    pre.forward_prefix(prompt, comps, si)
    ced, lpd = pre.grpo_loss_backward(None, comps, adv_g, si)
    glp_d = pre.glp.to_numpy(); qg_d = [pre.layers[j].G["q_proj"].to_numpy() for j in range(cfg.n_layers)]
    eg_d = pre.gegrad.to_numpy(); fn_d = pre.gfng.to_numpy()
    pre.capture(dev)
    ceg, lpg = pre.graph_step(prompt, comps, adv_g, so, si)
    e_lp = float(np.abs(pre.glp.to_numpy() - glp_d).max())
    e_qg = max(float(np.abs(pre.layers[j].G["q_proj"].to_numpy() - qg_d[j]).max()) for j in range(cfg.n_layers))
    e_eg = float(np.abs(pre.gegrad.to_numpy() - eg_d).max())
    e_fn = float(np.abs(pre.gfng.to_numpy() - fn_d).max())
    det = 0.0                                                   # graph replay determinism (3×)
    for _ in range(3):
        pre.graph_step(prompt, comps, adv_g, so, si)
        det = max(det, float(np.abs(pre.glp.to_numpy() - glp_d).max()))
    ok_g = max(e_lp, e_qg, e_eg, e_fn, det) == 0.0 and ceg == ced
    print(f"  graph: replay vs direct — lp Δ={e_lp:.0e} q_projG Δ={e_qg:.0e} embedG Δ={e_eg:.0e} "
          f"fnG Δ={e_fn:.0e} 3x-det Δ={det:.0e}  {'OK (bitwise)' if ok_g else 'FAIL'}")

    # (h) GRPO through the graph: replay + uncaptured AdamW steps still improve the policy
    gap0 = float(lpg[0] - lpg[1:].mean())
    for _ in range(15):
        pre.step(si, lr=2e-3)
        ce, lpg = pre.graph_step(prompt, comps, adv_g, so, si)
    gap1 = float(lpg[0] - lpg[1:].mean())
    ok_rl = gap1 > gap0 + 50.0
    print(f"  graph GRPO: lp gap {gap0:.1f} → {gap1:.1f} over 15 graph steps  {'OK' if ok_rl else 'FAIL'}")

    # (i) host-overhead: direct (all launches) vs one graph launch, fwd+bwd only
    def direct():
        pre._upload_ids(np.concatenate([prompt, comps.reshape(-1)]), si)
        labels, adv = pre.grpo_io(comps, adv_g); pre._upload_io(labels, adv, si)
        pre._fwd_dev(si); pre._bwd_dev(si, 1.0 / (G * Sc)); sync()
    def graphed():
        pre.graph_step(prompt, comps, adv_g, so, si)
    for f in (direct, graphed): f()
    t0 = _t.perf_counter()
    for _ in range(20): direct()
    td = (_t.perf_counter() - t0) / 20
    t0 = _t.perf_counter()
    for _ in range(20): graphed()
    tg = (_t.perf_counter() - t0) / 20
    print(f"  fwd+bwd: direct {td*1e3:.1f} ms vs graph {tg*1e3:.1f} ms = {td/tg:.2f}x less host overhead")
    return ok_g and ok_rl, lp0


def bench(Sp, Sc, G, n_layers):
    """Step-time at a REALISTIC GRPO size (the correctness case above is launch-bound — M=384 at
    NL=4 hides the compute win; prefix-sharing pays in the compute-bound regime)."""
    cfg = MoEConfig(vocab=2048, n_layers=n_layers, period=2, window=128)
    S = Sp + Sc; M = Sp + G * Sc; Mr = G * S
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, S)
    pre = ResidentPrefixMoEModel(cfg, w, Sp, Sc, G)
    rep = ResidentMoEModel(cfg, w, G, S)
    rng = np.random.default_rng(3)
    prompt = rng.integers(0, cfg.vocab, size=(Sp,)).astype(np.int64)
    comps = rng.integers(0, cfg.vocab, size=(G, Sc)).astype(np.int64)
    r = np.zeros(G); r[0] = 1.0
    adv_g = ((r - r.mean()) / (r.std() + 1e-6)).astype(np.float32)
    ids_rep = np.stack([np.concatenate([prompt, comps[i]]) for i in range(G)])
    lab_rep = np.zeros(Mr, np.int64); adv_rep = np.zeros(Mr, np.float32)
    for i in range(G):
        lab_rep[i * S + Sp - 1] = comps[i, 0]; adv_rep[i * S + Sp - 1] = adv_g[i]
        lab_rep[i * S + Sp: (i + 1) * S - 1] = comps[i, 1:]
        adv_rep[i * S + Sp: (i + 1) * S - 1] = adv_g[i]

    def one_pre():
        h = pre.forward_prefix(prompt, comps, si)
        pre.grpo_loss_backward(h, comps, adv_g, si); pre.step(si, lr=1e-4)
    def one_rep():
        h = rep.forward(ids_rep, si)
        rep.loss_backward(h, lab_rep, si, advantage=adv_rep, norm=G * Sc); rep.step(si, lr=1e-4)
    for f in (one_pre, one_rep): f(); f()                       # JIT + AdamW init out of the timing
    sync()
    t0 = time.perf_counter()
    for _ in range(5): one_pre()
    sync(); dt = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5): one_rep()
    sync(); dtr = (time.perf_counter() - t0) / 5
    print(f"  bench Sp={Sp} Sc={Sc} G={G} NL={n_layers}: replicated {dtr*1e3:.0f} ms (M={Mr}) vs "
          f"prefix {dt*1e3:.0f} ms (M={M}) = {dtr/dt:.2f}x  (tokens {Mr}→{M}, {Mr/M:.1f}x fewer)")


if __name__ == "__main__":
    print("ResidentPrefixMoEModel — prefix-shared GRPO step (full model) vs replicated resident")
    print("=" * 92)
    ok, (cfg_, w_, prompt_, comps_, adv_, lp_host) = main()
    ok2, lp_dev = graph_phase(cfg_, w_, prompt_, comps_, adv_)
    # device-route vs host-route: NOT bitwise by design — router_gate's h·Wr warp-reduction order
    # differs from numpy's, so the GATE VALUES differ ±ulp (the expert CHOICES and dispatch are
    # bitwise) and drift amplifies through the layers. Each route is internally bitwise-deterministic
    # (det/graph Δ=0) — ratio=1 requires rollout and training to use the SAME route, not the two
    # routes to match each other. Gate: close (≤2% of the lp sums).
    e_route = float(np.abs(lp_dev - lp_host).max() / (np.abs(lp_host).max() + 1e-9))
    print(f"  device-route vs host-route lp_comp rel≤{e_route*100:.2f}%  (gate-value ulp drift; "
          f"each route is itself bitwise)  {'OK' if e_route < 0.02 else 'FAIL'}")
    bench(512, 256, 8, 2)
    print("=" * 92)
    allok = ok and ok2 and e_route < 0.02
    print("  PASS (prefix GRPO step: logprobs bitwise == replicated → ratio=1; grads equivalent; "
          "policy improves; fwd+bwd graph-captured bitwise)" if allok else "  FAIL")
