"""Grouped MoE FFN (ancora/kernels/moe.py) — drop-in for moe_layer.MoEFFN — vs fp64 reference
and the host-loop MoEFFN, for BOTH forward and backward, plus the two properties it exists to
deliver: determinism (no alloc-churn flakiness) and batch invariance."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.moe import GroupedMoEFFN
from ancora.model.moe_layer import MoEConfig, MoEFFN
from ancora.model.qwen3_layer import _bf
from ancora.kernels.norm import f32_to_bf16_bits as f32bf
import tests.model.test_moe_layer as TM        # reuse moe_ref / moe_bwd_f64 (fp64)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

f64 = lambda a: a.astype(np.float64)
def rel(a, b): return np.abs(f64(a) - f64(b)).max() / (np.abs(f64(b)).max() + 1e-9)


def test_forward():
    print("--- grouped forward vs fp64 + host-loop ---")
    cfg = MoEConfig(); ffn = MoEFFN(cfg, np.random.default_rng(0))
    gm = GroupedMoEFFN(ffn.w, cfg.top_k, si); ok = True
    for M in (128, 256, 384):
        h = _bf((np.random.default_rng(M).standard_normal((M, cfg.hidden)) * 0.5).astype(np.float32))
        _, _, topi, topw = ffn._route(h, si)
        out_g, _ = gm.forward(h)
        out_ref, _, _ = TM.moe_ref(h, ffn.w, cfg, topi, topw)
        out_loop, _ = ffn.forward(h, si)
        e_ref, e_loop = rel(out_g, out_ref), rel(out_g, out_loop)
        o = e_ref < 0.02 and e_loop < 0.02; ok &= o
        print(f"  M={M}: vs fp64={e_ref*100:.2f}%  vs host-loop={e_loop*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_backward():
    print("--- grouped backward vs fp64 ---")
    cfg = MoEConfig(); ffn = MoEFFN(cfg, np.random.default_rng(1))
    gm = GroupedMoEFFN(ffn.w, cfg.top_k, si); ok = True
    for M in (128, 256):
        h = _bf((np.random.default_rng(7 + M).standard_normal((M, cfg.hidden)) * 0.5).astype(np.float32))
        d_out = np.random.default_rng(100 + M).standard_normal((M, cfg.hidden)).astype(np.float32)
        gm.forward(h)
        d_h, g = gm.backward(d_out)
        _, probs, topi, topw = ffn._route(h, si)
        d_h_r, g_r = TM.moe_bwd_f64(h, ffn.w, cfg, d_out, probs, topi, topw)
        e_act = int(np.bincount(topi.reshape(-1), minlength=cfg.n_experts).argmax())
        e_dh = rel(d_h, d_h_r); e_rt = rel(g["router"], g_r["router"])
        e_dn = rel(g["down_proj"][e_act], g_r["down_proj"][e_act])
        e_gt = rel(g["gate_proj"][e_act], g_r["gate_proj"][e_act])
        e_up = rel(g["up_proj"][e_act], g_r["up_proj"][e_act])
        o = max(e_dh, e_rt, e_dn, e_gt, e_up) < 0.03; ok &= o
        print(f"  M={M}: d_h={e_dh*100:.2f}% d_router={e_rt*100:.2f}% d_down={e_dn*100:.2f}% "
              f"d_gate={e_gt*100:.2f}% d_up={e_up*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_determinism():
    """The reason this kernel exists: the host-loop backward flaked ~1-in-4 from alloc churn.
    The grouped path preallocates → fwd AND bwd identical across repeated runs."""
    print("--- determinism: fwd+bwd bitwise identical across runs (churn gone) ---")
    cfg = MoEConfig(); ffn = MoEFFN(cfg, np.random.default_rng(2)); gm = GroupedMoEFFN(ffn.w, cfg.top_k, si)
    h = _bf((np.random.default_rng(9).standard_normal((256, cfg.hidden)) * 0.5).astype(np.float32))
    d_out = np.random.default_rng(5).standard_normal((256, cfg.hidden)).astype(np.float32)
    fs, bs = [], []
    for _ in range(6):
        fs.append(gm.forward(h)[0].copy()); bs.append(gm.backward(d_out)[0].copy())
    of = all(np.array_equal(fs[0], x) for x in fs[1:])
    ob = all(np.array_equal(bs[0], x) for x in bs[1:])
    print(f"  6× forward identical={of}  6× backward d_h identical={ob}  {'OK' if of and ob else 'FAIL'}")
    return of and ob


def test_batch_invariance():
    print("--- batch invariance (token 0..S forward bitwise, M=S vs 2S) ---")
    cfg = MoEConfig(); ffn = MoEFFN(cfg, np.random.default_rng(3)); gm = GroupedMoEFFN(ffn.w, cfg.top_k, si)
    S = 128; r = np.random.default_rng(3)
    h0 = _bf((r.standard_normal((S, cfg.hidden)) * 0.5).astype(np.float32))
    h1 = _bf((r.standard_normal((S, cfg.hidden)) * 0.5).astype(np.float32))
    y1, _ = gm.forward(h0)
    y2, _ = gm.forward(np.concatenate([h0, h1], 0))
    same = np.array_equal(f32bf(y1), f32bf(y2[:S]))
    print(f"  tokens 0..{S} bitwise identical: {same}  {'OK' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    cfg = MoEConfig()
    print(f"grouped MoE — E={cfg.n_experts} top-k={cfg.top_k} H={cfg.hidden} Ie={cfg.expert_inter}")
    print("=" * 70)
    r = [test_forward(), test_backward(), test_determinism(), test_batch_invariance()]
    print("=" * 70)
    print("  ALL PASS" if all(r) else "  SOME FAILED: " + str(r))
