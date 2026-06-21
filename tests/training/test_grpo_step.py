"""End-to-end GRPO on-policy loop — the project's north star, validated on a toy task:
"emit the target token". Each iteration:
  rollout (sample G completions/prompt) → reward (fraction == target) → group-relative
  advantage (rl/grpo.py, DeepSeek (r-mean)/std) → policy-gradient update (model.loss_backward
  with per-token advantage = linear_ce) → Muon/AdamW.
The MEAN REWARD must climb (the GRPO analog of "loss goes down") — proof the RL loop learns.

β=0 (no KL / ref model). Fully on-policy (rollout & update use the same weights → ratio≈1,
no clipping needed; batch-invariance keeps logprobs consistent). Small model, no KV-cache."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.qwen3_model import Qwen3Model
from ancora.optim.hybrid import HybridOptimizer
from ancora.rl.rollout import generate, build_grpo_targets
from ancora.rl.grpo import grpo_advantage

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def main():
    cfg = Qwen3Config(); cfg._B, cfg._S = 1, 1   # _B,_S set per-call by the model
    V, NL = 512, 2
    P, GEN = 32, 32                  # prompt 32 + generate 32 → S=64 (tile-aligned)
    NUM_PROMPTS, G = 4, 6            # G completions/prompt → group-relative advantage
    TARGET = 7                       # reward = fraction of generated tokens == TARGET
    ITERS = 16

    model = Qwen3Model(cfg, n_layers=NL, vocab=V, seed=0)
    rng = np.random.default_rng(0)
    prompts = rng.integers(0, V, (NUM_PROMPTS, P)).astype(np.int64)
    prompts_rep = np.repeat(prompts, G, axis=0)            # (NUM_PROMPTS*G, P), groups contiguous
    opt = HybridOptimizer(model.params(),
                          adamw_kw=dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, wd=0.0))

    print(f"GRPO: {NUM_PROMPTS} prompts × G={G}, gen={GEN}, V={V}, target token={TARGET}")
    print(f"  baseline reward (random) ≈ 1/V = {1/V:.4f}")
    print("-" * 64)

    rewards_hist = []
    for it in range(ITERS):
        model.load(opt.weights())
        ids = generate(model, prompts_rep, GEN, si, temp=1.0, seed=1000 + it)   # rollout
        reward = (ids[:, P:] == TARGET).mean(1).astype(np.float32)              # (B,)
        A = grpo_advantage(reward, group_size=G, norm="std")                    # group-relative
        labels, adv_tok = build_grpo_targets(ids, A, P)

        hidden, cache = model.forward(ids, si)
        pg, grads = model.loss_backward(hidden, labels, cache, si, advantage=adv_tok)
        opt.step(grads, si)

        rewards_hist.append(float(reward.mean()))
        if it % 2 == 0 or it == ITERS - 1:
            print(f"  iter {it:2d}  mean reward = {reward.mean():.3f}  (max {reward.max():.2f})  pg_loss={pg:+.4f}")

    print("-" * 64)
    init = np.mean(rewards_hist[:3]); final = np.mean(rewards_hist[-3:])
    print(f"  mean reward: {init:.3f} (first 3) → {final:.3f} (last 3)")
    ok = final > init + 0.05 and final > 5 * (1 / V)
    opt.free()
    print("=" * 64)
    print(f"  {'PASS — GRPO loop learns (reward climbs)' if ok else 'FAIL — reward did not climb'}")
    return ok


if __name__ == "__main__":
    print("GRPO on-policy loop: rollout → reward → advantage → policy-grad → Muon/AdamW")
    print("=" * 64)
    main()
