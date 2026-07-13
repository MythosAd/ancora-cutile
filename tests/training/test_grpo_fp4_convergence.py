"""THE RL-correct FP4 test: does an FP4-forward policy CONVERGE under GRPO? (Not "does FP4 match
BF16" — in on-policy RL the FP4 model IS the policy; the question is whether it learns.)

QAT-style: BF16 master weights, forward quantizes to NVFP4 (the same forward used for sampling
AND the policy gradient → batch-invariant by construction), straight-through backward updates the
master. Task = the framework's "emit target token": N prompts each map to one target token; reward
1 if a sampled token hits its target. Advantage = rl/grpo.py default (ML form (r-mean)/(|mean|+eps)).
Run BF16 vs NVFP4 and compare the reward curves. NVFP4 GEMM ≡ dequant product (proven 0.166%),
so the numpy quant→dequant matmul is a faithful proxy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from ancora.rl.grpo import grpo_advantage
from ancora.kernels.quant_nvfp4 import quantize_nvfp4_rowblock, dequantize_nvfp4

bfv = lambda x: ((x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint32) << 16).view(np.float32)
relu = lambda z: np.maximum(z, 0.0)
def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

# precision-swappable operand prep (quant along the contraction K, 16-blocks)
def qa(x, fp4): return dequantize_nvfp4(*quantize_nvfp4_rowblock(x.astype(np.float32))) if fp4 else bfv(x)
def qw(W, fp4): return dequantize_nvfp4(*quantize_nvfp4_rowblock(np.ascontiguousarray(W.T).astype(np.float32))).T if fp4 else bfv(W)
def mm(x, W, fp4): return qa(x, fp4).astype(np.float64) @ qw(W, fp4).astype(np.float64)


def fwd(X, Ws, fp4):
    z1 = mm(X, Ws[0], fp4); h1 = relu(z1)
    z2 = mm(h1, Ws[1], fp4); h2 = relu(z2)
    logits = mm(h2, Ws[2], fp4)
    return logits, (X, z1, h1, z2, h2)

def bwd(dlogits, cache, Ws, fp4):   # straight-through: backward uses the dequantized values
    X, z1, h1, z2, h2 = cache
    dW3 = qa(h2, fp4).T.astype(np.float64) @ dlogits;  dh2 = dlogits @ qw(Ws[2], fp4).T.astype(np.float64)
    dz2 = dh2 * (z2 > 0)
    dW2 = qa(h1, fp4).T.astype(np.float64) @ dz2;      dh1 = dz2 @ qw(Ws[1], fp4).T.astype(np.float64)
    dz1 = dh1 * (z1 > 0)
    dW1 = qa(X, fp4).T.astype(np.float64) @ dz1
    return [dW1.astype(np.float32), dW2.astype(np.float32), dW3.astype(np.float32)]


def grpo_train(fp4, steps=400, N=32, G=16, in_dim=128, H=256, vocab=64, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((N, in_dim)) * 0.5).astype(np.float32)         # fixed prompts
    target = rng.integers(0, vocab, N)                                     # one target token / prompt
    Ws = [(rng.standard_normal((in_dim, H)) / np.sqrt(in_dim)).astype(np.float32),
          (rng.standard_normal((H, H)) / np.sqrt(H)).astype(np.float32),
          (rng.standard_normal((H, vocab)) / np.sqrt(H)).astype(np.float32)]
    m = [np.zeros_like(w) for w in Ws]; v = [np.zeros_like(w) for w in Ws]
    b1, b2, eps = 0.9, 0.999, 1e-8
    curve = []
    for t in range(1, steps + 1):
        logits, cache = fwd(X, Ws, fp4)                                    # (N, vocab)
        probs = softmax(logits)
        tok = np.array([rng.choice(vocab, size=G, p=probs[n]) for n in range(N)])   # (N,G) sample
        rew = (tok == target[:, None]).astype(np.float32)                  # (N,G) reward
        adv = grpo_advantage(rew.reshape(-1), G).reshape(N, G)             # group-normalised
        # policy gradient w.r.t. logits: Σ_g adv[n,g]·(softmax - onehot(tok))
        dlogits = np.zeros((N, vocab), np.float64)
        for g in range(G):
            oneh = np.zeros((N, vocab)); oneh[np.arange(N), tok[:, g]] = 1.0
            dlogits += adv[:, g:g + 1] * (probs - oneh)
        dlogits /= (N * G)
        grads = bwd(dlogits, cache, Ws, fp4)
        for i in range(3):                                                 # AdamW
            m[i] = b1 * m[i] + (1 - b1) * grads[i]; v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
            mh = m[i] / (1 - b1 ** t); vh = v[i] / (1 - b2 ** t)
            Ws[i] -= lr * mh / (np.sqrt(vh) + eps)
        curve.append(rew.mean())
    return curve


if __name__ == "__main__":
    print("GRPO end-to-end convergence: BF16 vs NVFP4 forward (emit-target-token)"); print("=" * 68)
    HP = dict(steps=1000, N=16, G=32, vocab=64, lr=0.004)
    cb = grpo_train(fp4=False, **HP); cn = grpo_train(fp4=True, **HP)
    print(f"  random reward ≈ {100/HP['vocab']:.1f}%   N={HP['N']} prompts  G={HP['G']} samples  lr={HP['lr']}")
    print(f"  {'step':>5s} {'BF16 reward':>12s} {'NVFP4 reward':>13s}")
    for s in [0, 99, 249, 499, 749, 999]:
        print(f"  {s+1:5d} {cb[s]*100:11.1f}% {cn[s]*100:12.1f}%")
    okb = cb[-1] > 0.7; okn = cn[-1] > 0.7
    print("=" * 68)
    print(f"  BF16 converged (>70%): {'YES' if okb else 'NO'} ({cb[-1]*100:.0f}%)  |  "
          f"NVFP4 converged: {'YES' if okn else 'NO'} ({cn[-1]*100:.0f}%)")
    if not okb:
        print("  ⚠️ BF16 baseline didn't converge → harness/hparam issue, FP4 verdict not yet valid.")
    else:
        print(f"  → FP4 policy {'LEARNS the task (RL-viable)' if okn else 'FAILS to learn (not RL-viable as-is)'}")
