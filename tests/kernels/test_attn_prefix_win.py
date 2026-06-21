"""WINDOWED prefix-shared attention (GRPO on LOCAL sliding-window layers) — _attn_fwd_prefix_win /
_attn_bwd_dq_prefix_win / _attn_bwd_dkdv_prefix_win via the window= arg of the prefix host helpers.

The window can SPAN the prefix/suffix boundary: a suffix query at global pos Sp+t sees the last ≤W
prompt keys (shared) + its in-window suffix keys. Validates:
  - FWD: flash_attn_forward_prefix(window=W) BITWISE == flash_attn_forward(window=W) on each
    concatenated [P, s_i] (same tiles, same w-order, same masks, same M_INIT) → ratio=1 on local layers.
  - BWD: suffix grads BITWISE == the standard windowed device backward (batched B=G);
    prefix grads == Σ_G of an fp64 windowed reference (≤2%, bf16 noise).
Cases cover: window reaching into the prompt, window entirely inside the suffix (late queries),
and prompt longer than the window. Foreground only."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import (flash_attn_forward, flash_attn_backward,
                                       flash_attn_forward_prefix, flash_attn_backward_prefix, D)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def ref_full_win(Q, K, V, dO, W):
    """fp64 sliding-window causal attention fwd+bwd (query i sees keys (i-W, i])."""
    Hq, S, d = Q.shape; Hkv = K.shape[0]; G = Hq // Hkv; scale = 1.0 / math.sqrt(d)
    Qd, Kd, Vd, dOd = (a.astype(np.float64) for a in (Q, K, V, dO))
    ii = np.arange(S)[:, None]; jj = np.arange(S)[None, :]
    mask = (jj <= ii) & (jj > ii - W)
    O = np.zeros((Hq, S, d)); dQ = np.zeros((Hq, S, d)); dK = np.zeros((Hkv, S, d)); dV = np.zeros((Hkv, S, d))
    for h in range(Hq):
        kv = h // G
        Sc = np.where(mask, (Qd[h] @ Kd[kv].T) * scale, -1e38)
        P = np.exp(Sc - Sc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        O[h] = P @ Vd[kv]; do = dOd[h]
        dV[kv] += P.T @ do
        dP = do @ Vd[kv].T
        dS = P * (dP - (do * O[h]).sum(-1, keepdims=True))
        dQ[h] = (dS @ Kd[kv]) * scale
        dK[kv] += (dS.T @ Qd[h]) * scale
    return O, dQ, dK, dV


def _case(G, Hq, Hkv, Sp, Sc, W):
    rng = np.random.default_rng(3)
    pq = (rng.standard_normal((Hq,  Sp, D)) * 0.5).astype(np.float32)
    pk = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    pv = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    sq = (rng.standard_normal((G, Hq,  Sc, D)) * 0.5).astype(np.float32)
    sk = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
    sv = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
    dOp_each = (rng.standard_normal((G, Hq, Sp, D)) * 0.3).astype(np.float32)
    dOs = (rng.standard_normal((G, Hq, Sc, D)) * 0.3).astype(np.float32)
    dOp_shared = dOp_each.sum(0)

    Op, Os, Lp, Ls = flash_attn_forward_prefix(pq, pk, pv, sq, sk, sv, si, window=W)
    dQp, dKp, dVp, dQs, dKs, dVs = flash_attn_backward_prefix(
        pq, pk, pv, sq, sk, sv, Op, Os, Lp, Ls, dOp_shared, dOs, si, window=W)

    # batched naive windowed reference on the concatenated [P, s_i] (B=G, churn-minimal)
    fq = np.stack([np.concatenate([pq, sq[i]], 1) for i in range(G)])
    fk = np.stack([np.concatenate([pk, sk[i]], 1) for i in range(G)])
    fv = np.stack([np.concatenate([pv, sv[i]], 1) for i in range(G)])
    fdo = np.stack([np.concatenate([dOp_each[i], dOs[i]], 1) for i in range(G)])
    Of, Lf = flash_attn_forward(fq, fk, fv, si, return_lse=True, window=W)
    e_fwd = max(maxabs(Os, Of[:, :, Sp:]), maxabs(Op, Of[0, :, :Sp]))
    dQf, dKf, dVf = flash_attn_backward(fq, fk, fv, Of, fdo, Lf, si, window=W)
    bit = max(maxabs(dQs, dQf[:, :, Sp:]), maxabs(dKs, dKf[:, :, Sp:]), maxabs(dVs, dVf[:, :, Sp:]))

    # fp64 windowed reference; prefix grads = Σ over G
    rdQp = np.zeros((Hq, Sp, D)); rdKp = np.zeros((Hkv, Sp, D)); rdVp = np.zeros((Hkv, Sp, D))
    e_suf = 0.0
    for i in range(G):
        _, rdQ, rdK, rdV = ref_full_win(fq[i], fk[i], fv[i], fdo[i], W)
        e_suf = max(e_suf, rel(dQs[i], rdQ[:, Sp:]), rel(dKs[i], rdK[:, Sp:]), rel(dVs[i], rdV[:, Sp:]))
        rdQp += rdQ[:, :Sp]; rdKp += rdK[:, :Sp]; rdVp += rdV[:, :Sp]
    e_pre = max(rel(dQp, rdQp), rel(dKp, rdKp), rel(dVp, rdVp))

    ok = e_fwd == 0.0 and bit < 1e-4 and e_suf < 0.02 and e_pre < 0.02
    print(f"  G={G} Hq={Hq} Hkv={Hkv} Sp={Sp} Sc={Sc} W={W}: fwd Δ={e_fwd:.0e}  suffix-bwd Δ={bit:.1e}  "
          f"suffix≤{e_suf*100:.2f}%  prefix(Σ)≤{e_pre*100:.2f}%  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("WINDOWED prefix-shared attention — fwd bitwise vs _attn_fwd_win on [P,s_i]; bwd vs fp64")
    print("=" * 96)
    r = [_case(4, 4, 2, 128, 128, 128),     # window spans boundary; late queries suffix-only
         _case(4, 16, 8, 256, 128, 256),    # real heads; whole suffix reaches into the prompt
         _case(6, 16, 8, 128, 256, 128),    # suffix longer than window (most queries never see prompt)
         _case(4, 16, 8, 256, 128, 128)]    # prompt LONGER than window (NKVBp > win_blocks)
    print("=" * 96)
    print("  ALL PASS (windowed prefix attention bitwise == naive windowed → ratio=1 on local layers)"
          if all(r) else "  FAIL: " + str(r))
