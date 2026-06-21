"""Device-resident prefix-shared attention (NoPE/global) — rl/prefix_resident.PrefixGlobalAttn.
Persistent buffers (no per-call alloc) → (a) BITWISE-equal to the host flash_attn_forward_prefix
(same kernels) and to standard attn on each [P,s_i], (b) CHURN-FREE: N repeated forwards are all
bitwise-identical (the host helper's alloc/free is what made the GRPO-step test flaky). This is the
clean building block for the ResidentMoE GLOBAL layers (global = NoPE, so no offset-RoPE)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import (flash_attn_forward, flash_attn_forward_prefix,
                                       flash_attn_backward_prefix, D, BQ)
from ancora.rl.prefix_resident import PrefixGlobalAttn

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _case(G, Hq, Hkv, Sp, Sc):
    rng = np.random.default_rng(0)
    pq = (rng.standard_normal((Hq,  Sp, D)) * 0.5).astype(np.float32)
    pk = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    pv = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    sq = (rng.standard_normal((G, Hq,  Sc, D)) * 0.5).astype(np.float32)
    sk = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
    sv = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)

    dOp = (rng.standard_normal((Hq, Sp, D)) * 0.3).astype(np.float32)   # shared prompt-output grad
    dOs = (rng.standard_normal((G, Hq, Sc, D)) * 0.3).astype(np.float32)

    pa = PrefixGlobalAttn(Hq, Hkv, D, Sp, Sc, G)
    Op, Os = pa.forward(pq, pk, pv, sq, sk, sv, si)
    grad = pa.backward(Op, Os, dOp, dOs, si)                           # (dQp,dKp,dVp, dQs,dKs,dVs)

    # (a) FORWARD bitwise vs host helper + vs standard attn on [P, s_i]
    hOp, hOs, hLp, hLs = flash_attn_forward_prefix(pq, pk, pv, sq, sk, sv, si)
    e_host = max(maxabs(Op, hOp), maxabs(Os, hOs))
    e_ref = 0.0
    for i in range(G):
        fq = np.concatenate([pq, sq[i]], 1)[None]; fk = np.concatenate([pk, sk[i]], 1)[None]
        fv = np.concatenate([pv, sv[i]], 1)[None]
        O_full = flash_attn_forward(fq, fk, fv, si)[0]
        e_ref = max(e_ref, maxabs(Os[i], O_full[:, Sp:]), maxabs(Op, O_full[:, :Sp]))
    # (b) BACKWARD bitwise vs host flash_attn_backward_prefix
    hg = flash_attn_backward_prefix(pq, pk, pv, sq, sk, sv, hOp, hOs, hLp, hLs, dOp, dOs, si)
    e_bwd = max(maxabs(grad[j], hg[j]) for j in range(6))
    # (c) churn-free determinism: 5 repeated resident fwd+bwd bitwise-identical
    det = 0.0
    for _ in range(5):
        Op2, Os2 = pa.forward(pq, pk, pv, sq, sk, sv, si)
        g2 = pa.backward(Op2, Os2, dOp, dOs, si)
        det = max(det, maxabs(Op2, Op), maxabs(g2[1], grad[1]), maxabs(g2[3], grad[3]))

    ok = e_host == 0.0 and e_ref == 0.0 and e_bwd == 0.0 and det == 0.0
    print(f"  G={G} Hq={Hq} Hkv={Hkv} Sp={Sp} Sc={Sc}: fwd vs-host/ref Δ={max(e_host,e_ref):.0e}  bwd vs-host Δ={e_bwd:.0e}  "
          f"5x-det Δ={det:.0e}  {'OK (bitwise + churn-free)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Device-resident prefix-shared attention (NoPE/global) — bitwise + churn-free")
    print("=" * 88)
    r = [_case(4, 16, 8, 128, 128), _case(8, 16, 8, 256, 128), _case(6, 16, 8, 192, 64)]
    print("=" * 88)
    print("  ALL PASS (resident prefix attn bitwise == host/[P,s_i] + churn-free deterministic)" if all(r)
          else "  FAIL: " + str(r))
