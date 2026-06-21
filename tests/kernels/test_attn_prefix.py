"""Prefix-shared attention (GRPO / Prefix Grouper): the prompt PREFIX is encoded once and each of
the G completions' SUFFIX queries attend over the shared prefix KV then their own suffix KV. The
acceptance test is BITWISE equivalence to standard _attn_fwd run on each concatenated [prefix,
suffix_i] — if that holds, prefix-sharing is training-equivalent ⇒ ratio=1 preserved.

  (A) O_prefix (run _attn_fwd on the prompt alone)  == [P,s_i] reference's prefix rows, every i
  (B) O_suffix_i (run _attn_fwd_prefix)             == [P,s_i] reference's suffix rows
  + determinism (run twice → bitwise).
Foreground only."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import (flash_attn_forward, _attn_fwd_prefix, _GpuArray,
                                       _f32_to_bf16_bits as f32bf, D, BQ, BKV)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _prefix_shared(pq, pk, pv, sq, sk, sv):
    """pq/pk/pv: prefix (Hq|Hkv, Sp, D); sq/sk/sv: suffix (G, Hq|Hkv, Sc, D). Returns O_suffix
    (G, Hq, Sc, D) f32 from _attn_fwd_prefix + O_prefix (Hq, Sp, D) from _attn_fwd on the prompt."""
    Hq, Sp, _ = pq.shape; Hkv = pk.shape[0]; G, _, Sc, _ = sq.shape
    NQBs, NKVBp, NKVBs = Sc // BQ, Sp // BKV, Sc // BKV
    scale = float(1.0 / math.sqrt(D))
    gQs = _GpuArray(f32bf(sq.reshape(G * Hq * Sc, D)))
    gKp = _GpuArray(f32bf(pk.reshape(Hkv * Sp, D))); gVp = _GpuArray(f32bf(pv.reshape(Hkv * Sp, D)))
    gKs = _GpuArray(f32bf(sk.reshape(G * Hkv * Sc, D))); gVs = _GpuArray(f32bf(sv.reshape(G * Hkv * Sc, D)))
    gO = _GpuArray(np.zeros((G * Hq * Sc, D), np.float32)); gL = _GpuArray(np.zeros((G * Hq * Sc, 1), np.float32))
    ct.launch(si, (NQBs, G * Hq, 1), _attn_fwd_prefix,
              (gQs, gKp, gVp, gKs, gVs, gO, gL, NQBs, NKVBp, NKVBs, Hq, Hkv, scale))
    sync()
    O_suffix = gO.to_numpy().reshape(G, Hq, Sc, D)
    O_prefix = flash_attn_forward(pq[None], pk[None], pv[None], si)[0]      # _attn_fwd on the prompt
    for g in (gQs, gKp, gVp, gKs, gVs, gO, gL): g.free()
    return O_suffix, O_prefix


def _case(G, Hq, Hkv, Sp, Sc):
    rng = np.random.default_rng(0)
    pq = (rng.standard_normal((Hq,  Sp, D)) * 0.5).astype(np.float32)       # shared prompt prefix
    pk = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    pv = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
    sq = (rng.standard_normal((G, Hq,  Sc, D)) * 0.5).astype(np.float32)    # per-completion suffix
    sk = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
    sv = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)

    O_suffix, O_prefix = _prefix_shared(pq, pk, pv, sq, sk, sv)
    O_suffix2, _ = _prefix_shared(pq, pk, pv, sq, sk, sv)                   # determinism
    det = maxabs(O_suffix, O_suffix2)

    e_pre = 0.0; e_suf = 0.0
    for i in range(G):                                                     # reference: standard attn on [P, s_i]
        fq = np.concatenate([pq, sq[i]], 1)[None]; fk = np.concatenate([pk, sk[i]], 1)[None]
        fv = np.concatenate([pv, sv[i]], 1)[None]
        O_full = flash_attn_forward(fq, fk, fv, si)[0]                      # (Hq, Sp+Sc, D)
        e_pre = max(e_pre, maxabs(O_prefix, O_full[:, :Sp]))               # prefix rows (same every i)
        e_suf = max(e_suf, maxabs(O_suffix[i], O_full[:, Sp:]))            # suffix rows
    ok = e_pre == 0.0 and e_suf == 0.0 and det == 0.0
    print(f"  G={G} Hq={Hq} Hkv={Hkv} Sp={Sp} Sc={Sc}: prefix Δ={e_pre:.2e}  suffix Δ={e_suf:.2e}  "
          f"det Δ={det:.2e}  {'OK (bitwise)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Prefix-shared attention forward — BITWISE == standard attn on [prefix, suffix_i]")
    print("=" * 84)
    r = [_case(4, 4, 2, 128, 64), _case(4, 16, 8, 128, 128), _case(8, 16, 8, 256, 128), _case(2, 16, 8, 64, 64)]
    print("=" * 84)
    print("  ALL PASS (prefix-shared fwd bitwise-equal → training-equivalent, ratio=1)" if all(r)
          else "  FAIL: " + str(r))
