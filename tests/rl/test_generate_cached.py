"""KV-cache decode validation: teacher-forced decode hidden must match the full-prefill
forward hidden at every generated position (proves the cache + decode forward compute the
same thing as full causal attention). This is the correctness gate before using the
rollout engine. Then a tiny coherence check that greedy cached generation runs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.qwen3_model import Qwen3Model
from ancora.rl.rollout import generate_cached

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def test_decode_matches_prefill():
    print("--- teacher-forced decode hidden vs full-prefill hidden ---")
    cfg = Qwen3Config()
    V, NL, B, P, GEN = 512, 2, 8, 64, 64    # P,S multiples of BQ=64 (prefill/reference constraint)
    S, H = P + GEN, cfg.hidden
    model = Qwen3Model(cfg, n_layers=NL, vocab=V, seed=0)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, (B, S)).astype(np.int64)

    ref_hidden = model.forward(ids, si)[0].reshape(B, S, H)[:, P:]          # (B, GEN, H)
    _, dec_hidden = generate_cached(model, ids[:, :P], GEN, si, teacher=ids, return_hidden=True)

    denom = np.abs(ref_hidden).max() + 1e-9
    pe = np.abs(dec_hidden - ref_hidden).reshape(B, GEN, H).max(axis=(0, 2)) / denom  # (GEN,)
    e = pe.max()
    bad = np.argsort(pe)[-5:]
    print(f"  overall {e*100:.2f}%  worst 5 positions (offset from P): "
          + ", ".join(f"+{int(t)}:{pe[t]*100:.1f}%" for t in bad))
    ok = e < 0.05
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_greedy_runs():
    print("--- greedy cached generation runs ---")
    cfg = Qwen3Config()
    model = Qwen3Model(cfg, n_layers=2, vocab=512, seed=1)
    rng = np.random.default_rng(1)
    prompt = rng.integers(0, 512, (4, 64)).astype(np.int64)
    ids = generate_cached(model, prompt, 64, si, temp=0.0)   # greedy
    ok = ids.shape == (4, 128) and np.array_equal(ids[:, :64], prompt)
    print(f"  output shape {ids.shape}, prompt preserved={np.array_equal(ids[:, :64], prompt)}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("KV-cache decode engine validation")
    print("=" * 64)
    ok = test_decode_matches_prefill()
    ok &= test_greedy_runs()
    print("=" * 64)
    print(f"  {'PASS' if ok else 'FAIL'}")
