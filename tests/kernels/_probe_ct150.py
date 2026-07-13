"""cuda-tile 1.5.0 feature probe — retest the 1.4.0 walls that shaped this codebase.

1. cat/extract → rotate_half in registers (the RoPE-fusion wall: 1.4.0 had NO tile
   slice/concat, blocking rope-in-epilogue; a pure data-movement+negate must be BITWISE).
2. Python list of tiles in a kernel (1.4.0: `xs=[]` → TileSyntaxError; this is the
   "hold chunks in registers across loops" pattern pitfall-0c wanted).
3. Pitfall 0c: ONE kernel with reduce-loop + reload-loop (RMSNorm shape) silently
   miscompiled ONE chunk in 1.4.0 → compare fused vs split-kernels per chunk, bitwise.
4. ct.gather: the embed-gather / data-dependent row load pattern, vs numpy.
5. atomic_add + num_blocks: persistent-kernel work-counter smoke (each block takes a
   ticket; final counter == grid size).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.loss import _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
print(f"cuda-tile {ct.__version__}")
results = {}

# ── 1. cat/extract rotate_half ────────────────────────────────────────────────
@ct.kernel
def _rot_half(x, y, TT_: ct.Constant[int], D_: ct.Constant[int]):
    r = ct.bid(0)
    t  = ct.load(x, index=(r, 0), shape=(TT_, D_))
    x1 = ct.extract(t, (0, 0), shape=(TT_, D_ // 2))
    x2 = ct.extract(t, (0, 1), shape=(TT_, D_ // 2))
    ct.store(y, index=(r, 0), tile=ct.cat((-x2, x1), 1))

try:
    TT, D = 64, 128
    rng = np.random.default_rng(0)
    xh = rng.standard_normal((256, D)).astype(np.float32)
    gx, gy = _GpuArray(xh), _GpuArray.zeros((256, D), np.float32)
    ct.launch(si, (256 // TT,), _rot_half, (gx, gy, TT, D)); sync()
    ref = np.concatenate([-xh[:, D // 2:], xh[:, :D // 2]], 1)
    nbad = int((gy.to_numpy() != ref).sum())
    results["cat/extract rotate_half"] = f"{'PASS (BITWISE)' if nbad == 0 else f'FAIL ({nbad} diffs)'}"
    gx.free(); gy.free()
except Exception as e:
    results["cat/extract rotate_half"] = f"FAIL ({type(e).__name__}: {str(e)[:120]})"

# ── 2. Python list of tiles across loops ─────────────────────────────────────
def _try_list():
    @ct.kernel
    def _list_hold(x, y, NC: ct.Constant[int]):
        r = ct.bid(0)
        xs = []
        for k in range(NC):
            xs.append(ct.load(x, index=(r, k), shape=(64, 128)))
        acc = ct.zeros((64, 1), ct.float32)
        for k in range(NC):
            acc = acc + ct.sum(xs[k] * xs[k], axis=-1, keepdims=True)
        rstd = ct.rsqrt(acc / (NC * 128.0) + 1e-6)
        for k in range(NC):
            ct.store(y, index=(r, k), tile=xs[k] * ct.broadcast_to(rstd, (64, 128)))
    NC = 8
    xh = np.random.default_rng(1).standard_normal((128, NC * 128)).astype(np.float32)
    gx, gy = _GpuArray(xh), _GpuArray.zeros((128, NC * 128), np.float32)
    ct.launch(si, (2,), _list_hold, (gx, gy, NC)); sync()
    rstd = 1.0 / np.sqrt((xh.astype(np.float64) ** 2).mean(1, keepdims=True) + 1e-6)
    rel = float(np.abs(gy.to_numpy() - xh * rstd).max() / np.abs(xh * rstd).max())
    gx.free(); gy.free()
    return f"PASS (list-of-tiles works, rel {rel:.1e})" if rel < 1e-5 else f"FAIL rel={rel:.1e}"

try:
    results["list-of-tiles in kernel"] = _try_list()
except Exception as e:
    results["list-of-tiles in kernel"] = f"STILL UNSUPPORTED ({type(e).__name__}: {str(e)[:100]})"

# ── 3. pitfall 0c: fused reduce+reload vs split kernels, per-chunk bitwise ───
@ct.kernel
def _fused_norm(x, y, HB: ct.Constant[int]):
    r = ct.bid(0)
    ss = ct.zeros((128, 1), ct.float32)
    for h in range(HB):                                   # loop 1: reduce
        t = ct.load(x, index=(r, h), shape=(128, 128))
        ss = ss + ct.sum(t * t, axis=-1, keepdims=True)
    rstd = ct.rsqrt(ss / (HB * 128.0) + 1e-6)
    for h in range(HB):                                   # loop 2: RELOAD + apply
        t = ct.load(x, index=(r, h), shape=(128, 128))
        ct.store(y, index=(r, h), tile=t * ct.broadcast_to(rstd, (128, 128)))

@ct.kernel
def _stats_only(x, rs, HB: ct.Constant[int]):
    r = ct.bid(0)
    ss = ct.zeros((128, 1), ct.float32)
    for h in range(HB):
        t = ct.load(x, index=(r, h), shape=(128, 128))
        ss = ss + ct.sum(t * t, axis=-1, keepdims=True)
    ct.store(rs, index=(r, 0), tile=ct.rsqrt(ss / (HB * 128.0) + 1e-6))

@ct.kernel
def _apply_only(x, rs, y, HB: ct.Constant[int]):
    r = ct.bid(0)
    rstd = ct.load(rs, index=(r, 0), shape=(128, 1))
    for h in range(HB):
        t = ct.load(x, index=(r, h), shape=(128, 128))
        ct.store(y, index=(r, h), tile=t * ct.broadcast_to(rstd, (128, 128)))

try:
    HB, M = 8, 1024
    xh = np.random.default_rng(2).standard_normal((M, HB * 128)).astype(np.float32)
    gx = _GpuArray(xh)
    gyf, gys = _GpuArray.zeros((M, HB * 128), np.float32), _GpuArray.zeros((M, HB * 128), np.float32)
    grs = _GpuArray.zeros((M, 1), np.float32)
    ct.launch(si, (M // 128,), _fused_norm, (gx, gyf, HB))
    ct.launch(si, (M // 128,), _stats_only, (gx, grs, HB))
    ct.launch(si, (M // 128,), _apply_only, (gx, grs, gys, HB)); sync()
    yf, ys = gyf.to_numpy(), gys.to_numpy()
    per_chunk = [int((yf[:, h * 128:(h + 1) * 128] != ys[:, h * 128:(h + 1) * 128]).sum()) for h in range(HB)]
    results["pitfall-0c two-loop reload"] = ("PASS (all 8 chunks bitwise == split kernels)"
                                             if sum(per_chunk) == 0 else f"STILL BROKEN per-chunk diffs {per_chunk}")
    for g in (gx, gyf, gys, grs): g.free()
except Exception as e:
    results["pitfall-0c two-loop reload"] = f"FAIL ({type(e).__name__}: {str(e)[:100]})"

# ── 4. gather: embed-gather pattern ──────────────────────────────────────────
@ct.kernel
def _gather_rows(emb, ids, out, TH_: ct.Constant[int]):
    r = ct.bid(0)
    ind0 = ct.load(ids, index=(r, 0), shape=(64, 1))
    ind1 = ct.reshape(ct.arange(TH_, dtype=ct.int32), (1, TH_))
    ct.store(out, index=(r, 0), tile=ct.gather(emb, (ind0, ind1)))

try:
    V, H, M = 512, 128, 256
    emb = np.random.default_rng(3).standard_normal((V, H)).astype(np.float32)
    ids = np.random.default_rng(4).integers(0, V, (M, 1)).astype(np.int32)
    ge, gi, go = _GpuArray(emb), _GpuArray(ids), _GpuArray.zeros((M, H), np.float32)
    ct.launch(si, (M // 64,), _gather_rows, (ge, gi, go, H)); sync()
    nbad = int((go.to_numpy() != emb[ids[:, 0]]).sum())
    results["gather (embed rows)"] = "PASS (BITWISE)" if nbad == 0 else f"FAIL ({nbad} diffs)"
    for g in (ge, gi, go): g.free()
except Exception as e:
    results["gather (embed rows)"] = f"FAIL ({type(e).__name__}: {str(e)[:100]})"

# ── 5. atomic_add work counter (persistent-kernel building block) ────────────
@ct.kernel
def _ticket(counter, out):
    b = ct.bid(0)
    t = ct.atomic_add(counter, 0, 1)          # 1-D array → single index
    ct.store(out, index=(b, 0), tile=ct.reshape(t, (1, 1)))

try:
    G = 60
    gc = _GpuArray(np.zeros(1, np.int32)); go = _GpuArray.zeros((G, 1), np.int32)
    ct.launch(si, (G,), _ticket, (gc, go)); sync()
    fin = int(gc.to_numpy()[0]); tk = np.sort(go.to_numpy()[:, 0])
    ok = fin == G and (tk == np.arange(G)).all()
    results["atomic_add work-counter"] = ("PASS (tickets 0..59 unique, counter==grid)"
                                          if ok else f"FAIL (counter={fin})")
    gc.free(); go.free()
except Exception as e:
    results["atomic_add work-counter"] = f"FAIL ({type(e).__name__}: {str(e)[:100]})"

print("=" * 76)
for k, v in results.items():
    print(f"  {k:32s} {v}")
