"""_attn_fwd_tok (token-major attention + inline RoPE, TE+FlashInfer pattern) vs the existing
validated head-major chain (tok→head transpose → head-major RoPE → _attn_fwd → head→tok).
Proves the fused kernel matches, so it can replace 6 "other" kernels in the resident layer."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.attention import _attn_fwd, _attn_fwd_tok, BQ, D as DH
from ancora.kernels.rope import _rope_fwd, _rope_fwd_tok, RTM, build_cos_sin
from ancora.kernels.fused import _tok_to_head, _head_to_tok_f32, TT

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)

class GA:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))

B, S, Hq, Hkv, D = 1, 128, 16, 8, 64
M, qd, kd = B * S, Hq * D, Hkv * D
NQB, NKVB, NSB = S // BQ, S // BQ, S // TT
scale = 1.0 / math.sqrt(D)
rng = np.random.default_rng(0)
Qt = GA(bf((rng.standard_normal((M, qd)) * 0.5).astype(np.float32)))   # token-major
Kt = GA(bf((rng.standard_normal((M, kd)) * 0.5).astype(np.float32)))
Vt = GA(bf((rng.standard_normal((M, kd)) * 0.5).astype(np.float32)))
cosv, sinv = build_cos_sin(S, D, 1e6); gcos = GA(cosv); gsin = GA(sinv)

# ── reference: head-major transpose → rope → attn → head→tok ──
Qh = GA.z((M * Hq, D), np.uint16); Kh = GA.z((M * Hkv, D), np.uint16); Vh = GA.z((M * Hkv, D), np.uint16)
Qr = GA.z((M * Hq, D), np.uint16); Kr = GA.z((M * Hkv, D), np.uint16)
Ohm = GA.z((M * Hq, D), np.float32); Lhm = GA.z((M * Hq, 1), np.float32)
Otok_ref = GA.z((M, qd), np.uint16)   # _head_to_tok_f32 outputs bf16 bits
ct.launch(SI, (B * Hq, NSB, 1), _tok_to_head, (Qt, Qh, Hq, NSB))
ct.launch(SI, (B * Hkv, NSB, 1), _tok_to_head, (Kt, Kh, Hkv, NSB))
ct.launch(SI, (B * Hkv, NSB, 1), _tok_to_head, (Vt, Vh, Hkv, NSB))
ct.launch(SI, (S // RTM, B * Hq, 1), _rope_fwd, (Qh, gcos, gsin, Qr, S // RTM, D // 2))
ct.launch(SI, (S // RTM, B * Hkv, 1), _rope_fwd, (Kh, gcos, gsin, Kr, S // RTM, D // 2))
ct.launch(SI, (NQB, B * Hq, 1), _attn_fwd, (Qr, Kr, Vh, Ohm, Lhm, NQB, NKVB, Hq, Hkv, scale))
ct.launch(SI, (B * Hq, NSB, 1), _head_to_tok_f32, (Ohm, Otok_ref, Hq, NSB))   # note: f32→bf16 bits

# ── new: token-major RoPE (separate) + token-major attention ──
Qrt = GA.z((M, qd), np.uint16); Krt = GA.z((M, kd), np.uint16)
Otok = GA.z((M, qd), np.uint16); Lnew = GA.z((M * Hq, 1), np.float32)
ct.launch(SI, (M // RTM, Hq, 1), _rope_fwd_tok, (Qt, gcos, gsin, Qrt, S // RTM, D // 2))
ct.launch(SI, (M // RTM, Hkv, 1), _rope_fwd_tok, (Kt, gcos, gsin, Krt, S // RTM, D // 2))
ct.launch(SI, (NQB, B * Hq, 1), _attn_fwd_tok, (Qrt, Krt, Vt, Otok, Lnew, NQB, NKVB, Hq, Hkv, scale))
cudart.cudaStreamSynchronize(SI)

if __name__ == "__main__":
    print(f"_attn_fwd_tok vs head-major transpose+rope+attn  B={B} S={S}"); print("=" * 64)
    ref = (Otok_ref.np().astype(np.uint32) << 16).view(np.float32)   # decode bf16 bits → f32
    newO = (Otok.np().astype(np.uint32) << 16).view(np.float32)      # new O also bf16 bits now
    e = rel(newO, ref)
    print(f"  O(token-major) match: {e*100:.3f}%  {'OK' if e < 0.02 else 'FAIL'}")
    print(f"  sample new {newO[0,:3]}  ref {ref[0,:3]}")
    # also validate the fused-output-quant variant end-to-end via the resident class elsewhere;
    # here just confirm the bf16-output token-major kernel matches (the fp8 variant shares the loop).
    print("=" * 64); print(f"  {'PASS' if e < 0.02 else 'FAIL'}")
