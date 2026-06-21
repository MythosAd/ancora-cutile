"""
ancora/optim/adamw.py — AdamW optimizer (cuda-tile kernel, fp32 master + BF16 view).

Routed (in the Muon+AdamW hybrid) to 1D params (RMSNorm gains) and embedding / LM-head.
Keeps an fp32 master weight + fp32 m,v moments; emits a BF16 (uint16) copy each step
for the forward. Elementwise → trivially batch-invariant.

AdamW:  m=β1 m+(1-β1)g ;  v=β2 v+(1-β2)g² ;  θ -= lr(m̂/(√v̂+eps) + wd·θ),
        m̂=m/(1-β1ᵗ), v̂=v/(1-β2ᵗ).  Bias corrections change every step → passed as
        RUNTIME float scalars (NOT ct.Constant, which would recompile per step).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # sets CUDA_PATH

C = 64   # flatten every param to (numel//64, 64); 64 divides all Qwen3 param numels


@ct.kernel
def _adamw(grad, m, v, p32, p16,
           OTM: ct.Constant[int],
           beta1: float, beta2: float, eps: float, lr: float, wd: float,
           ibc1: float, ibc2: float):
    """One AdamW step on a (OTM, 64) tile.  ibc1=1/(1-β1ᵗ), ibc2=1/(1-β2ᵗ) (runtime)."""
    r = ct.bid(0)
    g  = ct.load(grad, index=(r, 0), shape=(OTM, C))
    tm = ct.load(m,    index=(r, 0), shape=(OTM, C))
    tv = ct.load(v,    index=(r, 0), shape=(OTM, C))
    tp = ct.load(p32,  index=(r, 0), shape=(OTM, C))
    mn = beta1 * tm + (1.0 - beta1) * g
    vn = beta2 * tv + (1.0 - beta2) * g * g
    upd = (mn * ibc1) / (ct.sqrt(vn * ibc2) + eps) + wd * tp
    pn  = tp - lr * upd
    ct.store(m,   index=(r, 0), tile=mn)
    ct.store(v,   index=(r, 0), tile=vn)
    ct.store(p32, index=(r, 0), tile=pn)
    ct.store(p16, index=(r, 0), tile=ct.bitcast(ct.astype(pn, ct.bfloat16), ct.uint16))


def _pick_otm(R):
    for cand in (128, 64, 32, 16, 8, 4, 2):
        if R % cand == 0:
            return cand
    return 1


class _Buf:
    def __init__(self, arr):
        self.shape, self.dtype, self.nb = arr.shape, arr.dtype, arr.nbytes
        _, self.p = cdrv.cuMemAlloc(arr.nbytes); cdrv.cuMemcpyHtoD(self.p, arr, arr.nbytes)
        self.__cuda_array_interface__ = {"shape": arr.shape, "typestr": arr.dtype.str,
                                         "data": (int(self.p), False), "version": 3}
    def upload(self, arr): cdrv.cuMemcpyHtoD(self.p, np.ascontiguousarray(arr), arr.nbytes)
    def numpy(self):
        o = np.empty(self.shape, self.dtype); cdrv.cuMemcpyDtoH(o, self.p, self.nb); return o
    def free(self): cdrv.cuMemFree(self.p)


def _bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32); u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)
def _bits_f32(u): return (u.astype(np.uint32) << 16).view(np.float32)


class AdamW:
    """fp32-master AdamW. params: {name: fp32 ndarray}. no_decay: names with wd=0
    (norm gains / biases). step(grads, stream) updates in place; .weights() returns the
    BF16-valued fp32 weights for the forward."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.01, no_decay=()):
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, wd
        self.t = 0
        self.st = {}
        for name, p in params.items():
            R = p.size // C
            otm = _pick_otm(R)
            flat = p.astype(np.float32).reshape(R, C)
            self.st[name] = dict(
                shape=p.shape, R=R, otm=otm, wd=0.0 if name in no_decay else wd,
                g=_Buf(np.zeros((R, C), np.float32)),
                m=_Buf(np.zeros((R, C), np.float32)), v=_Buf(np.zeros((R, C), np.float32)),
                p32=_Buf(flat.copy()), p16=_Buf(np.zeros((R, C), np.uint16)))

    def step(self, grads, stream_int):
        self.t += 1
        ibc1 = 1.0 / (1.0 - self.b1 ** self.t)
        ibc2 = 1.0 / (1.0 - self.b2 ** self.t)
        for name, s in self.st.items():
            s["g"].upload(grads[name].astype(np.float32).reshape(s["R"], C))
            ct.launch(stream_int, (s["R"] // s["otm"], 1, 1), _adamw,
                      (s["g"], s["m"], s["v"], s["p32"], s["p16"], s["otm"],
                       float(self.b1), float(self.b2), float(self.eps), float(self.lr),
                       float(s["wd"]), float(ibc1), float(ibc2)))
        cudart.cudaStreamSynchronize(stream_int)

    def weights(self):
        """BF16-valued fp32 weights (what the forward consumes), per name."""
        return {n: _bits_f32(s["p16"].numpy()).reshape(s["shape"]) for n, s in self.st.items()}

    def master(self):
        return {n: s["p32"].numpy().reshape(s["shape"]) for n, s in self.st.items()}

    def free(self):
        for s in self.st.values():
            for k in ("g", "m", "v", "p32", "p16"): s[k].free()
