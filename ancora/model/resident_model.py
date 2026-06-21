"""
ancora/model/resident_model.py — ResidentModel: device-resident multi-layer Qwen3-0.6B training.

The bulk (every decoder layer) is a chain of validated device-resident ResidentLayerTrain instances:
within a layer all ~15 kernels pipeline on ONE stream with no host round-trips, and between layers
the activation handoff is a device buffer (gx_out → next gx_in, FP32 residual stream).

The model BOUNDARY is now DEVICE-RESIDENT and TIED (Qwen3 tie_word_embeddings=True): embed and the
LM head are ONE (V,H) param living on the GPU with device AdamW. That kills the two per-step
bottlenecks the host boundary had: the 311 MB lm_head upload every linear_ce, and the 2.5 s host
numpy AdamW on the 622 MB embed/lm_head arrays. The tied head uses (validated in
tests/kernels/test_tied_head.py):
  logits        = hidden @ embed.T          via _gemm_nt_f32(hidden, embed)
  CE            (_ce_stats / _ce_grad)       → logprob, glogit (M,V) bf16
  dhidden       = glogit @ embed             via _gemm(glogit, embed)
  embed grad    = glogitᵀ@hidden (LM head)   via _gemm_dW(glogit, hidden)   (V,H)
                + onehotᵀ@d_x0  (input embed) via _gemm_dW(onehot, gd) + _acc_f32
  input gather  x0 = onehot @ embed          via _gemm(onehot, embed)        (== embed[ids])
Only the final RMSNorm stays host (a tiny (H,) vector — never was the bottleneck).

  m  = ResidentModel(cfg, weights, B, S, vocab)
  h  = m.forward(input_ids, si)
  ce = m.loss_backward(h, labels, si, advantage)
  m.step(si, lr)
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.resident import _DBuf, _f32bf
from ancora.model.resident_train import ResidentLayerTrain, _bits2f
from ancora.kernels.norm import rmsnorm_forward, rmsnorm_backward
from ancora.kernels.loss import _gemm, _ce_stats, _ce_grad, GTM, GTN, GTK, CTM, TV
from ancora.kernels.fused import _gemm_nt_f32, _gemm_dW, _acc_f32, ACM, ACN
from ancora.optim.adamw import _adamw, _pick_otm, C as ADAM_C

_DWT = 64   # _gemm_dW output tile (embed grad (V,H))


class ResidentModel:
    def __init__(self, cfg, weights: dict, B: int, S: int, vocab: int,
                 optimizer: str = "adamw", muon_lr: float = 0.02):
        self.cfg, self.B, self.S, self.V = cfg, B, S, vocab
        self.H, self.M, self.eps = cfg.hidden, B * S, cfg.eps
        self.NL = len(weights["layers"])
        H, M, V = self.H, self.M, self.V
        assert V % GTN == 0 and V % GTK == 0 and V % TV == 0 and V % _DWT == 0, "pad vocab to the tiles"
        assert M % GTM == 0 and M % CTM == 0 and M % _DWT == 0 and M % 128 == 0

        # ── the bulk: device-resident layers (FP32 residual, SR grad, per-layer lid) ──
        # optimizer="muon": the 2D PROJ matrices use the resident Muon (momentum-only state) sharing
        # ONE NS scratch across ALL layers; the tied embed/LM-head + 1D gains stay on AdamW below
        # (Muon there hurts). Drops the per-PROJ v buffer (~1.7 GB on Qwen3-0.6B) → lower VRAM floor.
        self.optimizer, self.muon_lr, self.muon_scratch = optimizer, muon_lr, None
        if optimizer == "muon":
            from ancora.optim.muon import MuonScratch
            PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            self.muon_scratch = MuonScratch([weights["layers"][0][n].shape for n in PROJ])
        self.layers = [ResidentLayerTrain(cfg, weights["layers"][i], B, S, lid=i,
                                          optimizer=optimizer, muon_scratch=self.muon_scratch)
                       for i in range(self.NL)]

        # ── TIED embed = LM head: one (V,H) device param + device AdamW (fp32 master + m,v) ──
        embed = weights["embed"].astype(np.float32)                  # (V, H)
        self.gembed = _DBuf(_f32bf(embed))                           # (V,H) bf16 — the tied weight
        Re = (V * H) // ADAM_C
        self.eopt = dict(R=Re, otm=_pick_otm(Re),
                         p32=_DBuf(embed.reshape(Re, ADAM_C).copy()),
                         m=_DBuf(np.zeros((Re, ADAM_C), np.float32)),
                         v=_DBuf(np.zeros((Re, ADAM_C), np.float32)),
                         p16=self.gembed.view((Re, ADAM_C)))

        # ── final RMSNorm: host fp32 master + host AdamW (tiny (H,) vector) ──
        self.final_norm = weights["final_norm"].astype(np.float32).copy()
        self.fn_m = np.zeros_like(self.final_norm); self.fn_v = np.zeros_like(self.final_norm)
        self.t = 0

        # ── persistent boundary buffers (no per-step alloc churn) ──
        Z = _DBuf.zeros
        self.gin   = Z((M, H), np.float32)     # x0 = embed[ids] (fp32 residual) → layer 0
        self.gh    = Z((M, H), np.uint16)      # post-final-norm hidden (bf16) for logits
        self.gohot = Z((M, V), np.uint16)      # input one-hot (bf16) for gather + input-embed grad
        self.glog  = Z((M, V), np.float32)     # materialized logits
        self.gglog = Z((M, V), np.uint16)      # glogit (bf16)
        self.glp   = Z((M, 1), np.float32)     # logprob
        self.glse  = Z((M, 1), np.float32)     # logsumexp
        self.gadv  = Z((M, 1), np.float32)     # advantage
        self.glab  = Z((M, 1), np.int32)       # labels
        self.gdhid = Z((M, H), np.float32)     # dhidden (grad wrt post-norm hidden)
        self.gegrad = Z((V, H), np.float32)    # embed grad: LM-head part, then += input-embed part
        self.giegr  = Z((V, H), np.float32)    # input-embed grad (onehotᵀ @ gd)
        self.gdin  = Z((M, H), np.uint16)      # final-norm dx → top layer's gdout

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, input_ids: np.ndarray, si: int):
        """input_ids (B,S) int → h (M,H) f32 after the final RMSNorm. Device-resident gather."""
        M, H, V = self.M, self.H, self.V
        ids = input_ids.reshape(-1)
        onehot = np.zeros((M, V), np.float32); onehot[np.arange(M), ids] = 1.0
        self._oh_bits = _f32bf(onehot)                                   # keep host buffer alive until sync
        cdrv.cuMemcpyHtoDAsync(self.gohot.ptr, self._oh_bits, self.gohot.nbytes, si)
        # x0 = onehot @ embed  → gin (fp32, == embed[ids])
        ct.launch(si, (M // GTM, H // GTN, 1), _gemm, (self.gohot, self.gembed, self.gin, V // GTK, GTM, GTN, GTK))

        gx = self.gin
        for l in self.layers:
            gx = l.forward(gx, si)
        cudart.cudaStreamSynchronize(si)                                 # onehot/_oh_bits alive until here

        hpre = gx.to_numpy()                                             # (M,H) f32 residual
        h, rstd = rmsnorm_forward(hpre, self.final_norm, si, self.eps)   # host final norm
        cdrv.cuMemcpyHtoD(self.gh.ptr, _f32bf(h), self.gh.nbytes)        # upload h (bf16) for the logits GEMM
        self._cache = dict(hpre=hpre, rstd=rstd)
        return h

    # ── loss + backward ──────────────────────────────────────────────────────
    def loss_backward(self, h, labels, si: int, advantage=None):
        """h (M,H), labels (M,). Returns CE. Layer grads land in layer.G (device); embed grad in
        self.gegrad (device); final_norm grad cached for step()."""
        M, H, V = self.M, self.H, self.V; T = _DWT
        adv = np.ones(M, np.float32) if advantage is None else advantage.astype(np.float32)
        cdrv.cuMemcpyHtoD(self.gadv.ptr, np.ascontiguousarray(adv.reshape(M, 1)), self.gadv.nbytes)
        cdrv.cuMemcpyHtoD(self.glab.ptr, np.ascontiguousarray(labels.astype(np.int32).reshape(M, 1)), self.glab.nbytes)

        # logits = h @ embed.T ; CE → logprob, lse, glogit
        ct.launch(si, (M // 128, V // 128, 1), _gemm_nt_f32, (self.gh, self.gembed, self.glog, H // 64, 128, 128, 64))
        ct.launch(si, (M // CTM, 1, 1), _ce_stats, (self.glog, self.glab, self.glp, self.glse, V // TV))
        ct.launch(si, (M // CTM, 1, 1), _ce_grad, (self.glog, self.glse, self.glab, self.gadv, self.gglog, V // TV, 1.0 / M))
        # dhidden = glogit @ embed  ;  embed-grad (LM-head part) = glogitᵀ @ h
        ct.launch(si, (M // GTM, H // GTN, 1), _gemm, (self.gglog, self.gembed, self.gdhid, V // GTK, GTM, GTN, GTK))
        ct.launch(si, (V // T, H // T, 1), _gemm_dW, (self.gglog, self.gh, self.gegrad, M // T, T, T, T))
        cudart.cudaStreamSynchronize(si)

        lp = self.glp.to_numpy().reshape(M)
        ce = float(-(adv * lp).mean())
        dhidden = self.gdhid.to_numpy()                                  # (M,H) grad wrt post-norm hidden

        # host final-norm backward
        d_xpre, d_final = rmsnorm_backward(self._cache["hpre"], self.final_norm, dhidden, self._cache["rstd"], si)
        cdrv.cuMemcpyHtoDAsync(self.gdin.ptr, _f32bf(d_xpre), self.gdin.nbytes, si)

        gd = self.gdin
        for i in reversed(range(self.NL)):
            gd = self.layers[i].backward(gd, si)                         # gd → grad wrt x0 (embed output)
        # input-embed grad = onehotᵀ @ gd ; total embed grad = LM-head part + input part
        ct.launch(si, (V // T, H // T, 1), _gemm_dW, (self.gohot, gd, self.giegr, M // T, T, T, T))
        ct.launch(si, (V // ACM, H // ACN, 1), _acc_f32, (self.giegr, self.gegrad))
        cudart.cudaStreamSynchronize(si)                                 # d_xpre host buffer alive until here
        self._fn_grad = d_final.reshape(self.final_norm.shape)
        return ce

    # ── optimizer step (AdamW) ───────────────────────────────────────────────
    def step(self, si: int, lr: float = 1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, muon_lr=None):
        ml = self.muon_lr if muon_lr is None else muon_lr
        for l in self.layers:
            l.step(si, lr, b1, b2, eps, wd, ml)
        self.t += 1
        ibc1 = 1.0 / (1.0 - b1 ** self.t); ibc2 = 1.0 / (1.0 - b2 ** self.t)
        # device AdamW on the tied embed (no weight decay on embeddings)
        s = self.eopt
        ct.launch(si, (s["R"] // s["otm"], 1, 1), _adamw,
                  (self.gegrad.view((s["R"], ADAM_C)), s["m"], s["v"], s["p32"], s["p16"], s["otm"],
                   float(b1), float(b2), float(eps), float(lr), 0.0, float(ibc1), float(ibc2)))
        # host AdamW on the (tiny) final-norm gain
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        g = self._fn_grad
        self.fn_m = b1 * self.fn_m + (1 - b1) * g
        self.fn_v = b2 * self.fn_v + (1 - b2) * g * g
        self.final_norm -= lr * ((self.fn_m / bc1) / (np.sqrt(self.fn_v / bc2) + eps))

    def free(self):
        """Free every device buffer once (dedup by ptr: eopt['p16'] shares gembed's ptr; the Muon
        scratch is shared across layers so dedup also prevents a double-free)."""
        seen, seen_obj = set(), set()
        def visit(o):
            if isinstance(o, _DBuf):
                if o.ptr not in seen:
                    seen.add(o.ptr); o.free()
            elif isinstance(o, dict):
                for v in o.values(): visit(v)
            elif isinstance(o, (list, tuple)):
                for v in o: visit(v)
            elif type(o).__name__ in ("ResidentMuon", "MuonScratch") and id(o) not in seen_obj:
                seen_obj.add(id(o)); visit(vars(o))   # reach ResidentMuon.buf + MuonScratch pools
        for l in self.layers:
            visit(l.__dict__)
        visit(self.__dict__)
