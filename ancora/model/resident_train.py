"""
ancora/model/resident_train.py — ResidentLayerTrain: the device-resident TRAINING counterpart
to ResidentLayer (inference). One clean fwd→bwd→update unit on persistent buffers.

The validated backward (tests/model/test_resident_layer_bwd.py) is HEAD-MAJOR (uses
_attn_bwd_dq/dkdv on head-major Q/K/V), so the training forward here is the cache-materializing
HEAD-MAJOR forward (tests/model/test_resident_layer.py), NOT ResidentLayer's token-major inference
forward. Both forwards are numerically equivalent; the head-major one keeps the bf16 intermediates
the backward needs. GEMMs are BF16 (_gemm_bf16 fwd, _gemm_dx/_gemm_dW bwd) — param grads in fp32.

  tl = ResidentLayerTrain(cfg, weights, B, S)
  out = tl.forward(gx, si)            # → out + cache (self buffers)
  tl.backward(gdout, si)             # → weight grads in tl.G[name]
  tl.step(si, lr)                    # AdamW update of the bf16 weights (fp32 master)
"""
import sys, os, math, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.resident import _DBuf, _f32bf, _load_cutlass
from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, _rmsnorm_bwd_dx, _rmsnorm_dw_part,
                                  _rmsnorm_dw_reduce, _rmsnorm_dw_reduce_acc, _rmsnorm_stats_f32,
                                  _rmsnorm_apply_f32, _rmsnorm_bwd_dx_f32, _rmsnorm_dw_part_f32,
                                  TM as NTM, TH, TD, PART)
from ancora.kernels.activation import _swiglu_fwd, _swiglu_bwd, TM as STM, TI
from ancora.kernels.rope import _rope_fwd, _rope_bwd, RTM as RRTM, build_cos_sin
from ancora.kernels.attention import _attn_fwd, _attn_bwd_dq, _attn_bwd_dkdv, BQ
from ancora.kernels.fused import (_gemm_bf16, _gemm_dx, _gemm_dx_sr, _gemm_dx_fp8, _gemm_dW,
                                  _gemm_dW_acc, _residual_add, _residual_add_rf32, _tok_to_head,
                                  _head_to_tok, _head_to_tok_f32, _attn_delta, _cast64, RTM, RTN, TT, DTM)
from ancora.kernels.loss import GTM, GTN, GTK
from ancora.kernels.quant import _quant_mxfp8, _quant_mxfp8_w, _quant_mxfp8_w_cm, QTM, QWN, B as QB
from ancora.kernels.linear import _fwd_mxfp8_bf16, mxfp8_tile
from ancora.optim.adamw import _adamw, _pick_otm, C as ADAM_C

_bits2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
T = 64   # backward GEMM tile


class ResidentLayerTrain:
    def __init__(self, cfg, weights: dict, B: int, S: int, lid: int = 0, sr_grad: bool = True,
                 mxfp8: bool = False, cutlass: bool = False, fp8_bwd: bool = False,
                 optimizer: str = "adamw", muon_scratch=None, muon_exclude=None):
        self.cfg, self.B, self.S = cfg, B, S
        # sr_grad: stochastically round the fp32→bf16 activation-gradient downcasts (_gemm_dx_sr),
        # unbiasing them (the precision recipe's prescription). lid = layer index (salts the SR seed
        # so layers decorrelate). _bstep increments per backward() → SR dither varies per step.
        # mxfp8: run the forward projection GEMMs in MXFP8 (the precision recipe's forward; backward
        # stays BF16). Quant + mma_scaled are per-row → the MXFP8 rollout decode is bitwise-identical
        # to this MXFP8 prefill forward ⇒ ratio=1 under MXFP8 ([[precision-format-decision]]).
        # cutlass: same MXFP8 forward but the 7 projection GEMMs run on the CUTLASS sm_120a blockscaled
        # kernel (~1.3× cuda-tile) instead of _fwd_mxfp8_bf16. The CUTLASS GEMM is M-invariant / no
        # split-K (tests/hardware/test_cutlass_minvariant.py), so swapping it into BOTH this prefill
        # path and ResidentDecodeLayer keeps the frontier row bitwise-equal ⇒ ratio=1 still holds.
        self.lid, self.sr_grad, self._bstep, self.mxfp8, self.cutlass = lid, sr_grad, 0, mxfp8, cutlass
        self.gacc = False   # gradient accumulation: micro-batch ≥1 ADDS weight grads in place
        # fp8_bwd: run the DATA-GRADIENT (dx=dy@Wᵀ) in FP8 E4M3 + E8M0 block scaling (MAI/DeepSeek dgrad;
        # E4M3 not E5M2 — our fine block scaling handles the range, E4M3 ~2× more accurate, probed). The
        # weight-gradient stays BF16+FP32 (permanent grad). Backward-ONLY → not in the rollout forward →
        # ratio=1 untouched. The SFT/pretrain precision lever; RL keeps BF16 (rollout-bound + stability).
        self.fp8_bwd = fp8_bwd
        H, Hq, Hkv, Dh, I, eps = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.eps
        M, qd, kd = B * S, Hq * Dh, Hkv * Dh
        self.M, self.qd, self.kd, self.H, self.I, self.eps = M, qd, kd, H, I, eps
        self.Hq, self.Hkv, self.Dh, self.Ggrp = Hq, Hkv, Dh, Hq // Hkv
        self.NSB, self.NQB = S // TT, S // BQ
        self.scale = 1.0 / math.sqrt(Dh)
        Z = _DBuf.zeros

        # ── weights: bf16 (the GEMMs) + device-resident AdamW state ──
        self.PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.NORM = ["input_ln", "post_ln", "q_norm", "k_norm"]
        self.w = {n: _DBuf(_f32bf(weights[n])) for n in self.PROJ}                       # (K,N) bf16
        self.wn = {n: _DBuf(_f32bf(weights[n].reshape(1, -1))) for n in self.NORM}        # (1,·) bf16
        cosv, sinv = build_cos_sin(S, Dh, cfg.rope_theta); self.gcos = _DBuf(cosv); self.gsin = _DBuf(sinv)
        # ── device-resident AdamW state: fp32 master + m,v ON DEVICE; the bf16 weight the GEMM reads
        #    IS the AdamW p16 output (updated in place). step() = N elementwise kernel launches on si,
        #    NO host download/upload — that host churn was the 5.7s/step per-step bottleneck. ──
        # optimizer="muon" → the 2D PROJ matrices use the resident Muon (momentum-only state, no v);
        # NORM (1D gains) stay on AdamW (Muon on gains/embed/head HURTS — [[mfu-strategy]]). The Muon
        # master IS this same p32 (viewed (K,N)); only the v second-moment buffer is dropped (~1.7 GB
        # over the model's 2D weights — the optimizer-memory win that lowers the long_context floor).
        # muon_exclude = PROJ names to KEEP on AdamW even under muon (e.g. a MoE layer's DUMMY dense-FFN
        # gate/up/down — zero grad, routing them to Muon would waste 3 NS chains/layer on zeros).
        self.optimizer = optimizer
        self._proj_ext = set()   # proj names handled EXTERNALLY (the model's BatchedProjMuon) → step skips
        self._muon_excl = set(muon_exclude or ())
        muon_names = [n for n in self.PROJ if n not in self._muon_excl] if optimizer == "muon" else []
        self.opt = {}
        for n in self.PROJ + self.NORM:
            flat = weights[n].astype(np.float32).reshape(-1)
            R = flat.size // ADAM_C
            p16 = (self.w if n in self.PROJ else self.wn)[n]
            is_muon = (n in muon_names)
            self.opt[n] = dict(R=R, otm=_pick_otm(R),
                               p32=_DBuf(flat.reshape(R, ADAM_C).copy()),
                               m=None if is_muon else _DBuf(np.zeros((R, ADAM_C), np.float32)),
                               v=None if is_muon else _DBuf(np.zeros((R, ADAM_C), np.float32)),
                               p16=p16.view((R, ADAM_C)))
        self.t = 0
        if optimizer == "muon":
            from ancora.optim.muon import MuonScratch, ResidentMuon
            self.muon_scratch = muon_scratch or MuonScratch([weights[n].shape for n in muon_names])
            self._own_scratch = muon_scratch is None     # only free a scratch we created (not a shared one)
            self.muon = {n: ResidentMuon(self.opt[n]["p32"], self.w[n], *weights[n].shape, self.muon_scratch)
                         for n in muon_names}

        # ── MXFP8 forward: device-quantized projection weights (colblock) + per-input-width activation
        #    -quant scratch. The bf16 self.w stays for the BF16 backward AND is the quant SOURCE — the
        #    weight fp8 is (re)quantized ON DEVICE from self.w, lazily before a forward whenever a step
        #    has dirtied the weights (_wq_dirty). So the MXFP8 forward tracks the AdamW updates with no
        #    host round-trip / alloc churn (the closed loop). ──
        if mxfp8 or cutlass:
            self._wq_dirty, self._qsrc = True, {}
            self._q = {k: (Z((M, k), np.uint8), Z((M, k // QB), np.uint8)) for k in {H, qd, I}}
        if mxfp8 and not cutlass:
            self.wf8, self.ws8 = {}, {}
            for n in self.PROJ:
                K, N = weights[n].shape
                self.wf8[n] = Z((K, N), np.uint8); self.ws8[n] = Z((K // QB, N), np.uint8)
        # ── FP8 dgrad: each weight ALSO quantized per-32 along N (axis 1) for dx=dy@Wᵀ (the forward's
        #    wf8/ws8 are along K — a different orientation), + per-N-width dy-quant scratch. ──
        if fp8_bwd:
            self._wq_dirty_n = True
            self.wf8_n, self.ws8_n = {}, {}
            for n in self.PROJ:
                K, N = weights[n].shape
                self.wf8_n[n] = Z((K, N), np.uint8); self.ws8_n[n] = Z((K, N // QB), np.uint8)
            self._qdy = {nn: (Z((M, nn), np.uint8), Z((M, nn // QB), np.uint8)) for nn in {qd, kd, H, I}}

        # ── forward buffers (= the backward's cache) ──
        self.gh = Z((M, H)); self.gq = Z((M, qd)); self.gk = Z((M, kd)); self.gv = Z((M, kd))
        self.gqn = Z((M, qd)); self.gkn = Z((M, kd))
        self.gqh = Z((M * Hq, Dh)); self.gkh = Z((M * Hkv, Dh)); self.gvh = Z((M * Hkv, Dh))
        self.gqr = Z((M * Hq, Dh)); self.gkr = Z((M * Hkv, Dh))
        self.gO = Z((M * Hq, Dh), np.float32); self.gL = Z((M * Hq, 1), np.float32)
        # gx2/gout = the FP32 residual carry (gx in → +attn → gx2 → +mlp → gout out). Keeping the
        # residual stream fp32 stops the ~6912 massive activation from being coarsely bf16-rounded
        # across 28 layers (the late-layer drift, [[resident-layer]]). gattn/gh2/gmlp stay bf16
        # (they are GEMM/norm outputs feeding the next matmul); only the residual itself is fp32.
        self.gotok = Z((M, qd)); self.gattn = Z((M, H)); self.gx2 = Z((M, H), np.float32); self.gh2 = Z((M, H))
        self.gg = Z((M, I)); self.gu = Z((M, I)); self.ga = Z((M, I)); self.gmlp = Z((M, H)); self.gout = Z((M, H), np.float32)
        self.r1 = Z((M, 1), np.float32); self.rq = Z((M * Hq, 1), np.float32)
        self.rk = Z((M * Hkv, 1), np.float32); self.r2 = Z((M, 1), np.float32)

        # ── backward grad buffers ──
        self.gda = Z((M, I)); self.gdg = Z((M, I)); self.gdu = Z((M, I))
        self.gdh2a = Z((M, H)); self.gdh2b = Z((M, H)); self.gdh2 = Z((M, H)); self.gdx2m = Z((M, H)); self.gdx2 = Z((M, H))
        self.gdotok = Z((M, qd)); self.gdohm = Z((M * Hq, Dh)); self.gDelta = Z((M * Hq, 1), np.float32)
        self.gdqr = Z((M * Hq, Dh), np.float32); self.gdkr = Z((M * Hkv, Dh), np.float32); self.gdvh = Z((M * Hkv, Dh), np.float32)
        self.gdqrb = Z((M * Hq, Dh)); self.gdkrb = Z((M * Hkv, Dh)); self.gdqnhm = Z((M * Hq, Dh)); self.gdknhm = Z((M * Hkv, Dh))
        self.gdqn = Z((M, qd)); self.gdkn = Z((M, kd)); self.gdv = Z((M, kd)); self.gdq = Z((M, qd)); self.gdk = Z((M, kd))
        self.gdh1q = Z((M, H)); self.gdh1k = Z((M, H)); self.gdh1v = Z((M, H)); self.gdh1t = Z((M, H)); self.gdh1 = Z((M, H))
        self.gdxa = Z((M, H)); self.gdx = Z((M, H))
        self.part = Z((PART, max(H, Dh)), np.float32)
        # weight grads (fp32), shape matches the weight
        self.G = {n: Z(weights[n].shape, np.float32) for n in self.PROJ}
        self.G["input_ln"] = Z((1, H), np.float32); self.G["post_ln"] = Z((1, H), np.float32)
        self.G["q_norm"] = Z((1, Dh), np.float32); self.G["k_norm"] = Z((1, Dh), np.float32)

        if cutlass:
            self._setup_cutlass()

    # ── helpers ──
    def _V(self, g, sh): return g.view(sh)
    def _setup_cutlass(self):
        """One persistent CUTLASS MXFP8 handle per projection (init bakes the persistent A-fp8 / weight
        -fp8 / output pointers; the hot path is set_scales(scatter SFA/SFB) + run). Weights are laid
        COLUMN-major (CUTLASS B = (N,K) row-major) + scale (N,K//32), re-derived on device by
        _quant_mxfp8_w_cm from the bf16 master each time it changes (same kernel as the decode layer →
        byte-identical weight ⇒ ratio=1). Must run after the output buffers (self.gq … gmlp) exist."""
        self.dll = _load_cutlass(); Z = _DBuf.zeros
        self.wf8_cm, self.ws8_t = {}, {}
        for n in self.PROJ:
            K, N = self.w[n].shape
            self.wf8_cm[n] = Z((N, K), np.uint8); self.ws8_t[n] = Z((N, K // QB), np.uint8)
        cv = lambda p: ctypes.c_void_p(int(p))
        outbuf = {"q_proj": self.gq, "k_proj": self.gk, "v_proj": self.gv, "o_proj": self.gattn,
                  "gate_proj": self.gg, "up_proj": self.gu, "down_proj": self.gmlp}
        self._cu = {}
        for n in self.PROJ:
            K, N = self.w[n].shape
            af = self._q[K][0]   # A-fp8 (M,K); width K = the projection input dim
            h = self.dll.cutlass_mxfp8_init(self.M, N, K, cv(af.ptr), cv(self.wf8_cm[n].ptr), cv(outbuf[n].ptr))
            assert h, f"CUTLASS init failed for {n} (M={self.M} N={N} K={K})"
            self._cu[n] = h
    def _requant_w(self, si):
        """(Re)quantize the bf16 projection weights → MXFP8 on device (no host round-trip). Called
        lazily before a forward when _wq_dirty (after construction and after each AdamW step)."""
        for n in self.PROJ:
            K, N = self.w[n].shape
            if self.cutlass:   # column-major (N,K) fp8 + (N,K//32) scale for CUTLASS B
                ct.launch(si, (K // QB, N // QWN, 1), _quant_mxfp8_w_cm, (self.w[n], self.wf8_cm[n], self.ws8_t[n]))
            else:
                ct.launch(si, (K // QB, N // QWN, 1), _quant_mxfp8_w, (self.w[n], self.wf8[n], self.ws8[n]))
        self._wq_dirty = False
    def _gemm(self, si, A, wname, C, K, N):
        if self.cutlass or self.mxfp8:   # MXFP8 forward: quant A per-row-per-32-block (fixed K, no split-K)
            af, asc = self._q[K]
            if self._qsrc.get(K) != id(A):   # quant A ONCE: q/k/v reuse h's fp8, gate/up reuse gh2's (profiler: quant was 30% / 3 of 7 redundant)
                ct.launch(si, (self.M // QTM, 1, 1), _quant_mxfp8, (A, af, asc, K // QB))
                self._qsrc[K] = id(A)
        if self.cutlass:   # CUTLASS blockscaled GEMM: scatter our linear E8M0 scales → atom layout, run (both on si)
            h, cv = self._cu[wname], lambda p: ctypes.c_void_p(int(p))
            self.dll.cutlass_mxfp8_set_scales(h, cv(asc.ptr), cv(self.ws8_t[wname].ptr), cv(si))
            self.dll.cutlass_mxfp8_run(h, cv(si))
        elif self.mxfp8:
            tm, tn, tk = mxfp8_tile(N, K)
            ct.launch(si, (self.M // tm, N // tn, 1), _fwd_mxfp8_bf16,
                      (af, self.wf8[wname], asc, self.ws8[wname], C, K // tk, tm, tn, tk))
        else:
            # adaptive column tile: at small M the 128-wide TN leaves 60 SMs underfilled
            # (M=1024,N=1024 → 64 blocks, ncu DRAM 6%/SM 47%); _gemm_bf16 is BITWISE-invariant
            # to TN (probed, tests/kernels/_probe_decode_tiles.py) so narrow it until the grid
            # covers the machine. Large-M grids are untouched (tn stays GTN).
            tn = GTN
            while tn > 32 and (self.M // GTM) * (N // tn) < 240:
                tn //= 2
            ct.launch(si, (self.M // GTM, N // tn, 1), _gemm_bf16, (A, self.w[wname], C, K // GTK, GTM, tn, GTK))
    def _rms(self, si, xb, wb, rstd, yb, rows, hh, xf32=False):
        """xf32=True → x (the residual) is fp32 (input_ln/post_ln); else bf16 (q_norm/k_norm)."""
        stats = _rmsnorm_stats_f32 if xf32 else _rmsnorm_stats
        apply = _rmsnorm_apply_f32 if xf32 else _rmsnorm_apply
        ct.launch(si, (rows // NTM, 1, 1), stats, (xb, rstd, hh // TH, 1.0 / hh, self.eps))
        ct.launch(si, (rows // NTM, 1, 1), apply, (xb, wb, rstd, yb, hh // TH))

    def forward(self, gx, si: int):
        """gx: _DBuf (M,H) bf16 bits → self.gout. Materializes the cache the backward needs."""
        if self.mxfp8 or self.cutlass:
            if self._wq_dirty: self._requant_w(si)   # track the latest AdamW weight update in the fp8 forward weights
            self._qsrc.clear()                       # fresh activations this forward → first GEMM at each width re-quants
        self.gx = gx
        B, Hq, Hkv, Dh, H, I, qd = self.B, self.Hq, self.Hkv, self.Dh, self.H, self.I, self.qd
        M, NSB, NQB = self.M, self.NSB, self.NQB; V = self._V
        self._rms(si, gx, self.wn["input_ln"], self.r1, self.gh, M, H, xf32=True)
        self._gemm(si, self.gh, "q_proj", self.gq, H, qd)
        self._gemm(si, self.gh, "k_proj", self.gk, H, kd := self.kd)
        self._gemm(si, self.gh, "v_proj", self.gv, H, kd)
        self._rms(si, V(self.gq, (M * Hq, Dh)), self.wn["q_norm"], self.rq, V(self.gqn, (M * Hq, Dh)), M * Hq, Dh)
        self._rms(si, V(self.gk, (M * Hkv, Dh)), self.wn["k_norm"], self.rk, V(self.gkn, (M * Hkv, Dh)), M * Hkv, Dh)
        ct.launch(si, (B * Hq, NSB, 1), _tok_to_head, (self.gqn, self.gqh, Hq, NSB))
        ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (self.gkn, self.gkh, Hkv, NSB))
        ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (self.gv, self.gvh, Hkv, NSB))
        ct.launch(si, (self.S // RRTM, B * Hq, 1), _rope_fwd, (self.gqh, self.gcos, self.gsin, self.gqr, self.S // RRTM, Dh // 2))
        ct.launch(si, (self.S // RRTM, B * Hkv, 1), _rope_fwd, (self.gkh, self.gcos, self.gsin, self.gkr, self.S // RRTM, Dh // 2))
        ct.launch(si, (NQB, B * Hq, 1), _attn_fwd, (self.gqr, self.gkr, self.gvh, self.gO, self.gL, NQB, NQB, Hq, Hkv, self.scale))
        ct.launch(si, (B * Hq, NSB, 1), _head_to_tok_f32, (self.gO, self.gotok, Hq, NSB))
        self._gemm(si, self.gotok, "o_proj", self.gattn, qd, H)
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (gx, self.gattn, self.gx2))
        self._rms(si, self.gx2, self.wn["post_ln"], self.r2, self.gh2, M, H, xf32=True)
        self._gemm(si, self.gh2, "gate_proj", self.gg, H, I)
        self._gemm(si, self.gh2, "up_proj", self.gu, H, I)
        ct.launch(si, (M // STM, I // TI, 1), _swiglu_fwd, (self.gg, self.gu, self.ga))
        self._gemm(si, self.ga, "down_proj", self.gmlp, I, H)
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (self.gx2, self.gmlp, self.gout))
        return self.gout

    # ── backward helpers ──
    def _requant_w_n(self, si):
        """(Re)quantize each projection weight per-32 along N (axis 1) → wf8_n/ws8_n for the FP8 dgrad.
        Different orientation from the forward's along-K wf8/ws8. Lazy (only when an AdamW step dirtied
        the weights). _quant_mxfp8 reads the bf16 weight and blocks along its 2nd axis (= N)."""
        for n in self.PROJ:
            K, N = self.w[n].shape
            ct.launch(si, (K // QTM, 1, 1), _quant_mxfp8, (self.w[n], self.wf8_n[n], self.ws8_n[n], N // QB))
        self._wq_dirty_n = False

    def _dx(self, si, dy, wname, out, K, N):
        if self.fp8_bwd:                          # FP8 E4M3 dgrad: quant dy along N, mma_scaled(dy, Wᵀ)
            if self._wq_dirty_n: self._requant_w_n(si)   # lazy along-N weight quant (runs regardless of
            af, asc = self._qdy[N]                       # which backward() override calls _dx — once/step)
            ct.launch(si, (self.M // QTM, 1, 1), _quant_mxfp8, (dy, af, asc, N // QB))
            ct.launch(si, (self.M // T, K // T, 1), _gemm_dx_fp8,
                      (af, asc, self.wf8_n[wname], self.ws8_n[wname], out, N // T, T, T, T))
        elif self.sr_grad:
            seed = ((self._bstep * 64 + self._dxc) * 31 + self.lid) & 0x7FFFFFFF
            self._dxc += 1
            ct.launch(si, (self.M // T, K // T, 1), _gemm_dx_sr,
                      (dy, self.w[wname], out, N // T, T, T, T, int(seed)))
        else:
            ct.launch(si, (self.M // T, K // T, 1), _gemm_dx, (dy, self.w[wname], out, N // T, T, T, T))
    def _dW(self, si, xb, dy, out, K, N):
        dw = _gemm_dW_acc if self.gacc else _gemm_dW   # gacc: micro-batch ≥1 adds in place
        ct.launch(si, (K // T, N // T, 1), dw, (xb, dy, out, self.M // T, T, T, T))
    def _radd(self, si, a, b, o):
        ct.launch(si, (self.M // RTM, self.H // RTN, 1), _residual_add, (a, b, o))
    def _rms_bwd(self, si, xb, wname, dy, rstd, dxo, rows, hh, xf32=False):
        """xf32=True → x (the residual) is fp32 (input_ln/post_ln re-read the fp32 residual)."""
        bwd_dx  = _rmsnorm_bwd_dx_f32 if xf32 else _rmsnorm_bwd_dx
        dw_part = _rmsnorm_dw_part_f32 if xf32 else _rmsnorm_dw_part
        ct.launch(si, (rows // NTM, 1, 1), bwd_dx, (xb, self.wn[wname], dy, rstd, dxo, hh // TH, 1.0 / hh))
        mb = rows // NTM; bpp = (mb + PART - 1) // PART
        ct.launch(si, (hh // TD, PART, 1), dw_part, (xb, dy, rstd, self._V(self.part, (PART, hh)), mb, bpp))
        reduce = _rmsnorm_dw_reduce_acc if self.gacc else _rmsnorm_dw_reduce
        ct.launch(si, (hh // TD, 1, 1), reduce, (self._V(self.part, (PART, hh)), self.G[wname]))

    def backward(self, gdout, si: int):
        """gdout: _DBuf (M,H) bf16 bits (grad of self.gout). Fills self.G[name] (fp32) + self.gdx."""
        B, Hq, Hkv, Dh, H, I, qd, kd = self.B, self.Hq, self.Hkv, self.Dh, self.H, self.I, self.qd, self.kd
        M, NSB, NQB, G = self.M, self.NSB, self.NQB, self.G; V = self._V; sc = self.scale
        self._bstep += 1; self._dxc = 0          # SR seed: new per backward (step), per-_dx-call salt
        # ── MLP ── (fp8_bwd weight-along-N requant is lazy in _dx → covers all backward() overrides)
        self._dx(si, gdout, "down_proj", self.gda, I, H); self._dW(si, self.ga, gdout, G["down_proj"], I, H)
        ct.launch(si, (M // STM, I // TI, 1), _swiglu_bwd, (self.gg, self.gu, self.gda, self.gdg, self.gdu))
        self._dx(si, self.gdg, "gate_proj", self.gdh2a, H, I); self._dW(si, self.gh2, self.gdg, G["gate_proj"], H, I)
        self._dx(si, self.gdu, "up_proj", self.gdh2b, H, I);   self._dW(si, self.gh2, self.gdu, G["up_proj"], H, I)
        self._radd(si, self.gdh2a, self.gdh2b, self.gdh2)
        self._rms_bwd(si, self.gx2, "post_ln", self.gdh2, self.r2, self.gdx2m, M, H, xf32=True)
        self._radd(si, gdout, self.gdx2m, self.gdx2)
        # ── Attention ──
        self._dx(si, self.gdx2, "o_proj", self.gdotok, qd, H); self._dW(si, self.gotok, self.gdx2, G["o_proj"], qd, H)
        ct.launch(si, (B * Hq, NSB, 1), _tok_to_head, (self.gdotok, self.gdohm, Hq, NSB))
        ct.launch(si, (M * Hq // DTM, 1, 1), _attn_delta, (self.gO, self.gdohm, self.gDelta))
        ct.launch(si, (NQB, B * Hq, 1), _attn_bwd_dq, (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdqr, NQB, NQB, Hq, Hkv, sc))
        ct.launch(si, (NQB, B * Hkv, 1), _attn_bwd_dkdv, (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdkr, self.gdvh, NQB, NQB, Hq, Hkv, self.Ggrp, sc))
        ct.launch(si, (M * Hq // DTM, 1, 1), _cast64, (self.gdqr, self.gdqrb)); ct.launch(si, (M * Hkv // DTM, 1, 1), _cast64, (self.gdkr, self.gdkrb))
        ct.launch(si, (self.S // RRTM, B * Hq, 1), _rope_bwd, (self.gdqrb, self.gcos, self.gsin, self.gdqnhm, self.S // RRTM, Dh // 2))
        ct.launch(si, (self.S // RRTM, B * Hkv, 1), _rope_bwd, (self.gdkrb, self.gcos, self.gsin, self.gdknhm, self.S // RRTM, Dh // 2))
        ct.launch(si, (B * Hq, NSB, 1), _head_to_tok, (self.gdqnhm, self.gdqn, Hq, NSB))
        ct.launch(si, (B * Hkv, NSB, 1), _head_to_tok, (self.gdknhm, self.gdkn, Hkv, NSB))
        ct.launch(si, (B * Hkv, NSB, 1), _head_to_tok_f32, (self.gdvh, self.gdv, Hkv, NSB))
        self._rms_bwd(si, V(self.gq, (M * Hq, Dh)), "q_norm", V(self.gdqn, (M * Hq, Dh)), self.rq, V(self.gdq, (M * Hq, Dh)), M * Hq, Dh)
        self._rms_bwd(si, V(self.gk, (M * Hkv, Dh)), "k_norm", V(self.gdkn, (M * Hkv, Dh)), self.rk, V(self.gdk, (M * Hkv, Dh)), M * Hkv, Dh)
        self._dx(si, self.gdq, "q_proj", self.gdh1q, H, qd); self._dW(si, self.gh, self.gdq, G["q_proj"], H, qd)
        self._dx(si, self.gdk, "k_proj", self.gdh1k, H, kd); self._dW(si, self.gh, self.gdk, G["k_proj"], H, kd)
        self._dx(si, self.gdv, "v_proj", self.gdh1v, H, kd); self._dW(si, self.gh, self.gdv, G["v_proj"], H, kd)
        self._radd(si, self.gdh1q, self.gdh1k, self.gdh1t); self._radd(si, self.gdh1t, self.gdh1v, self.gdh1)
        self._rms_bwd(si, self.gx, "input_ln", self.gdh1, self.r1, self.gdxa, M, H, xf32=True)
        self._radd(si, self.gdx2, self.gdxa, self.gdx)
        return self.gdx

    def step(self, si: int, lr: float = 1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, muon_lr: float = 0.02):
        """Device-resident AdamW: grads are already in self.G (device), the fp32 master + m,v live on
        device, and the bf16 weight the GEMM reads IS the AdamW p16 output → just N elementwise kernel
        launches on si, no host transfer. Backward wrote G on si and step reads G on si (same-stream
        order), so no sync is needed either. Norms never weight-decay (wd=0).
        optimizer="muon": the PROJ matrices route to the resident Muon (lr=muon_lr); NORM stays AdamW."""
        self.t += 1
        ibc1 = 1.0 / (1.0 - b1 ** self.t); ibc2 = 1.0 / (1.0 - b2 ** self.t)
        for n in self.PROJ + self.NORM:
            if n in self._proj_ext:                           # batched by the model's BatchedProjMuon
                continue
            if self.optimizer == "muon" and n in self.muon:   # muon-managed PROJ (excludes dummy FFN)
                self.muon[n].step(self.G[n], si, muon_lr)
                continue
            s = self.opt[n]; R, otm = s["R"], s["otm"]
            ct.launch(si, (R // otm, 1, 1), _adamw,
                      (self.G[n].view((R, ADAM_C)), s["m"], s["v"], s["p32"], s["p16"], otm,
                       float(b1), float(b2), float(eps), float(lr),
                       float(wd if n in self.PROJ else 0.0), float(ibc1), float(ibc2)))
        if self.mxfp8 or self.cutlass:
            self._wq_dirty = True      # AdamW changed the bf16 weights → re-quant the fp8 forward weights next forward
        if self.fp8_bwd:
            self._wq_dirty_n = True    # ditto for the along-N fp8 weights the dgrad reads
