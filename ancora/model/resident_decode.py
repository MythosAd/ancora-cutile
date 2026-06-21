"""
ancora/model/resident_decode.py — ResidentDecodeLayer: device-resident single-token DECODE,
the rollout counterpart to ResidentLayerTrain (prefill/training). One frontier token per
sequence, a per-layer KV-cache, kernels chained on persistent buffers (no host round-trips,
no per-step alloc → no churn race, [[batch-invariance]]).

THE POINT — ratio = 1 exactly. The decode forward mirrors training's prefill forward kernel-for
-kernel using the *unified* decode pieces that were proven bitwise-equal to prefill at a position:
  • _rope_fwd_dec        single-position RoPE  ==  _rope_fwd row at pos        (test_device_decode_plumbing)
  • _attn_decode_blk     full BQ-block at runtime q_blk; frontier row pmod  ==  _attn_fwd row pos
                         (single-row MMA is 1 ULP off → MUST run the full block, test_attn_decode_mma)
  • _gemm_bf16/_rms/_swiglu/_residual_add_rf32  are per-row/per-position → row-independent.
By induction over positions AND layers, the decode hidden at position t is BITWISE-identical to
training's prefill hidden at t ⇒ rollout logprob == training prefill logprob exactly (no importance
sampling needed — the single-codebase advantage, [[batch-invariance]]).

tok↔head is the IDENTITY at one position (S=1): token-major (Md, Hq*Dh) reshaped (Md*Hq, Dh) is
head-major, so the prefill _tok_to_head/_head_to_tok transposes become free reshapes (.view).

Layout / sizing:
  • GEMM batch  Md = 128 (a full 128-row MMA tile, so each frontier row is computed in the same
    fragment structure as prefill → bitwise). Real sequences live in rows [0:Bp); the rest are
    padding (per-row ops ⇒ harmless). attention/cache are sized for the REAL Bp only (no 128× cache).
  • per-layer cache Kc,Vc : (Bp*Hkv*maxS, D) uint16, sequence-major within (seq,kv-head); _append_kv
    writes pos via cache.at_pos(pos); _attn_decode_blk reads blocks 0..q_blk (future slots masked).

  dl  = ResidentDecodeLayer(cfg, weights, Bp, maxS)
  out = dl.forward(gx, pos, si)     # gx (Md,H) f32 residual (row 0 = the frontier token) → gout (Md,H) f32
"""
import sys, os, math, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.resident import _DBuf, _f32bf, _load_cutlass
from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, _rmsnorm_stats_f32,
                                  _rmsnorm_apply_f32, f32_to_bf16_bits, TM as NTM, TH)
from ancora.kernels.activation import _swiglu_fwd, TM as STM, TI
from ancora.kernels.rope import _rope_fwd_dec, build_cos_sin, RTM as RRTM
from ancora.kernels.attention import (_attn_decode_blk, _scatter_blk, _gather_blk, _append_kv,
                                       BQ, BKV, D as ATTN_D)
from ancora.kernels.fused import _gemm_bf16, _cast_bf16, _trunc_bf16, _residual_add_rf32, _gemm_nt_f32, RTM, RTN
from ancora.kernels.loss import _ce_stats, _argmax_id, GTM, GTN, GTK, CTM, TV
from ancora.kernels.quant import _quant_mxfp8, _quant_mxfp8_w, _quant_mxfp8_w_cm, QTM, QWN, B as QB
from ancora.kernels.linear import _fwd_mxfp8_bf16, mxfp8_tile

MGEMM = 128   # decode GEMM batch (a full 128-row MMA tile → frontier row is bitwise-equal to prefill)


class ResidentDecodeLayer:
    def __init__(self, cfg, weights: dict, Bp: int, maxS: int, mxfp8: bool = False, cutlass: bool = False):
        self.cfg, self.Bp, self.maxS, self.mxfp8, self.cutlass = cfg, Bp, maxS, mxfp8, cutlass
        H, Hq, Hkv, Dh, I, eps = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.eps
        assert Dh == ATTN_D, f"head_dim must be {ATTN_D}"
        assert Bp <= MGEMM, "Bp must fit in one padded GEMM tile"
        assert maxS % BKV == 0, "maxS must be a multiple of BKV"
        Md = MGEMM
        qd, kd = Hq * Dh, Hkv * Dh
        self.Md, self.qd, self.kd, self.H, self.I, self.eps = Md, qd, kd, H, I, eps
        self.Hq, self.Hkv, self.Dh = Hq, Hkv, Dh
        self.NKVB = maxS // BKV
        self.scale = 1.0 / math.sqrt(Dh)
        Z = _DBuf.zeros

        # ── weights: bf16 bits (same _f32bf truncation as ResidentLayerTrain → identical bits) ──
        self.PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.NORM = ["input_ln", "post_ln", "q_norm", "k_norm"]
        self.w = {n: _DBuf(_f32bf(weights[n])) for n in self.PROJ}
        self.wn = {n: _DBuf(_f32bf(weights[n].reshape(1, -1))) for n in self.NORM}
        cosv, sinv = build_cos_sin(maxS, Dh, cfg.rope_theta)
        self.gcos, self.gsin = _DBuf(cosv), _DBuf(sinv)             # (maxS, Dh/2) f32

        # ── MXFP8 forward (matches ResidentLayerTrain(mxfp8=True) kernel-for-kernel): device-quantized
        #    projection weights (colblock, same _quant_mxfp8_w as training → identical fp8 bytes from the
        #    same bf16 weight ⇒ bitwise) + per-input-width activation-quant scratch (M=Md). Weights are
        #    quantized lazily on the first forward (_wq_dirty); a caller updating the bf16 weights between
        #    rollouts re-dirties to re-quant. quant + mma_scaled are per-row → frontier row bitwise-equal
        #    to the MXFP8 prefill forward. ──
        if mxfp8 or cutlass:
            self._wq_dirty, self._qsrc = True, {}
            self._q = {k: (Z((Md, k), np.uint8), Z((Md, k // QB), np.uint8)) for k in {H, qd, I}}
        if mxfp8 and not cutlass:
            self.wf8, self.ws8 = {}, {}
            for n in self.PROJ:
                K, N = weights[n].shape
                self.wf8[n] = Z((K, N), np.uint8); self.ws8[n] = Z((K // QB, N), np.uint8)

        # ── per-layer KV-cache (head-major, sequence-major within head) ──
        self.Kc = Z((Bp * Hkv * maxS, Dh)); self.Vc = Z((Bp * Hkv * maxS, Dh))

        # ── intermediate buffers (token-major (Md, ·); head-major access is a free .view) ──
        self.gh = Z((Md, H)); self.gq = Z((Md, qd)); self.gk = Z((Md, kd)); self.gv = Z((Md, kd))
        self.gqn = Z((Md, qd)); self.gkn = Z((Md, kd)); self.gqr = Z((Md, qd)); self.gkr = Z((Md, kd))
        self.r1 = Z((Md, 1), np.float32); self.rq = Z((Md * Hq, 1), np.float32)
        self.rk = Z((Md * Hkv, 1), np.float32); self.r2 = Z((Md, 1), np.float32)
        # attention staging (only the real Bp sequences; frontier query at in-block row pmod)
        self.blockQ = Z((Bp * Hq * BQ, Dh)); self.blockO = Z((Bp * Hq * BQ, Dh), np.float32)
        self.gO = Z((Md * Hq, Dh), np.float32)                     # f32 attn output, head-major (padded → zeros)
        # post-attention / MLP (gx2/gout = FP32 residual carry, matching ResidentLayerTrain)
        self.gotok = Z((Md, qd)); self.gattn = Z((Md, H)); self.gx2 = Z((Md, H), np.float32)
        self.gh2 = Z((Md, H)); self.gg = Z((Md, I)); self.gu = Z((Md, I)); self.ga = Z((Md, I))
        self.gmlp = Z((Md, H)); self.gout = Z((Md, H), np.float32)

        if cutlass:
            self._setup_cutlass()

    # ── helpers (identical reductions to ResidentLayerTrain) ──
    def _V(self, g, sh): return g.view(sh)

    def _setup_cutlass(self):
        """Per-projection CUTLASS MXFP8 handles (M=Md=128). SAME column-major weight quant
        (_quant_mxfp8_w_cm) + SAME CUTLASS kernel config as ResidentLayerTrain._setup_cutlass, so the
        frontier row is bitwise-equal to the prefill GEMM (the GEMM is M-invariant). Must run after the
        output buffers (self.gq … gmlp) exist."""
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
            af = self._q[K][0]
            h = self.dll.cutlass_mxfp8_init(self.Md, N, K, cv(af.ptr), cv(self.wf8_cm[n].ptr), cv(outbuf[n].ptr))
            assert h, f"CUTLASS init failed for {n} (M={self.Md} N={N} K={K})"
            self._cu[n] = h

    def _requant_w(self, si):
        """(Re)quantize the bf16 projection weights → MXFP8 on device (same kernel as training)."""
        for n in self.PROJ:
            K, N = self.w[n].shape
            if self.cutlass:
                ct.launch(si, (K // QB, N // QWN, 1), _quant_mxfp8_w_cm, (self.w[n], self.wf8_cm[n], self.ws8_t[n]))
            else:
                ct.launch(si, (K // QB, N // QWN, 1), _quant_mxfp8_w, (self.w[n], self.wf8[n], self.ws8[n]))
        self._wq_dirty = False

    def _gemm(self, si, A, wname, C, K, N):
        if self.cutlass or self.mxfp8:    # MXFP8 forward: per-row-per-32-block quant — identical to the MXFP8 prefill
            af, asc = self._q[K]
            if self._qsrc.get(K) != id(A):   # quant A ONCE (q/k/v share h, gate/up share gh2) — same fp8 as before → bitwise
                ct.launch(si, (self.Md // QTM, 1, 1), _quant_mxfp8, (A, af, asc, K // QB))
                self._qsrc[K] = id(A)
        if self.cutlass:
            h, cv = self._cu[wname], lambda p: ctypes.c_void_p(int(p))
            self.dll.cutlass_mxfp8_set_scales(h, cv(asc.ptr), cv(self.ws8_t[wname].ptr), cv(si))
            self.dll.cutlass_mxfp8_run(h, cv(si))
        elif self.mxfp8:
            tm, tn, tk = mxfp8_tile(N, K)
            ct.launch(si, (self.Md // tm, N // tn, 1), _fwd_mxfp8_bf16,
                      (af, self.wf8[wname], asc, self.ws8[wname], C, K // tk, tm, tn, tk))
        else:
            ct.launch(si, (self.Md // GTM, N // GTN, 1), _gemm_bf16, (A, self.w[wname], C, K // GTK, GTM, GTN, GTK))

    def _rms(self, si, xb, wb, rstd, yb, rows, hh, xf32=False):
        stats = _rmsnorm_stats_f32 if xf32 else _rmsnorm_stats
        apply = _rmsnorm_apply_f32 if xf32 else _rmsnorm_apply
        ct.launch(si, (rows // NTM, 1, 1), stats, (xb, rstd, hh // TH, 1.0 / hh, self.eps))
        ct.launch(si, (rows // NTM, 1, 1), apply, (xb, wb, rstd, yb, hh // TH))

    def forward(self, gx, pos: int, si: int):
        """gx (Md,H) f32 residual, row 0 = the frontier token (rows 1.. are padding). Appends this
        token's K/V to the cache at position `pos`, attends over the cache, returns gout (Md,H) f32
        whose row 0 is BITWISE-equal to training's prefill hidden at position `pos`."""
        Md, Hq, Hkv, Dh, H, I, qd, kd = self.Md, self.Hq, self.Hkv, self.Dh, self.H, self.I, self.qd, self.kd
        Bp, V = self.Bp, self._V
        q_blk, pmod = pos // BQ, pos % BQ
        hd = Dh // 2
        if self.mxfp8 or self.cutlass:
            if self._wq_dirty: self._requant_w(si)
            self._qsrc.clear()

        # ── attention block ──
        self._rms(si, gx, self.wn["input_ln"], self.r1, self.gh, Md, H, xf32=True)
        self._gemm(si, self.gh, "q_proj", self.gq, H, qd)
        self._gemm(si, self.gh, "k_proj", self.gk, H, kd)
        self._gemm(si, self.gh, "v_proj", self.gv, H, kd)
        self._rms(si, V(self.gq, (Md * Hq, Dh)), self.wn["q_norm"], self.rq, V(self.gqn, (Md * Hq, Dh)), Md * Hq, Dh)
        self._rms(si, V(self.gk, (Md * Hkv, Dh)), self.wn["k_norm"], self.rk, V(self.gkn, (Md * Hkv, Dh)), Md * Hkv, Dh)
        # single-position RoPE (cos/sin[pos] broadcast); reshape token-major→head-major is free at S=1
        ct.launch(si, (Md * Hq // RRTM, 1, 1), _rope_fwd_dec,
                  (V(self.gqn, (Md * Hq, Dh)), self.gcos.at_pos(pos), self.gsin.at_pos(pos), V(self.gqr, (Md * Hq, Dh)), hd))
        ct.launch(si, (Md * Hkv // RRTM, 1, 1), _rope_fwd_dec,
                  (V(self.gkn, (Md * Hkv, Dh)), self.gcos.at_pos(pos), self.gsin.at_pos(pos), V(self.gkr, (Md * Hkv, Dh)), hd))
        # append this token's K (RoPE'd) and V to the cache at `pos` (only the real Bp sequences)
        ct.launch(si, (Bp, Hkv, 1), _append_kv,
                  (V(self.gkr, (Md * Hkv, Dh)), V(self.gv, (Md * Hkv, Dh)),
                   self.Kc.at_pos(pos), self.Vc.at_pos(pos), self.maxS, Hkv))
        # scatter frontier query → blockQ in-block row pmod; decode-attend the cache; gather row pmod back
        ct.launch(si, (Bp * Hq, 1, 1), _scatter_blk, (V(self.gqr, (Md * Hq, Dh)), self.blockQ.at_pos(pmod), BQ))
        ct.launch(si, (Bp * Hq, 1, 1), _attn_decode_blk,
                  (self.blockQ, self.Kc, self.Vc, self.blockO, self.NKVB, Hq, Hkv, self.scale, int(q_blk)))
        ct.launch(si, (Bp * Hq, 1, 1), _gather_blk, (self.blockO.at_pos(pmod), self.gO, BQ))
        # head-major f32 O → token-major bf16 (head→tok identity at S=1) → o_proj + residual
        ct.launch(si, (Md * Hq // RTM, Dh // RTN, 1), _cast_bf16, (V(self.gO, (Md * Hq, Dh)), V(self.gotok, (Md * Hq, Dh))))
        self._gemm(si, self.gotok, "o_proj", self.gattn, qd, H)
        ct.launch(si, (Md // RTM, H // RTN, 1), _residual_add_rf32, (gx, self.gattn, self.gx2))

        # ── MLP block ──
        self._rms(si, self.gx2, self.wn["post_ln"], self.r2, self.gh2, Md, H, xf32=True)
        self._gemm(si, self.gh2, "gate_proj", self.gg, H, I)
        self._gemm(si, self.gh2, "up_proj", self.gu, H, I)
        ct.launch(si, (Md // STM, I // TI, 1), _swiglu_fwd, (self.gg, self.gu, self.ga))
        self._gemm(si, self.ga, "down_proj", self.gmlp, I, H)
        ct.launch(si, (Md // RTM, H // RTN, 1), _residual_add_rf32, (self.gx2, self.gmlp, self.gout))
        return self.gout

    def free(self):
        seen = set()
        bufs = list(self.w.values()) + list(self.wn.values()) + [
                self.gcos, self.gsin, self.Kc, self.Vc, self.gh, self.gq, self.gk, self.gv,
                self.gqn, self.gkn, self.gqr, self.gkr, self.r1, self.rq, self.rk, self.r2,
                self.blockQ, self.blockO, self.gO, self.gotok, self.gattn, self.gx2, self.gh2,
                self.gg, self.gu, self.ga, self.gmlp, self.gout]
        if self.mxfp8 and not self.cutlass:
            bufs += list(self.wf8.values()) + list(self.ws8.values())
        if self.cutlass:
            bufs += list(self.wf8_cm.values()) + list(self.ws8_t.values())
            for h in self._cu.values(): self.dll.cutlass_mxfp8_free(ctypes.c_void_p(int(h)))
        if self.mxfp8 or self.cutlass:
            bufs += [b for pair in self._q.values() for b in pair]
        for o in bufs:
            if int(o.ptr) not in seen:
                seen.add(int(o.ptr)); o.free()


class ResidentDecodeModel:
    """Device-resident autoregressive ROLLOUT engine — the inference counterpart to ResidentModel.

    A stack of ResidentDecodeLayer (per-layer KV-cache) + the SAME tied embed/LM-head boundary
    ResidentModel uses (Qwen3 tie_word_embeddings): embed gather, host final RMSNorm, tied-head
    logits via _gemm_nt_f32, CE via _ce_stats. Because the decode hidden at position t is bitwise
    -equal to ResidentModel's training prefill at t and the boundary is identical, the per-token
    logprob this engine reports during rollout is BITWISE-equal to the logprob training assigns to
    the same token ⇒ π_train/π_infer = 1 exactly (no importance sampling — [[batch-invariance]]).

      dm = ResidentDecodeModel(cfg, weights, Bp, maxS, vocab)      # weights = load_qwen3 format
      lp = dm.score(ids, labels, si)                               # teacher-forced per-token logprob
      gen_ids, gen_lp = dm.generate(prompt_ids, n_new, si)         # greedy autoregressive (Bp=1)
    """

    def __init__(self, cfg, weights: dict, Bp: int, maxS: int, vocab: int, mxfp8: bool = False, cutlass: bool = False):
        self.cfg, self.Bp, self.maxS, self.V, self.H, self.eps = cfg, Bp, maxS, vocab, cfg.hidden, cfg.eps
        self.NL = len(weights["layers"])
        H, V, Md = self.H, vocab, MGEMM
        assert V % GTN == 0 and V % GTK == 0 and V % TV == 0, "pad vocab to the tiles"
        self.Md = Md
        self.layers = [ResidentDecodeLayer(cfg, weights["layers"][i], Bp, maxS, mxfp8=mxfp8, cutlass=cutlass) for i in range(self.NL)]
        embed = weights["embed"].astype(np.float32)                 # (V,H)
        self.gembed = _DBuf(_f32bf(embed))                          # tied (V,H) bf16 — logits read it
        self.embed_bf = _f32bf(embed)                               # host (V,H) for the per-step gather
        self.gwfn = _DBuf(f32_to_bf16_bits(weights["final_norm"].astype(np.float32).reshape(1, H)))  # final-norm gain (bf16)
        Z = _DBuf.zeros
        self.gin  = Z((Md, H), np.float32)                          # frontier embed → layer-0 input (row 0)
        self.gxbf = Z((Md, H), np.uint16)                           # bf16-rounded pre-norm hidden (final-norm input)
        self.grfn = Z((Md, 1), np.float32)                          # final-norm rstd
        self.gh   = Z((Md, H), np.uint16)                           # post-final-norm hidden (bf16) for logits
        self.glog = Z((Md, V), np.float32)                          # logits
        self.glp  = Z((Md, 1), np.float32)                          # logprob
        self.glse = Z((Md, 1), np.float32)                          # logsumexp
        self.glab = Z((Md, 1), np.int32)                            # labels

    def _gather(self, toks, si):
        """toks (Bp,) ints → gin rows [0:Bp] = embed[tok] (fp32 residual, == ResidentModel's GEMM-onehot
        gather bitwise: both are f32(bf16(embed[tok]))). Async upload from a persistent host buffer."""
        self._ginh = np.zeros((self.Bp, self.H), np.float32)
        for b, t in enumerate(toks):
            self._ginh[b] = (self.embed_bf[int(t)].astype(np.uint32) << 16).view(np.float32)
        cdrv.cuMemcpyHtoDAsync(self.gin.ptr, np.ascontiguousarray(self._ginh), self.Bp * self.H * 4, si)

    def decode_step(self, toks, pos: int, si: int):
        """toks (Bp,) ints, sequence position `pos` → frontier hidden (Md,H) f32 (rows[0:Bp] real).
        Appends each token's K/V to every layer's cache at `pos`."""
        self._gather(toks, si)
        x = self.gin
        for l in self.layers:
            x = l.forward(x, pos, si)
        return x

    def _logits(self, hidden, si):
        """Final RMSNorm + tied-head logits → self.glog (Md,V) f32, ALL ON DEVICE (no glog DtoH). Mirrors
        ResidentModel's boundary kernel-for-kernel so the frontier-row logits are bitwise-equal to training.
        The final norm runs on-device with PERSISTENT buffers (NOT host rmsnorm_forward — that alloc-churn
        RACES the decode kernels, [[batch-invariance]]); the only host op is the bf16 TRUNCATION round of the
        pre-norm residual (== rmsnorm_forward's f32_to_bf16_bits), uploaded to a persistent buffer."""
        Md, H, V = self.Md, self.H, self.V
        # NOTE: the residual bf16 round stays HOST — SOLVED (2026-06-11, resident_moe_model device
        # final-norm work): norm.py's f32_to_bf16_bits is RNE, NOT truncation (attention.py's is) —
        # the old `_trunc_bf16` swap broke ratio=1 because of that 1-ulp ROUNDING-MODE mismatch
        # (~47% of elements), never a sync/race. The correct device replacement is fused._cast_bf16
        # (RNE) — proven bitwise-equal end-to-end in ResidentMoEModel. Wire it here when decode
        # perf matters again ([[grpo-step-profile]]).
        cudart.cudaStreamSynchronize(si)
        self._xbf = f32_to_bf16_bits(hidden.to_numpy())            # (Md,H) bf16 bits of the residual (== RM)
        cdrv.cuMemcpyHtoDAsync(self.gxbf.ptr, self._xbf, self.gxbf.nbytes, si)
        ct.launch(si, (Md // NTM, 1, 1), _rmsnorm_stats, (self.gxbf, self.grfn, H // TH, 1.0 / H, float(self.eps)))
        ct.launch(si, (Md // NTM, 1, 1), _rmsnorm_apply, (self.gxbf, self.gwfn, self.grfn, self.gh, H // TH))
        ct.launch(si, (Md // 128, V // 128, 1), _gemm_nt_f32, (self.gh, self.gembed, self.glog, H // 64, 128, 128, 64))

    def _score(self, hidden, labels, si):
        """Teacher-forced logprob of `labels` (≤Bp,). Transfers only glp (Md floats), NOT the (Md,V) logits
        — the 78 MB glog DtoH was ~half the per-step boundary cost ([[grpo-step-profile]])."""
        Md, V, Bp = self.Md, self.V, self.Bp
        self._logits(hidden, si)
        self._lab = np.zeros((Md, 1), np.int32); self._lab[:Bp, 0] = np.asarray(labels, np.int32)
        cdrv.cuMemcpyHtoDAsync(self.glab.ptr, self._lab, self.glab.nbytes, si)
        ct.launch(si, (Md // CTM, 1, 1), _ce_stats, (self.glog, self.glab, self.glp, self.glse, V // TV))
        cudart.cudaStreamSynchronize(si)
        return self.glp.to_numpy()

    def _generate_step(self, hidden, si):
        """Greedy next token + its logprob, ALL ON DEVICE: _argmax_id picks argmax_v logits → glab, then
        _ce_stats(glab) gives that token's logprob (bitwise-equal to training's CE on the same token).
        Transfers only Md ints + Md floats (no (Md,V) DtoH). Returns (ids[:Bp], logprob[:Bp])."""
        Md, V, Bp = self.Md, self.V, self.Bp
        self._logits(hidden, si)
        ct.launch(si, (Md // CTM, 1, 1), _argmax_id, (self.glog, self.glab, V // TV))                 # argmax → glab
        ct.launch(si, (Md // CTM, 1, 1), _ce_stats, (self.glog, self.glab, self.glp, self.glse, V // TV))
        cudart.cudaStreamSynchronize(si)
        return self.glab.to_numpy().reshape(Md)[:Bp].copy(), self.glp.to_numpy().reshape(Md)[:Bp].copy()

    def score(self, ids, labels, si: int):
        """Teacher-forced (Bp=1): decode ids[t] one at a time, return per-position logprob of labels[t]
        (S,). BITWISE-equal to ResidentModel(B=1,S).forward+loss_backward's per-token logprob."""
        ids = np.asarray(ids).reshape(-1); labels = np.asarray(labels).reshape(-1)
        S = ids.size; lps = np.zeros(S, np.float32)
        for t in range(S):
            hidden = self.decode_step([int(ids[t])], t, si)
            lp = self._score(hidden, [int(labels[t])], si)
            lps[t] = lp[0, 0]
        return lps

    def generate(self, prompts, n_new: int, si: int):
        """Greedy autoregressive generation, BATCHED over Bp sequences in ONE Md=128 tile (B≤128 decode
        in the same wall-time as 1 → pack Bp to amortize the fixed per-token decode cost, [[grpo-step-profile]]).
        prompts (P,) [Bp=1] or (Bp, P) [same length]. Returns (gen_ids, gen_logprob) shaped (n_new,) for a
        1-D prompt else (Bp, n_new). gen_logprob[b,i] is the on-policy rollout logprob of the generated token
        — BITWISE-equal to what training assigns to that token."""
        prompts = np.asarray(prompts); one = prompts.ndim == 1
        if one: prompts = prompts[None, :]
        Bp, P = prompts.shape
        assert Bp == self.Bp, f"prompt batch {Bp} != model Bp {self.Bp}"
        gen = np.zeros((Bp, n_new), np.int64); glp = np.zeros((Bp, n_new), np.float32)
        nxt = None; gi = 0
        for t in range(P + n_new - 1):
            toks = prompts[:, t] if t < P else nxt
            hidden = self.decode_step(np.asarray(toks, np.int64), t, si)
            ids, lp = self._generate_step(hidden, si)
            nxt = ids
            if t >= P - 1:                                          # from the last prompt token on, predict a new token
                gen[:, gi] = ids; glp[:, gi] = lp; gi += 1
        return (gen[0], glp[0]) if one else (gen, glp)

    def free(self):
        for l in self.layers:
            l.free()
        for o in (self.gembed, self.gwfn, self.gin, self.gxbf, self.grfn, self.gh,
                  self.glog, self.glp, self.glse, self.glab):
            o.free()
