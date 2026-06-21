"""
ancora/model/resident_moe_decode.py — ResidentMoEDecodeModel: the device-resident ROLLOUT engine
for the MoE family (the decode side of GRPO; Qwen3 counterpart: resident_decode.py).

THE FOUR REQUIREMENTS:
  • RESIDENT — persistent buffers, per-layer KV-cache, no per-step alloc (no churn race).
  • RATIO=1  — kernel-for-kernel mirror of the TRAINING forward: global=NoPE+full-causal decode,
    local=RoPE@pos+windowed decode (bitwise == _attn_fwd/_attn_fwd_win rows), MoE FFN = weight-
    shared GroupedMoEFFN (device_route, same route as training), boundary = _cast_bf16(RNE)+
    _rmsnorm_*+_gemm_nt_f32 == ResidentMoEModel._fwd_dev. Decode hidden at t BITWISE == prefill.
  • SHARED WEIGHTS — every weight buffer ALIASES the trainer's _DBuf (proj/norm, packed expert
    Wg/Wu/Wd, device router Wr_dev, tied embed, final-norm gain): device AdamW updates in place ⇒
    rollout reads post-step weights with ZERO copy.
  • EFFICIENT — all four levers:
      (1) Bp≤128 sequences in ONE Md=128 GEMM tile (batch ~free);
      (2) local layers: O(window) attention + a RING KV-cache (pow2 blocks; memory O(window),
          not O(maxS) — the values the window reads are identical to a full cache ⇒ bitwise);
      (3) DEVICE-POSITION kernels (pos in a (1,1) i32 buffer, advanced in-stream by _inc1; scalar
          store-index/bit-ops/loop-bounds probed OK) ⇒ the WHOLE token step is position-free pure
          launches → captured ONCE into a CUDA GRAPH and replayed per token (1 launch/token);
      (4) device closed loop: pick → glab → _embed_gather feeds the next step; ids/lp stream to
          device history (DtoDAsync) ⇒ ONE sync per ROLLOUT;
      (5) DECODE MEGAKERNEL (2026-06-12): fewer, fatter kernels — every fusion probed BITWISE
          (tests/kernels/_probe_decode_{tiles,fused,attn}.py) before wiring:
            • projection GEMMs at DTN=32 (N=1024 → 32 blocks, not 8; _gemm_bf16 is
              TN-bitwise-invariant);
            • o_proj = _gemm_af32_res (f32→bf16 cast + GEMM + residual, ONE kernel); dense
              down = _gemm_bf16_res (GEMM + residual);
            • MoE FFN = forward_resident_dec: _ggemm_gus (gate+up+SwiGLU fused, Gg/Ug never
              hit HBM) + _ggemm_b(TN=64) + _combine_rf32 (combine + residual);
            • pick+CE = _argmax_ce_b / _sample_ce_b: ONE pass over the 78 MB logits (2.3×);
            • hidden norms at DNTM=8 row tile (16 blocks; TMb=4 flips bits — rejected), vocab
              stream at DCTM=4 (CTMb≤2 flips bits — rejected);
            • q/k norm + RoPE run on the REAL Bp·H* rows only (row-independent; the Md-pad
              rows feed nothing).
          REJECTED by measurement/risk: GQA-paired attention (bitwise ✓ but 2× register state
          → SLOWER, the training ping-pong lesson again), vocab-GEMM TN<128 (slower), rope
          refactors (x·cos−y·sin is FMA-contraction territory — the 1-ULP dQ precedent).
          Real size: 8.6 → 5.4 ms/step graph (Bp=32, 5890 tok/s); Bp=64 → 9410 tok/s.
    Sampling: Gumbel-max with a coord-hash RNG (the SR-grad recipe; seed in device memory,
    incremented in-graph) — batch-invariant, deterministic given the seed, graph-replayable.

  eng = ResidentMoEDecodeModel(train_model, Bp, maxS, si)
  ids, lp = eng.generate(prompts, n_new, si, so=so, dev=dev, use_graph=True)          # greedy
  ids, lp = eng.generate(..., sample=True, temperature=1.0, seed=k)                   # sampled
  lps     = eng.score(ids, labels, si)                                                # teacher-forced
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.resident import _DBuf
from ancora.model.resident_decode import ResidentDecodeLayer, MGEMM
from ancora.model.resident_moe import _shim_cfg
from ancora.kernels.moe import GroupedMoEFFN
from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, _rmsnorm_stats_f32_b,
                                  _rmsnorm_apply_f32_b, TM as NTM, TH)
from ancora.kernels.activation import _swiglu_fwd, TM as STM, TI
from ancora.kernels.rope import _rope_fwd_dec_p, RTM as RRTM
from ancora.kernels.attention import (_attn_decode_blk_pd, _attn_decode_blk_win_pd, _scatter_blk_p,
                                       _gather_blk_p, _append_kv_p, BQ, BKV)
from ancora.kernels.fused import (_cast_bf16, _residual_add_rf32, _gemm_bf16, _gemm_bf16_res,
                                  _gemm_af32_res, _gemm_nt_f32, _embed_gather, _inc1, RTM, RTN)
from ancora.kernels.loss import _ce_stats_b, _argmax_ce_b, _sample_ce_b, TV, GTM, GTK

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NORM = ["input_ln", "post_ln", "q_norm", "k_norm"]
NOMASK = (1 << 30) - 1     # `& NOMASK` is the identity for any position — full-cache mode
DCTM = 4                   # vocab-kernel row tile: Md=128 → 32 blocks (CTM=64 gave 2, 8 gave 16 —
                           # SMs idle on the 39MB V=151936 stream; CTMb=4 is the smallest BITWISE
                           # row tile, CTMb≤2 flips the in-tile reduce — _probe_decode_tiles.py)
DTN = 32                   # decode projection-GEMM column tile: N=1024 → 32 blocks instead of 8
                           # (2× at Md=128; _gemm_bf16 is BITWISE-invariant to TN — same probe)
DNTM = 8                   # decode hidden-norm row tile: Md=128 → 16 blocks (TMb=8 BITWISE,
                           # TMb=4 flips bits — _probe_decode_fused.py)


class ResidentMoEDecodeLayer(ResidentDecodeLayer):
    """Decode layer for the MoE family: local(RoPE+window+RING)/global(NoPE+full) attention,
    dense or grouped-MoE FFN — ALL WEIGHTS ALIASED from the training layer; position from device."""
    def __init__(self, cfg, tl, Bp, maxS, si, window=512):
        is_global = tl.is_global
        ffn_dense = not hasattr(tl, "moe")
        mx = bool(getattr(tl, "mxfp8", False))             # mirror the TRAINER's precision (ratio=1)
        dummy = {n: np.zeros(tl.w[n].shape, np.float32) for n in PROJ}
        dummy |= {n: np.zeros(tl.wn[n].shape[1], np.float32) for n in NORM}
        super().__init__(_shim_cfg(cfg, is_global), dummy, Bp, maxS, mxfp8=mx)
        for n in PROJ: self.w[n].free()
        for n in NORM: self.wn[n].free()
        self.w, self.wn = tl.w, tl.wn                      # ALIAS — AdamW updates visible zero-copy
        self._tl = tl
        if mx:                                             # ALIAS the trainer's fp8 weights too — the
            for n in PROJ:                                 # trainer's _wq_dirty stays the ONE authority
                self.wf8[n].free(); self.ws8[n].free()     # (its step() dirties; whoever runs first
            self.wf8, self.ws8 = tl.wf8, tl.ws8            # requants the SAME buffers → same bytes)
        self.is_global, self.ffn_dense = is_global, ffn_dense
        self.win_blocks = 0 if is_global else window // BKV
        if is_global:                                      # full cache: every position lives at its row
            self.CROWS, self.CMASK, self.NRB, self.SMASK = maxS, NOMASK, self.NKVB, NOMASK
        else:                                              # RING cache: pow2 blocks ≥ win_blocks+1
            rb2 = 1
            while rb2 < self.win_blocks + 1: rb2 <<= 1
            self.NRB, self.SMASK = rb2, rb2 - 1
            self.CROWS, self.CMASK = rb2 * BKV, rb2 * BKV - 1
            self.Kc.free(); self.Vc.free()                 # shrink the parent's maxS cache to O(window)
            self.Kc = _DBuf.zeros((Bp * self.Hkv * self.CROWS, self.Dh))
            self.Vc = _DBuf.zeros((Bp * self.Hkv * self.CROWS, self.Dh))
        if not ffn_dense:
            tm = tl.moe
            assert tm.device_route, "training MoE layers must run device_route=True (rollout and " \
                                    "training must use the SAME route implementation for ratio=1)"
            if not tm._packed: tm._pack(si)                # ensure the trainer's weight buffers exist
            if not hasattr(tm, "Wr_dev"): tm._prealloc(tl.M)
            if not getattr(tm, "_router_dev_adam", False):  # device AdamW not up yet → seed Wr_dev
                self._wrk = tm.w["router"].astype(np.float32)
                cdrv.cuMemcpyHtoDAsync(tm.Wr_dev._ptr, self._wrk, self._wrk.nbytes, si)
            m = GroupedMoEFFN(tm.w, tm.k, device_route=True, norm_topk=tm.norm_topk, mxfp8=mx)
            m._prealloc(self.Md)                           # own Md=128 activation scratch
            m.Wr_dev.free(); m.Wr_dev = tm.Wr_dev          # ALIAS the device router weights
            for nm in ("Wg", "Wu", "Wd", "WdT", "WgT", "WuT"):
                setattr(m, nm, getattr(tm, nm))            # ALIAS the packed expert weights
            m._packed = True                               # never pack (stale host would clobber)
            m._router_dev_adam = True                      # never re-seed Wr_dev from the host master
            if mx:                                         # ALIAS the trainer's fp8 expert weights —
                tm._ensure_mx()                            # the trainer's dirty flag is the authority
                for nm in ("Wgq", "Wgs", "Wuq", "Wus", "Wdq", "Wds"):
                    setattr(m, nm, getattr(tm, nm))
                m._mx_master = tm
            self.moe = m
        # head-norm/RoPE row counts: only rows [0, Bp·H*) are real (scatter/append read b<Bp and
        # nothing else consumes gqn/gkn/gqr/gkr) → run them on the REAL rows when tile-aligned
        # (4× less work at Bp=32/Md=128; per-row kernels ⇒ real rows bitwise-unchanged)
        # Mq/Mk drive BOTH the norm grid (//NTM) and the RoPE grid (//RRTM) → require divisibility
        # by both (NTM==RRTM==32 today, but don't silently skip real rows if they ever diverge)
        _ok = lambda r: r % NTM == 0 and r % RRTM == 0
        self.Mq = Bp * self.Hq if _ok(Bp * self.Hq) else self.Md * self.Hq
        self.Mk = Bp * self.Hkv if _ok(Bp * self.Hkv) else self.Md * self.Hkv

    # ── decode-megakernel overrides (2026-06-11): fewer, fatter kernels ──────
    def _gemm(self, si, A, wname, C, K, N):
        """Projection GEMM at the decode column tile DTN=32 (N=1024 → 32 blocks, not 8).
        Same _gemm_bf16 kernel — BITWISE-invariant to TN (probed). MXFP8 mode defers to the
        parent (quant-once activation + mma_scaled at the training mxfp8_tile — ratio=1)."""
        if self.mxfp8:
            return super()._gemm(si, A, wname, C, K, N)
        ct.launch(si, (self.Md // GTM, N // DTN, 1), _gemm_bf16,
                  (A, self.w[wname], C, K // GTK, GTM, DTN, GTK))

    def _rms(self, si, xb, wb, rstd, yb, rows, hh, xf32=False):
        """Hidden-state norms (xf32, Md=128 rows) use the TMb=8 row-tile variants (16 blocks,
        BITWISE — probed); the head norms (bf16, ≥1024 rows) keep the parent path."""
        if xf32:
            ct.launch(si, (rows // DNTM, 1, 1), _rmsnorm_stats_f32_b,
                      (xb, rstd, hh // TH, 1.0 / hh, self.eps, DNTM))
            ct.launch(si, (rows // DNTM, 1, 1), _rmsnorm_apply_f32_b,
                      (xb, wb, rstd, yb, hh // TH, DNTM))
        else:
            super()._rms(si, xb, wb, rstd, yb, rows, hh)

    def forward(self, gx, gpos, si: int):
        """gx (Md,H) f32 residual (rows [0:Bp) real), position from the DEVICE buffer gpos →
        gout (Md,H) f32, real rows BITWISE == training prefill at that position. Pure launches."""
        Md, Hq, Hkv, Dh, H, I, qd, kd = self.Md, self.Hq, self.Hkv, self.Dh, self.H, self.I, self.qd, self.kd
        Bp, V = self.Bp, self._V
        hd = Dh // 2
        if self.mxfp8:                                     # trainer's step() dirtied the weights →
            if self._tl._wq_dirty: self._tl._requant_w(si) # requant the SHARED fp8 buffers (one authority)
            self._qsrc.clear()
        # ── attention ──
        self._rms(si, gx, self.wn["input_ln"], self.r1, self.gh, Md, H, xf32=True)
        self._gemm(si, self.gh, "q_proj", self.gq, H, qd)
        self._gemm(si, self.gh, "k_proj", self.gk, H, kd)
        self._gemm(si, self.gh, "v_proj", self.gv, H, kd)
        self._rms(si, V(self.gq, (Md*Hq, Dh)), self.wn["q_norm"], self.rq, V(self.gqn, (Md*Hq, Dh)), self.Mq, Dh)
        self._rms(si, V(self.gk, (Md*Hkv, Dh)), self.wn["k_norm"], self.rk, V(self.gkn, (Md*Hkv, Dh)), self.Mk, Dh)
        if self.is_global:                                 # NoPE
            qsrc, ksrc = self.gqn, self.gkn
        else:                                              # LOCAL: device-position RoPE (theta_local)
            ct.launch(si, (self.Mq // RRTM, 1, 1), _rope_fwd_dec_p,
                      (V(self.gqn, (Md*Hq, Dh)), self.gcos, self.gsin, V(self.gqr, (Md*Hq, Dh)), gpos, hd))
            ct.launch(si, (self.Mk // RRTM, 1, 1), _rope_fwd_dec_p,
                      (V(self.gkn, (Md*Hkv, Dh)), self.gcos, self.gsin, V(self.gkr, (Md*Hkv, Dh)), gpos, hd))
            qsrc, ksrc = self.gqr, self.gkr
        ct.launch(si, (Bp, Hkv, 1), _append_kv_p,
                  (V(ksrc, (Md*Hkv, Dh)), V(self.gv, (Md*Hkv, Dh)), self.Kc, self.Vc, gpos,
                   self.CROWS, Hkv, self.CMASK))
        ct.launch(si, (Bp*Hq, 1, 1), _scatter_blk_p, (V(qsrc, (Md*Hq, Dh)), self.blockQ, gpos, BQ))
        if self.is_global:
            ct.launch(si, (Bp*Hq, 1, 1), _attn_decode_blk_pd,
                      (self.blockQ, self.Kc, self.Vc, self.blockO, gpos, self.NKVB, Hq, Hkv, self.scale))
        else:
            ct.launch(si, (Bp*Hq, 1, 1), _attn_decode_blk_win_pd,
                      (self.blockQ, self.Kc, self.Vc, self.blockO, gpos, self.NRB, Hq, Hkv, self.scale,
                       self.win_blocks, self.SMASK))
        ct.launch(si, (Bp*Hq, 1, 1), _gather_blk_p, (self.blockO, self.gO, gpos, BQ))
        if self.mxfp8:    # mirror the trainer kernel-for-kernel: cast → quant+mma_scaled → residual
            ct.launch(si, (Md*Hq // RTM, Dh // RTN, 1), _cast_bf16,
                      (V(self.gO, (Md*Hq, Dh)), V(self.gotok, (Md*Hq, Dh))))
            self._gemm(si, self.gotok, "o_proj", self.gattn, qd, H)
            ct.launch(si, (Md // RTM, H // RTN, 1), _residual_add_rf32, (gx, self.gattn, self.gx2))
        else:             # o_proj MEGAKERNEL: f32→bf16 cast of gO + GEMM + residual add in ONE launch
            ct.launch(si, (Md // GTM, H // DTN, 1), _gemm_af32_res,
                      (V(self.gO, (Md, qd)), self.w["o_proj"], gx, self.gx2, qd // GTK, GTM, DTN, GTK))
        # ── FFN ──
        self._rms(si, self.gx2, self.wn["post_ln"], self.r2, self.gh2, Md, H, xf32=True)
        if self.ffn_dense:
            self._gemm(si, self.gh2, "gate_proj", self.gg, H, I)
            self._gemm(si, self.gh2, "up_proj", self.gu, H, I)
            ct.launch(si, (Md // STM, I // TI, 1), _swiglu_fwd, (self.gg, self.gu, self.ga))
            if self.mxfp8:
                self._gemm(si, self.ga, "down_proj", self.gmlp, I, H)
                ct.launch(si, (Md // RTM, H // RTN, 1), _residual_add_rf32, (self.gx2, self.gmlp, self.gout))
            else:         # down MEGAKERNEL: GEMM + residual add in ONE launch
                ct.launch(si, (Md // GTM, H // DTN, 1), _gemm_bf16_res,
                          (self.ga, self.w["down_proj"], self.gx2, self.gout, I // GTK, GTM, DTN, GTK))
        else:
            # MoE MEGAKERNEL path: gate+up+SwiGLU fused (mx variant under mxfp8), combine+residual fused
            self.moe.forward_resident_dec(self.gh2, self.gx2, self.gout, si)
        return self.gout


class ResidentMoEDecodeModel:
    """The rollout engine: ResidentMoEDecodeLayer stack over the TRAINING model's layers (weights
    aliased), device-position decode loop, optional one-token CUDA-graph replay + Gumbel sampling."""
    def __init__(self, train, Bp: int, maxS: int, si: int):
        cfg = train.cfg
        self.cfg, self.Bp, self.maxS, self.V, self.H, self.eps = cfg, Bp, maxS, train.V, train.H, cfg.eps
        H, V, Md = self.H, self.V, MGEMM
        self.Md, self.NL = Md, len(train.layers)
        self.layers = [ResidentMoEDecodeLayer(cfg, tl, Bp, maxS, si, window=cfg.window)
                       for tl in train.layers]
        self.gembed = train.gembed                          # ALIAS (tied embed = LM head)
        self.gwfn = train.gfnw                              # ALIAS (final-norm gain)
        Z = _DBuf.zeros
        self.gin  = Z((Md, H), np.float32)
        self.gxbf = Z((Md, H), np.uint16)
        self.grfn = Z((Md, 1), np.float32)
        self.gh   = Z((Md, H), np.uint16)
        self.glog = Z((Md, V), np.float32)
        self.glp  = Z((Md, 1), np.float32); self.glse = Z((Md, 1), np.float32)
        self.glab = Z((Md, 1), np.int32)                    # input ids / argmax-sample out / ce labels
        self.gpos = Z((1, 1), np.int32)                     # DEVICE position (graph-replayable)
        self.gseed = Z((1, 1), np.int32)                    # sampling seed (incremented in-graph)
        self.ghist_i = Z((maxS * Md, 1), np.int32)
        self.ghist_l = Z((maxS * Md, 1), np.float32)
        self._graphs = {}

    # ── device-side pieces (pure launches — graph-capturable) ────────────────
    def _step_dev(self, si: int):
        Md, H, V = self.Md, self.H, self.V
        ct.launch(si, (Md, 1, 1), _embed_gather, (self.glab, self.gembed, self.gin, H // 128))
        x = self.gin
        for l in self.layers:
            x = l.forward(x, self.gpos, si)
        ct.launch(si, (Md // RTM, H // RTN, 1), _cast_bf16, (x, self.gxbf))
        ct.launch(si, (Md // NTM, 1, 1), _rmsnorm_stats, (self.gxbf, self.grfn, H // TH, 1.0 / H, float(self.eps)))
        ct.launch(si, (Md // NTM, 1, 1), _rmsnorm_apply, (self.gxbf, self.gwfn, self.grfn, self.gh, H // TH))
        ct.launch(si, (Md // 128, V // 128, 1), _gemm_nt_f32, (self.gh, self.gembed, self.glog, H // 64, 128, 128, 64))

    def _token_step(self, si: int, sample: bool, temperature: float):
        """ONE decode token: embeds→layers→logits→pick(argmax|gumbel)→lp→pos++. Position-free
        (reads/advances gpos on device) ⇒ capturable once, replayable every position."""
        Md, V = self.Md, self.V
        self._step_dev(si)
        # fused pick+CE: ONE pass over the 78 MB logits instead of two (pick kernel + _ce_stats_b)
        # — BITWISE == the two-kernel path (probed, _probe_decode_attn.py)
        if sample:
            ct.launch(si, (Md // DCTM, 1, 1), _sample_ce_b,
                      (self.glog, self.gseed, self.glab, self.glp, self.glse,
                       V // TV, float(1.0 / temperature), DCTM))
            ct.launch(si, (1, 1, 1), _inc1, (self.gseed,))
        else:
            ct.launch(si, (Md // DCTM, 1, 1), _argmax_ce_b,
                      (self.glog, self.glab, self.glp, self.glse, V // TV, DCTM))
        ct.launch(si, (1, 1, 1), _inc1, (self.gpos,))

    def capture(self, dev, sample: bool = False, temperature: float = 1.0):
        """Capture ONE token step into a CUDA graph (per pick-mode). Run a warm direct rollout
        first (JIT + MoE pack/prealloc). Replay = 1 graph launch per token instead of ~20·NL."""
        key = ("s", float(temperature)) if sample else "g"
        gb = dev.create_graph_builder(); gb.begin_building()
        self._token_step(int(gb.__cuda_stream__()[1]), sample, temperature)
        gb.end_building()
        self._graphs[key] = gb.complete()
        return key

    # ── host-side plumbing ────────────────────────────────────────────────────
    def _mx_refresh(self, si: int):
        """MXFP8: requantize the shared fp8 weights ONCE per rollout if the trainer's AdamW
        dirtied them — so a captured GRAPH (which cannot re-check the host dirty flag) always
        replays over fresh bytes. No-op when clean / in BF16 mode."""
        for l in self.layers:
            if l.mxfp8:
                if l._tl._wq_dirty:
                    l._tl._requant_w(si)
                if not l.ffn_dense:
                    l.moe._mx_ready(si)

    def _put_ids(self, toks, si: int):
        buf = np.zeros((self.Md, 1), np.int32); buf[:self.Bp, 0] = np.asarray(toks, np.int32)
        self._keep.append(buf)
        cdrv.cuMemcpyHtoDAsync(self.glab.ptr, buf, self.glab.nbytes, si)

    def _reset(self, si: int, seed: int = 0):
        self._keep = [np.zeros((1, 1), np.int32), np.array([[seed]], np.int32)]
        cdrv.cuMemcpyHtoDAsync(self.gpos.ptr, self._keep[0], 4, si)
        cdrv.cuMemcpyHtoDAsync(self.gseed.ptr, self._keep[1], 4, si)

    # ── rollout APIs ─────────────────────────────────────────────────────────
    def generate(self, prompts, n_new: int, si: int, so=None, dev=None, use_graph: bool = False,
                 sample: bool = False, temperature: float = 1.0, seed: int = 0):
        """Autoregressive generation (greedy or Gumbel-sampled), Bp sequences per Md=128 tile,
        device closed loop (ONE sync per rollout). use_graph=True → one captured graph launch per
        token (pass so=Stream object + dev=Device; warm with a direct rollout first). Returns
        (ids, logprob); logprob is the on-policy rollout lp — BITWISE == training's."""
        prompts = np.asarray(prompts); one = prompts.ndim == 1
        if one: prompts = prompts[None, :]
        Bp, P = prompts.shape
        assert Bp == self.Bp and P + n_new <= self.maxS
        Md = self.Md
        self._mx_refresh(si)
        self._reset(si, seed); gi = 0
        if use_graph:
            key = ("s", float(temperature)) if sample else "g"
            if key not in self._graphs:
                self.capture(dev, sample, temperature)
        for t in range(P + n_new - 1):
            if t < P:
                self._put_ids(prompts[:, t], si)
            if use_graph:
                self._graphs[key].launch(so)
            else:
                self._token_step(si, sample, temperature)
            if t >= P - 1:
                cdrv.cuMemcpyDtoDAsync(int(self.ghist_i.ptr) + gi * Md * 4, self.glab.ptr, Md * 4, si)
                cdrv.cuMemcpyDtoDAsync(int(self.ghist_l.ptr) + gi * Md * 4, self.glp.ptr, Md * 4, si)
                gi += 1
        cudart.cudaStreamSynchronize(si)                    # the ONE rollout sync
        self._keep = []
        ids = self.ghist_i.to_numpy().reshape(self.maxS, Md)[:gi, :Bp].T.astype(np.int64)
        lps = self.ghist_l.to_numpy().reshape(self.maxS, Md)[:gi, :Bp].T
        return (ids[0], lps[0]) if one else (ids, lps)

    def score(self, ids, labels, si: int):
        """Teacher-forced per-position logprob of labels[t] (direct path — argmax must not
        clobber the labels, so no graph). ONE sync. BITWISE == the training model's glp."""
        ids = np.asarray(ids); one = ids.ndim == 1
        if one: ids = ids[None, :]; labels = np.asarray(labels)[None, :]
        Bp, S = ids.shape
        assert Bp == self.Bp and S <= self.maxS
        Md, V = self.Md, self.V
        self._mx_refresh(si)
        self._reset(si)
        for t in range(S):
            self._put_ids(ids[:, t], si)
            self._step_dev(si)
            self._put_ids(labels[:, t], si)
            ct.launch(si, (Md // DCTM, 1, 1), _ce_stats_b, (self.glog, self.glab, self.glp, self.glse, V // TV, DCTM))
            cdrv.cuMemcpyDtoDAsync(int(self.ghist_l.ptr) + t * Md * 4, self.glp.ptr, Md * 4, si)
            ct.launch(si, (1, 1, 1), _inc1, (self.gpos,))
        cudart.cudaStreamSynchronize(si)
        self._keep = []
        lps = self.ghist_l.to_numpy().reshape(self.maxS, Md)[:S, :Bp].T
        return lps[0] if one else lps
