"""
ancora/model/resident_prefix.py — DEVICE-RESIDENT prefix-shared decoder layer for GRPO.

Completes the prefix-sharing stack: the host-glue prefix_layer (rl/prefix_resident.py) proved the
math (suffix forward bitwise == naive, grads training-equivalent) but its per-token ops are
self-allocating host helpers — the documented alloc-churn race made its tests need process
isolation. Here EVERYTHING runs on persistent buffers (the real fix): per-token ops are inherited
verbatim from ResidentMoEDenseLayer (they are row-independent over M = Sp + G·Sc), and only the
sequence-structure-aware pieces are overridden for the ragged [prompt(Sp); G·suffix(Sc)] layout:

  • tok↔head transposes — TWO launches each: prompt (B=1,S=Sp) into head-major rows [0, H_·Sp),
    suffix (B=G,S=Sc) into rows [H_·Sp, …) via _DBuf.at_pos row-offset views. The kernels are
    plain tile copies, so a row-offset view is all they need.
  • RoPE (local layers) — prompt at positions 0..Sp-1 (table rows [0,Sp)) and suffix at the
    OFFSET positions Sp..Sp+Sc-1 (pre-sliced gcos_s/gsin_s tables), same _rope_fwd kernel.
    Global layers are NoPE (no rotation) — per the MoE architecture.
  • attention — the prefix kernels on regions of the SAME persistent buffers: prompt self-attn
    (_attn_fwd[_win]) on rows [0, Hq·Sp), suffix (_attn_fwd_prefix[_win]) reading the shared
    prompt K/V + its own suffix K/V. Backward = the 5-kernel prefix decomposition; prompt
    dK/dV = self + Σ_G cross via an f32 _add_f32 (bitwise == the host helper's f32 add).

Layout invariant: head-major buffers (M·H_, Dh) hold [prompt-block (H_·Sp rows); suffix-block
(G·H_·Sc rows)] — exactly the layouts the (validated) prefix kernels expect, so no data movement
beyond the split transposes. Forward/backward FFN, AdamW step(): inherited unchanged.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct

from ancora.model.resident_moe import ResidentMoEDenseLayer
from ancora.model.resident import _DBuf
from ancora.kernels.attention import (_attn_fwd, _attn_fwd_win, _attn_bwd_dq, _attn_bwd_dkdv,
                                      _attn_bwd_dq_win, _attn_bwd_dkdv_win,
                                      _attn_fwd_prefix, _attn_bwd_dq_prefix, _attn_bwd_dkdv_prefix,
                                      _attn_fwd_prefix_win, _attn_bwd_dq_prefix_win,
                                      _attn_bwd_dkdv_prefix_win, BQ, BKV)
from ancora.kernels.rope import _rope_fwd, _rope_bwd, RTM as RRTM, build_cos_sin
from ancora.kernels.fused import (_tok_to_head, _head_to_tok, _head_to_tok_f32, _residual_add_rf32,
                                  _attn_delta, _cast64, _add_f32, RTM, RTN, TT, DTM)


class ResidentPrefixDenseLayer(ResidentMoEDenseLayer):
    """Device-resident prefix-shared dense layer (local or global)."""
    def __init__(self, cfg, weights, Sp, Sc, G, is_global, window=512, lid=0, mxfp8=False,
                 optimizer="adamw", muon_scratch=None, muon_exclude=None):
        assert Sp % TT == 0 and Sc % TT == 0 and (Sp + G * Sc) % 128 == 0
        super().__init__(cfg, weights, 1, Sp + G * Sc, is_global, window, lid, mxfp8=mxfp8,
                         optimizer=optimizer, muon_scratch=muon_scratch, muon_exclude=muon_exclude)
        self.Sp, self.Sc, self.Gc = Sp, Sc, G
        self.NQBp, self.NKVBp = Sp // BQ, Sp // BKV
        self.NQBs, self.NKVBs = Sc // BQ, Sc // BKV
        self.NSBp, self.NSBs = Sp // TT, Sc // TT
        # rope tables: prompt = rows [0,Sp) of the full table; suffix = rows [Sp,Sp+Sc) pre-sliced
        # (positions are absolute, so slicing == the host _rope_seg offset path → bitwise)
        for b in (self.gcos, self.gsin): b.free()              # replace the (M-sized) parent tables
        cosv, sinv = build_cos_sin(Sp + Sc, cfg.head_dim, cfg.rope_theta_local)
        self.gcos = _DBuf(np.ascontiguousarray(cosv[:Sp])); self.gsin = _DBuf(np.ascontiguousarray(sinv[:Sp]))
        self.gcos_s = _DBuf(np.ascontiguousarray(cosv[Sp:])); self.gsin_s = _DBuf(np.ascontiguousarray(sinv[Sp:]))
        # prompt CROSS dK/dV accumulators (Σ_G over suffix queries); prompt total = self + cross
        self.gdkc = _DBuf.zeros((self.Hkv * Sp, self.Dh), np.float32)
        self.gdvc = _DBuf.zeros((self.Hkv * Sp, self.Dh), np.float32)

    # ── split tok↔head transposes: prompt (B=1,S=Sp) + suffix (B=G,S=Sc) ──
    def _t2h(self, si, tok, head, Hh):
        ct.launch(si, (Hh, self.NSBp, 1), _tok_to_head, (tok, head, Hh, self.NSBp))
        ct.launch(si, (self.Gc * Hh, self.NSBs, 1), _tok_to_head,
                  (tok.at_pos(self.Sp), head.at_pos(Hh * self.Sp), Hh, self.NSBs))

    def _h2t(self, si, head, tok, Hh, kern):
        ct.launch(si, (Hh, self.NSBp, 1), kern, (head, tok, Hh, self.NSBp))
        ct.launch(si, (self.Gc * Hh, self.NSBs, 1), kern,
                  (head.at_pos(Hh * self.Sp), tok.at_pos(self.Sp), Hh, self.NSBs))

    def _rope(self, si, src, dst, Hh, kern):
        ct.launch(si, (self.Sp // RRTM, Hh, 1), kern,
                  (src, self.gcos, self.gsin, dst, self.Sp // RRTM, self.Dh // 2))
        ct.launch(si, (self.Sc // RRTM, self.Gc * Hh, 1), kern,
                  (src.at_pos(Hh * self.Sp), self.gcos_s, self.gsin_s, dst.at_pos(Hh * self.Sp),
                   self.Sc // RRTM, self.Dh // 2))

    # ── attention sublayer forward (overrides the uniform-batch chain) ──
    def _attn_fwd_chain(self, gx, si):
        Hq, Hkv, Dh, H, qd, kd = self.Hq, self.Hkv, self.Dh, self.H, self.qd, self.kd
        M, Sp, Gc = self.M, self.Sp, self.Gc; V = self._V; sc = self.scale
        self._rms(si, gx, self.wn["input_ln"], self.r1, self.gh, M, H, xf32=True)
        self._gemm(si, self.gh, "q_proj", self.gq, H, qd)
        self._gemm(si, self.gh, "k_proj", self.gk, H, kd)
        self._gemm(si, self.gh, "v_proj", self.gv, H, kd)
        self._rms(si, V(self.gq, (M*Hq, Dh)), self.wn["q_norm"], self.rq, V(self.gqn, (M*Hq, Dh)), M*Hq, Dh)
        self._rms(si, V(self.gk, (M*Hkv, Dh)), self.wn["k_norm"], self.rk, V(self.gkn, (M*Hkv, Dh)), M*Hkv, Dh)
        self._t2h(si, self.gqn, self.gqh, Hq)
        self._t2h(si, self.gkn, self.gkh, Hkv)
        self._t2h(si, self.gv,  self.gvh, Hkv)
        if self.is_global:                                     # NoPE
            self.gqr, self.gkr = self.gqh, self.gkh
        else:                                                  # LOCAL: prompt@0.. + suffix@Sp..
            self._rope(si, self.gqh, self.gqr, Hq,  _rope_fwd)
            self._rope(si, self.gkh, self.gkr, Hkv, _rope_fwd)
        qs = self.gqr.at_pos(Hq * Sp); ks = self.gkr.at_pos(Hkv * Sp); vs = self.gvh.at_pos(Hkv * Sp)
        Os = self.gO.at_pos(Hq * Sp); Ls = self.gL.at_pos(Hq * Sp)
        if self.is_global:
            ct.launch(si, (self.NQBp, Hq, 1), _attn_fwd,
                      (self.gqr, self.gkr, self.gvh, self.gO, self.gL, self.NQBp, self.NKVBp, Hq, Hkv, sc))
            ct.launch(si, (self.NQBs, Gc * Hq, 1), _attn_fwd_prefix,
                      (qs, self.gkr, self.gvh, ks, vs, Os, Ls,
                       self.NQBs, self.NKVBp, self.NKVBs, Hq, Hkv, sc))
        else:
            WB = self.win_blocks
            ct.launch(si, (self.NQBp, Hq, 1), _attn_fwd_win,
                      (self.gqr, self.gkr, self.gvh, self.gO, self.gL, self.NQBp, self.NKVBp, Hq, Hkv, sc, WB))
            ct.launch(si, (self.NQBs, Gc * Hq, 1), _attn_fwd_prefix_win,
                      (qs, self.gkr, self.gvh, ks, vs, Os, Ls,
                       self.NQBs, self.NKVBp, self.NKVBs, Hq, Hkv, sc, WB))
        self._h2t(si, self.gO, self.gotok, Hq, _head_to_tok_f32)
        self._gemm(si, self.gotok, "o_proj", self.gattn, qd, H)
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (gx, self.gattn, self.gx2))

    # ── attention sublayer backward ──
    def _attn_bwd_chain(self, si):
        Hq, Hkv, Dh, H, qd, kd = self.Hq, self.Hkv, self.Dh, self.H, self.qd, self.kd
        M, Sp, Gc, GG = self.M, self.Sp, self.Gc, self.Ggrp; V = self._V; sc = self.scale; G = self.G
        NQBp, NKVBp, NQBs, NKVBs = self.NQBp, self.NKVBp, self.NQBs, self.NKVBs
        self._dx(si, self.gdx2, "o_proj", self.gdotok, qd, H); self._dW(si, self.gotok, self.gdx2, G["o_proj"], qd, H)
        self._t2h(si, self.gdotok, self.gdohm, Hq)
        ct.launch(si, (M * Hq // DTM, 1, 1), _attn_delta, (self.gO, self.gdohm, self.gDelta))
        qsv = self.gqr.at_pos(Hq * Sp); ksv = self.gkr.at_pos(Hkv * Sp); vsv = self.gvh.at_pos(Hkv * Sp)
        dOs = self.gdohm.at_pos(Hq * Sp); Lsv = self.gL.at_pos(Hq * Sp); Dsv = self.gDelta.at_pos(Hq * Sp)
        dQs = self.gdqr.at_pos(Hq * Sp); dKs = self.gdkr.at_pos(Hkv * Sp); dVs = self.gdvh.at_pos(Hkv * Sp)
        if self.is_global:
            ct.launch(si, (NQBs, Gc * Hq, 1), _attn_bwd_dq_prefix,         # suffix dQ
                      (qsv, self.gkr, self.gvh, ksv, vsv, dOs, Lsv, Dsv, dQs, NQBs, NKVBp, NKVBs, Hq, Hkv, sc))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv_prefix,          # prompt CROSS dK/dV (Σ_G)
                      (qsv, self.gkr, self.gvh, dOs, Lsv, Dsv, self.gdkc, self.gdvc, NQBs, NKVBp, Hq, Hkv, Gc, GG, sc))
            ct.launch(si, (NQBp, Hq, 1), _attn_bwd_dq,                     # prompt dQ
                      (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdqr, NQBp, NKVBp, Hq, Hkv, sc))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv,                 # prompt SELF dK/dV
                      (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdkr, self.gdvh,
                       NQBp, NKVBp, Hq, Hkv, GG, sc))
            ct.launch(si, (NKVBs, Gc * Hkv, 1), _attn_bwd_dkdv,            # suffix dK/dV (B=G)
                      (qsv, ksv, vsv, dOs, Lsv, Dsv, dKs, dVs, NQBs, NKVBs, Hq, Hkv, GG, sc))
        else:
            WB = self.win_blocks
            ct.launch(si, (NQBs, Gc * Hq, 1), _attn_bwd_dq_prefix_win,
                      (qsv, self.gkr, self.gvh, ksv, vsv, dOs, Lsv, Dsv, dQs, NQBs, NKVBp, NKVBs, Hq, Hkv, sc, WB))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv_prefix_win,
                      (qsv, self.gkr, self.gvh, dOs, Lsv, Dsv, self.gdkc, self.gdvc, NQBs, NKVBp, Hq, Hkv, Gc, GG, sc, WB))
            ct.launch(si, (NQBp, Hq, 1), _attn_bwd_dq_win,
                      (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdqr, NQBp, NKVBp, Hq, Hkv, sc, WB))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv_win,
                      (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdkr, self.gdvh,
                       NQBp, NKVBp, Hq, Hkv, GG, sc, WB))
            ct.launch(si, (NKVBs, Gc * Hkv, 1), _attn_bwd_dkdv_win,
                      (qsv, ksv, vsv, dOs, Lsv, Dsv, dKs, dVs, NQBs, NKVBs, Hq, Hkv, GG, sc, WB))
        # prompt dK/dV = self + Σ_G cross (f32+f32, fixed order — same as the host helper)
        ct.launch(si, (Hkv * Sp // DTM, 1, 1), _add_f32, (self.gdkr, self.gdkc, self.gdkr))
        ct.launch(si, (Hkv * Sp // DTM, 1, 1), _add_f32, (self.gdvh, self.gdvc, self.gdvh))
        # dV head→tok; dq/dk: cast → (rope-bwd for local) → head→tok — split launches like the fwd
        self._h2t(si, self.gdvh, self.gdv, Hkv, _head_to_tok_f32)
        ct.launch(si, (M * Hq // DTM, 1, 1), _cast64, (self.gdqr, self.gdqrb))
        ct.launch(si, (M * Hkv // DTM, 1, 1), _cast64, (self.gdkr, self.gdkrb))
        if self.is_global:
            self._h2t(si, self.gdqrb, self.gdqn, Hq, _head_to_tok)
            self._h2t(si, self.gdkrb, self.gdkn, Hkv, _head_to_tok)
        else:
            self._rope(si, self.gdqrb, self.gdqnhm, Hq,  _rope_bwd)
            self._rope(si, self.gdkrb, self.gdknhm, Hkv, _rope_bwd)
            self._h2t(si, self.gdqnhm, self.gdqn, Hq, _head_to_tok)
            self._h2t(si, self.gdknhm, self.gdkn, Hkv, _head_to_tok)
        self._rms_bwd(si, V(self.gq, (M*Hq, Dh)), "q_norm", V(self.gdqn, (M*Hq, Dh)), self.rq, V(self.gdq, (M*Hq, Dh)), M*Hq, Dh)
        self._rms_bwd(si, V(self.gk, (M*Hkv, Dh)), "k_norm", V(self.gdkn, (M*Hkv, Dh)), self.rk, V(self.gdk, (M*Hkv, Dh)), M*Hkv, Dh)
        self._dx(si, self.gdq, "q_proj", self.gdh1q, H, qd); self._dW(si, self.gh, self.gdq, G["q_proj"], H, qd)
        self._dx(si, self.gdk, "k_proj", self.gdh1k, H, kd); self._dW(si, self.gh, self.gdk, G["k_proj"], H, kd)
        self._dx(si, self.gdv, "v_proj", self.gdh1v, H, kd); self._dW(si, self.gh, self.gdv, G["v_proj"], H, kd)
        self._radd(si, self.gdh1q, self.gdh1k, self.gdh1t); self._radd(si, self.gdh1t, self.gdh1v, self.gdh1)
        self._rms_bwd(si, self.gx, "input_ln", self.gdh1, self.r1, self.gdxa, M, H, xf32=True)
        self._radd(si, self.gdx2, self.gdxa, self.gdx)
        return self.gdx


class ResidentPrefixMoELayer(ResidentPrefixDenseLayer):
    """Device-resident prefix-shared MoE-FFN layer — ResidentMoELayer's FFN over the prefix
    attention chain (the MoE model's real GRPO training layer)."""
    def __init__(self, cfg, attn_weights, moe_w, Sp, Sc, G, is_global, window=512, lid=0,
                 device_route=False, mxfp8=False,
                 optimizer="adamw", muon_scratch=None, muon_scratch_e=None, muon_lr=0.02):
        H, I = cfg.hidden, cfg.dense_inter
        dummy = {"gate_proj": np.zeros((H, I), np.float32),
                 "up_proj":   np.zeros((H, I), np.float32),
                 "down_proj": np.zeros((I, H), np.float32)}
        super().__init__(cfg, {**attn_weights, **dummy}, Sp, Sc, G, is_global, window, lid, mxfp8=mxfp8,
                         optimizer=optimizer, muon_scratch=muon_scratch,
                         muon_exclude=("gate_proj", "up_proj", "down_proj"))
        from ancora.kernels.moe import GroupedMoEFFN
        self.moe = GroupedMoEFFN(moe_w, cfg.top_k, device_route=device_route, mxfp8=mxfp8,
                                 optimizer=optimizer, muon_scratch_e=muon_scratch_e, muon_lr=muon_lr)
        self.gmlp_moe = _DBuf.zeros((self.M, H))

    def forward(self, gx, si):
        self.gx = gx
        self._mx_fwd_prologue(si)
        self._attn_fwd_chain(gx, si)
        M, H = self.M, self.H
        self._rms(si, self.gx2, self.wn["post_ln"], self.r2, self.gh2, M, H, xf32=True)
        self.moe.forward_resident(self.gh2, self.gmlp_moe, si)
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (self.gx2, self.gmlp_moe, self.gout))
        return self.gout

    def backward(self, gdout, si):
        M, H = self.M, self.H
        self._bstep += 1; self._dxc = 0
        self.moe.backward_resident(gdout, self.gdh2, si)
        self._rms_bwd(si, self.gx2, "post_ln", self.gdh2, self.r2, self.gdx2m, M, H, xf32=True)
        self._radd(si, gdout, self.gdx2m, self.gdx2)
        return self._attn_bwd_chain(si)

    def step(self, si, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, muon_lr=0.02):
        super().step(si, lr, b1, b2, eps, wd, muon_lr)
        if not hasattr(self.moe, "_eopt"): self.moe.init_adamw(si, betas=(b1, b2), eps=eps, wd=wd)
        self.moe.step(si, lr, muon_lr=muon_lr)
