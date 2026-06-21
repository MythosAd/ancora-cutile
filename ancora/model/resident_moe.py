"""
ancora/model/resident_moe.py — device-resident MoE training layer (the fix for the
host-orchestration MFU wall: keep weights/activations/AdamW on-device, transfer nothing
per-kernel). Milestone 1: dense-FFN layer with LOCAL (windowed) / GLOBAL (NoPE) attention,
built on ResidentLayerTrain (reuses its persistent buffers, dense FFN, and the device-resident
AdamW where the bf16 weight the GEMM reads IS the optimizer's p16 output).

  - LOCAL  layer: RoPE(base 1e4) + sliding-window attention (_attn_fwd_win / _win backward).
  - GLOBAL layer: NoPE (skip RoPE) + full-causal attention (_attn_fwd).
The dense FFN, norms, residual-fp32 stream, and step() are inherited unchanged.

Milestone 2 (TODO): the MoE-FFN path (grouped kernels + device-AdamW over the 3D expert
weights + router) — this file is the foundation it plugs into.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from types import SimpleNamespace
import numpy as np
import cuda.tile as ct

from ancora.model.resident_train import ResidentLayerTrain, RTM, RTN
from ancora.model.resident import _DBuf
from ancora.kernels.attention import (_attn_fwd, _attn_fwd_win, _attn_bwd_dq, _attn_bwd_dkdv,
                                      _attn_bwd_dq_win, _attn_bwd_dkdv_win, BQ, BKV)
from ancora.kernels.rope import _rope_fwd, _rope_bwd, RTM as RRTM
from ancora.kernels.fused import (_tok_to_head, _head_to_tok, _head_to_tok_f32, _residual_add_rf32,
                                  _attn_delta, _cast64, DTM)


def _shim_cfg(cfg, is_global):
    """ResidentLayerTrain reads cfg.intermediate + cfg.rope_theta; MoEConfig uses dense_inter +
    rope_theta_local. Map them (global layers are NoPE, rope_theta irrelevant)."""
    return SimpleNamespace(hidden=cfg.hidden, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                           head_dim=cfg.head_dim, intermediate=cfg.dense_inter, eps=cfg.eps,
                           rope_theta=cfg.rope_theta_local)


class ResidentMoEDenseLayer(ResidentLayerTrain):
    """Resident dense-FFN decoder layer with local/global attention (MoE model's dense layers)."""
    def __init__(self, cfg, weights, B, S, is_global, window=512, lid=0, mxfp8=False, fp8_bwd=False,
                 optimizer="adamw", muon_scratch=None, muon_exclude=None):
        super().__init__(_shim_cfg(cfg, is_global), weights, B, S, lid=lid, sr_grad=False,
                         mxfp8=mxfp8, fp8_bwd=fp8_bwd, optimizer=optimizer, muon_scratch=muon_scratch,
                         muon_exclude=muon_exclude)
        self.is_global = is_global
        self.win_blocks = 0 if is_global else window // BKV

    def _mx_fwd_prologue(self, si):
        """MXFP8: requant the projection weights if an AdamW step dirtied them + fresh-activation
        quant cache (== ResidentLayerTrain.forward's prologue — our forwards override it)."""
        if self.mxfp8:
            if self._wq_dirty: self._requant_w(si)
            self._qsrc.clear()

    # ── attention sublayer with local(window+RoPE) / global(NoPE+full-causal) ──
    def _attn_fwd_chain(self, gx, si):
        B, Hq, Hkv, Dh, H, qd, kd = self.B, self.Hq, self.Hkv, self.Dh, self.H, self.qd, self.kd
        M, NSB, NQB = self.M, self.NSB, self.NQB; V = self._V
        self._rms(si, gx, self.wn["input_ln"], self.r1, self.gh, M, H, xf32=True)
        self._gemm(si, self.gh, "q_proj", self.gq, H, qd)
        self._gemm(si, self.gh, "k_proj", self.gk, H, kd)
        self._gemm(si, self.gh, "v_proj", self.gv, H, kd)
        self._rms(si, V(self.gq, (M*Hq, Dh)), self.wn["q_norm"], self.rq, V(self.gqn, (M*Hq, Dh)), M*Hq, Dh)
        self._rms(si, V(self.gk, (M*Hkv, Dh)), self.wn["k_norm"], self.rk, V(self.gkn, (M*Hkv, Dh)), M*Hkv, Dh)
        ct.launch(si, (B*Hq, NSB, 1), _tok_to_head, (self.gqn, self.gqh, Hq, NSB))
        ct.launch(si, (B*Hkv, NSB, 1), _tok_to_head, (self.gkn, self.gkh, Hkv, NSB))
        ct.launch(si, (B*Hkv, NSB, 1), _tok_to_head, (self.gv, self.gvh, Hkv, NSB))
        if self.is_global:                                   # NoPE: qr/kr = qh/kh (no rotation)
            self.gqr, self.gkr = self.gqh, self.gkh
            ct.launch(si, (NQB, B*Hq, 1), _attn_fwd,
                      (self.gqr, self.gkr, self.gvh, self.gO, self.gL, NQB, NQB, Hq, Hkv, self.scale))
        else:                                                # LOCAL: RoPE then windowed attention
            ct.launch(si, (self.S//RRTM, B*Hq, 1), _rope_fwd, (self.gqh, self.gcos, self.gsin, self.gqr, self.S//RRTM, Dh//2))
            ct.launch(si, (self.S//RRTM, B*Hkv, 1), _rope_fwd, (self.gkh, self.gcos, self.gsin, self.gkr, self.S//RRTM, Dh//2))
            ct.launch(si, (NQB, B*Hq, 1), _attn_fwd_win,
                      (self.gqr, self.gkr, self.gvh, self.gO, self.gL, NQB, NQB, Hq, Hkv, self.scale, self.win_blocks))
        ct.launch(si, (B*Hq, NSB, 1), _head_to_tok_f32, (self.gO, self.gotok, Hq, NSB))
        self._gemm(si, self.gotok, "o_proj", self.gattn, qd, H)
        ct.launch(si, (M//RTM, H//RTN, 1), _residual_add_rf32, (gx, self.gattn, self.gx2))

    def forward(self, gx, si):
        self.gx = gx
        self._mx_fwd_prologue(si)
        self._attn_fwd_chain(gx, si)
        # dense FFN + final residual (identical to ResidentLayerTrain)
        H, I, M = self.H, self.I, self.M
        from ancora.kernels.activation import _swiglu_fwd, TM as STM, TI
        self._rms(si, self.gx2, self.wn["post_ln"], self.r2, self.gh2, M, H, xf32=True)
        self._gemm(si, self.gh2, "gate_proj", self.gg, H, I)
        self._gemm(si, self.gh2, "up_proj", self.gu, H, I)
        ct.launch(si, (M//STM, I//TI, 1), _swiglu_fwd, (self.gg, self.gu, self.ga))
        self._gemm(si, self.ga, "down_proj", self.gmlp, I, H)
        ct.launch(si, (M//RTM, H//RTN, 1), _residual_add_rf32, (self.gx2, self.gmlp, self.gout))
        return self.gout

    # ── attention sublayer backward: reads self.gdx2 (grad into attn residual), writes self.gdx.
    #    Shared verbatim by the dense and MoE resident layers (only the FFN above it differs). ──
    def _attn_bwd_chain(self, si):
        B, Hq, Hkv, Dh, H, qd, kd = self.B, self.Hq, self.Hkv, self.Dh, self.H, self.qd, self.kd
        M, NSB, NQB, G = self.M, self.NSB, self.NQB, self.G; V = self._V; sc = self.scale
        self._dx(si, self.gdx2, "o_proj", self.gdotok, qd, H); self._dW(si, self.gotok, self.gdx2, G["o_proj"], qd, H)
        ct.launch(si, (B*Hq, NSB, 1), _tok_to_head, (self.gdotok, self.gdohm, Hq, NSB))
        ct.launch(si, (M*Hq//DTM, 1, 1), _attn_delta, (self.gO, self.gdohm, self.gDelta))
        if self.is_global:
            ct.launch(si, (NQB, B*Hq, 1), _attn_bwd_dq, (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdqr, NQB, NQB, Hq, Hkv, sc))
            ct.launch(si, (NQB, B*Hkv, 1), _attn_bwd_dkdv, (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdkr, self.gdvh, NQB, NQB, Hq, Hkv, self.Ggrp, sc))
        else:
            ct.launch(si, (NQB, B*Hq, 1), _attn_bwd_dq_win, (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdqr, NQB, NQB, Hq, Hkv, sc, self.win_blocks))
            ct.launch(si, (NQB, B*Hkv, 1), _attn_bwd_dkdv_win, (self.gqr, self.gkr, self.gvh, self.gdohm, self.gL, self.gDelta, self.gdkr, self.gdvh, NQB, NQB, Hq, Hkv, self.Ggrp, sc, self.win_blocks))
        # dV is fp32 head-major → token-major; dq/dk: rope_bwd for local, identity for global
        ct.launch(si, (B*Hkv, NSB, 1), _head_to_tok_f32, (self.gdvh, self.gdv, Hkv, NSB))
        ct.launch(si, (M*Hq//DTM, 1, 1), _cast64, (self.gdqr, self.gdqrb)); ct.launch(si, (M*Hkv//DTM, 1, 1), _cast64, (self.gdkr, self.gdkrb))
        if self.is_global:                                   # NoPE: dqn/dkn = dqr/dkr (head→tok)
            ct.launch(si, (B*Hq, NSB, 1), _head_to_tok, (self.gdqrb, self.gdqn, Hq, NSB))
            ct.launch(si, (B*Hkv, NSB, 1), _head_to_tok, (self.gdkrb, self.gdkn, Hkv, NSB))
        else:                                                # LOCAL: rope_bwd then head→tok
            ct.launch(si, (self.S//RRTM, B*Hq, 1), _rope_bwd, (self.gdqrb, self.gcos, self.gsin, self.gdqnhm, self.S//RRTM, Dh//2))
            ct.launch(si, (self.S//RRTM, B*Hkv, 1), _rope_bwd, (self.gdkrb, self.gcos, self.gsin, self.gdknhm, self.S//RRTM, Dh//2))
            ct.launch(si, (B*Hq, NSB, 1), _head_to_tok, (self.gdqnhm, self.gdqn, Hq, NSB))
            ct.launch(si, (B*Hkv, NSB, 1), _head_to_tok, (self.gdknhm, self.gdkn, Hkv, NSB))
        self._rms_bwd(si, V(self.gq, (M*Hq, Dh)), "q_norm", V(self.gdqn, (M*Hq, Dh)), self.rq, V(self.gdq, (M*Hq, Dh)), M*Hq, Dh)
        self._rms_bwd(si, V(self.gk, (M*Hkv, Dh)), "k_norm", V(self.gdkn, (M*Hkv, Dh)), self.rk, V(self.gdk, (M*Hkv, Dh)), M*Hkv, Dh)
        self._dx(si, self.gdq, "q_proj", self.gdh1q, H, qd); self._dW(si, self.gh, self.gdq, G["q_proj"], H, qd)
        self._dx(si, self.gdk, "k_proj", self.gdh1k, H, kd); self._dW(si, self.gh, self.gdk, G["k_proj"], H, kd)
        self._dx(si, self.gdv, "v_proj", self.gdh1v, H, kd); self._dW(si, self.gh, self.gdv, G["v_proj"], H, kd)
        self._radd(si, self.gdh1q, self.gdh1k, self.gdh1t); self._radd(si, self.gdh1t, self.gdh1v, self.gdh1)
        self._rms_bwd(si, self.gx, "input_ln", self.gdh1, self.r1, self.gdxa, M, H, xf32=True)
        self._radd(si, self.gdx2, self.gdxa, self.gdx)
        return self.gdx

    def backward(self, gdout, si):
        from ancora.kernels.activation import _swiglu_bwd, TM as STM, TI
        H, I, M, G = self.H, self.I, self.M, self.G
        self._bstep += 1; self._dxc = 0
        # ── dense MLP backward → grad-wrt-gh2 (post_ln input) → gdx2 (attn residual grad) ──
        self._dx(si, gdout, "down_proj", self.gda, I, H); self._dW(si, self.ga, gdout, G["down_proj"], I, H)
        ct.launch(si, (M//STM, I//TI, 1), _swiglu_bwd, (self.gg, self.gu, self.gda, self.gdg, self.gdu))
        self._dx(si, self.gdg, "gate_proj", self.gdh2a, H, I); self._dW(si, self.gh2, self.gdg, G["gate_proj"], H, I)
        self._dx(si, self.gdu, "up_proj", self.gdh2b, H, I);   self._dW(si, self.gh2, self.gdu, G["up_proj"], H, I)
        self._radd(si, self.gdh2a, self.gdh2b, self.gdh2)
        self._rms_bwd(si, self.gx2, "post_ln", self.gdh2, self.r2, self.gdx2m, M, H, xf32=True)
        self._radd(si, gdout, self.gdx2m, self.gdx2)
        return self._attn_bwd_chain(si)


class ResidentMoELayer(ResidentMoEDenseLayer):
    """Resident MoE-FFN decoder layer (milestone 2): attention (local/global) resident + the grouped
    MoE FFN resident (GroupedMoEFFN.forward_resident — router = the ONLY host round-trip/layer).
    Forward done; backward + device-AdamW over the 3D expert weights = milestone 2b/3."""
    def __init__(self, cfg, attn_weights, moe_w, B, S, is_global, window=512, lid=0,
                 device_route=False, mxfp8=False, fp8_bwd=False,
                 optimizer="adamw", muon_scratch=None, muon_scratch_e=None, muon_lr=0.02):
        H, I = cfg.hidden, cfg.dense_inter
        dummy = {"gate_proj": np.zeros((H, I), np.float32),     # satisfy ResidentLayerTrain's dense-FFN
                 "up_proj":   np.zeros((H, I), np.float32),     # buffer/AdamW setup (unused on a MoE layer)
                 "down_proj": np.zeros((I, H), np.float32)}
        # the dummy dense-FFN weights stay on AdamW under muon (zero grad → no-op; excludes 3 NS/layer)
        super().__init__(cfg, {**attn_weights, **dummy}, B, S, is_global, window, lid,
                         mxfp8=mxfp8, fp8_bwd=fp8_bwd, optimizer=optimizer, muon_scratch=muon_scratch,
                         muon_exclude=("gate_proj", "up_proj", "down_proj"))
        from ancora.kernels.moe import GroupedMoEFFN
        self.moe = GroupedMoEFFN(moe_w, cfg.top_k, device_route=device_route, mxfp8=mxfp8,
                                 optimizer=optimizer, muon_scratch_e=muon_scratch_e, muon_lr=muon_lr)
        self.gmlp_moe = _DBuf.zeros((B * S, H))                 # MoE FFN output (bf16 bits)

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
        """gdout: _DBuf (M,H) bf16 bits (grad of gout). gout = gx2 + gmlp_moe, so grad-of-gmlp = gdout
        (→ MoE FFN backward) and grad-of-gx2 gets the residual gdout + the FFN path through post_ln.
        Expert weight grads + router grad land in self.moe (device dWd/dWg/dWu, host G_router); the
        attention-path backward is the shared _attn_bwd_chain. Caller syncs si."""
        M, H = self.M, self.H
        self._bstep += 1; self._dxc = 0
        self.moe.backward_resident(gdout, self.gdh2, si)         # grad-of-gmlp=gdout → grad-wrt-gh2 in self.gdh2
        self._rms_bwd(si, self.gx2, "post_ln", self.gdh2, self.r2, self.gdx2m, M, H, xf32=True)
        self._radd(si, gdout, self.gdx2m, self.gdx2)             # gdx2 = gdout(residual) + FFN-path
        return self._attn_bwd_chain(si)

    def step(self, si, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, muon_lr=0.02):
        """Optimizer step: the inherited update for the attention proj + norms (proj→Muon under
        optimizer="muon"; the dummy dense FFN stays AdamW, zero grad → no-op), then the MoE FFN's own
        optimizer (experts→batched Muon under "muon", else AdamW; router always AdamW). init_adamw is
        lazy (first step, after the weights are packed)."""
        super().step(si, lr, b1, b2, eps, wd, muon_lr)
        if not hasattr(self.moe, "_eopt"): self.moe.init_adamw(si, betas=(b1, b2), eps=eps, wd=wd)
        self.moe.step(si, lr, muon_lr=muon_lr)
