"""
ancora/model/resident_prefix_model.py — ResidentPrefixMoEModel: the device-resident PREFIX-SHARED
GRPO training step for the MoE model. The prompt is encoded ONCE; the G completions share its KV
through every layer (ResidentPrefixDense/MoELayer); the tied embed/LM-head boundary, the DEVICE
final RMSNorm, and all AdamW state are inherited from ResidentMoEModel.

BOUNDARY-ROW DUPLICATION (the PrefixGrouper `include_prefix_last` trick, made device-resident):
completion token 0 is predicted by the SHARED prompt row Sp-1, whose single device label/adv slot
cannot carry G different (label_i, adv_i). The head-side row count grows to Mh = align128(M + G):
the forward DtoD-copies hidden row Sp-1 into G tail rows (one per completion), each carrying its
own (comp_i[0], adv_i[0]) — the STANDARD _ce_stats/_ce_grad and GEMMs then treat them like any
other row, reproducing the replicated model's per-row math (and bf16 rounding) EXACTLY. The
backward sums the G duplicate rows' dhidden back into row Sp-1 (fused._bnd_acc, fixed order —
autograd's implicit duplicate-sum made explicit). The layers only see the M = Sp + G·Sc real rows.

CUDA-GRAPH STEP: fwd + bwd are pure device launch chains (no sync / host round-trip anywhere —
needs device_route=True so the MoE router never syncs), so capture() records them into ONE graph;
graph_step() = upload inputs → graph.launch → sync → (ce, lp_comp). AdamW step() stays OUTSIDE
the graph: its per-step bias-correction scalars are runtime kernel args a graph would freeze.

  m = ResidentPrefixMoEModel(cfg, weights, Sp, Sc, G, device_route=True)
  h = m.forward_prefix(prompt_ids, comp_ids, si)             # direct path
  ce, lp_comp = m.grpo_loss_backward(h, comp_ids, comp_adv, si)
  m.step(si, lr)                                             # warm step → then:
  m.capture(dev)                                             # one-time graph capture
  ce, lp_comp = m.graph_step(prompt_ids, comp_ids, comp_adv, so, si)   # replay per step
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.resident import _DBuf
from ancora.model.resident_moe_model import ResidentMoEModel, _DWT
from ancora.model.resident_prefix import ResidentPrefixDenseLayer, ResidentPrefixMoELayer
from ancora.kernels.loss import _gemm, _ce_stats_b, _ce_grad_b, GTM, GTN, GTK, CTM, TV
from ancora.kernels.fused import _gemm_nt_f32, _gemm_dW, _gemm_dW_acc, _bnd_acc
from ancora.model.resident_moe_model import _DWN, _CTB


class ResidentPrefixMoEModel(ResidentMoEModel):
    def __init__(self, cfg, weights: dict, Sp: int, Sc: int, G: int, device_route: bool = False,
                 mxfp8: bool = False, optimizer: str = "adamw", muon_lr: float = 0.02,
                 long_context: bool = False):
        self.Sp, self.Sc, self.Gc = Sp, Sc, G
        # long_context: activation checkpointing also works on the prefix GRPO path — the parent's
        # _setup_checkpoint aliases each prefix layer's _SHARE scratch to layer 0's + gives it a gx_in
        # (the prefix layers inherit those buffer names); _fwd_dev stores gx_in, _bwd_dev RECOMPUTES
        # each prefix layer's forward from it. The Mh head/boundary buffers are allocated below, after
        # super().__init__ runs _setup_checkpoint on the M=Sp+G·Sc layer buffers → no conflict. Lets
        # RL training reach 16K (the SFT path's wall) instead of OOMing at full-store.
        super().__init__(cfg, weights, 1, Sp + G * Sc, device_route=device_route, mxfp8=mxfp8,
                         optimizer=optimizer, muon_lr=muon_lr, long_context=long_context)
        # head-side rows: M real + G boundary duplicates, padded to the GEMM/CE tiles. The parent
        # allocated the head buffers at M — replace them (init-time only, no churn).
        H, V = self.H, self.V
        self.Mh = ((self.M + G + 127) // 128) * 128
        # (glog/gglog stay at the parent's CHUNK size MC — the boundary is chunked; only the
        # per-row buffers grow to Mh for the G duplicate rows)
        for n in ("gh", "glp", "glse", "gadv", "glab", "gdhid"):
            getattr(self, n).free()
        Z = _DBuf.zeros; Mh = self.Mh
        self.gh = Z((Mh, H), np.uint16);     self.glp = Z((Mh, 1), np.float32)
        self.glse = Z((Mh, 1), np.float32);  self.gadv = Z((Mh, 1), np.float32)
        self.glab = Z((Mh, 1), np.int32);    self.gdhid = Z((Mh, H), np.float32)
        self._graph = None

    def _build_layer(self, cfg, lw, i):
        if lw["ffn_dense"]:
            return ResidentPrefixDenseLayer(cfg, {**lw["attn"], **lw["ffn"]}, self.Sp, self.Sc, self.Gc,
                                            is_global=lw["is_global"], window=cfg.window, lid=i,
                                            mxfp8=self.mxfp8,
                                            optimizer=self.optimizer, muon_scratch=self._muon_scratch)
        return ResidentPrefixMoELayer(cfg, lw["attn"], lw["ffn"], self.Sp, self.Sc, self.Gc,
                                      is_global=lw["is_global"], window=cfg.window, lid=i,
                                      device_route=self.device_route, mxfp8=self.mxfp8,
                                      optimizer=self.optimizer, muon_scratch=self._muon_scratch,
                                      muon_scratch_e=self._muon_scratch_e, muon_lr=self.muon_lr)

    # ── forward: parent chain + the G boundary duplicate rows (async DtoD — capturable) ──
    def _fwd_dev(self, si: int):
        super()._fwd_dev(si)
        src = int(self.gh.ptr) + (self.Sp - 1) * self.H * 2
        for g in range(self.Gc):
            cdrv.cuMemcpyDtoDAsync(int(self.gh.ptr) + (self.M + g) * self.H * 2, src, self.H * 2, si)

    def forward_prefix(self, prompt_ids: np.ndarray, comp_ids: np.ndarray, si: int):
        """prompt_ids (Sp,), comp_ids (G,Sc) → post-final-norm hidden bf16 BITS (M,H)."""
        self._upload_ids(np.concatenate([prompt_ids.reshape(-1), comp_ids.reshape(-1)]), si)
        self._fwd_dev(si)
        cudart.cudaStreamSynchronize(si)
        return self.gh.to_numpy()[:self.M]

    # ── head backward: parent's at Mh + the boundary duplicate-row gradient sum ──
    def _bwd_head_dev(self, si: int, inv_nrm: float, gacc: bool = False):
        M, Mh, H, V, Sp, G = self.M, self.Mh, self.H, self.V, self.Sp, self.Gc; T = _DWT
        for ci, (m0, mc) in enumerate(self._chunks(Mh)):
            gh_c = self.gh.at_pos(m0)
            ct.launch(si, (mc // 128, V // 128, 1), _gemm_nt_f32, (gh_c, self.gembed, self.glog, H // 64, 128, 128, 64))
            ct.launch(si, (mc // _CTB, 1, 1), _ce_stats_b,
                      (self.glog, self.glab.at_pos(m0), self.glp.at_pos(m0), self.glse.at_pos(m0), V // TV, _CTB))
            ct.launch(si, (mc // _CTB, 1, 1), _ce_grad_b,
                      (self.glog, self.glse.at_pos(m0), self.glab.at_pos(m0), self.gadv.at_pos(m0), self.gglog, V // TV, inv_nrm, _CTB))
            ct.launch(si, (mc // GTM, H // 32, 1), _gemm, (self.gglog, self.gembed, self.gdhid.at_pos(m0), V // GTK, GTM, 32, GTK))
            dw = _gemm_dW if (ci == 0 and not gacc) else _gemm_dW_acc
            ct.launch(si, (V // T, H // _DWN, 1), dw, (self.gglog, gh_c, self.gegrad, mc // T, T, _DWN, T))
        ct.launch(si, (H // 128, 1, 1), _bnd_acc, (self.gdhid, Sp - 1, M, G))   # Σ duplicates → row Sp-1

    # ── GRPO loss + backward ──────────────────────────────────────────────────
    def grpo_io(self, comp_ids, comp_adv):
        """(Mh,) labels/adv. Suffix row Sp+i·Sc+t (t<Sc-1) predicts comp_i[t+1]; the duplicate
        row M+i predicts comp_i[0]. Prompt / final-suffix / padding rows: adv=0 (zero grad).
        comp_adv: (G,) per-completion or (G,Sc) per-token."""
        Sp, Sc, G, M, Mh = self.Sp, self.Sc, self.Gc, self.M, self.Mh
        adv2 = np.broadcast_to(np.asarray(comp_adv, np.float32).reshape(G, -1), (G, Sc)).copy()
        labels = np.zeros(Mh, np.int64); adv = np.zeros(Mh, np.float32)
        for i in range(G):
            r = Sp + i * Sc
            labels[r:r + Sc - 1] = comp_ids[i, 1:]
            adv[r:r + Sc - 1] = adv2[i, 1:]
            labels[M + i] = comp_ids[i, 0]                    # boundary duplicate row
            adv[M + i] = adv2[i, 0]
        return labels, adv

    def _upload_io(self, labels, adv, si: int):
        Mh = self.Mh
        cdrv.cuMemcpyHtoD(self.gadv.ptr, np.ascontiguousarray(adv.reshape(Mh, 1)), self.gadv.nbytes)
        cdrv.cuMemcpyHtoD(self.glab.ptr, np.ascontiguousarray(labels.astype(np.int32).reshape(Mh, 1)), self.glab.nbytes)

    def _ce_lp(self, adv, nrm):
        M, Sp, Sc, G = self.M, self.Sp, self.Sc, self.Gc
        lp = self.glp.to_numpy().reshape(self.Mh)
        lp_comp = np.array([lp[M + i] + lp[Sp + i * Sc: Sp + (i + 1) * Sc - 1].sum() for i in range(G)])
        return float(-(adv * lp).sum() / nrm), lp_comp

    def grpo_loss_backward(self, h, comp_ids, comp_adv, si: int, norm=None, accumulate=False):
        """h from forward_prefix (unused — state on self). comp_ids (G,Sc), comp_adv (G,) or (G,Sc).
        Returns (ce, lp_comp (G,)), lp_comp[i] = Σ_t log π(comp_i[t]) (all Sc tokens — the boundary
        token's lp comes from the duplicate row's device _ce_stats like any other). Grads land like
        loss_backward (layers / embed / gfng). accumulate=True: ADD onto the existing grads
        (GRADIENT ACCUMULATION across prompt groups — pass the same total-token `norm` each call)."""
        nrm = float(self.Gc * self.Sc if norm is None else norm)
        labels, adv = self.grpo_io(comp_ids, comp_adv)
        self._upload_io(labels, adv, si)
        self._bwd_dev(si, 1.0 / nrm, accumulate)              # head (Mh) + final-norm bwd + layers
        cudart.cudaStreamSynchronize(si)
        return self._ce_lp(adv, nrm)

    def loss_backward(self, h, labels, si, advantage=None, norm=None):
        raise NotImplementedError("head buffers are Mh-sized (boundary duplicates) — "
                                  "use grpo_loss_backward (SFT = comp_adv of ones)")

    # ── CUDA-graph capture of the ENTIRE fwd+bwd ──────────────────────────────
    def capture(self, dev):
        """Capture forward + backward into ONE CUDA graph. Call AFTER a warm fwd+bwd+step():
        all kernels JIT-compiled, MoE weights packed / buffers preallocated, and the device-route
        router AdamW owns Wr_dev (so the captured forward contains no host-pointer upload).
        step() is NOT captured — its AdamW bias-correction scalars change per step and a graph
        would freeze them."""
        assert self.device_route, "graph capture needs device_route=True (host routing syncs)"
        for l in self.layers:
            if hasattr(l, "moe"):
                assert getattr(l.moe, "_router_dev_adam", False), \
                    "run one warm fwd+bwd+step() before capture (device router AdamW must own Wr_dev)"
        gb = dev.create_graph_builder(); gb.begin_building()
        gsi = int(gb.__cuda_stream__()[1])
        self._fwd_dev(gsi)
        self._bwd_dev(gsi, 1.0 / float(self.Gc * self.Sc))
        gb.end_building()
        self._graph = gb.complete()

    def graph_step(self, prompt_ids, comp_ids, comp_adv, so, si: int):
        """One captured fwd+bwd replay: upload inputs → graph.launch → sync → (ce, lp_comp).
        so = the cuda.core Stream OBJECT the graph launches on; si = its int (uploads).
        Identical math to forward_prefix + grpo_loss_backward — same kernels/order/buffers."""
        self._upload_ids(np.concatenate([prompt_ids.reshape(-1), comp_ids.reshape(-1)]), si)
        labels, adv = self.grpo_io(comp_ids, comp_adv)
        self._upload_io(labels, adv, si)
        self._graph.launch(so)
        cudart.cudaStreamSynchronize(si)
        return self._ce_lp(adv, float(self.Gc * self.Sc))
