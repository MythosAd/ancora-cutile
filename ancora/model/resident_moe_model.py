"""
ancora/model/resident_moe_model.py — ResidentMoEModel: device-resident multi-layer training for the
interleaved dense/MoE + local/global model (the MoE counterpart of resident_model.ResidentModel).

The bulk is a chain of device-resident layers per layer_schedule(cfg):
  - dense-FFN layer  → ResidentMoEDenseLayer  (dense SwiGLU + local/global attention, device AdamW)
  - MoE-FFN  layer   → ResidentMoELayer       (grouped MoE FFN resident + device AdamW over the 3D
                                               expert weights + host router AdamW)
Between layers the activation handoff is a device buffer (fp32 residual stream). The model BOUNDARY
(tied embed = LM head + final RMSNorm) is reused VERBATIM from ResidentModel's design:
  logits = h @ embedᵀ (_gemm_nt_f32) ; CE (_ce_stats/_ce_grad) ; dhidden = glogit @ embed ;
  embed grad = glogitᵀ@h (LM head) + onehotᵀ@d_x0 (input embed, _acc_f32) ; final RMSNorm host.

  m  = ResidentMoEModel(cfg, weights, B, S)
  h  = m.forward(input_ids, si)
  ce = m.loss_backward(h, labels, si, advantage)
  m.step(si, lr)

weights = {"embed": (V,H), "final_norm": (H,),
           "layers": [{"is_global": bool, "ffn_dense": bool, "attn": {...}, "ffn": {...}}, ...]}
where ffn = {gate_proj,up_proj,down_proj} for dense, {router,gate_proj,up_proj,down_proj} for MoE.
`from_host(host_moemodel, B, S)` builds this dict from a moe_model.MoEModel.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.resident import _DBuf, _f32bf
from ancora.model.resident_moe import ResidentMoEDenseLayer, ResidentMoELayer
from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, _rmsnorm_bwd_dx,
                                 _rmsnorm_dw_part, _rmsnorm_dw_reduce, _rmsnorm_dw_reduce_acc,
                                 TM as NTM, TH, TD, PART,
                                 f32_to_bf16_bits as _f32bf_rne)   # RNE — the norm gain's rounding
from ancora.kernels.loss import (_gemm, _ce_stats, _ce_grad, _ce_stats_b, _ce_grad_b,
                                 GTM, GTN, GTK, CTM, TV)
from ancora.kernels.fused import (_gemm_nt_f32, _gemm_dW, _gemm_dW_acc, _acc_f32, _cast_bf16,
                                  _embed_gather, _embed_dw_scatter, build_id_groups,
                                  ACM, ACN, RTM, RTN)
from ancora.optim.adamw import _adamw, _pick_otm, C as ADAM_C

_DWT = 64    # _gemm_dW output row tile (embed grad (V,H))
_DWN = 128   # _gemm_dW output COLUMN tile at the boundary: x=(M,V) is re-read by every column
             # block, so H//128 (not //64) halves the dominant DRAM traffic (ncu: 76% DRAM)
_CTB = 4     # boundary CE row tile: CTM=64 → 16 blocks at M=1024 (ncu SM 5% on a ~1GB stream);
             # CTMb=4 is the smallest BITWISE row tile (probed — ≤2 flips the streamed reduce)


def from_host(host, B, S):
    """Build the ResidentMoEModel weights dict from a moe_model.MoEModel (tie=True assumed)."""
    layers = []
    for (is_global, ffn_dense), l in zip(host.sched, host.layers):
        ffn = dict(l.ffn.w)                                   # dense: g/u/d ; MoE: router+3D g/u/d
        layers.append(dict(is_global=is_global, ffn_dense=ffn_dense, attn=dict(l.attn), ffn=ffn))
    return dict(embed=host.embed, final_norm=host.final_norm, layers=layers)


class ResidentMoEModel:
    def __init__(self, cfg, weights: dict, B: int, S: int, device_route: bool = False,
                 mxfp8: bool = False, fp8_bwd: bool = False, long_context: bool = False,
                 optimizer: str = "adamw", muon_lr: float = 0.02, batch_proj: bool = True,
                 muon_scope=None):
        self.cfg, self.B, self.S, self.V = cfg, B, S, cfg.vocab
        self.device_route = device_route       # MoE router on device (sync-free; rollout must match)
        self.mxfp8 = mxfp8                     # MXFP8 forward GEMMs (proj + experts); bwd stays BF16
        self.fp8_bwd = fp8_bwd                 # FP8 E4M3 data-gradient (dgrad) on the dense/attn projections
        # long_context: ACTIVATION CHECKPOINTING for long sequences — the backward RECOMPUTES each
        # layer's forward (deterministic ⇒ bitwise-identical grads) instead of keeping all NL layers'
        # intermediates resident. Off by default (short training keeps the fast full-store resident
        # path). Saves the NL× activation multiplier (the part that scales with S); enables 8K-16K
        # single-sequence training on 16 GB. The local/global split makes recompute cheap (10/12 layers
        # are windowed O(S·window); only 2 global are O(S²)). See [[mfu-strategy]].
        self.long_context = long_context
        self.H, self.M, self.eps = cfg.hidden, B * S, cfg.eps
        H, M, V = self.H, self.M, self.V
        self.NL = len(weights["layers"])
        assert V % GTN == 0 and V % GTK == 0 and V % TV == 0 and V % _DWT == 0, "pad vocab to the tiles"
        assert M % GTM == 0 and M % CTM == 0 and M % _DWT == 0 and M % 128 == 0

        # ── Muon/AdamW hybrid: 2D proj → resident Muon, MoE EXPERTS → batched Muon (Kimi recipe), the
        #    tied embed/LM-head + 1D gains + router → AdamW. ONE proj scratch + ONE expert scratch shared
        #    across all layers (built BEFORE the layers; _build_layer hands them down). Drops the experts'
        #    v buffer (the FFN's optimizer bulk) + the proj v → lowers the long_context VRAM floor. ──
        self.optimizer, self.muon_lr, self.muon_scope = optimizer, muon_lr, muon_scope
        self._muon_scratch = self._muon_scratch_e = None
        if optimizer == "muon":
            from ancora.optim.muon import MuonScratch
            from ancora.kernels.moe import ExpertMuonScratch
            proj_shapes, has_moe = [], False
            for lw in weights["layers"]:
                proj_shapes += [lw["attn"][n].shape for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
                if lw["ffn_dense"]:
                    proj_shapes += [lw["ffn"][n].shape for n in ("gate_proj", "up_proj", "down_proj")]
                else:
                    has_moe = True
            self._muon_scratch = MuonScratch(proj_shapes)
            if has_moe:
                self._muon_scratch_e = ExpertMuonScratch(cfg.n_experts, cfg.hidden)

        # ── the bulk: device-resident dense/MoE layers per the schedule ──
        # (_build_layer is a hook: ResidentPrefixMoEModel swaps in the prefix-shared layers)
        # long_context: alias each layer's _SHARE scratch to layer 0's AS IT IS BUILT (free-as-you-go)
        # so the construction peak is ~2 layers' scratch, NOT all NL — the all-NL peak was what OOM'd
        # the (heavier) prefix layers at 16K before _setup_checkpoint could free the duplicates.
        self.layers = []
        for i, lw in enumerate(weights["layers"]):
            l = self._build_layer(cfg, lw, i)
            if long_context and i > 0:
                self._alias_scratch(l, self.layers[0])
            self.layers.append(l)
        if long_context:
            self._setup_checkpoint()

        # ── BATCHED proj Muon: pull ALL 2D proj weights out of each layer's per-weight muon and run
        #    batched Newton-Schulz chains grouped by NS shape across layers (industrial Keller/Kimi
        #    pattern) — kills the per-weight NS launch overhead (~195ms→floor). Square k/v/gate/up/
        #    down (1024²) form one group; the RECTANGULAR q/o both batch as (1024,2048) (o transposed
        #    in/out by the fused _muon_mom_t/_muon_update_cast_t — bit-identical to the per-weight
        #    path). muon_scope=None → global groups; int k → sub-group by lid//k (pipeline-parallel
        #    granularity: each k-layer stage steps its own batches). ──
        self.batched_proj_muon = None
        if optimizer == "muon" and batch_proj:
            allw = []
            for li, l in enumerate(self.layers):
                if not hasattr(l, "muon"):
                    continue
                for n in list(l.muon):
                    rm = l.muon.pop(n)                        # remove from per-weight muon …
                    l._proj_ext.add(n)                        # … and tell layer.step() to skip it
                    K, N = l.w[n].shape
                    allw.append(dict(buf=rm.buf, p32=l.opt[n]["p32"].view((K, N)), p16=l.w[n],
                                     G=l.G[n], K=K, N=N, lid=li))
            if allw:
                from ancora.optim.muon import BatchedProjMuon
                self.batched_proj_muon = BatchedProjMuon(allw, scope=muon_scope)
                # every proj weight is batched → the per-weight MuonScratch is dead weight; free it
                if self._muon_scratch is not None and all(not l.muon for l in self.layers
                                                          if hasattr(l, "muon")):
                    self._muon_scratch.free(); self._muon_scratch = None

        # ── TIED embed = LM head: one (V,H) device param + device AdamW (fp32 master + m,v) ──
        embed = weights["embed"].astype(np.float32)                  # (V,H)
        self.gembed = _DBuf(_f32bf(embed))
        Re = (V * H) // ADAM_C
        self.eopt = dict(R=Re, otm=_pick_otm(Re),
                         p32=_DBuf(embed.reshape(Re, ADAM_C).copy()),
                         m=_DBuf(np.zeros((Re, ADAM_C), np.float32)),
                         v=_DBuf(np.zeros((Re, ADAM_C), np.float32)),
                         p16=self.gembed.view((Re, ADAM_C)))
        # ── final RMSNorm: DEVICE fwd+bwd (bf16 gain + persistent buffers — graph-capturable;
        #    bitwise == the old host rmsnorm_forward path: _cast_bf16 (RNE) == norm.py's
        #    f32_to_bf16_bits — NOT attention.py's truncating one / fused._trunc_bf16! That RNE-vs
        #    -trunc mismatch was also the decode model's parked "removed sync changes gout" mystery
        #    — then the same _rmsnorm_* kernels). AdamW for the tiny (H,) gain stays host in step()
        #    (gfng readback + gfnw re-upload, outside any captured graph). ──
        self.final_norm = weights["final_norm"].astype(np.float32).copy()
        self.fn_m = np.zeros_like(self.final_norm); self.fn_v = np.zeros_like(self.final_norm)
        self.gfnw = _DBuf(_f32bf_rne(self.final_norm.reshape(1, H)))  # bf16 gain, RNE (== host helper)
        self.gnx  = _DBuf.zeros((M, H), np.uint16)                    # bf16-trunc pre-norm residual
        self.grfn = _DBuf.zeros((M, 1), np.float32)                   # final-norm rstd
        self.gdy  = _DBuf.zeros((M, H), np.uint16)                    # bf16 dhidden (norm-bwd input)
        self.gfng = _DBuf.zeros((1, H), np.float32)                   # final-norm gain grad
        self.gprt = _DBuf.zeros((PART, H), np.float32)                # dw 2-pass partials
        self.t = 0

        # ── persistent boundary buffers (no per-step alloc churn) ──
        # The (·,V) vocab buffers are CHUNKED to MC rows (Liger-style chunked boundary): glog/
        # gglog/gohot were 2.5 GB at M=2048 — the #1 VRAM pressure that paged the AdamW sweep
        # (WDDM, 16 GB). Per-row boundary math is chunk-invariant (logits/CE/dhidden bitwise);
        # only the (V,H) dW's M-reduction GROUPING changes (≤1 ulp on the embed grad — grads
        # need determinism, not bitwise; lp is untouched ⇒ ratio=1 holds).
        Z = _DBuf.zeros
        self.MC = min(M, 1024)
        self.gin   = Z((M, H), np.float32)
        self.gh    = Z((M, H), np.uint16)
        self.gids  = Z((M, 1), np.int32)
        # input-embed dW group layout (host stable-sort of ids, uploaded per micro-batch):
        # sorted row order + per-unique-id (start, count, id), padded to M with count=0
        self.gsrt, self.ggst, self.ggcnt, self.ggid = (Z((M, 1), np.int32) for _ in range(4))
        self.glog  = Z((self.MC, V), np.float32)
        self.gglog = Z((self.MC, V), np.uint16)
        self.glp   = Z((M, 1), np.float32)
        self.glse  = Z((M, 1), np.float32)
        self.gadv  = Z((M, 1), np.float32)
        self.glab  = Z((M, 1), np.int32)
        self.gdhid = Z((M, H), np.float32)
        self.gegrad = Z((V, H), np.float32)
        self.gdin  = Z((M, H), np.uint16)

    def _build_layer(self, cfg, lw, i):
        if lw["ffn_dense"]:
            return ResidentMoEDenseLayer(cfg, {**lw["attn"], **lw["ffn"]}, self.B, self.S,
                                         is_global=lw["is_global"], window=cfg.window, lid=i,
                                         mxfp8=self.mxfp8, fp8_bwd=self.fp8_bwd,
                                         optimizer=self.optimizer, muon_scratch=self._muon_scratch)
        return ResidentMoELayer(cfg, lw["attn"], lw["ffn"], self.B, self.S,
                                is_global=lw["is_global"], window=cfg.window, lid=i,
                                device_route=self.device_route, mxfp8=self.mxfp8, fp8_bwd=self.fp8_bwd,
                                optimizer=self.optimizer, muon_scratch=self._muon_scratch,
                                muon_scratch_e=self._muon_scratch_e, muon_lr=self.muon_lr)

    # intermediate scratch (present on every layer via ResidentLayerTrain) the backward RECOMPUTES →
    # shareable across layers in long_context. NOT shared: w/wn (weights), opt (AdamW), G (weight grads),
    # gcos/gsin (rope tables), the fp8 quant buffers, and the per-layer checkpoint input gx_in.
    _SHARE = ("gh gq gk gv gqn gkn gqh gkh gvh gqr gkr gO gL gotok gattn gx2 gh2 gg gu ga gmlp gout "
              "r1 rq rk r2 gda gdg gdu gdh2a gdh2b gdh2 gdx2m gdx2 gdotok gdohm gDelta gdqr gdkr gdvh "
              "gdqrb gdkrb gdqnhm gdknhm gdqn gdkn gdv gdq gdk gdh1q gdh1k gdh1v gdh1t gdh1 gdxa gdx "
              "part").split()

    # GroupedMoEFFN scratch (lazily prealloc'd) that the backward recomputes → shareable across MoE
    # layers. NOT shared: the expert weights (Wg/Wu/Wd + transposes), weight grads (dWg/dWu/dWd), and
    # the router weight/grad (Wr_dev/G_router_dev/Gr_acc/G_router_part).
    _MOE_SHARE = ("Xg Ag Gg Ug Yg Out hbits gsrc ggate gtile gtoks goff dOut dYg dsg dAg dGg dUg dXg "
                  "dH gdh2_e dhr dtopi dtopw dprobs dlogits").split()

    def _alias_scratch(self, l, L0):
        """Free layer l's _SHARE intermediate scratch and point it at layer 0's. Called as EACH layer
        is built (free-as-you-go) so the construction peak is ~2 layers' scratch, not all NL — only
        ONE layer's intermediates are ever resident. The backward recomputes l's forward into this
        shared scratch (deterministic ⇒ bitwise-identical to full-store)."""
        for n in self._SHARE:
            buf = getattr(l, n, None)
            if buf is not None and hasattr(L0, n):
                buf.free(); setattr(l, n, getattr(L0, n))

    def _setup_checkpoint(self):
        """long_context finalize (the per-layer _SHARE aliasing already happened during the build via
        _alias_scratch): (1) a PERSISTENT input buffer gx_in per layer (the shared gout is reused), and
        (2) share the MoE-FFN scratch across MoE layers. Bitwise == full-store — gated test_checkpoint.py."""
        M, H = self.M, self.H
        for l in self.layers:
            l.gx_in = _DBuf.zeros((M, H), np.float32)
        # share the MoE FFN scratch across MoE layers (prealloc early — it's lazy on first forward)
        moe = [l.moe for l in self.layers if hasattr(l, "moe")]
        if len(moe) > 1:
            for m in moe:
                if m._M != M: m._prealloc(M)
            for m in moe[1:]:
                for n in self._MOE_SHARE:
                    buf = getattr(m, n, None)
                    if buf is not None and hasattr(moe[0], n):
                        buf.free(); setattr(m, n, getattr(moe[0], n))

    # ── forward ──────────────────────────────────────────────────────────────
    def _fwd_dev(self, si: int):
        """Pure device launch chain (no sync, no host round-trip — graph-capturable):
        embed gather → layers → final RMSNorm (trunc + stats + apply → gh bf16 bits)."""
        M, H, V = self.M, self.H, self.V
        ct.launch(si, (M, 1, 1), _embed_gather, (self.gids, self.gembed, self.gin, H // 128))
        gx = self.gin
        for l in self.layers:
            if self.long_context:                      # store this layer's input (shared gout is reused)
                cdrv.cuMemcpyDtoDAsync(l.gx_in.ptr, gx.ptr, gx.nbytes, si)
            gx = l.forward(gx, si)
        ct.launch(si, (M // RTM, H // RTN, 1), _cast_bf16, (gx, self.gnx))
        ct.launch(si, (M // NTM, 1, 1), _rmsnorm_stats, (self.gnx, self.grfn, H // TH, 1.0 / H, float(self.eps)))
        ct.launch(si, (M // NTM, 1, 1), _rmsnorm_apply, (self.gnx, self.gfnw, self.grfn, self.gh, H // TH))

    def _upload_ids(self, ids, si: int):
        """Upload (M,) token ids. The old HOST (M,V) onehot build + upload (1.2 GB numpy
        scatter + 0.6 GB PCIe at M=1024) dominated the forward (~0.3 s/step); the forward
        gather is now _embed_gather (bitwise == the onehot GEMM — decode-proven) and the
        backward builds a per-CHUNK device onehot only where the input-embed dW needs it."""
        self._ids_keep = np.ascontiguousarray(np.asarray(ids).reshape(self.M, 1).astype(np.int32))
        cdrv.cuMemcpyHtoDAsync(self.gids.ptr, self._ids_keep, self.gids.nbytes, si)
        # input-embed dW group layout for _embed_dw_scatter (stable sort ⇒ deterministic order)
        self._grp_keep = build_id_groups(self._ids_keep[:, 0])
        for buf, arr in zip((self.gsrt, self.ggst, self.ggcnt, self.ggid), self._grp_keep):
            cdrv.cuMemcpyHtoDAsync(buf.ptr, arr, buf.nbytes, si)

    def _chunks(self, rows):
        m0 = 0
        while m0 < rows:
            yield m0, min(self.MC, rows - m0)
            m0 += self.MC

    def forward(self, input_ids: np.ndarray, si: int):
        """input_ids (B,S) int → post-final-norm hidden as bf16 BITS (M,H) uint16."""
        self._upload_ids(input_ids, si)
        self._fwd_dev(si)
        cudart.cudaStreamSynchronize(si)
        return self.gh.to_numpy()[:self.M]

    # ── loss + backward ──────────────────────────────────────────────────────
    def _bwd_head_dev(self, si: int, inv_nrm: float, gacc: bool = False):
        """Head: logits → CE stats/grad → dhidden + LM-head dW. Pure launches (graph-capturable).
        Subclass hook — the prefix model widens the grids to Mh and adds the boundary _bnd_acc.
        gacc: gradient accumulation — chunk 0 also ADDS onto gegrad instead of overwriting."""
        M, H, V = self.M, self.H, self.V; T = _DWT
        for ci, (m0, mc) in enumerate(self._chunks(M)):
            gh_c = self.gh.at_pos(m0)
            ct.launch(si, (mc // 128, V // 128, 1), _gemm_nt_f32, (gh_c, self.gembed, self.glog, H // 64, 128, 128, 64))
            ct.launch(si, (mc // _CTB, 1, 1), _ce_stats_b,
                      (self.glog, self.glab.at_pos(m0), self.glp.at_pos(m0), self.glse.at_pos(m0), V // TV, _CTB))
            ct.launch(si, (mc // _CTB, 1, 1), _ce_grad_b,
                      (self.glog, self.glse.at_pos(m0), self.glab.at_pos(m0), self.gadv.at_pos(m0), self.gglog, V // TV, inv_nrm, _CTB))
            ct.launch(si, (mc // GTM, H // 32, 1), _gemm, (self.gglog, self.gembed, self.gdhid.at_pos(m0), V // GTK, GTM, 32, GTK))
            dw = _gemm_dW if (ci == 0 and not gacc) else _gemm_dW_acc
            ct.launch(si, (V // T, H // _DWN, 1), dw, (self.gglog, gh_c, self.gegrad, mc // T, T, _DWN, T))

    def _set_gacc(self, gacc: bool):
        """Flip every grad producer between overwrite (micro-batch 0) and add-in-place (≥1)."""
        for l in self.layers:
            l.gacc = gacc
            if hasattr(l, "moe"):
                l.moe.gacc = gacc

    def _bwd_dev(self, si: int, inv_nrm: float, gacc: bool = False):
        """The ENTIRE backward as pure launches (graph-capturable): head → final-norm bwd (device:
        RNE-cast(dhidden) → dx into gdin, dw 2-pass into gfng — bitwise == the old host
        rmsnorm_backward) → layers backward → input-embed dW + accumulate."""
        M, H = self.M, self.H
        self._set_gacc(gacc)
        self._bwd_head_dev(si, inv_nrm, gacc)
        ct.launch(si, (M // RTM, H // RTN, 1), _cast_bf16, (self.gdhid, self.gdy))
        ct.launch(si, (M // NTM, 1, 1), _rmsnorm_bwd_dx, (self.gnx, self.gfnw, self.gdy, self.grfn, self.gdin, H // TH, 1.0 / H))
        mb = M // NTM; bpp = (mb + PART - 1) // PART
        ct.launch(si, (H // TD, PART, 1), _rmsnorm_dw_part, (self.gnx, self.gdy, self.grfn, self.gprt, mb, bpp))
        ct.launch(si, (H // TD, 1, 1), _rmsnorm_dw_reduce_acc if gacc else _rmsnorm_dw_reduce,
                  (self.gprt, self.gfng))
        gd = self.gdin
        for i in reversed(range(self.NL)):
            if self.long_context:                              # recompute this layer's forward from its
                self.layers[i].forward(self.layers[i].gx_in, si)   # PERSISTENT input → repopulates the
            gd = self.layers[i].backward(gd, si)                   # shared scratch (deterministic ⇒ exact)
        # input-embed dW accumulates DIRECTLY onto the LM-head grad — deterministic sorted-
        # scatter (2026-07-13): out[id] += Σ f32(gd rows of that id) in stable-sorted order,
        # ONE launch. Replaces the (MC,V) onehot GEMM (2·M·V·H FLOPs + 311MB gohot for M·H
        # useful adds — 11.43→0.147 ms at M=2048, 78×; bit-identical to the GEMM at real V,
        # ≤1e-7 regroup-ulp in the duplicate-heavy worst case; test_embed_dw_scatter.py).
        ct.launch(si, (M, H // 128), _embed_dw_scatter,
                  (self.ggcnt, self.ggst, self.ggid, self.gsrt, gd, self.gegrad, 128))

    def loss_backward(self, h, labels, si: int, advantage=None, norm=None, accumulate=False):
        """labels (M,). Returns CE. Layer grads land in-layer (device); embed grad in self.gegrad;
        expert/router grads in each MoE layer's .moe; final-norm grad in self.gfng (device).
        norm: loss-normalization count (default M → mean over all rows). GRPO passes the number of
        SCORED completion tokens so prefix and replicated paths normalize identically.
        accumulate=True (GRADIENT ACCUMULATION, micro-batch ≥1): ADD this micro-batch's weight
        grads onto the existing ones in place — run forward+loss_backward per micro-batch (each a
        fresh sequence batch at this model's (B,S)), first with accumulate=False, then step() once.
        Pass the same `norm` (= total tokens over ALL micro-batches) to every call so the summed
        grads equal the big-batch mean."""
        M = self.M
        nrm = float(M if norm is None else norm)
        adv = np.ones(M, np.float32) if advantage is None else advantage.astype(np.float32)
        cdrv.cuMemcpyHtoD(self.gadv.ptr, np.ascontiguousarray(adv.reshape(M, 1)), self.gadv.nbytes)
        cdrv.cuMemcpyHtoD(self.glab.ptr, np.ascontiguousarray(labels.astype(np.int32).reshape(M, 1)), self.glab.nbytes)
        self._bwd_dev(si, 1.0 / nrm, accumulate)
        cudart.cudaStreamSynchronize(si)
        lp = self.glp.to_numpy().reshape(M)
        return float(-(adv * lp).sum() / nrm)

    # ── optimizer step (Muon/AdamW hybrid; embed/LM-head + final-norm always AdamW) ──────────
    def step(self, si: int, lr: float = 1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, muon_lr=None):
        ml = self.muon_lr if muon_lr is None else muon_lr
        if self.batched_proj_muon is not None:               # batched square proj NS (all layers, one shot)
            self.batched_proj_muon.step(si, ml)
        for l in self.layers:
            l.step(si, lr, b1, b2, eps, wd, ml)              # q/o per-weight muon + norms AdamW + experts
        self.t += 1
        ibc1 = 1.0 / (1.0 - b1 ** self.t); ibc2 = 1.0 / (1.0 - b2 ** self.t)
        s = self.eopt
        ct.launch(si, (s["R"] // s["otm"], 1, 1), _adamw,
                  (self.gegrad.view((s["R"], ADAM_C)), s["m"], s["v"], s["p32"], s["p16"], s["otm"],
                   float(b1), float(b2), float(eps), float(lr), 0.0, float(ibc1), float(ibc2)))
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        g = self.gfng.to_numpy().reshape(self.final_norm.shape)       # device final-norm grad (4 KB)
        self.fn_m = b1 * self.fn_m + (1 - b1) * g
        self.fn_v = b2 * self.fn_v + (1 - b2) * g * g
        self.final_norm -= lr * ((self.fn_m / bc1) / (np.sqrt(self.fn_v / bc2) + eps))
        cdrv.cuMemcpyHtoD(self.gfnw.ptr, _f32bf_rne(self.final_norm.reshape(1, -1)), self.gfnw.nbytes)
