"""
ancora/model/resident.py — ResidentLayer: a device-resident Qwen3 transformer layer.

Holds persistent device buffers + pre-quantized MXFP8 weights; forward() chains every
kernel on-device (no host round-trips, no per-call alloc) with the megakernel fusions:
  • input_ln / post_ln → _rmsnorm_apply_q  (emit FP8+E8M0 directly, no bf16 round-trip)
  • gate/up GEMM + SwiGLU + quant          → _gateup_swiglu_q   (gate/up never touch HBM)
  • o_proj / down_proj + residual          → _fwd_mxfp8_bf16_res (residual in epilogue)

This is the clean base the persistent megakernel + CUDA-graph capture build on: one object
owns the buffers and the kernel schedule, so capturing/replaying or fusing the schedule is a
local change. Validated full-layer forward MFU ~55% of BF16 peak (gemm_mfu_ceiling memory).

Scope: forward (inference/rollout path, the megakernel target). The training path needs the
cache (gate/up materialized) for backward — that lives in the validated bwd chain
(tests/model/test_resident_layer_bwd.py) and folds in here later via forward(training=True).
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import ctypes
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv
import ancora.env  # noqa: F401

from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, _rmsnorm_apply_q,
                                  _rmsnorm_stats_f32, _rmsnorm_apply_q_f32, TM as NTM, TH, QB)
from ancora.kernels.rope import _rope_fwd_tok, RTM as RRTM, build_cos_sin
from ancora.kernels.attention import _attn_fwd_tok_q, BQ
from ancora.kernels.fused import _gateup_swiglu_q, _residual_add, _residual_add_rf32, RTM, RTN, TT
from ancora.kernels.activation import _swiglu_fwd_q
from ancora.kernels.linear import _fwd_mxfp8_bf16, _fwd_mxfp8_bf16_res, _fwd_mxfp8_f32res, mxfp8_tile
from ancora.kernels.quant import _quant_mxfp8, QTM, B as QB32, quantize_colblock

# ── optional CUTLASS MXFP8 GEMM hybrid (1.2-1.4× over cuda-tile, see gemm_mfu_ceiling memory) ──
_CUTLASS = None
def _load_cutlass(dll_path=r"C:\project\cutlass\cutlass_mxfp8.dll"):
    """Load the CUTLASS MXFP8 GEMM DLL (init/set_scales/run/free). Returns None if unavailable."""
    global _CUTLASS
    if _CUTLASS is None:
        d = ctypes.CDLL(dll_path)
        d.cutlass_mxfp8_init.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3; d.cutlass_mxfp8_init.restype = ctypes.c_void_p
        d.cutlass_mxfp8_set_scales.argtypes = [ctypes.c_void_p] * 4
        d.cutlass_mxfp8_run.argtypes = [ctypes.c_void_p] * 2; d.cutlass_mxfp8_run.restype = ctypes.c_int
        d.cutlass_mxfp8_free.argtypes = [ctypes.c_void_p]
        _CUTLASS = d
    return _CUTLASS

_f32bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


class _DBuf:
    """Persistent device buffer exposing __cuda_array_interface__ (+ cheap reshaped views)."""
    def __init__(self, arr):
        arr = np.ascontiguousarray(arr)
        self.shape, self.dtype, self.nbytes = arr.shape, arr.dtype, arr.nbytes
        err, self.ptr = cdrv.cuMemAlloc(arr.nbytes)
        if err.value:   # NEVER silently accept a bad ptr — an OOM here = garbage numbers downstream
            raise RuntimeError(f"_DBuf cuMemAlloc({arr.nbytes}) failed: {err}")
        cdrv.cuMemcpyHtoD(self.ptr, arr, arr.nbytes)
        self.__cuda_array_interface__ = {"shape": arr.shape, "typestr": arr.dtype.str,
                                         "data": (int(self.ptr), False), "version": 3}

    @classmethod
    def zeros(cls, shape, dtype=np.uint16):
        return cls(np.zeros(shape, dtype))

    def view(self, shape):
        v = _DBuf.__new__(_DBuf)
        v.shape, v.dtype, v.nbytes, v.ptr = shape, self.dtype, self.nbytes, self.ptr
        v.__cuda_array_interface__ = {"shape": shape, "typestr": np.dtype(self.dtype).str,
                                      "data": (int(self.ptr), False), "version": 3}
        return v

    def at_pos(self, pos):
        """View with the data pointer offset by `pos` ROWS (pos * row_width elements), where
        row_width = shape[-1]. The KV-cache append writes the frontier token's K/V at sequence
        position `pos` (cache laid out (B*Hkv*maxS, D)); the single-position decode RoPE picks
        cos/sin[pos] from a (maxS, D/2) table — both are a base pointer + pos*row_width offset,
        mirroring the host pointer-offset trick in attention._append_kv. NEVER .free() a view."""
        if pos == 0:
            return self
        v = _DBuf.__new__(_DBuf)
        off = pos * self.shape[-1] * np.dtype(self.dtype).itemsize
        v.shape, v.dtype, v.nbytes, v.ptr = self.shape, self.dtype, self.nbytes - off, int(self.ptr) + off
        v.__cuda_array_interface__ = {"shape": self.shape, "typestr": np.dtype(self.dtype).str,
                                      "data": (int(self.ptr) + off, False), "version": 3}
        return v

    def to_numpy(self):
        o = np.empty(self.shape, self.dtype); cdrv.cuMemcpyDtoH(o, self.ptr, self.nbytes); return o

    def free(self):
        cdrv.cuMemFree(self.ptr)


class ResidentLayer:
    """Device-resident Qwen3 transformer layer (fused MXFP8 forward).

    ⚠️ NOT THE ON-POLICY RL PATH. This is the FAST inference forward — token-major + heavily fused
    (norm→fp8, gate/up+SwiGLU+quant, residual-in-epilogue, optional CUTLASS hybrid). Those fusions
    quantize from f32 directly and reorder work, so it is only NUMERICALLY EQUIVALENT (~4.6%) to the
    training forward — it is NOT bitwise. For SFT/RL use the bitwise pair instead:
      training  → ResidentLayerTrain / ResidentModel   (head-major, materializes the bwd cache)
      rollout   → ResidentDecodeLayer / ResidentDecodeModel  (mirrors training kernel-for-kernel ⇒ ratio=1)
    Kept as the validated reference for the fused-MXFP8 forward + CUDA-graph capture + CUTLASS MXFP8
    hybrid — i.e. the substrate the persistent MEGAKERNEL (cross-operator MFU) will build on
    ([[mfu-strategy]], [[gemm-mfu-ceiling]]). A bitwise megakernel would have to fuse the bitwise
    (norm→bf16→fp8, head-major) primitives, not reuse these directly.

        layer = ResidentLayer(cfg, weights, B, S)        # weights: name -> f32 ndarray
        gout  = layer.forward(gx)                          # gx,gout: _DBuf (M,H) FP32 (residual stream)

    Buffers are sized for (B,S) at construction and reused across forward() calls.
    """

    def __init__(self, cfg, weights: dict, B: int, S: int, use_cutlass: bool = False):
        self.cfg, self.B, self.S = cfg, B, S
        self._weights = weights
        H, Hq, Hkv, Dh, I = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate
        self.eps = cfg.eps
        M, qd, kd = B * S, Hq * Dh, Hkv * Dh
        self.M, self.qd, self.kd, self.H, self.I = M, qd, kd, H, I
        self.Hq, self.Hkv, self.Dh = Hq, Hkv, Dh
        self.NSB, self.NQB = S // TT, S // BQ
        self.scale = 1.0 / math.sqrt(Dh)
        assert M % 128 == 0, "M=B*S must be divisible by 128 (gate/up fusion TM)"

        # ── weights: MXFP8 (fp8+E8M0) for the projections, bf16 bits for the norm gains ──
        self.W = {}   # name -> (w_fp8 _DBuf, w_scale _DBuf, K, N)
        for nm in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            wf32 = weights[nm].astype(np.float32); K, N = wf32.shape
            wfp8, wsc = quantize_colblock(wf32)
            self.W[nm] = (_DBuf(wfp8), _DBuf(wsc), K, N)
        self.wln  = _DBuf(_f32bf(weights["input_ln"].reshape(1, H)))
        self.wpln = _DBuf(_f32bf(weights["post_ln"].reshape(1, H)))
        self.wqn  = _DBuf(_f32bf(weights["q_norm"].reshape(1, Dh)))
        self.wkn  = _DBuf(_f32bf(weights["k_norm"].reshape(1, Dh)))
        cosv, sinv = build_cos_sin(S, Dh, cfg.rope_theta)
        self.gcos, self.gsin = _DBuf(cosv), _DBuf(sinv)

        # ── persistent intermediate buffers ──
        # token-major attention (_attn_fwd_tok) reads gqn/gkn/gv directly + applies RoPE inline
        # → NO tok↔head transposes (gqh/gkh/gvh/gqr/gkr) and NO separate RoPE buffers needed.
        Z = _DBuf.zeros
        self.gq = Z((M, qd)); self.gk = Z((M, kd)); self.gv = Z((M, kd))
        self.gqn = Z((M, qd)); self.gkn = Z((M, kd))
        self.gqr = Z((M, qd)); self.gkr = Z((M, kd))   # token-major RoPE'd q/k
        self.gL = Z((M * Hq, 1), np.float32)
        # gx2/gout = FP32 residual carry (matches the training path; keeps the ~6912 massive
        # activation out of bf16 rounding across layers — [[resident-layer]]).
        self.gotok = Z((M, qd)); self.gx2 = Z((M, H), np.float32); self.gout = Z((M, H), np.float32)
        self.r1 = Z((M, 1), np.float32); self.rq = Z((M * Hq, 1), np.float32)
        self.rk = Z((M * Hkv, 1), np.float32); self.r2 = Z((M, 1), np.float32)
        # MXFP8 activation-quant buffers (one per distinct GEMM input)
        self.qh_f = Z((M, H), np.uint8);  self.qh_s = Z((M, H // QB32), np.uint8)    # → q/k/v
        self.qo_f = Z((M, qd), np.uint8); self.qo_s = Z((M, qd // QB32), np.uint8)   # → o
        self.q2_f = Z((M, H), np.uint8);  self.q2_s = Z((M, H // QB32), np.uint8)    # → gate/up
        self.qa_f = Z((M, I), np.uint8);  self.qa_s = Z((M, I // QB32), np.uint8)    # → down

        self.use_cutlass = use_cutlass
        if use_cutlass:
            self._setup_cutlass(weights)

    def _setup_cutlass(self, weights):
        """Build one persistent CUTLASS MXFP8 handle per projection (the GEMMs that are a clear
        1.2-1.4× win vs cuda-tile). Weights pre-quantized + laid COLUMN-major (CUTLASS B), scales
        transposed; our fused-quant kernels already emit the (M,K//32) linear SFA set_scales wants."""
        self.dll = _load_cutlass()
        M, H, I, qd, kd = self.M, self.H, self.I, self.qd, self.kd
        Z = _DBuf.zeros
        self.gg = Z((M, I)); self.gu = Z((M, I))          # gate/up outputs (un-fused for CUTLASS)
        self.o_tmp = Z((M, H)); self.mlp_tmp = Z((M, H))  # CUTLASS o/down output (residual added after)
        # per-projection: (A-fp8 buf, output buf, SFA-linear buf)
        io = {"q_proj": (self.qh_f, self.gq, self.qh_s), "k_proj": (self.qh_f, self.gk, self.qh_s),
              "v_proj": (self.qh_f, self.gv, self.qh_s), "o_proj": (self.qo_f, self.o_tmp, self.qo_s),
              "gate_proj": (self.q2_f, self.gg, self.q2_s), "up_proj": (self.q2_f, self.gu, self.q2_s),
              "down_proj": (self.qa_f, self.mlp_tmp, self.qa_s)}
        cv = lambda p: ctypes.c_void_p(int(p))
        self._cu = {}   # name -> (handle, w_cm _DBuf, sfb_t _DBuf, A_fp8, out, sfa)
        for nm, (Af, Of, Sf) in io.items():
            wf32 = weights[nm].astype(np.float32); K, N = wf32.shape
            wfp8, wsc = quantize_colblock(wf32)
            w_cm  = _DBuf(np.ascontiguousarray(wfp8.T))   # (N,K) row-major == (K,N) column-major
            sfb_t = _DBuf(np.ascontiguousarray(wsc.T))    # (N, K//32)
            h = self.dll.cutlass_mxfp8_init(M, N, K, cv(Af.ptr), cv(w_cm.ptr), cv(Of.ptr))
            assert h, f"CUTLASS init failed for {nm}"
            self._cu[nm] = (h, w_cm, sfb_t, Sf)

    def _cgemm(self, si, nm):
        """Run the CUTLASS GEMM for projection `nm` on stream si (scatter SFA + run)."""
        h, w_cm, sfb_t, sfa = self._cu[nm]
        cv = lambda p: ctypes.c_void_p(int(p))
        self.dll.cutlass_mxfp8_set_scales(h, cv(sfa.ptr), cv(sfb_t.ptr), cv(si))
        self.dll.cutlass_mxfp8_run(h, cv(si))

    # ── kernel-launch helpers (all on the caller's stream) ──
    def _mxg(self, si, a_fp8, a_sc, nm, C):
        wf, ws, K, N = self.W[nm]; TM, TN, TK = mxfp8_tile(N, K)
        ct.launch(si, (self.M // TM, N // TN, 1), _fwd_mxfp8_bf16, (a_fp8, wf, a_sc, ws, C, K // TK, TM, TN, TK))

    def _mxg_res(self, si, a_fp8, a_sc, nm, res, C):
        wf, ws, K, N = self.W[nm]; TM, TN, TK = mxfp8_tile(N, K)   # res/C are FP32 (residual stream)
        ct.launch(si, (self.M // TM, N // TN, 1), _fwd_mxfp8_f32res, (a_fp8, wf, a_sc, ws, res, C, K // TK, TM, TN, TK))

    def _rms(self, si, xb, wb, rstd, yb, rows, hh):
        ct.launch(si, (rows // NTM, 1, 1), _rmsnorm_stats, (xb, rstd, hh // TH, 1.0 / hh, self.eps))
        ct.launch(si, (rows // NTM, 1, 1), _rmsnorm_apply, (xb, wb, rstd, yb, hh // TH))

    def _rms_q(self, si, xb, wb, rstd, fp8, sc, hh):   # fused stats + apply→fp8, FP32 residual input
        ct.launch(si, (self.M // NTM, 1, 1), _rmsnorm_stats_f32, (xb, rstd, hh // TH, 1.0 / hh, self.eps))
        ct.launch(si, (self.M // NTM, 1, 1), _rmsnorm_apply_q_f32, (xb, wb, rstd, fp8, sc, hh // QB))

    def forward(self, gx, si: int):
        """gx: _DBuf (M,H) FP32 residual → self.gout (M,H) FP32. Launches on stream `si`.
        All buffers persistent → safe to CUDA-graph-capture or megakernel-fuse this schedule."""
        B, Hq, Hkv, Dh, H, I, qd = self.B, self.Hq, self.Hkv, self.Dh, self.H, self.I, self.qd
        M, NQB = self.M, self.NQB
        V = lambda g, sh: g.view(sh)
        # ── attention block (token-major + inline RoPE → no transposes, no RoPE kernel) ──
        self._rms_q(si, gx, self.wln, self.r1, self.qh_f, self.qh_s, H)           # input_ln → fp8
        self._mxg(si, self.qh_f, self.qh_s, "q_proj", self.gq)
        self._mxg(si, self.qh_f, self.qh_s, "k_proj", self.gk)
        self._mxg(si, self.qh_f, self.qh_s, "v_proj", self.gv)
        self._rms(si, V(self.gq, (M * Hq, Dh)), self.wqn, self.rq, V(self.gqn, (M * Hq, Dh)), M * Hq, Dh)   # q_norm
        self._rms(si, V(self.gk, (M * Hkv, Dh)), self.wkn, self.rk, V(self.gkn, (M * Hkv, Dh)), M * Hkv, Dh)  # k_norm
        ct.launch(si, (M // RRTM, Hq, 1), _rope_fwd_tok, (self.gqn, self.gcos, self.gsin, self.gqr, self.S // RRTM, Dh // 2))
        ct.launch(si, (M // RRTM, Hkv, 1), _rope_fwd_tok, (self.gkn, self.gcos, self.gsin, self.gkr, self.S // RRTM, Dh // 2))
        # attention with FUSED output MXFP8 quant → writes o_proj's fp8 input directly (no bf16 O, no quant kernel)
        ct.launch(si, (NQB, B * Hq, 1), _attn_fwd_tok_q,
                  (self.gqr, self.gkr, self.gv, self.qo_f, self.qo_s, self.gL, NQB, NQB, Hq, Hkv, self.scale))
        self._mxg_res(si, self.qo_f, self.qo_s, "o_proj", gx, self.gx2)            # o_proj + residual → gx2
        # ── MLP block ──
        self._rms_q(si, self.gx2, self.wpln, self.r2, self.q2_f, self.q2_s, H)     # post_ln → fp8
        wgf, wgs, _, _ = self.W["gate_proj"]; wuf, wus, _, _ = self.W["up_proj"]
        ct.launch(si, (M // 128, I // 32, 1), _gateup_swiglu_q,
                  (self.q2_f, self.q2_s, wgf, wgs, wuf, wus, self.qa_f, self.qa_s, H // 128, 128, 128))
        self._mxg_res(si, self.qa_f, self.qa_s, "down_proj", self.gx2, self.gout)  # down + residual → gout
        return self.gout

    def forward_cutlass(self, gx, si: int):
        """Same schedule as forward() but the 7 projections run on CUTLASS MXFP8 (1.2-1.4× each).
        cuda-tile keeps everything it's already near-optimal at (norms, RoPE, attention, SwiGLU).
        gate/up un-fuse from the megakernel; o/down residuals add separately (CUTLASS has no residual
        epilogue) — still a net win at training size since the CUTLASS GEMMs dominate. Requires use_cutlass=True."""
        assert self.use_cutlass, "construct with use_cutlass=True"
        B, Hq, Hkv, Dh, H, I, qd = self.B, self.Hq, self.Hkv, self.Dh, self.H, self.I, self.qd
        M, NQB = self.M, self.NQB
        V = lambda g, sh: g.view(sh)
        # ── attention block ──
        self._rms_q(si, gx, self.wln, self.r1, self.qh_f, self.qh_s, H)
        self._cgemm(si, "q_proj"); self._cgemm(si, "k_proj"); self._cgemm(si, "v_proj")
        self._rms(si, V(self.gq, (M * Hq, Dh)), self.wqn, self.rq, V(self.gqn, (M * Hq, Dh)), M * Hq, Dh)
        self._rms(si, V(self.gk, (M * Hkv, Dh)), self.wkn, self.rk, V(self.gkn, (M * Hkv, Dh)), M * Hkv, Dh)
        ct.launch(si, (M // RRTM, Hq, 1), _rope_fwd_tok, (self.gqn, self.gcos, self.gsin, self.gqr, self.S // RRTM, Dh // 2))
        ct.launch(si, (M // RRTM, Hkv, 1), _rope_fwd_tok, (self.gkn, self.gcos, self.gsin, self.gkr, self.S // RRTM, Dh // 2))
        ct.launch(si, (NQB, B * Hq, 1), _attn_fwd_tok_q,
                  (self.gqr, self.gkr, self.gv, self.qo_f, self.qo_s, self.gL, NQB, NQB, Hq, Hkv, self.scale))
        self._cgemm(si, "o_proj")                                                  # → o_tmp
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (gx, self.o_tmp, self.gx2))  # fp32 residual
        # ── MLP block ──
        self._rms_q(si, self.gx2, self.wpln, self.r2, self.q2_f, self.q2_s, H)
        self._cgemm(si, "gate_proj"); self._cgemm(si, "up_proj")                    # → gg, gu
        ct.launch(si, (M // 64, 1, 1), _swiglu_fwd_q, (self.gg, self.gu, self.qa_f, self.qa_s, I // QB))
        self._cgemm(si, "down_proj")                                               # → mlp_tmp
        ct.launch(si, (M // RTM, H // RTN, 1), _residual_add_rf32, (self.gx2, self.mlp_tmp, self.gout))  # fp32 residual
        return self.gout
