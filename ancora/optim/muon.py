"""
ancora/optim/muon.py — Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Muon (Keller Jordan 2024; nanoGPT speedrun; Moonshot Kimi) updates 2D weight MATRICES
with an orthogonalized momentum: the update is the nearest semi-orthogonal matrix to the
momentum (via a quintic Newton-Schulz iteration). Empirically ~2× token-efficiency vs
AdamW on the matmul weights. Used ONLY for 2D hidden matrices; embeddings / LM-head / 1D
gains stay on AdamW (→ the hybrid). See [[mfu-strategy]].

  buf = μ·buf + g                       # heavy-ball momentum
  u   = g + μ·buf   (Nesterov)          # look-ahead
  O   = NewtonSchulz5(u)                # ≈ U Vᵀ of u (orthogonalize, kill the spectrum)
  W  -= lr · √max(1, rows/cols) · O     # shape-aware scale (Keller's repo)

v1 NOTE: Newton-Schulz runs in fp32 on host (numpy) — it's small (≤3072²) and runs once
per optimizer step, not per token. The 3 matmuls/iter (X Xᵀ, A², B X) move to the BF16
`loss._gemm` on-device as the perf follow-up (that's where cuda-tile gets reused here).
"""
import numpy as np
import cuda.tile as ct
from cuda.bindings import runtime as cudart
from ancora.kernels.loss import _gemm, GTM, GTN, GTK, _GpuArray, f32_to_bf16_bits as _f32bf


def _dmm(Ah, Bh, si):
    """C = A @ B on device via the compiled BF16 _gemm (f32 accumulate). A:(M,K) B:(K,N), both
    bf16-valued f32 host arrays. Requires M%128, K%64, N%128. Used for the Newton-Schulz matmuls."""
    M, K = Ah.shape; K2, N = Bh.shape
    gA = _GpuArray(_f32bf(Ah)); gB = _GpuArray(_f32bf(Bh)); gC = _GpuArray.zeros((M, N), np.float32)
    ct.launch(si, (M // GTM, N // GTN, 1), _gemm, (gA, gB, gC, K // GTK, GTM, GTN, GTK))
    cudart.cudaStreamSynchronize(si)
    C = gC.to_numpy()
    for g in (gA, gB, gC): g.free()
    return C


def _ns_ok(shape):
    """The big NS matmuls fit the GEMM tiling iff both dims are tile-aligned (M%128,N%128,K%64).
    Small matrices (e.g. router (H,E)) fail → stay on the host NS (they're cheap anyway)."""
    r, c = shape
    return r % GTM == 0 and c % GTM == 0 and (r % GTK == 0) and (c % GTK == 0)


def newton_schulz5_device(G: np.ndarray, si: int, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    """Device Newton-Schulz via the host-API _gemm. MEASURED 2.5× SLOWER than the numpy/BLAS host
    version (2026-06-06): each of the 360 _gemm calls (3 matmuls × 5 iters × 24 matrices) round-trips
    (alloc 3 bufs + upload 2 + download 1 + sync), and that overhead dwarfs the tiny matmul (≤2048²)
    that CPU BLAS already does at ~770 GFLOP/s. Kept as the empirical proof that PIECEMEAL kernel-
    moving loses — the round-trip is the wall. A real win needs a RESIDENT NS (upload G once, chain
    the 15 GEMMs + axpy/cast on-device, download O once → ~50× est.), which needs an A@Bᵀ GEMM + axpy
    kernels — part of the device-residency (megakernel) effort. So default device=False. bf16 NS is
    also ~3% off the fp32 host NS and occasionally under-orthogonalizes (σ_min≈0.02 seen)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = np.ascontiguousarray(X.T)
    X = X / (np.linalg.norm(X) + eps)
    for _ in range(steps):
        A  = _dmm(X, np.ascontiguousarray(X.T), si)      # X @ Xᵀ
        A2 = _dmm(A, A, si)                                # A @ A
        B  = b * A + c * A2
        X  = a * X + _dmm(B, X, si)                        # a·X + B@X
    return X.T if transposed else X


def newton_schulz5(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    """Orthogonalize G via the quintic Newton-Schulz iteration (Keller Jordan coeffs).
    Returns a matrix with the same shape whose singular values are ≈1."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (np.linalg.norm(X) + eps)        # ‖·‖_F normalize → spectral norm ≤ 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


def _bf(x):
    u = x.astype(np.float32).view(np.uint32); u = u + 0x7FFF + ((u >> 16) & 1)
    return ((u >> 16).astype(np.uint32) << 16).view(np.float32)


class DeviceMuon:
    """DEVICE-RESIDENT Muon over one 2D weight (the resident-layer integration manages many of these
    sharing the NS scratch). State = fp32 master `p32` + ONE momentum buffer `buf` (NO second moment
    → ~1/3 less optimizer memory than AdamW's master+m+v). The step runs ENTIRELY on device:
    momentum+Nesterov → (transpose tall) → resident Newton-Schulz → (transpose back) → master update →
    bf16 refresh. `p16` is the BF16 weight the forward GEMM reads (== AdamW's p16 role)."""
    def __init__(self, W: np.ndarray, lr=0.02, momentum=0.95, ns_steps=5):
        from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf
        from ancora.kernels.muon_ns import NTM
        K, N = W.shape
        assert K % NTM == 0 and N % NTM == 0 and K % 64 == 0 and N % 64 == 0, f"Muon weight {W.shape} tile-misaligned"
        self.K, self.N, self.tr = K, N, K > N           # transpose tall (NS works on rows≤cols)
        self.M, self.Nn = (N, K) if self.tr else (K, N)  # NS operates on (M, Nn), M ≤ Nn
        self.lr_scale = lr * max(1.0, K / N) ** 0.5      # Keller's shape-aware scale
        self.mom, self.ns = momentum, ns_steps
        self.p32 = _GpuArray(W.astype(np.float32).copy())
        self.buf = _GpuArray.zeros((K, N), np.float32)
        self.p16 = _GpuArray(_f32bf(W))
        Z = lambda *s: _GpuArray.zeros(s, np.uint16)
        self.u = Z(K, N)                                 # momentum-Nesterov (bf16); also O in the tall path
        self.gA, self.gA2, self.gB = Z(self.M, self.M), Z(self.M, self.M), Z(self.M, self.M)
        self.scr = Z(self.M, self.Nn)                    # NS B@X scratch — MUST be (M,Nn), never aliased to u
        self.gBX = Z(self.M, self.Nn) if self.tr else None  # holds uᵀ (the NS input) only when transposing
        self.recip = _GpuArray.zeros((1, 1), np.float32)

    def step(self, g_dev, si):
        """g_dev: (K,N) f32 device gradient. Updates p32 (master) + p16 (bf16 weight) in place."""
        from ancora.kernels.muon_ns import (_muon_mom, _muon_update, _transpose_mat,
                                             newton_schulz_resident, NTM, NTN)
        from ancora.kernels.fused import _cast_bf16, RTM, RTN
        K, N = self.K, self.N
        ct.launch(si, (K // NTM, N // NTN, 1), _muon_mom, (self.buf, g_dev, self.u, float(self.mom), NTM, NTN))
        if self.tr:                                      # u(K,N) → NS input X(M=N, Nn=K)
            ct.launch(si, (K // 64, N // 64, 1), _transpose_mat, (self.u, self.gBX, 64))
            X = self.gBX
        else:
            X = self.u
        newton_schulz_resident(X, self.gA, self.gA2, self.gB, self.scr, self.recip, self.M, self.Nn, si, self.ns)
        if self.tr:                                      # orthogonalized X(M,Nn) → O(K,N)
            ct.launch(si, (self.M // 64, self.Nn // 64, 1), _transpose_mat, (X, self.u, 64))
            O = self.u
        else:
            O = X
        ct.launch(si, (K // NTM, N // NTN, 1), _muon_update, (self.p32, O, float(self.lr_scale), NTM, NTN))
        ct.launch(si, (K // RTM, N // RTN, 1), _cast_bf16, (self.p32, self.p16))

    def master(self):
        return self.p32.to_numpy()


class MuonScratch:
    """SHARED transient Newton-Schulz scratch for the resident Muon. Every 2D weight in a step runs
    its NS SEQUENTIALLY on `si`, so ONE scratch serves all of them — per-weight scratch would cost
    ~2.7 GB (the (M,M)+(M,Nn) buffers × ~196 weights) and ERASE the v-buffer saving. Sized for the
    largest weight; each weight .view()s the flat pools to its own (M,Nn)/(M,M) (the declared shape
    sets the cuda-tile row stride, so a smaller view over a bigger pool is correct). One-time ~25 MB
    for the whole model. The only PER-WEIGHT persistent state lives in ResidentMuon.buf."""
    def __init__(self, shapes):
        from ancora.model.resident import _DBuf
        kn = max(int(K) * int(N) for K, N in shapes)              # u / scr / X each need ≤ K·N = M·Nn elems
        mm = max(min(int(K), int(N)) ** 2 for K, N in shapes)     # gA/gA2/gB are (M,M), M=min(K,N)
        Zf = lambda n: _DBuf.zeros((1, int(n)), np.uint16)
        self.u, self.scr, self.X = Zf(kn), Zf(kn), Zf(kn)
        self.gA, self.gA2, self.gB = Zf(mm), Zf(mm), Zf(mm)
        self.recip = _DBuf.zeros((1, 1), np.float32)

    def nbytes(self):
        return self.u.nbytes * 3 + self.gA.nbytes * 3 + self.recip.nbytes

    def free(self):
        for b in (self.u, self.scr, self.X, self.gA, self.gA2, self.gB, self.recip):
            b.free()


class ResidentMuon:
    """Per-2D-weight Muon over the resident layer's EXISTING buffers (p32 master + p16 the GEMM reads),
    with a SHARED MuonScratch. Persistent state = ONE momentum buffer `buf` (NO second moment) →
    p32+buf = 8 B/param vs AdamW's p32+m+v = 12 B/param: it drops the `v` buffer (~4 B/param ≈ 1.7 GB
    on the framework's 2D weights). p32/p16 are the SAME _DBufs the layer already holds, so the forward
    GEMM reads the Muon-updated weight in place (zero extra weight memory). lr is per-step (schedulable).
    Shares the validated muon_ns kernels with DeviceMuon; the only delta is shared-scratch views."""
    def __init__(self, p32, p16, K, N, scratch, momentum=0.95, ns_steps=5):
        from ancora.model.resident import _DBuf
        from ancora.kernels.muon_ns import NTM
        assert K % NTM == 0 and N % NTM == 0 and K % 64 == 0 and N % 64 == 0, f"Muon weight ({K},{N}) tile-misaligned"
        self.p32, self.p16, self.K, self.N, self.sc = p32, p16, K, N, scratch
        self.tr = K > N                                   # transpose tall (NS needs rows ≤ cols)
        self.M, self.Nn = (N, K) if self.tr else (K, N)
        self.shape_scale = max(1.0, K / N) ** 0.5         # Keller's shape-aware scale (× lr at step time)
        self.mom, self.ns = momentum, ns_steps
        self.buf = _DBuf.zeros((K, N), np.float32)        # the ONLY persistent per-weight state

    def step(self, g_dev, si, lr=0.02):
        """g_dev: (K,N) f32 device weight-grad. Updates p32 + p16 in place. Uses the shared scratch."""
        from ancora.kernels.muon_ns import (_muon_mom, _muon_update, _transpose_mat,
                                             newton_schulz_resident, NTM, NTN)
        from ancora.kernels.fused import _cast_bf16, RTM, RTN
        K, N, sc, M, Nn = self.K, self.N, self.sc, self.M, self.Nn
        p32, p16, u = self.p32.view((K, N)), self.p16.view((K, N)), sc.u.view((K, N))
        ct.launch(si, (K // NTM, N // NTN, 1), _muon_mom, (self.buf, g_dev, u, float(self.mom), NTM, NTN))
        if self.tr:                                       # u(K,N) → NS input X(M=N, Nn=K)
            X = sc.X.view((M, Nn))
            ct.launch(si, (K // 64, N // 64, 1), _transpose_mat, (u, X, 64))
        else:
            X = u
        newton_schulz_resident(X, sc.gA.view((M, M)), sc.gA2.view((M, M)), sc.gB.view((M, M)),
                               sc.scr.view((M, Nn)), sc.recip, M, Nn, si, self.ns)
        if self.tr:                                       # orthogonalized X(M,Nn) → O(K,N)
            ct.launch(si, (M // 64, Nn // 64, 1), _transpose_mat, (X, u, 64))
            O = u
        else:
            O = X
        ct.launch(si, (K // NTM, N // NTN, 1), _muon_update, (p32, O, float(lr * self.shape_scale), NTM, NTN))
        ct.launch(si, (K // RTM, N // RTN, 1), _cast_bf16, (p32, p16))

    def free(self):
        self.buf.free()


class BatchedProjMuon:
    """Model-level BATCHED Muon over SAME-SHAPE square projection weights ACROSS ALL layers — the
    industrial single-GPU Muon pattern (Keller/Kimi: group by shape, batch the Newton-Schulz). The
    per-weight NS was launch-overhead-dominated (~195 ms/step proj at NL=12, ~1950 tiny launches);
    batching the NS over `chunk` weights at a time pushes it toward the GEMM compute floor. SQUARE
    (K==N) only — k/v/gate/up/down are 1024² in this model, EXACTLY the expert shape → reuses the
    batched-expert NS (newton_schulz_resident_e) verbatim; q/o (rectangular) stay per-weight (phase
    1b). The per-weight momentum/update stay single-kernel (cheap) and write to / read from packed `u`
    slices via at_pos; only the expensive 5-iter NS chain is batched. Operates IN PLACE on the layers'
    existing buf/p32/p16/G — drop-in for the per-weight ResidentMuon proj step.

    weights: list of {buf:(K,N)f32, p32:(K,N)f32 view, p16:(K,N)bf16, G:(K,N)f32, K, N} for square proj."""
    def __init__(self, weights, momentum=0.95, ns_steps=5, chunk=8):
        from ancora.model.resident import _DBuf
        from ancora.kernels.muon_ns import NTM
        self.mom, self.ns = momentum, ns_steps
        self.groups = {}                                     # K -> [weight dicts] (all K==N here)
        for w in weights:
            assert w["K"] == w["N"], f"BatchedProjMuon: square only, got ({w['K']},{w['N']})"
            assert w["K"] % NTM == 0, f"K={w['K']} not 128-aligned"
            self.groups.setdefault(w["K"], []).append(w)
        self.scr = {}                                        # one chunk's packed u + NS scratch per K
        for K, ws in self.groups.items():
            c = min(chunk, len(ws))
            Z = lambda *s: _DBuf.zeros(s, np.uint16)
            self.scr[K] = dict(c=c, u=Z(c * K, K), gA=Z(c * K, K), gA2=Z(c * K, K),
                               gB=Z(c * K, K), gBX=Z(c * K, K), recip=_DBuf.zeros((c, 1), np.float32))

    def step(self, si, lr=0.02):
        from ancora.kernels.muon_ns import _muon_mom, _muon_update, newton_schulz_resident_e, NTM, NTN
        from ancora.kernels.fused import _cast_bf16, RTM, RTN
        for K, ws in self.groups.items():
            s = self.scr[K]; c0 = s["c"]; u = s["u"]
            for i0 in range(0, len(ws), c0):                 # chunk the group to cap the NS scratch
                grp = ws[i0:i0 + c0]; c = len(grp)
                for i, w in enumerate(grp):                  # momentum+Nesterov → packed u slice i
                    ct.launch(si, (K // NTM, K // NTN, 1), _muon_mom,
                              (w["buf"], w["G"], u.at_pos(i * K), float(self.mom), NTM, NTN))
                newton_schulz_resident_e(u, s["gA"], s["gA2"], s["gB"], s["gBX"], s["recip"], c, K, K, si, self.ns)
                for i, w in enumerate(grp):                  # fp32 master update + bf16 weight refresh
                    ct.launch(si, (K // NTM, K // NTN, 1), _muon_update,
                              (w["p32"], u.at_pos(i * K), float(lr), NTM, NTN))
                    ct.launch(si, (K // RTM, K // RTN, 1), _cast_bf16, (w["p32"], w["p16"]))

    def free(self):
        for s in self.scr.values():
            for b in (s["u"], s["gA"], s["gA2"], s["gB"], s["gBX"], s["recip"]): b.free()


class Muon:
    """fp32-master Muon for 2D matrix params. params: {name: (R,C) fp32 ndarray}."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 device=False, si=None):
        self.lr, self.mom, self.nesterov, self.ns = lr, momentum, nesterov, ns_steps
        self.device, self.si = device, si               # device=True → Newton-Schulz matmuls on GPU
        self.w = {n: p.astype(np.float32).copy() for n, p in params.items()}
        self.buf = {n: np.zeros_like(self.w[n]) for n in self.w}
        for n, p in params.items():
            assert p.ndim == 2, f"Muon is for 2D matrices, {n} is {p.shape}"

    def step(self, grads, stream_int=None):
        si = stream_int if stream_int is not None else self.si
        for n, W in self.w.items():
            g = grads[n].astype(np.float32)
            buf = self.buf[n]
            buf *= self.mom; buf += g                       # buf = μ·buf + g
            u = g + self.mom * buf if self.nesterov else buf
            o = (newton_schulz5_device(u, si, self.ns)
                 if (self.device and si is not None and _ns_ok(u.shape)) else newton_schulz5(u, self.ns))
            scale = max(1.0, W.shape[0] / W.shape[1]) ** 0.5
            W -= self.lr * scale * o

    def weights(self):
        return {n: _bf(W) for n, W in self.w.items()}       # BF16-valued for the forward

    def master(self):
        return {n: W.copy() for n, W in self.w.items()}

    def free(self):
        pass
