"""RE-TEST the inline-RoPE-in-attention verdict under cuda-tile 1.5.0 (ct.cat/ct.extract).

The 2026-06-02 verdict (D=64, NO tile-concat): folding rope into attention forced QK^T to
split into two half-wide MMAs → 58→33 TF, net loss → rope stayed a separate kernel.
1.5.0 gives cat/extract → the rotated full-width Q/K tile CAN be rebuilt in registers.

Fused kernel = _attn_fwd_tok + in-register NEOX rotation of Q (once per q-tile) and K
(per kv-tile, inside the loop) with the SAME math as _rope_fwd_tok (f32 rotate → RNE bf16
via ct.astype) so the MMA consumes identical bf16 values if FMA contraction matches.
Reference = production _rope_fwd_tok(Q) + _rope_fwd_tok(K) + _attn_fwd_tok.

Checks: (a) O bitwise vs reference (or ulp count — the 1-ULP separate-compile precedent),
(b) wall time fused vs (ropeQ + ropeK + attn), B=1 S=2048 Hq=16 Hkv=8 D=128."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.attention import _attn_fwd_tok, BQ, BKV, D
from ancora.kernels.rope import _rope_fwd_tok, build_cos_sin, RTM
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

DH = D // 2


@ct.kernel(occupancy=2)
def _attn_fwd_tok_rope(Q, K, V, cos, sin, O, L_out,
                       NQB: ct.Constant[int], NKVB: ct.Constant[int],
                       Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float]):
    """_attn_fwd_tok with RoPE applied IN REGISTERS to raw Q/K (B=1: position == token row).
    Rotation math == _rope_fwd_tok: f32 halves, y1=x1c−x2s, y2=x2c+x1s, RNE→bf16, cat."""
    q_blk = ct.bid(0); hb = ct.bid(1)
    q_head = hb % Hq; batch = hb // Hq
    kv_head = q_head * Hkv // Hq
    qr = batch * NQB + q_blk

    LOG2E = 1.4426950408889634
    qk_scale = scale * LOG2E; NEG_INF = -1e38

    tQraw = ct.load(Q, index=(qr, q_head), shape=(BQ, D))
    q1 = ct.astype(ct.bitcast(ct.extract(tQraw, (0, 0), shape=(BQ, DH)), ct.bfloat16), ct.float32)
    q2 = ct.astype(ct.bitcast(ct.extract(tQraw, (0, 1), shape=(BQ, DH)), ct.bfloat16), ct.float32)
    cq = ct.load(cos, index=(q_blk, 0), shape=(BQ, DH))
    sq = ct.load(sin, index=(q_blk, 0), shape=(BQ, DH))
    tQ = ct.astype(ct.cat((q1 * cq - q2 * sq, q2 * cq + q1 * sq), 1), ct.bfloat16)

    m = ct.full((BQ, 1), NEG_INF, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    for kv in range(q_blk + 1):
        kr = batch * NKVB + kv
        tKraw = ct.load(K, index=(kr, kv_head), shape=(BKV, D), latency=10)
        k1 = ct.astype(ct.bitcast(ct.extract(tKraw, (0, 0), shape=(BKV, DH)), ct.bfloat16), ct.float32)
        k2 = ct.astype(ct.bitcast(ct.extract(tKraw, (0, 1), shape=(BKV, DH)), ct.bfloat16), ct.float32)
        ck = ct.load(cos, index=(kv, 0), shape=(BKV, DH))
        sk = ct.load(sin, index=(kv, 0), shape=(BKV, DH))
        tK = ct.astype(ct.cat((k1 * ck - k2 * sk, k2 * ck + k1 * sk), 1), ct.bfloat16)
        tV = ct.bitcast(ct.load(V, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new

    res = O_acc / ct.broadcast_to(l, (BQ, D))
    ct.store(O, index=(qr, q_head), tile=ct.bitcast(ct.astype(res, ct.bfloat16), ct.uint16))
    ct.store(L_out, index=(hb * NQB + q_blk, 0), tile=m + ct.log2(l))


@ct.kernel(occupancy=2)
def _attn_fwd_tok_ropeq(Q, K, V, cos, sin, O, L_out,
                        NQB: ct.Constant[int], NKVB: ct.Constant[int],
                        Hq: ct.Constant[int], Hkv: ct.Constant[int], scale: ct.Constant[float]):
    """Q-ONLY inline rope (once per tile, OUTSIDE the kv loop); K arrives pre-rotated.
    Isolates loop-register-pressure/K-redundancy from the one-time Q rotation cost."""
    q_blk = ct.bid(0); hb = ct.bid(1)
    q_head = hb % Hq; batch = hb // Hq
    kv_head = q_head * Hkv // Hq
    qr = batch * NQB + q_blk

    LOG2E = 1.4426950408889634
    qk_scale = scale * LOG2E; NEG_INF = -1e38

    tQraw = ct.load(Q, index=(qr, q_head), shape=(BQ, D))
    q1 = ct.astype(ct.bitcast(ct.extract(tQraw, (0, 0), shape=(BQ, DH)), ct.bfloat16), ct.float32)
    q2 = ct.astype(ct.bitcast(ct.extract(tQraw, (0, 1), shape=(BQ, DH)), ct.bfloat16), ct.float32)
    cq = ct.load(cos, index=(q_blk, 0), shape=(BQ, DH))
    sq = ct.load(sin, index=(q_blk, 0), shape=(BQ, DH))
    tQ = ct.astype(ct.cat((q1 * cq - q2 * sq, q2 * cq + q1 * sq), 1), ct.bfloat16)

    m = ct.full((BQ, 1), NEG_INF, ct.float32); l = ct.zeros((BQ, 1), ct.float32)
    O_acc = ct.zeros((BQ, D), ct.float32)
    ii  = ct.broadcast_to(ct.expand_dims(ct.arange(BQ,  dtype=ct.int32), -1), (BQ, BKV))
    jj  = ct.broadcast_to(ct.expand_dims(ct.arange(BKV, dtype=ct.int32),  0), (BQ, BKV))
    tri = ct.greater_equal(ii, jj)

    for kv in range(q_blk + 1):
        kr = batch * NKVB + kv
        tK = ct.bitcast(ct.load(K, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16)
        tV = ct.bitcast(ct.load(V, index=(kr, kv_head), shape=(BKV, D), latency=10), ct.bfloat16)
        S = ct.mma(tQ, ct.transpose(tK), ct.zeros((BQ, BKV), ct.float32)) * qk_scale
        if kv == q_blk:
            S = ct.where(tri, S, NEG_INF)
        m_new = ct.maximum(m, ct.max(S, axis=-1, keepdims=True))
        alpha = ct.exp2(m - m_new)
        P     = ct.exp2(S - ct.broadcast_to(m_new, (BQ, BKV)))
        l     = alpha * l + ct.sum(P, axis=-1, keepdims=True)
        O_acc = ct.broadcast_to(alpha, (BQ, D)) * O_acc + ct.mma(ct.astype(P, ct.bfloat16), tV, ct.zeros((BQ, D), ct.float32))
        m = m_new

    res = O_acc / ct.broadcast_to(l, (BQ, D))
    ct.store(O, index=(qr, q_head), tile=ct.bitcast(ct.astype(res, ct.bfloat16), ct.uint16))
    ct.store(L_out, index=(hb * NQB + q_blk, 0), tile=m + ct.log2(l))


def main():
    S, Hq, Hkv = 2048, 16, 8
    M, NQB, NKVB = S, S // BQ, S // BKV
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    qh = _f32bf(rng.standard_normal((M, Hq * D)).astype(np.float32) * 0.5)
    kh = _f32bf(rng.standard_normal((M, Hkv * D)).astype(np.float32) * 0.5)
    vh = _f32bf(rng.standard_normal((M, Hkv * D)).astype(np.float32) * 0.5)
    cos, sin = build_cos_sin(S, D)

    gq, gk, gv = _GpuArray(qh), _GpuArray(kh), _GpuArray(vh)
    gqr, gkr = _GpuArray.zeros((M, Hq * D), np.uint16), _GpuArray.zeros((M, Hkv * D), np.uint16)
    gc, gs = _GpuArray(cos), _GpuArray(sin)
    go_ref, go_fus = _GpuArray.zeros((M, Hq * D), np.uint16), _GpuArray.zeros((M, Hq * D), np.uint16)
    gl_ref, gl_fus = _GpuArray.zeros((Hq * NQB, 1), np.float32), _GpuArray.zeros((Hq * NQB, 1), np.float32)

    def ref_path():
        ct.launch(si, (M // RTM, Hq), _rope_fwd_tok, (gq, gc, gs, gqr, S // RTM, DH))
        ct.launch(si, (M // RTM, Hkv), _rope_fwd_tok, (gk, gc, gs, gkr, S // RTM, DH))
        ct.launch(si, (NQB, Hq), _attn_fwd_tok, (gqr, gkr, gv, go_ref, gl_ref, NQB, NKVB, Hq, Hkv, float(scale)))

    def fused_path():
        ct.launch(si, (NQB, Hq), _attn_fwd_tok_rope,
                  (gq, gk, gv, gc, gs, go_fus, gl_fus, NQB, NKVB, Hq, Hkv, float(scale)))

    ref_path(); fused_path(); sync()
    o_ref, o_fus = go_ref.to_numpy(), go_fus.to_numpy()
    nbad = int((o_ref != o_fus).sum())
    b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    rel = float(np.abs(b2f(o_fus) - b2f(o_ref)).max() / (np.abs(b2f(o_ref)).max() + 1e-30))
    lbad = int((gl_ref.to_numpy() != gl_fus.to_numpy()).sum())
    print(f"  O bitwise: {nbad} diffs of {o_ref.size} ({100*nbad/o_ref.size:.2f}%)  max-rel {rel:.2e}   L diffs {lbad}")
    verdict_bits = "BITWISE" if nbad == 0 else ("~ULP (FMA contraction)" if rel < 1e-2 else "NUMERIC MISMATCH")

    def tim(fn, reps=30):
        fn(); sync()
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(reps): fn()
            sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
        return best

    t_ref = tim(ref_path)
    t_att = tim(lambda: ct.launch(si, (NQB, Hq), _attn_fwd_tok,
                                  (gqr, gkr, gv, go_ref, gl_ref, NQB, NKVB, Hq, Hkv, float(scale))))
    t_rope = tim(lambda: (ct.launch(si, (M // RTM, Hq), _rope_fwd_tok, (gq, gc, gs, gqr, S // RTM, DH)),
                          ct.launch(si, (M // RTM, Hkv), _rope_fwd_tok, (gk, gc, gs, gkr, S // RTM, DH))))
    t_fus = tim(fused_path)
    # Q-only variant: rotate Q in registers (pre-loop), K pre-rotated by the standalone kernel
    go_q = _GpuArray.zeros((M, Hq * D), np.uint16); gl_q = _GpuArray.zeros((Hq * NQB, 1), np.float32)
    def qonly_path():
        ct.launch(si, (M // RTM, Hkv), _rope_fwd_tok, (gk, gc, gs, gkr, S // RTM, DH))
        ct.launch(si, (NQB, Hq), _attn_fwd_tok_ropeq,
                  (gq, gkr, gv, gc, gs, go_q, gl_q, NQB, NKVB, Hq, Hkv, float(scale)))
    qonly_path(); sync()
    nbad_q = int((go_q.to_numpy() != o_ref).sum())
    t_qon = tim(qonly_path)

    flop = 2 * 2 * Hq * D * (M * (M + BQ) / 2)          # QK^T + PV over causal pairs
    print(f"  ref   : rope(Q+K) {t_rope:6.1f} + attn {t_att:6.1f} = chain {t_ref:6.1f} us  ({flop/t_ref/1e6:5.1f} TF chain)")
    print(f"  fused : {t_fus:6.1f} us  ({flop/t_fus/1e6:5.1f} TF)   vs chain {t_ref/t_fus:.2f}x  (attn-only ratio {t_att/t_fus:.2f}x)")
    print(f"  Q-only: {t_qon:6.1f} us  ({flop/t_qon/1e6:5.1f} TF)   vs chain {t_ref/t_qon:.2f}x  bitwise diffs {nbad_q}")
    print(f"  numerics (full fusion): {verdict_bits}")
    go_q.free(); gl_q.free()
    for g in (gq, gk, gv, gqr, gkr, gc, gs, go_ref, go_fus, gl_ref, gl_fus): g.free()


if __name__ == "__main__":
    print(f"inline-RoPE attention re-test (cuda-tile {ct.__version__}, cat/extract)")
    print("=" * 78)
    main()
