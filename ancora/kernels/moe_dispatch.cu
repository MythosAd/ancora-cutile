// ancora/kernels/moe_dispatch.cu — device-resident MoE dispatch (the "sort tokens by expert"
// that moe.build_layout does on host). Plain CUDA, NO cub / NO atomics → compiles via NVRTC
// (small, no system headers) and is DETERMINISTIC + STABLE (matches numpy's stable argsort
// grouping bitwise), which our batch-invariance rules require (a non-stable atomic scatter,
// like vLLM's, would reorder a group → break the fixed-order weight-grad reduction).
//
// Single block: parallel memset + a serial histogram→scan→stable-scatter on thread 0 (E=16 and
// M*k ≤ a few k → microseconds, off the critical path) + parallel tile_expert fill. Produces the
// SAME 5 arrays as build_layout, sized for a FIXED Rmax = M*k + E*TM grid (so the downstream
// grouped-GEMM grid is host-knowable without reading R back; padding tiles → expert 0, gate 0).
// ── stage A: gating (router projection + softmax + top-k + renorm), FP32, ONE WARP per token ──
// Mirrors GroupedMoEFFN._route (h@Wr fp32, softmax, stable top-k by descending prob with low-index
// tie-break, optional renorm). Per-token & FP32 ⇒ batch-invariant + deterministic. Writes the
// topi/topw the dispatch kernel below consumes, plus probs (M,E) for the router backward — so the
// whole forward router (gate → dispatch) is device-resident with NO host round-trip.
//
// COALESCED h load: a warp owns one token; the 32 lanes stride over H, so consecutive lanes read
// consecutive h addresses (one 128 B transaction/step) — NOT one-thread-per-token, which strides H
// (=4 KB) between adjacent threads and serializes the warp into 32 separate cache lines. Each lane
// keeps an E-wide partial over its H-stride; a warp-shuffle tree-reduce sums them; lane 0 finishes.
extern "C" __global__ void moe_router_gate(
    const unsigned short* h,   // (M,H) bf16 bits (the post-norm gh2)
    const float* Wr,           // (H,E) router weight, FP32
    int* topi, float* topw, float* probs,   // (M,k) int, (M,k) f32, (M,E) f32
    int M, int H, int E, int k, int norm)
{
    // RE is a COMPILE-TIME expert count (this model family is fixed at E=16). It must be compile-time
    // so lg[] stays in REGISTERS and the e-loops unroll — a runtime bound would force lg[] into local
    // (global-backed) memory, and `lg[topi[...]]` (runtime index) alone would spill it, dwarfing the
    // coalescing win. The runtime E arg is asserted == RE.
    constexpr int RE = 16;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // global warp id == token
    const int lane = threadIdx.x & 31;
    if (warp >= M) return;
    const unsigned short* hr = h + (long long)warp * H;
    float lg[RE];                                         // per-lane partial over its H-stride (registers)
    #pragma unroll
    for (int e = 0; e < RE; e++) lg[e] = 0.0f;
    for (int hh = lane; hh < H; hh += 32) {               // lane reads hr[lane], hr[lane+32], … → COALESCED
        float hv = __uint_as_float(((unsigned int)hr[hh]) << 16);
        const float4* w4 = (const float4*)(Wr + hh * RE);  // RE=16 contiguous → 4× LDG.128 (not 16 scalar)
        #pragma unroll
        for (int q = 0; q < RE / 4; q++) {
            float4 wv = w4[q];
            lg[q * 4 + 0] += hv * wv.x; lg[q * 4 + 1] += hv * wv.y;
            lg[q * 4 + 2] += hv * wv.z; lg[q * 4 + 3] += hv * wv.w;
        }
    }
    #pragma unroll
    for (int e = 0; e < RE; e++)                           // warp tree-reduce each expert → lane 0 holds logit
        for (int o = 16; o > 0; o >>= 1) lg[e] += __shfl_down_sync(0xffffffffu, lg[e], o);
    if (lane != 0) return;
    float mx = lg[0];
    #pragma unroll
    for (int e = 1; e < RE; e++) mx = fmaxf(mx, lg[e]);
    float s = 0.0f;
    #pragma unroll
    for (int e = 0; e < RE; e++) { lg[e] = __expf(lg[e] - mx); s += lg[e]; }
    #pragma unroll
    for (int e = 0; e < RE; e++) { lg[e] /= s; probs[warp * RE + e] = lg[e]; }
    float ws = 0.0f;
    for (int j = 0; j < k; j++) {                          // top-k: stable descending, low-index ties
        int best = 0; float bv = -1.0f;
        #pragma unroll
        for (int e = 0; e < RE; e++) {                     // already-picked check reads global topi (not lg)
            int used = 0; for (int jj = 0; jj < j; jj++) if (topi[warp * k + jj] == e) used = 1;
            if (!used && lg[e] > bv) { bv = lg[e]; best = e; }
        }
        topi[warp * k + j] = best; ws += bv;               // bv == lg[best] → no dynamic lg[] index
    }
    for (int j = 0; j < k; j++) {                          // renorm; gather lg[sel] via an unrolled e==sel scan
        int sel = topi[warp * k + j]; float w = 0.0f;
        #pragma unroll
        for (int e = 0; e < RE; e++) if (e == sel) w = lg[e];
        topw[warp * k + j] = norm ? w / ws : w;
    }
}


// ── router BACKWARD (device): gate-backward → d_logits, then the two router GEMMs ──
// Completes router residency: with these, GroupedMoEFFN.backward_resident does the whole router grad
// on device (no gh2 / topi / probs / dsg download). FP32, per-token / coalesced → batch-invariant.
__device__ __forceinline__ unsigned short f2bf(float x) {          // f32 → bf16 bits, round-nearest-even
    unsigned int u = __float_as_uint(x); u = u + 0x7fffu + ((u >> 16) & 1u); return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }

// (1) gate backward: dsg (per-slot <dOut,Yg>) → d_logits (M,E). Mirrors _gate_backward (top-k gather,
//     renorm bwd, softmax bwd). One thread/token; E compile-time (RE) → register arrays, no spill.
extern "C" __global__ void moe_router_gate_bwd(
    const float* dsg, const int* tok_slots, const int* topi, const float* probs,
    float* dlogits, int M, int E, int k, int norm)
{
    constexpr int RE = 16;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M) return;
    float pr[RE];
    #pragma unroll
    for (int e = 0; e < RE; e++) pr[e] = probs[t * RE + e];
    float dsel[8], raw[8], sm = 0.0f;
    for (int j = 0; j < k; j++) {
        int sel = topi[t * k + j]; dsel[j] = dsg[tok_slots[t * k + j]];
        float rj = 0.0f;
        #pragma unroll
        for (int e = 0; e < RE; e++) if (e == sel) rj = pr[e];      // gather pr[sel] (no dynamic index)
        raw[j] = rj; sm += rj;
    }
    if (norm) {
        float dot = 0.0f; for (int j = 0; j < k; j++) dot += dsel[j] * raw[j] / sm;
        for (int j = 0; j < k; j++) dsel[j] = (dsel[j] - dot) / sm;
    }
    float dprob[RE];
    #pragma unroll
    for (int e = 0; e < RE; e++) dprob[e] = 0.0f;
    for (int j = 0; j < k; j++) {
        int sel = topi[t * k + j];
        #pragma unroll
        for (int e = 0; e < RE; e++) if (e == sel) dprob[e] = dsel[j];
    }
    float pdot = 0.0f;
    #pragma unroll
    for (int e = 0; e < RE; e++) pdot += dprob[e] * pr[e];
    #pragma unroll
    for (int e = 0; e < RE; e++) dlogits[t * RE + e] = pr[e] * (dprob[e] - pdot);   // softmax bwd
}

// (2) router WEIGHT grad: G_router (H,E) = hᵀ @ d_logits, 2-PASS split-M for both occupancy AND
//     h-reuse. Pass A: one thread per hh accumulates ALL E experts (h[m,hh] read ONCE, reused for the
//     16 d_logits via float4), over its M-split chunk → Gpart[(s,hh,e)]; grid (H/blk, NSPL) gives
//     H/blk·NSPL blocks. Pass B: sum the NSPL partials (fixed order → deterministic). vs the prior
//     one-thread-per-(hh,e) (full occupancy but read h 16× redundantly) and the original H/128≈8 blocks.
extern "C" __global__ void moe_router_dW_part(
    const unsigned short* h, const float* dlogits, float* Gpart, int M, int H, int E, int NSPL)
{
    constexpr int RE = 16;
    int hh = blockIdx.x * blockDim.x + threadIdx.x;
    if (hh >= H) return;
    int s = blockIdx.y;
    int m0 = (int)((long long)s * M / NSPL), m1 = (int)((long long)(s + 1) * M / NSPL);
    float acc[RE];
    #pragma unroll
    for (int e = 0; e < RE; e++) acc[e] = 0.0f;
    for (int m = m0; m < m1; m++) {                                // COALESCED h over hh; reused for all E
        float hv = bf2f(h[(long long)m * H + hh]);
        const float4* dl = (const float4*)(dlogits + (long long)m * RE);
        #pragma unroll
        for (int q = 0; q < RE / 4; q++) { float4 d = dl[q];
            acc[q*4]+=hv*d.x; acc[q*4+1]+=hv*d.y; acc[q*4+2]+=hv*d.z; acc[q*4+3]+=hv*d.w; }
    }
    #pragma unroll
    for (int e = 0; e < RE; e++) Gpart[((long long)s * H + hh) * RE + e] = acc[e];   // (NSPL,H,E)
}

extern "C" __global__ void moe_router_dW_reduce(const float* Gpart, float* Grouter, int HE, int NSPL)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HE) return;
    float s = 0.0f;
    for (int sp = 0; sp < NSPL; sp++) s += Gpart[(long long)sp * HE + idx];          // fixed-order reduce
    Grouter[idx] = s;
}

// (3) router → gh2 path: gdh2 = gdh2_e (expert path) + d_logits @ Wrᵀ. One WARP/token; lanes stride
//     over H → COALESCED gdh2_e read / gdh2 write; Wr[hh,:] float4. Fuses the expert+router grad add.
extern "C" __global__ void moe_router_dh(
    const float* dlogits, const float* Wr, const unsigned short* gdh2_e, unsigned short* gdh2,
    int M, int H, int E)
{
    constexpr int RE = 16;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= M) return;
    float dl[RE];
    #pragma unroll
    for (int e = 0; e < RE; e++) dl[e] = dlogits[warp * RE + e];
    const unsigned short* ge = gdh2_e + (long long)warp * H;
    unsigned short* go = gdh2 + (long long)warp * H;
    for (int hh = lane; hh < H; hh += 32) {                         // COALESCED over hh
        const float4* rw = (const float4*)(Wr + hh * RE);
        float dh = 0.0f;
        #pragma unroll
        for (int q = 0; q < RE / 4; q++) { float4 w = rw[q];
            dh += dl[q*4]*w.x + dl[q*4+1]*w.y + dl[q*4+2]*w.z + dl[q*4+3]*w.w; }
        go[hh] = f2bf(bf2f(ge[hh]) + dh);
    }
}


extern "C" __global__ void moe_build_layout(
    const int*   topi,        // (M*k,) expert id per assignment (token-major: a = t*k + s)
    const float* topw,        // (M*k,) gate weight per assignment (already renormalized)
    int*         src_row,     // (Rmax,)  out: source token per grouped slot   (init 0)
    float*       slot_gate,   // (Rmax,)  out: gate per grouped slot           (init 0)
    int*         tile_expert, // (Rtmax,) out: expert id per m-tile            (init 0)
    int*         tok_slots,   // (M*k,)   out: token's k grouped slot indices  (init 0)
    int*         off_tiles,   // (E+1,)   out: per-expert m-tile offset
    int M, int k, int E, int TM, int Rmax, int Rtmax)
{
    // Parallel single block: atomic histogram (counts are order-free) → tiny scan → STABLE scatter
    // with one WARP per expert (ballot/popc ranks preserve assignment order ⇒ same grouping as the
    // host stable argsort, bitwise). All 32 lanes of an active warp hit the ballot (warp-uniform if).
    extern __shared__ int sh[];
    int* cnt = sh;             // (E)   per-expert counts
    int* off = sh + E;         // (E+1) exclusive padded-row offsets
    const int Mk = M * k;
    const int tid = threadIdx.x, nth = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31;

    for (int r = tid; r < Rmax;  r += nth) { src_row[r] = 0; slot_gate[r] = 0.0f; }
    for (int r = tid; r < Rtmax; r += nth)   tile_expert[r] = 0;
    for (int a = tid; a < Mk;    a += nth)   tok_slots[a] = 0;
    for (int e = tid; e < E;     e += nth)   cnt[e] = 0;
    __syncthreads();

    for (int a = tid; a < Mk; a += nth) { int e = topi[a]; if (e >= 0 && e < E) atomicAdd(&cnt[e], 1); }
    __syncthreads();

    if (tid == 0) {                                    // exclusive scan over TM-padded counts (E small)
        int acc = 0;
        for (int e = 0; e < E; e++) { off[e] = acc; acc += ((cnt[e] + TM - 1) / TM) * TM; }
        off[E] = acc;
        for (int e = 0; e <= E; e++) off_tiles[e] = off[e] / TM;
    }
    __syncthreads();

    if (warp < E) {                                    // warp `e` scatters expert e's assignments, in order
        int e = warp, base = off[e], running = 0;
        for (int c = 0; c < Mk; c += 32) {
            int a = c + lane;
            int my = (a < Mk) ? topi[a] : -1;          // COALESCED topi read
            unsigned int bm = __ballot_sync(0xffffffffu, my == e);
            if (my == e) {
                int slot = base + running + __popc(bm & (((unsigned)1 << lane) - 1));   // stable rank
                src_row[slot]   = a / k;
                slot_gate[slot] = topw[a];
                tok_slots[a]    = slot;
            }
            running += __popc(bm);
        }
    }
    __syncthreads();

    for (int mi = tid; mi < Rtmax; mi += nth) {         // expert per tile; padding tiles → E (sentinel:
        int e = E;                                      // _ggemm skips its mma when e==E, saving the
        for (int ee = 0; ee < E; ee++)                  // ~17% worst-case-padding compute of the fixed grid)
            if (mi >= off_tiles[ee] && mi < off_tiles[ee + 1]) { e = ee; break; }
        tile_expert[mi] = e;
    }
}
