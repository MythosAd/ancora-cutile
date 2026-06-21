# ANCORA — Development Notes

An engineering journal for building a from-scratch RL/SFT training stack in pure Python on `cuda.tile`, targeting a single RTX 5080 Laptop (`sm_120a`). This is the human-readable companion to [`CLAUDE.md`](CLAUDE.md), which is the raw, exhaustive working log. Here I try to distill *what was actually hard*, the traps that cost days, and the few ideas that made the whole thing work.

If you only read one section, read [Batch invariance → `ratio = 1`](#4-batch-invariance--ratio--1-the-one-idea-that-justifies-the-project).

---

## 0. The premise

Modern RL post-training (GRPO/PPO) usually runs as two systems: a fast inference engine for rollouts (vLLM/SGLang) and a separate trainer (Megatron/FSDP). They can't be made numerically identical, so the field accepts a small mismatch between the policy that *generated* a token (`π_infer`) and the policy that *scores* it during training (`π_train`), and patches the resulting gradient bias with importance sampling.

The bet behind ANCORA: on **one GPU**, in **one codebase**, you can make rollout and training **bitwise-identical** and delete that entire problem. Everything else — the kernels, the device-resident orchestration, the precision choices — is in service of that bet, and of doing it on hardware that costs less than a month of cloud H100 time.

The constraint that shaped everything: **no PyTorch, no CUDA C++.** Just NVIDIA's `cuda.tile` (a Python DSL that emits tensor-core code), `cuda.core`, and `cuda.bindings`. That keeps the stack small and inspectable, but it means every numerical primitive is hand-built and every DSL sharp edge is yours to find.

---

## 1. The hardware reality (`sm_120a`)

Blackwell consumer (`sm_120`) is **not** Hopper (`sm_90`) and **not** datacenter Blackwell (`sm_100`/B200). A lot of "modern attention" assumes one of those, and simply won't compile:

- ✅ Works: `mma` (BF16/FP8, FP32 accumulate), **`mma_scaled`** with MXFP8 (E4M3 × E8M0) and MXFP4, thread-block clusters, `griddepcontrol`, CUDA-graph capture of `ct.launch`.
- ❌ Rejected by `ptxas`: **WGMMA** (`wgmma.mma_async`, Hopper-only), **`tcgen05`** TMEM/MMA (sm_100/110 only), and therefore **FlashAttention-3/4** (they're built on exactly those).

**Lesson:** pick your reference implementations by *target architecture first*, not by benchmark numbers. FA3 is gorgeous and irrelevant here. The runnable references on `sm_120` were gau-nernst's raw-CUDA FlashAttention (94% SOL) and NVIDIA's own cuda-tile attention paper (~53% of FA2). Knowing this on day one would have saved a week of trying to port the wrong thing.

The one thing `sm_120` *does* have that Hopper doesn't: **block-scaling tensor cores** (`mma_scaled`). That single fact is why ANCORA's MXFP8 uses per-block E8M0 scales (DeepSeek-V3 regime) instead of the delayed per-tensor scaling that Hopper-era recipes mandate — see [§5](#5-precision-choosing-against-the-recipe).

---

## 2. The cuda-tile traps that cost the most

`cuda.tile` is young (1.4.0). Most of the lost time was not algorithmic — it was the DSL doing something subtly different from what the code said.

### `bitcast` vs `astype` — reinterpret vs convert

When you load quantized bytes (FP8/FP4/E8M0 stored as `uint8`), you want to **reinterpret the bits**, not convert the value:

```python
ct.astype(load(x_u8), float8_e4m3fn)   # WRONG: uint8(56) → FP8(56.0)
ct.bitcast(load(x_u8), float8_e4m3fn)  # RIGHT: byte 0x38 → FP8(1.0)
```

This one bug returned `401408` instead of `128` in a forward test (56² × 128). E8M0 scales have their own version: there is *no* integer→E8M0 conversion path, so you `bitcast` pre-encoded bytes (`byte = floor(log2(value)) + 127`) or build constants with `ct.full(..., float8_e8m0fnu)`.

### No implicit low-precision store

cuda-tile won't implicitly cast a `bfloat16` tile into a `uint16` output array — you need a real `bfloat16` numpy dtype (`ml_dtypes`) or a `float32` output. Sounds obvious; isn't, when you've been treating `uint16` as "BF16 bits" everywhere else.

### `mma_scaled` shape convention

The `y` operand must be `(K, N)`, not `(N, K)` — so `nn.Linear` weights stored `(N, K)` need an in-kernel transpose of *both* the weight and its scale. Get this wrong and it compiles fine and produces garbage.

**Meta-lesson:** in a young DSL, the compiler accepting your code tells you almost nothing. Test every tile against a numpy reference, and test *each output chunk*, not just an aggregate norm — see [§3](#3-the-compiler-will-lie-to-you).

---

## 3. The compiler will lie to you

Three distinct `cuda.tile` 1.4.0 backend failures, none of which were clean errors:

1. **Crash (return `0x80000003`):** reusing one loaded tile both as `transpose(t)` and `t` in two MMAs aborts the compiler. Fix: load it *twice* into two handles (cheap, same data).
2. **Crash:** an MMA output tile with **rows > 128** aborts. The limit is on the M-dimension of the output; tile the output so rows ≤ 128.
3. **Silent miscompile (the worst):** a single kernel that loops over chunks to reduce (Σx²), then loops *again* re-loading the input to normalize, produced **correct output for 7 of 8 chunks and pure garbage for one** — no crash, no error, ~300% error on that one tile. The fix was to split into two single-loop kernels (a stats kernel + an apply kernel). Single-loop kernels are reliable; the two-loop-reload pattern is not.

That third one is why a recurring rule in this codebase is: **always validate every output tile against a reference, never just the average.** A one-tile miscompile hides inside a low mean error if the other tiles are right. (Also: cuda-tile kernels don't support Python list literals, so "keep all chunks in registers across loops" isn't available — two kernels it is.)

---

## 3b. The bf16 rounding-mode trap (it bit twice)

This deserves its own section because it masqueraded as concurrency bugs for *months*.

There are, across this codebase, f32→bf16 converters using **two different rounding modes**:

- **Truncation** (`bits >> 16`) — used in `attention.py`, `fused._trunc_bf16`, etc.
- **Round-to-nearest-even** (`u + 0x7FFF + ((u>>16)&1)`) — used in `norm.py`, `rope.py`, in-kernel `ct.astype`.

**The rule that emerged: a device replacement for a host helper must match *that module's* rounding mode.** Two victims:

1. A prefix offset-RoPE used attention's truncation where RNE was expected → 1-ULP drift → **12–58% gradient error**.
2. A decode boundary's RMSNorm used truncation where `norm.py` uses RNE. Truncation is biased toward zero, so `Σx²` shrank systematically → `rstd` biased 0.4%. This was **misdiagnosed for months as a sync/race** ("I removed a sync and the output changed!"). It was never a race. ~47% of elements were off by exactly 1 bit, with a *biased* reduction.

**Signature to recognize:** a large fraction of elements off by *exactly one bit*, plus a *biased* (not random) reduction error. That's a rounding-mode mismatch, not corruption. Random corruption looks different.

---

## 4. Batch invariance → `ratio = 1` (the one idea that justifies the project)

The mismatch between rollout and training comes from kernels that **change their reduction order as a function of batch size or sequence length** (the Thinking Machines result). So the entire stack obeys a few hard rules:

1. **Fixed tile sizes**, never batch-dependent. No autotuning that switches tile shape on batch size.
2. **No split-K** in matmul — one block owns the full K reduction, accumulated sequentially.
3. **No split-KV** in attention (training/prefill) — one block owns the full per-query reduction.
4. **The same kernel for rollout and training.**
5. **Reductions stay on one core** (RMSNorm, log-softmax).

Rule 4 is the keystone, and the non-obvious realization that made it work:

> **Decode reuses the prefill attention kernel.** To get token *t*'s attention during generation, run the *prefill* kernel over the cache and take row *t*. The causal mask zeroes `j > t`, so not-yet-generated cache slots (even garbage) don't affect row *t*, and prefill is sequence-length invariant. Result: decode attention is **bitwise-identical** to training, even with garbage in future cache slots.

A hand-rolled single-query decode kernel computes `q·k` with a *different* reduction order (`ct.sum` of a broadcast vs `ct.mma`) and lands 0.216% off — fine for inference, fatal for `ratio = 1`. So the single-query path was thrown away and decode runs the block kernel.

Because GEMM/RMSNorm/RoPE/SwiGLU are per-row (already batch-invariant), a full rollout forward = batch-invariant GEMM/RMSNorm + prefill-kernel decode attention ⇒ **`ratio = 1` exactly, no importance sampling.** This is verified by double induction over positions × layers, and empirically to `max|Δ| = 0` for: SFT decode == training prefill, greedy *and* sampled GRPO rollout logprob == training logprob, and CUDA-graph replay == direct launch.

This is the thing two-system labs structurally cannot do. It's the entire reason to build a single codebase.

### The concurrency tax of going device-resident

Getting `ratio = 1` *in principle* is numerics; getting it *in practice* meant fighting the allocator and streams, because the host-orchestrated path is non-deterministic:

- **Stream discipline:** `ct.launch` runs on a chosen stream; `cuMemcpyDtoH` runs on the default stream. Without a `cudaStreamSynchronize` between them, the copy races the kernel → partial reads → "nondeterminism at S=256" that was actually a missing sync. The generalization: **one stream end-to-end for any dependency chain** — an HtoD that *feeds* a kernel must be on that kernel's stream too (`cuMemcpyHtoDAsync(..., si)`), and `cudaDeviceSynchronize` does *not* fix a cross-stream ordering hazard.
- **Alloc/free churn is a race.** Allocating + freeing GPU scratch every decode step (or per-expert in a MoE loop) intermittently corrupts *one* position's output — different positions on different runs, flaky enough that adding an unrelated line flips which run is hit. The driver hands back addresses still referenced by in-flight work. **Fix: preallocate once, reuse across all steps.** This took a 117% error down to 1.2%. Any self-allocating host helper (it allocs→launches→syncs→frees internally) in a hot loop reintroduces it.

These two — stream discipline and no per-step alloc churn — are why the whole forward/backward/decode became *device-resident on persistent buffers*. That wasn't a performance choice first; it was a *correctness* choice.

---

## 5. Precision: choosing against the recipe

I started from a well-regarded external precision recipe (MAI-Thinking-style: FP8-E4M3 forward with delayed per-tensor scaling, E5M2 dgrad, FP32 residual stream, stochastic rounding). It's a *reference to weigh, not a spec*, and the most useful exercise was figuring out where to **diverge** because the hardware is different:

- **Forward stays MXFP8, not delayed-scaling E4M3.** The recipe's "no MXFP8" is Hopper-era — Hopper has no block-scaling tensor cores. `sm_120a` does. Per-block E8M0 scaling is stateless (no 1024-step abs-max history to checkpoint) and outlier-robust (it handles the ~6912 massive activation per 32-block). Right call for *this* hardware; wrong call for Hopper.
- **FP8 dgrad uses E4M3, not the recipe's E5M2.** The recipe uses E5M2 because its coarse delayed scaling needs the wider element range. Our *fine* block scaling handles the range at the scale layer, so the more-precise E4M3 element wins — measured ~2× more accurate (3.8% vs 7.4% rel-err). Same destination (FP8 backward), opposite element choice, because the scaling granularity differs.
- **FP32 residual stream + stochastic-rounded downcast** were adopted. But honestly: in *our* design (FP32 weight-grad + FP32 AdamW already close the bias path) stochastic rounding is **marginal** — it only becomes load-bearing once the backward goes FP8. Worth knowing which "best practices" are actually load-bearing for *your* configuration vs. cargo-culted.

And the recurring verdict on low precision: **MXFP8 and FP8-dgrad both validated correct (and `ratio = 1` holds under MXFP8!), but both measured net-negative on performance** on this model. The forward is only ~10% of a training step, and the per-GEMM quantization launches eat the 2× tensor-core peak; FP8 weight copies worsen the 16 GB card's WDDM paging. They stay opt-in, default off. **The time is in the backward and the optimizer, not the forward** — so optimizing the forward's precision was optimizing the wrong thing. They'd only pay off with CODA-style quant-epilogue fusion (fuse the quantize into the producing kernel) and compute-bound sizes.

---

## 6. The MFU journey

Standalone, the kernels hit the cuda-tile ceiling fast: attention is ~90% compute-bound (near peak — leave it), but the GEMMs are **L2-bound at ~40% of native peak** with no SMEM tiling — that's the DSL's ceiling, not a tuning miss. A CUTLASS `sm_120a` reference GEMM hits ~244 TF MXFP8; the cuda-tile loop GEMM is 51–76% of that. You can't tile your way past it inside the DSL.

So the real MFU lever was never a single kernel — it was **removing host overhead and feeding idle tensor cores from neighbouring operators**:

1. **CUDA-graph capture** of the device-resident forward: the ~15-launch/layer chain becomes one graph, bitwise-identical on replay, 1.1–2.2× less host overhead.
2. **The training "megakernel" pass** got training MFU from **5.4% → 30.5%**. The single biggest win was embarrassing: a host-built `(M, V)` one-hot matrix was **75% of the step** (1.2 GB numpy + 0.6 GB PCIe *per step*). Killing it (device `_embed_gather` forward + per-chunk device one-hot backward) plus a Liger-style chunked vocab boundary, ncu-guided retiling, and adaptive tile counts to fill the SMs did the rest.
3. **Gradient accumulation** then pushed it to **~35% MFU / 14.7k tok/s** — and the insight there is that accumulation is a *training-MFU lever*, not just a memory one: it amortizes the BW-bound AdamW floor (a 15 GB state sweep on a 16 GB card) over more compute.

**Lesson:** profile the *step*, not the kernel. The fact that the forward took longer than the backward was the tell that something non-compute (the host one-hot) dominated. And `ncu` is worth the setup pain — it told me in one glance which kernels were compute- vs memory-bound so I stopped optimizing the wrong ones. (On Windows, GPU perf counters are admin-gated — `ERR_NVGPUCTRPERM` — which silently returns *no* metrics; that's a permissions wall, not a tooling bug.)

The remaining ceiling is real hardware: AdamW's ~32 ms BW floor (551M params × 3 FP32 states), attention at bandwidth peak, and the vocab GEMM at the BF16 compute wall. Past that, you'd have to leave the DSL for a hand-written persistent CUDA kernel.

---

## 7. Long context, on 16 GB

Activation checkpointing (recompute the forward in backward instead of storing it) is standard, but two things made it work cleanly here:

- **Recompute is deterministic ⇒ gradients are bitwise-identical to full-store.** That's the correctness gate, and it held (`Δ = 0`). Determinism pays off again.
- The **local/global attention split** (Gemma-3/MAI 5:1) makes recompute cheap — most layers are windowed `O(S·window)`, only a couple are global `O(S²)`.

The surprise was *where* the memory wall actually was. It wasn't the optimizer state floor and it wasn't the step's activation peak — it was the **construct-time prealloc**: setup was allocating all NL layers' shared scratch and *then* freeing the duplicates, so the peak was NL layers' worth. Freeing each layer's scratch to layer 0's *as it's built* (peak = 2 layers, end-state identical ⇒ still bitwise) moved SFT from ~16K to ~24K tokens and prefix-GRPO from 8K to 16K. **Lesson:** "out of memory" tells you the *peak*, and the peak is often in setup, not the hot loop — measure construct, step, and optimizer separately.

---

## 8. Meta-lessons

A few things I'd tell someone starting a similar project:

- **In a young DSL, "it compiled and the norm looks right" means nothing.** Validate every tile against numpy. The bugs that cost the most were silent — miscompiles, rounding modes, races — not crashes.
- **Determinism is a feature you build, then spend everywhere.** It's the basis of `ratio = 1`, it's how activation-checkpoint grads are verified, it's how CUDA-graph replay is trusted. Make every kernel bitwise-reproducible and a whole class of debugging disappears (and a whole class of "is this a race?" panics turns out to be rounding modes).
- **"Best practices" are configuration-dependent.** Stochastic rounding, MXFP8, FP8 dgrad — all correct, all recommended by serious recipes, all marginal-or-negative *for this hardware and this model size*. Re-derive whether each one is load-bearing for *you*.
- **Profile the system, not the component.** The kernels were near-ceiling while the *step* was at 5% MFU because of a host one-hot. The bottleneck is rarely where the interesting code is.
- **Pick references by hardware first.** Half of "modern attention" targets `sm_90`/`sm_100` and won't even compile on `sm_120`.

---

## References

The reference implementations and papers that shaped specific decisions:

- **Attention on `sm_120`:** [gau-nernst/learn-cuda (07_attention)](https://github.com/gau-nernst/learn-cuda) (94% SOL raw CUDA), NVIDIA CUDA-Tile attention ([arXiv 2604.23466](https://arxiv.org/abs/2604.23466)), [SageAttention](https://github.com/thu-ml/SageAttention) (MXFP4).
- **Megakernel:** [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels) and the "No Bubbles" blog.
- **Batch invariance / `ratio = 1`:** Thinking Machines' work on defeating nondeterminism in LLM inference; the broader unified-FP8 RL flow line.
- **Optimizer:** Muon (Keller Jordan); Polar Express coefficients (Amsel et al., [arXiv 2505.16932](https://arxiv.org/abs/2505.16932)).
- **Precision / architecture:** MAI-Thinking precision recipe (weighed, not followed), DeepSeek-V3 block scaling, Qwen3.
- **Prefix sharing:** DualKV ([arXiv 2605.15422](https://arxiv.org/abs/2605.15422)), Prefix Grouper ([arXiv 2506.05433](https://arxiv.org/abs/2506.05433)).
- **GEMM reference:** CUTLASS 4.5.2 (`sm_120a` blockscaled, with the alignment patch for the misaligned-address bug).

For the exhaustive, unabridged log — every kernel, every probe, every dead end — see [`CLAUDE.md`](CLAUDE.md).
