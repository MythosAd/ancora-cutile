# cuTile / ANCORA — Development Notes for Claude

## Project
On-policy RL (GRPO) + SFT training framework on RTX 5080 Laptop (sm_120a).
Paper: ANCORA. Package: `ancora/`. (Design + precision recipe are consolidated into this file.)

---

## Hardware facts (verified, do not re-test)

- GPU: RTX 5080 Laptop, CC 12.0 (sm_120 / sm_120a)
- Always compile with `--gpu-architecture=sm_120a` (strict superset of sm_120, no cost)
- Display driven by CPU iGPU → RTX 5080 is compute-only (display_active=Disabled)
- WDDM mode, but headless → near-TCC behavior for megakernel workloads

### What works on sm_120a (tested)
- `ct.mma_scaled` with MXFP8 (float8_e4m3fn × float8_e8m0fnu, B=32) → 256.0 ✓
- `ct.mma_scaled` ALSO accepts float8_e5m2 elements (probed 2026-06-14, tests/kernels/_probe_fp8_dgrad.py)
  — but for the FP8 backward (dgrad) under E8M0 block scaling, **E4M3 is ~2× more accurate than E5M2**
  (3.8% vs 7.4% rel-err; 4.0% vs 12.0% with within-block outliers). Fine block scaling handles the range
  at the scale layer (DeepSeek-V3 regime), so the dgrad element should be E4M3, NOT MAI's E5M2 (MAI uses
  E5M2 because its delayed per-tensor scaling is coarse and needs the wide element range). [[gemm-mfu-ceiling]]
- `ct.mma_scaled` with MXFP4 (float4_e2m1fn × float8_e8m0fnu, B=16/32) → 64.0 ✓
- `ct.mma` with BF16/FP8 → works, acc must be float32
- `ct.pack_to_bytes`, `ct.unpack_from_bytes`, `ct.bitcast` → all verified
- Thread Block Clusters (sm_90+) via `cooperative_groups::cluster_group` → D[0]=2 ✓
- `griddepcontrol.launch_dependents` → compiles and runs

### What does NOT work on sm_120a (confirmed, do not attempt)
- `wgmma.mma_async` (WGMMA) → Hopper sm_90 only, ptxas rejects on sm_120
- `tcgen05.alloc/dealloc` (TMEM allocation) → sm_100/B200 only
- `tcgen05.mma` → sm_100a/103a/110a only (confirmed from CUDA 13.3 CCCL header)
- FA4 (FlashAttention-4, arXiv 2603.05451) → uses tcgen05.mma → not portable to sm_120
- `cuda.tile` float6 types → do not exist in 1.4.0; FP6 needs inline PTX only
- `cc.LaunchConfig(cluster_dim=...)` → not exposed in cuda.core

### Toolchain update (2026-06-01): cuda.core 1.0.1 + bindings 13.3.1 (cuda-tile still 1.4.0)
cuda.core graduated from `experimental` to stable `cuda.core`. NEW + relevant:
- **CUDA graphs capture ct.launch** ✓ (TESTED, _probe_cuda_graph.py): `gb=dev.create_graph_builder();
  gb.begin_building(); ct.launch(int(gb.__cuda_stream__()[1]), ...); gb.end_building();
  graph=gb.complete(); graph.launch(stream)`. 8-kernel chain 80µs→28µs (2.8× less host overhead).
  **This is the megakernel mechanism** — capture a device-resident forward, replay. Graph control
  flow too (`while_loop` for decode, `if_then`). Needs persistent device buffers (no per-kernel
  alloc/upload/download).
- **DeviceMemoryResource** = async memory pool → fixes the alloc/free churn race cleanly.
- **TensorMapDescriptor** (TMA), **LaunchConfig** (check cluster_dim now).
- cuda-tile UNCHANGED 1.4.0 → FP4 load still blocked (MXFP8 decision stands [[precision-format-decision]]).

---

## Critical API pitfalls — read before writing any kernel

### 1. `ct.bitcast` vs `ct.astype` — the most important distinction

```python
# WRONG: value-converts uint8(56) → FP8(56.0), not FP8(1.0)
ta = ct.astype(ct.load(x_u8, ...), ct.float8_e4m3fn)

# CORRECT: reinterprets raw bytes uint8(0x38) → FP8(0x38) = 1.0
ta = ct.bitcast(ct.load(x_u8, ...), ct.float8_e4m3fn)
```

**Rule**: when loading raw quantized bytes (FP8/FP4/E8M0) stored as uint8 from device memory,
always use `ct.bitcast`. Use `ct.astype` only for numeric value conversion (e.g., float32 → FP8).

This caused `test_fwd_mxfp8_ones` to return 401408 instead of 128 (56² × 128).

### 1b. THREE same-named f32→bf16 converters with TWO rounding modes (bit us TWICE)

- **TRUNCATION** (`bits >> 16`): `attention.py _f32_to_bf16_bits`, `resident.py _f32bf`, `fused._trunc_bf16`, `loss.py f32_to_bf16_bits`
- **ROUND-TO-NEAREST-EVEN** (`u + 0x7FFF + ((u>>16)&1)`): `norm.py f32_to_bf16_bits`, `rope.py f32_to_bf16_bits`, `fused._cast_bf16` (ct.astype), every in-kernel `ct.astype` store

**Rule: a device replacement for a host helper must match THAT MODULE's rounding.** Victims:
(1) prefix offset-RoPE used attention's trunc → 1-ulp drift → 12-58% grad error (2026-06-09);
(2) the decode boundary's `_trunc_bf16` swap vs norm.py's RNE → rstd biased 0.4%, misdiagnosed
for months as a sync/race ("removed sync changes gout") — solved 2026-06-11 (47% of elements
1-ulp; trunc is biased toward zero so Σx² shrinks systematically). Symptom signature: a huge
fraction of elements off by exactly 1 bit + biased reductions, NOT random corruption.

### 2. E8M0 scale conversion

```python
# WRONG: no integer→E8M0 conversion path exists
ct.astype(int_tile, ct.float8_e8m0fnu)  # TileTypeError

# WRONG: float32→E8M0 requires explicit rounding mode
ct.astype(float32_tile, ct.float8_e8m0fnu)  # TileCompilerExecutionError

# CORRECT: reinterpret pre-encoded uint8 bytes
ct.bitcast(ct.load(scale_u8, ...), ct.float8_e8m0fnu)

# CORRECT for constants in kernel
ct.full(shape, value_float, ct.float8_e8m0fnu)  # e.g. ct.full((M,4), 1.0, ct.float8_e8m0fnu)
```

E8M0 encoding: `byte = floor(log2(value)) + 127`. So 1.0 → 0x7F, 2.0 → 0x80, 0.5 → 0x7E.

### 0. Tile compiler CRASHES (return code 2147483651 / 0x80000003)

Two confirmed cuda-tile 1.4.0 backend (tileiras) crashes — NOT clean errors, the
compiler aborts. Both hit during attention/loss backward:

**(a) Reusing ONE loaded tile both `transpose(t)` and `t` in two MMAs crashes.**
```python
tK = ct.bitcast(ct.load(K, ...), ct.bfloat16)
S   = ct.mma(tQ, ct.transpose(tK), acc)   # tK transposed
acc = ct.mma(dS, tK, acc)                  # tK NOT transposed  → CRASH
```
Fix: load it TWICE (two distinct tile handles, same data — cheap):
```python
tKt = ct.bitcast(ct.load(K, ...), ct.bfloat16)  # for the transposed use
tK  = ct.bitcast(ct.load(K, ...), ct.bfloat16)  # for the non-transposed use
```
Using a tile in the SAME orientation multiple times is fine (e.g. tQ non-transposed twice).

**(b) MMA output tile rows > 128 crashes** (loss.py dW_head). A `(512, TV)` accumulator
aborts; `(64, 512)` is fine. Limit is on ROWS (M-dim of the MMA output), not cols.
Fix: tile the output dimension so MMA output rows ≤ 128.

**(c) SILENT miscompile: a 2nd unrolled loop that RE-LOADS the same array writes
garbage in exactly ONE iteration** (found in RMSNorm fwd, norm.py). A single kernel
that loops over H-chunks to reduce (Σx²), then loops AGAIN re-loading x to normalize,
produced correct output for 7 of 8 chunks but pure garbage for chunk 5 (300% error) —
NO crash, NO error, just wrong numbers. rstd (the reduction) was correct; only the
2nd loop's reload corrupted. Oddly, attention/linear backward kernels with the same
two-loop-reload shape work — the trigger is subtle (load count / carried tile?).
Fix: SPLIT into two single-loop kernels (stats kernel + apply kernel). Single-loop
kernels are reliable. Cost: 1 extra launch + an (M,1) rstd round-trip (negligible).
LESSON: always test EACH output chunk/tile vs a reference, not just an aggregate norm
— a one-tile miscompile hides in a low average error if most tiles are right.
Note: cuda-tile kernels do NOT support Python list literals (`xs = []` → TileSyntaxError),
so "hold all chunks in registers across loops" is not available; two kernels it is.

### 3. Output dtype — no implicit BF16 store

```python
# WRONG: cuda-tile cannot implicitly cast bfloat16 tile to uint16 array
out = _GpuArray.zeros((M, N), np.uint16)   # pretending it's BF16
ct.store(out, ..., tile=ct.astype(acc, ct.bfloat16))  # TileTypeError

# CORRECT: use float32 output array
out = _GpuArray.zeros((M, N), np.float32)
ct.store(out, ..., tile=acc)  # acc is already float32

# For BF16 output: need ml_dtypes.bfloat16 as numpy dtype (not yet in stdlib)
```

### 4. ct.mma_scaled shape convention

```
x:       (M, K)       FP8 or FP4
x_scale: (M, K // B)  E8M0  (B=32 for FP8/FP4, or B=16 for FP4)
y:       (K, N)       FP8 or FP4   ← y must be (K, N), NOT (N, K)
y_scale: (K // B, N)  E8M0
acc:     (M, N)       float32
```

When weights are stored (N, K) — as in nn.Linear — transpose before mma_scaled:
```python
tw  = ct.bitcast(ct.load(w, index=(n, k), shape=(TN, TK)), ct.float8_e4m3fn)
acc = ct.mma_scaled(ta, txs, ct.transpose(tw), ct.transpose(tws), acc)
```

### 5. NVRTC hangs on Windows with large headers

```python
# HANGS on Windows: NVRTC parsing cuda_runtime.h / cooperative_groups.h
cc.Program(src_with_cuda_runtime_h, "c++", opts).compile("cubin")

# CORRECT: use nvcc offline compiler for files with system headers
# cmd /c "vcvars64.bat && nvcc -arch=sm_120a kernel.cu -cubin -o kernel.cubin"
# Then load: cdrv.cuModuleLoadData(open("kernel.cubin","rb").read())
```

NVRTC is for small kernels without system headers. nvcc for anything including
`cuda_runtime.h`, `cuda_bf16.h`, `cooperative_groups.h`.

### 6. cuLaunchKernelEx kernelParams format

```python
# Source: cuda.bindings._lib.utils.pxi _HelperKernelParams
# Device pointer: pass as (int(CUdeviceptr), ctypes.c_void_p)
params = ((int(d_ptr1), int(d_ptr2)), (ctypes.c_void_p, ctypes.c_void_p))
cdrv.cuLaunchKernelEx(cfg, cu_function, params, 0)

# Note: cu_function must be CUfunction (from cuModuleGetFunction),
# NOT CUkernel (from cc.Program().get_kernel()) — different types!
```

### 7. ct.launch stream must be int

```python
# WRONG: cuda.core.Stream object
ct.launch(stream_obj, ...)

# CORRECT: raw integer pointer
stream_int = int(stream_obj.__cuda_stream__()[1])
ct.launch(stream_int, ...)
```

### 8. Thread Block Clusters need nvcc path, not cc.Program

```python
# cc.Program returns CUkernel; cuLaunchKernelEx needs CUfunction → use nvcc
# cooperative_groups.h also triggers NVRTC hang on Windows
# Compile with nvcc, load cubin, get CUfunction via cuModuleGetFunction
```

---

## Reference implementations — study before writing kernels

### Flash Attention on sm_120a (RTX 5080/5090)

| Source | Approach | Perf sm_120 | Tile (Q, KV, D) | Precision |
|--------|----------|-------------|-----------------|-----------|
| [gau-nernst/fa-5090](https://github.com/gau-nernst/learn-cuda/tree/e83c256/07_attention) | Raw CUDA C++ PTX | **197 TFLOPS (94% SOL)** | 128, 64, 128 | BF16 |
| NVIDIA CUDA Tile [arXiv 2604.23466](https://arxiv.org/html/2604.23466v1) | cuda-tile Python | 179 TFLOPS (~53% FA2) | 64 (baseline) | BF16 |
| [SageAttention 3](https://github.com/thu-ml/SageAttention) | MXFP4 | **560 TFLOPS (RTX 5090)** | - | FP4+BF16 |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | CUTLASS | sm_89 fallback only | - | FP8/FP16 |

**Code read: [attention_v5.cu](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/attention_v5.cu)**

**Key findings for attention on sm_120a:**

1. **Actual 94% SOL config**: BLOCK_Q=**64**, BLOCK_KV=**64**, DIM=128, NUM_WARPS=4.
   SMEM = max(64, 192) × 128 × 2 = **49,152 B ≈ 48 KB** (no opt-in needed).
   Earlier search said BLOCK_Q=128 — that was WRONG. Source code says 64×64.

2. **Q SMEM reuse trick**: Q is loaded into SMEM once, moved to registers, then
   Q_smem address is REUSED for K double-buffer. Saves 16 KB SMEM.
   K uses double-buffer (2 × 64×128×2 = 32 KB), V uses single-buffer (16 KB).

3. **Pipeline** (hiding memory latency inside softmax):
   ```
   iter kv:
     __syncthreads(); issue load_V(kv)     ← async, no wait yet
     wait_group(1): K[kv] done → K→regs
     MMA: S = Q @ K^T
     issue load_K(kv+1)                    ← prefetch DURING softmax
     online softmax (exp, shfl_xor reduce)
     wait_group(1): V[kv] done → V→regs
     MMA: O += P @ V
   ```

4. **Async copy**: `cp.async.cg.shared.global` 16-byte (NOT TMA/cp.async.bulk).
   `ct.load` in cuda-tile handles this automatically.

5. **Online softmax** (register-level, per mma tile):
   ```
   m_new  = max(m, rowmax(S))             # scalar, 2 values per mma tile
   alpha  = exp(m - m_new)                # rescale O from prev iterations
   P      = exp(S - m_new)                # attention weights
   l      = alpha * l + rowsum(P)
   O      = alpha * O + P_bf16 @ V        # P cast bf16 for MMA input
   m      = m_new
   ```
   Warp-level reduction of rowmax/rowsumexp via `__shfl_xor_sync`.
   In cuda-tile: `ct.max(S, axis=-1)`, `ct.sum(P, axis=-1)`, `ct.maximum`.

6. **MMA**: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
   = `ct.mma(Q_bf16, K_bf16, acc_f32)`. No WGMMA or tcgen05 needed.

7. **Bank conflict**: XOR swizzle on SMEM row index. `ct.load` auto-handles.

8. **cuda-tile vs raw CUDA gap**: NVIDIA paper got 53% on sm_120 with suboptimal
   tiles; 64×64 + correct pipeline should get much closer to 94% SOL.

### Megakernel reference

- **[HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels)**
  "No Bubbles" blog: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
  - Persistent kernel with on-GPU instruction interpreter
  - 13×16 KB SMEM pages, counter-based global sync
  - 78% bandwidth utilization on H100
  - Not using WGMMA/TMEM — just standard MMA + careful scheduling

---

## Batch invariance — HARD REQUIREMENT for RL correctness

On-policy RL needs rollout (inference) and training to compute **bitwise-identical
logprobs** for the same tokens. Otherwise the importance ratio π_train/π_infer ≠ 1
and GRPO/PPO gradients are biased. Root cause of mismatch (Thinking Machines, 2025):
kernels change their **reduction order** as a function of batch size / seq length.

### The invariants — every kernel MUST follow these

1. **Fixed tile sizes, never batch-dependent.** TM/TN/TK, BQ/BKV are compile-time
   constants. NEVER autotune or switch tile shape based on batch size or seq length.

2. **No Split-K in matmul.** One block owns the full K reduction, accumulated
   sequentially: `for k in range(K_BLOCKS): acc = ct.mma_scaled(...)`. No atomic
   accumulation into a shared output. (linear.py already complies.)

3. **No split-KV in attention** (for the training/prefill path). One block owns the
   full per-query KV reduction, sequential `for kv in range(NKVB)`. (attention.py
   already complies.) If decode needs split-KV for SM occupancy later, use
   **fixed split SIZE** (e.g. 256), variable number of splits — NOT fixed number.

4. **Same kernel for rollout and training.** Do not use a fast vanilla softmax in
   rollout and online (flash) softmax in training — they differ in FP. Both paths
   call the same kernel.
   **DECODE MUST REUSE THE PREFILL ATTENTION KERNEL** (2026-06-04, proven in
   tests/kernels/test_attn_decode_unified.py). The hand-rolled single-query `_attn_decode`
   computes q·k / p·v with `ct.sum(broadcast·…)` (CUDA-core), a DIFFERENT reduction order than
   `_attn_fwd`'s `ct.mma` (tensor-core) → 0.216% off prefill, NOT bitwise → breaks π_train==π_infer.
   Fix: to get position t's attention during generation, run the PREFILL kernel (`_attn_fwd`) over
   the cache and take row t. The causal mask zeroes j>t so not-yet-generated cache slots don't
   affect row t, and prefill is seq-len invariant → **bitwise-identical to training (max|Δ|=0,
   even with garbage future slots).** Since GEMM/RMSNorm/RoPE/SwiGLU are per-row/per-position or
   already batch-invariant, a rollout forward = batch-invariant GEMM/RMSNorm + prefill-kernel
   decode attention ⇒ **ratio π_train/π_infer = 1 EXACTLY, no importance sampling needed** (the
   single-codebase advantage: the two-system labs vLLM+Megatron can't bitwise-match so they need
   TIS/MIS; we can). Research basis: Thinking Machines fixed-split-size attention, Jet-RL/NVIDIA/
   Cursor unified-FP8 flow — see [[batch-invariance]]. Precision plan: BF16 unify first, MXFP8 later.
   **Device-decode kernels DONE + bitwise-verified (2026-06-04):** `_attn_decode_blk` (full BQ-block,
   runtime q_blk; single-row MMA is 1-ULP off so MUST use the full block), `rope._rope_fwd_dec`
   (single-position), `attention._scatter_blk`/`_gather_blk` (place/pull the frontier query at in-block
   row pmod on-device). Full device decode-attention pipeline (scatter→attn_decode_blk→gather over a
   device KV-cache) is bitwise == prefill (test_device_decode_plumbing.py, test_attn_decode_mma.py).
   **✅ `ResidentDecodeLayer` DONE + end-to-end ratio=1 verified (2026-06-04):** `ancora/model/
   resident_decode.py` mirrors `ResidentLayerTrain.forward` kernel-for-kernel for a single decode token
   with a per-layer KV-cache (`Kc,Vc (Bp*Hkv*maxS,D)`, `_append_kv` at `cache.at_pos(pos)`; new
   `_DBuf.at_pos(pos)` = ptr += pos·row_width). GEMM batch padded to `MGEMM=128` (a full 128-row MMA
   tile → frontier row bitwise-equal to prefill; rows 1.. are harmless padding), attention/cache sized
   for the REAL `Bp` only (no 128× cache blow-up). tok↔head IS the identity at S=1 → the prefill
   `_tok_to_head`/`_head_to_tok` transposes become free `.view()` reshapes. **`tests/model/
   test_resident_decode.py`: autoregressive rollout (one ResidentDecodeLayer-stack token at a time,
   teacher-forced) vs training prefill (ResidentLayerTrain-stack, whole sequence) → (A) hidden[t]
   BITWISE == prefill[t] every position, (B) rollout logprob == training prefill logprob bitwise
   (max|Δ|=0), (C) decode deterministic across runs. Verified NL=1/2/4, S=128/256 — ratio π_train/
   π_infer = 1 EXACTLY, no importance sampling.** Proof = double induction over positions×layers: each
   decode step appends its own K/V (per-row == prefill's), `_attn_decode_blk` attends cache[0..t]
   (future slots masked) == `_attn_fwd` row t, GEMM/RMSNorm/RoPE/SwiGLU per-row.
   ⚠️ The HOST rollout (`rl/rollout.py` generate_cached) is inherently nondeterministic (alloc-churn) —
   ratio=1 only works on this device-resident path. Blueprint in [[batch-invariance]].
   **✅ `ResidentDecodeModel` (full rollout engine) DONE (2026-06-04):** the ResidentDecodeLayer stack + the
   SAME tied embed/LM-head boundary ResidentModel uses (`decode_step`/`score`(teacher-forced)/`generate`(greedy
   autoregressive)). `tests/model/test_resident_decode_model.py` (NL=2/4): teacher-forced `score` per-token
   logprob bitwise == `ResidentModel` training `glp`, AND **the logprob `generate` reports DURING rollout is
   bitwise == what training assigns to that generated token** (max|Δ|=0) — ratio=1 at the full-model level.
   ⚠️ LESSON (alloc-churn rule, again): the per-step decode boundary's final RMSNorm must run on-device with
   PERSISTENT buffers — NOT host `rmsnorm_forward`, which allocs/frees GPU scratch every step and that churn
   RACES the decode kernels on `si` → flaky single-position garbage (run1 OK / run2 pos43 off 0.45 / run3 OK;
   hidden was bitwise but logprob intermittently wrong). Replicate rmsnorm_forward's exact compute (host bf16
   round → `_rmsnorm_stats`+`_rmsnorm_apply`) into persistent buffers ⇒ bitwise-equal AND churn-free. Any
   host-API kernel (self-allocating) in a hot loop can introduce this race even though it self-syncs.
   **✅ MXFP8 forward unified rollout↔training DONE (2026-06-04):** `ResidentLayerTrain` and `ResidentDecodeLayer`
   both take `mxfp8=True` — the 7 projection GEMMs run in MXFP8 (`_quant_mxfp8` per-row-per-32-block activation
   quant + `_fwd_mxfp8_bf16` `mma_scaled`, weights pre-quantized `quantize_colblock`); attention stays BF16, the
   backward stays BF16 (uses the bf16 `self.w`, the recipe's forward-MXFP8/backward-BF16 split). Because quant +
   mma_scaled are per-row and `mxfp8_tile` is batch-independent, the MGEMM=128 full-tile trick still makes the
   MXFP8 decode bitwise-equal to the MXFP8 prefill. `tests/model/test_mxfp8_unified.py` (NL=1/2/4): **(A) MXFP8
   decode == MXFP8 prefill BITWISE every position (ratio=1 under MXFP8); (B) MXFP8-vs-BF16 forward drift 3% max /
   7% mean on REAL Qwen3-0.6B** (synthetic random is a harder ~8-13% case). So the forward can move to MXFP8 for
   speed without breaking ratio=1. (Default `mxfp8=False`.)
   **✅ MXFP8 training CLOSED LOOP DONE (2026-06-04):** the forward fp8 weights are (re)quantized ON DEVICE from
   the bf16 master via a new `quant._quant_mxfp8_w` (colblock, 32-along-K, same floor-E8M0 as `_quant_mxfp8` so
   train re-quant and rollout quant give identical bytes → bitwise stays). `ResidentLayerTrain(mxfp8=True)` quantizes
   lazily on the first forward and re-dirties after each AdamW `step()` → the next forward re-quantizes, so the MXFP8
   forward GEMMs track the weight updates with NO host round-trip / alloc churn. Backward stays BF16 (uses the bf16
   `self.w`; straight-through wrt the quant — the QAT recipe). `tests/model/test_mxfp8_train.py`: **MXFP8 fwd→BF16
   bwd→AdamW→re-quant overfit collapses 0.59→0.0046 (0.8%, beats BF16's 1.4%); a frozen-fp8 control stays flat at
   99.9%** — proving the device re-quant is what closes the loop. Remaining lever: FP8 (E5M2) backward (recipe's
   future dgrad perf lever; BF16 backward is the safe default now).
   **✅ Bitwise-safe MFU step 1 — CUDA-graph capture (2026-06-04):** the device-resident NL-layer forward
   (ResidentLayerTrain stack, ~15 launches/layer on persistent buffers) captures into ONE CUDA graph
   (`dev.create_graph_builder()` → forward on the builder stream → `complete()` → `graph.launch`). Replay is
   BITWISE-identical to direct launch (same kernels/order/buffers → max|Δ|=0, ratio=1 untouched) and cuts host
   launch overhead 1.1–2.2× (bigger at small/launch-bound sizes). `tests/benchmarks/bench_resident_graph.py`.
   This is the megakernel's host-overhead win without PTX; cross-operator compute overlap still needs a real
   persistent megakernel.
   **✅ Profiling + optimization (2026-06-04, ncu counters now ENABLED — no more ERR_NVGPUCTRPERM):** ncu
   SpeedOfLight (M=8192) → attention `_attn_fwd_tok` **90% compute / 32% L2** (near peak, leave it); MXFP8 GEMM
   `_fwd_mxfp8_bf16` **55% compute / 61% L2** and fused gate/up **40% / 57% L2** (both L2-bound = the cuda-tile
   no-SMEM-tiling ceiling, needs megakernel to break). Admin-free per-operator event timing
   (`tests/hardware/_profile_forward.py`) → BF16 forward is **`_gemm_bf16` 55%**; MXFP8 forward was **`_quant_mxfp8`
   30% > matmul 26%** because each activation was re-quantized per-GEMM (h 3×, gh2 2×) → MXFP8 was *slower* than BF16.
   **Fix = quant-once** (`_gemm` caches the last-quantized source buffer id per width, cleared each forward; bitwise
   -safe since quant is deterministic): quant 7→4 launches, and **MXFP8 forward goes from slower to 1.29× faster than
   BF16 at M=2048** (1369 vs 1768 µs; MXFP8 matmul 2.4× faster, quant down to 16%). Applied to both ResidentLayerTrain
   and ResidentDecodeLayer (ratio=1 preserved). GEMM L2-bound + the persistent megakernel remain the next MFU levers.

5. **Reductions stay on one core** (RMSNorm, log_softmax in loss). One program per
   row, sequential accumulation. Do not split a row's reduction across blocks.

### Verified (tests/kernels/test_batch_invariant.py — all bitwise IDENTICAL)

- determinism: same input twice → identical (requires `cudaStreamSynchronize`
  before DtoH copy — see pitfall below)
- batch-size invariance: batch 0 output identical for B=1 vs B=4
- **seq-len invariance: token at position t identical for S=256 vs S=512**
  (THE RL-critical one — proves rollout/training logprob match). Works because
  causal-masked future blocks contribute exp(-inf)=0 and alpha=exp2(m-m)=1.0 exactly.

### Pitfall: missing stream sync = nondeterminism

`ct.launch` runs on `stream_int`; `cuMemcpyDtoH` (in `to_numpy`) runs on the default
stream. Without `cudart.cudaStreamSynchronize(stream_int)` between them, the copy
races the kernel → partial reads → nondeterministic output. This looked like a
"logic bug at S=256" but was a missing sync. ALWAYS sync the launch stream before
copying results back.

**Generalization (found building the KV-cache append, 2026-05-31):** any HtoD that
FEEDS a kernel on `stream_int` must also be on `stream_int`. Staging the KV-cache
input with `cuMemcpyHtoD` (default stream) while the append kernel runs on `si` is a
cross-stream race → the cache is intermittently ~15% corrupt, FLAKY across runs, and
**`cudaDeviceSynchronize` does NOT fix it** (the hazard is the per-token upload↔kernel
order, not a final barrier). Fix: `cuMemcpyHtoDAsync(dst, host, nbytes, si)` — upload on
the SAME stream as the consuming kernel (and keep the host buffer alive until synced).
Rule: one stream end-to-end for a dependency chain; cross-stream needs explicit ordering.

**Alloc/free CHURN races async kernels (KV-cache decode, 2026-05-31):** allocating +
freeing GPU scratch EVERY step (per decode token) — even with same-stream uploads + a
per-step sync — intermittently corrupts results: scattered single-position garbage (decode
positions +55/+59 at 90-117% error, DIFFERENT positions across runs; nl=1 vs nl=2 differ).
The driver's free/realloc hands back addresses still referenced by in-flight work. Fix:
**PREALLOCATE scratch once and REUSE across all steps** (no per-step cuMemAlloc/cuMemFree).
After: 117% → 1.2%, all positions clean. The host-API kernels (rmsnorm_forward etc.) are
safe because each self-syncs (alloc→upload→launch→sync→download→free in one call). So:
reuse buffers in any hot multi-step loop; never alloc/free per iteration.

**RE-CONFIRMED on the MoE host-loop backward (2026-06-06, moe_layer.py):** `MoEFFN.backward`
loops over 16 experts calling self-allocating `linear_bf16_backward`/`swiglu_backward` (~110
alloc/free GPU buffers per call, ≈2× the forward). That churn intermittently corrupts ONE
expert's `dW` readback → that expert's grad jumps 0.45% → 80% on ~1-in-4 runs, FLAKY (adding
one unrelated line flips which run is hit). `d_h`/`d_router` aggregate over all experts so they
stay stable; only the per-expert `d_down` exposes it. The FORWARD churns half as much and is
reliable; the formula is proven correct (fp64 finite-difference = 0.000%) — this is purely the
alloc-churn race, NOT a logic bug. The grouped/segmented-GEMM perf kernel (preallocated expert-
capacity buffers, one launch instead of a 16-expert Python loop) removes both the churn and the
loop. test_moe_layer.py takes best-of-4 + prints clean_runs/4 so the arithmetic is verified while
the churn stays visible. LESSON: per-expert host loops over self-allocating kernels are the SAME
hazard as per-step decode loops — the eventual MoE kernel MUST preallocate. See [[moe-architecture]].
**RESOLVED (2026-06-06): `kernels/moe.py` GroupedMoEFFN** — stable-sort tokens by expert (host),
pad each group to a TM-tile, then ONE launch per stage over all experts on PREALLOCATED buffers
(grouped GEMM reads a per-tile expert id from an array — `ct.reshape(load(idx),())` data-dependent
index, probed-OK; weight-grad = per-expert reduction with data-dependent loop bounds). Fwd+bwd
vs fp64 ≤0.45%, and **6× fwd AND bwd bitwise-identical** (churn gone). Drop-in for MoEFFN.

### Notes

- BF16 input + FP32 accumulator is consistent within our stack (bitwise invariant).
  FP16 reduces mismatch vs an *external* engine (vLLM) but we control both paths.
- TIS (Truncated Importance Sampling): clip ratio `min(π_train/π_infer, C)` is cheap
  insurance for residual mismatch, but the kernel-level fix above is the real route.

---

## External precision recipe — a reference to weigh, NOT a directive

This is a digest of the **MAI-Thinking-1** training recipe (+ MiMo-V2.5 serving notes), originally via a Gemini
summary — a secondary reference, not a spec we must follow. Weigh each item against our own hardware (sm_120a)
and measurements; **diverge wherever ours disagree** (we kept MXFP8 + bf16 vocab GEMM against the recipe).
Mapped below — ✓ = we already comply, ⚠ = conflict (resolved per "Precision decisions").

**GEMM precision split the recipe uses:**
| GEMM | Recipe | Ours | Status |
|------|--------|------|--------|
| Forward | FP8 **E4M3** + delayed (1024-step abs-max) **tensor** scaling | **MXFP8** (per-block E8M0, `mma_scaled`) | ⚠ different scaling philosophy — DISCUSS |
| Data-gradient (dgrad) | FP8 **E5M2** + delayed scaling | **FP8 E4M3** (opt-in `fp8_bwd`) / BF16 default | ✓ DONE 2026-06-14 — E4M3 not E5M2 (our block scaling handles range) |
| Weight-gradient (wgrad) | BF16 compute + **FP32 accum** | BF16 + FP32 accum (`_gemm_dW`) | ✓ |

**FP32 "safe zones" the recipe MANDATES (never downcast these):**
- **The entire residual stream, embed→output, stays FP32.** ⚠ We currently do **all-BF16 device handoff** —
  this is precisely the massive-activation (~6912) divergence in [[resident-layer]]. The recipe confirms the
  parked "fp32 residual stream" follow-up is the *correct* design, not a nicety. → planned fix.
- **All pre-softmax / output activations** FP32 (attention scores, output logits). ✓ attn acc is f32; logits
  are f32-out and `_ce_stats` runs on f32.
- **Final vocabulary GEMM** fully FP32. ⚠ we feed bf16 hidden × bf16 embed (only accumulate/output is f32). DISCUSS.
- **Embedding + RMSNorm weights** FP32. Partial: we keep an fp32 *master* but the resident GEMM reads a bf16 view.
- **AdamW master + momentum + all AdamW math** FP32. ✓ device AdamW (fp32 p32/m/v).
- **Gradient-accum buffers** FP32. ✓ (`_acc_f32`, `_gemm_dW` f32 out).

**Casts & rounding:**
- **Fuse casts into adjacent ops** (esp. high→low into RMSNorm). ✓ `_rmsnorm_apply_q` emits FP8+E8M0 directly (CODA).
- **Stochastic rounding REQUIRED on every high→low gradient downcast in backward** (in-layer compute precision <
  residual precision). ⚠ **We do none** (RTN/truncation). Becomes load-bearing the instant the residual goes FP32
  (casting fp32 grad → bf16 matmul input). → companion to the fp32-residual work.
- **Fused quant of BOTH x and xᵀ** + scale-factor swizzling (one kernel builds both operands the backward needs).
  Only relevant once we adopt FP8 backward.

**Determinism — matches our batch-invariance rules (strong external confirmation):**
- No GPU atomics; **two-stage tiled reduction** (partial sums → fixed-order finalize). ✓ exactly our RMSNorm-dw
  Megatron 2-pass + no-split-K + no-split-KV.
- (MoE only) stable-sort top-k. N/A — Qwen3-0.6B is dense.

**RL train/infer consistency:**
- **BF16 for BOTH learner and inference** in RL (smaller numerics gap than lower precision). ✓ our batch-invariance
  rule #4 (same kernel both paths, BF16-in + FP32-acc). Note the tension with an MXFP8 *forward* — see decisions.
- MoE-routing replay, top-p mask replay, NVLink-SHARP off, NCCL topology pinning, length bucketing, Radix KV-cache
  affinity, SWA dual-pool — **out of scope** (MoE / multi-GPU / serving infra; we are single-GPU dense).

**Precision decisions (2026-06-03, reconciled with the recipe — user sign-off):**
1. **Forward stays MXFP8** (NOT delayed-scaling E4M3). The recipe's "no MXFP8" is Hopper-era — Hopper has no
   block-scaling tensor cores. sm_120a does (`mma_scaled`), and block-scaling is stateless (no 1024-step abs-max
   history to checkpoint) + outlier-robust (handles the ~6912 massive activation per-32-block). We already use
   E4M3 *elements*; only scaling granularity differs and per-block is the right call here. See [[precision-format-decision]].
2. **FP32 residual stream — DONE (2026-06-03).** The forward residual carry (embed→output + the two intra-layer
   adds) is fp32; matmul inputs stay bf16 (cast-in / f32-accum / add-back-into-fp32). New kernels: `norm.py`
   `_rmsnorm_{stats,apply,bwd_dx,dw_part}_f32` (read the residual x as native f32) + `fused.py` `_residual_add_rf32`
   (f32 residual + bf16 branch → f32). Wired into `ResidentLayerTrain` (`gx2`/`gout` fp32, `_rms`/`_rms_bwd` take an
   `xf32` flag for input_ln/post_ln), `ResidentModel` (`gin` fp32, embed uploaded f32, final-norm reads f32 hpre),
   AND the rollout `ResidentLayer` (kernels `norm._rmsnorm_apply_q_f32` + `linear._fwd_mxfp8_f32res`; cutlass path
   uses `_residual_add_rf32`; fwd 4.61% vs host, was 5.09% at bf16; determinism + CUDA-graph capture intact). So
   training AND rollout now carry fp32 residual; full bitwise rollout↔training match still needs unifying the
   forward GEMM precision (rollout MXFP8 vs training BF16) — a separate task.
   The GRADIENT residual stays bf16. **Stochastic rounding DONE (2026-06-03)** on the dominant fp32→bf16
   gradient downcast: `fused.py` `_gemm_dx_sr` rounds the f32 activation-gradient accumulator to bf16 with a
   per-element dither (an in-kernel counter hash of the global output coords XOR a per-step `seed` — cuda.tile has
   NO RNG primitive). Coord-keyed ⇒ batch-invariant for a fixed seed; seed varies per step ⇒ unbiased across
   steps (verified: mean over 600 seeds 0.017% vs RTN 0.211%, tests/kernels/test_gemm_dx_sr.py). Wired into
   `ResidentLayerTrain._dx` (`sr_grad=True` default, `lid` salts per layer); overfit still collapses, forward
   determinism unaffected (SR is backward-only). NB its value is **marginal in our design** (fp32 weight-grad +
   fp32 AdamW already close the bias-accumulation path) — it becomes load-bearing with FP8 (E5M2) backward, and
   the same SR-store extends to the rms-bwd / `_cast64` downcasts then. The rms-bwd dx + `_cast64` keep RTN for now.
   **Validated:** kernels exact vs numpy (tests/kernels/test_fp32_residual_kernels.py); single layer fwd 0.82% / grads
   ≤1.39% / MSE collapses; ResidentModel determinism **bitwise** + CE collapses at NL=4/8/28; per-layer drift vs host
   0.04-0.48% (clean). Cost negligible (0.16 s/layer-step, unchanged). **Caveat — host-match is the WRONG yardstick:**
   the host path uses bf16 residual AND is nondeterministic (alloc-churn). Interleaving host+device kernels on one
   stream RACES → bogus 70-96% "drift" (the _diag must run the two chains SEPARATELY). fp32-residual's real win is
   precision retention of the ~6912 massive activation + bitwise determinism, not a tighter bf16-host match.
   **Reframe:** for our RL goal this is FIDELITY, not correctness — rollout==training share the (now fp32) device
   path, so they're self-consistent regardless; fp32 matters for HF-reference faithfulness + not biasing real runs.
3. **Final-vocab GEMM stays bf16-in / f32-accum / f32-logits** (NOT full fp32-input). The logits + CE math are
   already f32 (the part that matters); full fp32 inputs would tank the single biggest GEMM for unmeasurable gain.
4. **FP8 dgrad DONE (2026-06-14), opt-in `fp8_bwd=True`** — the data-gradient `dx=dy@Wᵀ` runs in FP8
   **E4M3** (NOT MAI's E5M2): both dy and W quantized per-32 along N (contraction) + E8M0 block scales,
   `fused._gemm_dx_fp8` mma_scaled with the W operand+scale transposed IN-KERNEL (probed OK). E4M3 because
   our FINE block scaling handles the dynamic range at the SCALE layer (DeepSeek-V3 regime) → the more-
   precise E4M3 beats E5M2 ~2× (3.8% vs 7.4% rel-err; probed _probe_fp8_dgrad.py). The wgrad stays
   BF16+FP32 (permanent weight grad). **Backward-ONLY → forward bitwise-UNCHANGED → ratio=1 UNTOUCHED**
   (test_fp8_dgrad.py: forward Δ=0 vs BF16, grad cosine 0.997-0.999 = quant noise not a bug, SFT 7.8→
   0.0001). ⚠ The lazy along-N weight requant MUST live in `_dx` (not `backward()`) — the MoE-family
   layers OVERRIDE backward() and don't call super, so a requant in backward() never runs → fp8 reads
   zero weights → dead layer grads (caught: cosine 0.0). This is the SFT/pretrain lever; **RL keeps BF16
   both fwd AND bwd** (rollout-bound so the dgrad speedup is moot + stability). wf8/ws8 (forward, along-K)
   and wf8_n/ws8_n (dgrad, along-N) are the "quantize both orientations" MAI mentions. Probes:
   _probe_fp8_dgrad{,_transpose}.py, _probe_dx_fp8_kernel.py. ⚠ PERF MEASURED NET-NEGATIVE ~5% (M=2048
   step 163.7→171.8ms, **bwd UNCHANGED 96.1→97.5**): the dgrad GEMM isn't compute-bound at M=2048 so FP8's
   2× peak doesn't translate, and the per-backward dy-quant + 2nd-weight-orientation launches add latency —
   the SAME quant-tax verdict as forward MXFP8 / CUTLASS-hybrid. So BF16 stays the default; fp8_bwd's value
   is the MAI-faithful recipe + learning, not speed. (Would pay off only with CODA quant-epilogue fusion +
   compute-bound sizes.)

---

## Compilation guide

| Kernel type | Compiler | Note |
|-------------|----------|------|
| `@ct.kernel` Python functions | `ct.launch` auto-compiles | Default path |
| Inline PTX / no system headers | `cc.Program(...).compile("cubin")` | NVRTC, fast |
| `cooperative_groups.h`, `cuda_bf16.h`, `cuda_runtime.h` | `nvcc -arch=sm_120a` | NVRTC hangs on Windows |

nvcc requires MSVC: `cmd /c "vcvars64.bat && nvcc ..."`.
Load cubin: `cdrv.cuModuleLoadData(np.char.array(open("x.cubin","rb").read()))`.

---

## Profiling with Nsight Compute (ncu) — DON'T reinvent this

**ncu IS installed** (no download needed):
`"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\target\windows-desktop-win7-x64\ncu.exe"`
(also `ncu-ui.exe` under `host\windows-desktop-win7-x64\` for the GUI; `nvcc` is in CUDA v13.3\bin; `nsys` NOT installed).

**BLOCKER: `ERR_NVGPUCTRPERM`** — GPU perf counters are admin-only by default on Windows. ncu connects and runs the
script but reports NO metrics. Two fixes (USER must do, agent shell is NOT admin and can't):
  1. One-time: NVIDIA Control Panel → Desktop/Developer → **Manage GPU Performance Counters → "Allow access to GPU
     performance counters to all users"** → reboot. (Or set `HKLM\SYSTEM\...\NVIDIA Corporation\Global\NVTweak\
     RmProfilingAdminOnly = 0 (DWORD)` + reboot.) After this the agent can run ncu directly.
  2. Per-run: launch the ncu command from an **elevated (Run as administrator)** terminal.

**Profiling pattern** (helper: `tests/hardware/_ncu_target.py gemm|gateup|attn` — launches the kernel N=12× so ncu
skips JIT/warmup and profiles one steady-state launch):
```
ncu.exe --launch-skip 8 --launch-count 1 --section SpeedOfLight python tests/hardware/_ncu_target.py gemm
```
- `--section SpeedOfLight` = the decisive read: **Compute(SM) throughput % vs Memory throughput %** → compute- vs
  memory-bound in one glance. Add `--section MemoryWorkloadAnalysis` / `LaunchStats` / `Occupancy` for detail, or
  `--set full` (slow). cuda-tile kernels show up with a generated/mangled name — use `--launch-skip` to land past warmup.
- Key metrics if scripting: `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__warps_active.avg.pct_of_peak_sustained_active` (occupancy),
  `sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` (BF16 TC util; FP8/MXFP8 use the imma/`mma` variants).
- `-o report --force-overwrite` writes a `.ncu-rep` for ncu-ui. Profiling is SLOW (replays each kernel) → profile ONE
  launch of ONE shape, never a whole bench loop.

**Vendor-reference benchmarking status (for MFU ground-truth vs my own microbench peaks):**
- **FlashAttention-3 does NOT run on sm_120** — it's Hopper sm_90a (wgmma + tcgen05 + TMA warp-spec); ptxas rejects
  wgmma on sm_120 (see hardware facts). So FA3 is NOT a valid baseline on the RTX 5080. FA2 / FlashInfer (has sm_120
  JIT) are the runnable attention refs — but attention is already ~79% of its real BF16 peak, not the bottleneck.
- **cuBLAS/cuBLASLt NOT installed** (no `cublas*.dll` in CUDA bin; cuda-python exposes only driver/runtime/nvrtc).
  Options for a real GEMM ref: `pip install nvidia-cublas-cu13` then call via ctypes, OR build a **CUTLASS** sm_120a
  GEMM with nvcc (git + nvcc are available), OR PyTorch (cu13 build). cupy is a broken empty stub — ignore it.
- **CUTLASS reference IS BUILT** at `C:\project\cutlass` (4.5.2). `build_79c.bat` builds example 79c patched to PURE
  mxfp8×mxfp8: `nvcc -arch=sm_120a -std=c++17 -Xcompiler "/Zc:preprocessor /utf-8 /bigobj" -I include -I tools/util/include
  -I examples/common`. **Required patch for SM120 (bug #2905/#2906/#2820 "misaligned address"):** add `alignas(64)` to the
  TMA-descriptor members — `tma_load_a/b/sfa/sfb` in `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` and
  `tma_load_c/tma_store_d` in `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`. Run:
  `79c.exe --m=4096 --n=4096 --k=4096 --iterations=30`. **Measured MXFP8 (the real achievable peak): 4096³ 244 TF,
  layer shapes 177-243 TF — our cuda-tile GEMM is only 51-76% of this** (see memory [[gemm-mfu-ceiling]]).

---

## Performance baseline (2026-05-30)

### GEMM (linear.py, MXFP8, M=N=K=4096, no SMEM tiling)
| Scenario | TFLOPS | Note |
|----------|--------|------|
| float32 input | 40 | BW × 4 wasted |
| uint8 native FP8 input | **118** | True BW-limited baseline |
| + SMEM tiling (theory) | ~800–1200 | Next optimization target |

### Attention (attention.py, BF16, causal, GQA, batch-invariant)
**2026-06-03: head_dim HARDCODED to 128 (real Qwen3-0.6B; single-model target, per user — no D=64/128
dual abstraction).** `attention.py D=128`, `fused.py HD=128` (transposes), `Qwen3Config.head_dim=128`.
BQ=BKV=64 unchanged (gau-nernst 64×64×128 = 94% SOL). D=128 validated: fwd ≤0.72%, bwd ≤1.23%,
batch-inv 3/3, real-weights SFT collapses. D=128 perf ~57 TF @S=2048 (vs D=64's 60). The numbers
BELOW were measured at the OLD D=64 — kept as the optimization journey (the tuning lessons transfer;
re-measure absolute TFLOPS at D=128 if it matters).

Optimization journey at S=2048 / plateau (S=8192), measured at D=64:
| Version | S=2048 | plateau | win |
|---------|--------|---------|-----|
| naive (float32, full mask, no skip) | 14 | 14.5 | — |
| + causal-skip `range(q_blk+1)` + diagonal-only mask | 40 | 48 | 3.3× |
| + native BF16 input (uint16+bitcast) | 49.6 | 61 | 4.2× |
| + `occupancy=2` (ct.tune autotuned) | 53 | 72 | 5× |
| + `ct.load(..., latency=10)` prefetch hint | **60** | **75** | **5.2×** |

Full API audit results (what helps vs not, sm_120 head_dim=64):
- **`ct.load(latency=10)`**: +13% at S=2048. Hints heavy DRAM → compiler prefetches next
  iter's K/V aggressively, filling the softmax bubble. (latency 1-10; 10 best.) USE THIS.
- **`occupancy=2`** (ct.tune): +17%. occ 2-3 optimal, ≥4 worse, num_ctas=2/clusters
  36× slower, num_worker_warps no effect (kernel not warp-specialized).
- **TMA already ON** by default (`allow_tma` defaults True) — loads already use it.
- **Ping-pong (2 GQA heads/block, shared KV) HURTS**: S=2048 60→47.6. Doubled register
  state (2× m/l/O_acc/S/P) drops occupancy; sm_120 has no spare regs. Same lesson as BQ=128.
- **Conditional rescaling does NOT help**: `where()` computes both branches (tile-select,
  not a real branch) → saves no exp2. Only works in hand-written CUDA.
- **opt_level** default 3 (max). **memory_order/scope** are for atomics, not perf.

**Standalone attention is now genuinely at the cuda-tile ceiling: ~60/75 TFLOPS.**
Every exposed knob explored. Further gains require the MEGAKERNEL (cross-operator bubble
fill: attention's idle Tensor Cores filled by the neighbouring GEMM's matmul). Intra-
attention overlap is blocked by register pressure. See [[mfu-strategy]], [[coda-epilogue-fusion]].

Key tuning facts (sm_120, head_dim=64):
- **BQ=BKV=64 is optimal.** BQ=128 DROPPED to 13 TFLOPS (occupancy collapse).
  Matches NVIDIA CUDA-Tile paper: 64×64 baseline for RTX 50.
- **Causal skip is the biggest win** (~3×): loop `for kv in range(q_blk+1)`,
  data-dependent bound works in cuda-tile. Skips future blocks + masks only diagonal.
- **Native BF16 input** (+27% at plateau): store Q/K/V as uint16 BF16-bits, `ct.bitcast`.
- head_dim=64 caps achievable TFLOPS below head_dim=128 refs (gau-nernst 197,
  NVIDIA cuTile 179 — both head_dim=128, non-causal). NVIDIA's own cuTile attention
  is "53% of FA2, experimental" on sm_120, so 61 for batch-invariant causal GQA is OK.
- **Remaining levers (untried)**: `ct.tune` autotuning, explicit K double-buffering /
  prefetch during softmax (cuda-tile auto-pipelines but may need hints).

### RMSNorm + RoPE (norm.py, rope.py — memory-bound, GB/s; peak 896 GB/s)
| Kernel | GB/s (sustained) | % peak | Note |
|--------|------------------|--------|------|
| RoPE fwd / bwd | ~727 / ~748 | 81% / 83% | single elementwise kernel, near-optimal |
| RMSNorm bwd | ~665 | 74% | dx (2-pass on one block) + Megatron 2-pass dw |
| RMSNorm fwd | ~455 | 51% | 2-kernel stats+apply → 1.5× naive traffic (~72% real) |

- **RMSNorm = TWO kernels** (`_rmsnorm_stats` + `_rmsnorm_apply`), NOT one — a single
  kernel with reduce-loop + reload-loop silently miscompiles (pitfall 0c). Compute in
  fp32, BF16 I/O, save rstd for bwd (Unsloth). H is CHUNKED (TH=128); a (TM,1024) wide
  tile compiles pathologically slowly (dx hung >150 s).
- **RMSNorm dw = Megatron/Apex two-pass partial reduction** (`_rmsnorm_dw_part` →
  `_rmsnorm_dw_reduce`, PART=16). A single block looping all M tokens unrolls 128-256×
  → compiler hangs AND underfills the GPU (8 blocks). dx math = Apex FusedRMSNorm exactly.
- **RoPE = rotate-half / NEOX** (Qwen3/Llama, theta=1e6), cos/sin width-D/2 on host;
  elementwise → trivially batch-invariant. Both norm & RoPE verified bitwise seq-invariant.
- These fuse into GEMM epilogues in the megakernel ([[coda-epilogue-fusion]]) — standalone
  numbers are launch+L2 bound at this size; end-to-end they're nearly free.

---

## Project structure

```
ancora/
├── env.py              # Must import before any cuda.* — sets CUDA_PATH to v13.3
├── kernels/
│   ├── linear.py       # MXFP8 fwd (ct.mma_scaled) + BF16 bwd (ct.mma)  ✓
│   ├── attention.py    # FA fwd + bwd (dQ/dK/dV) causal GQA ✓ + sliding-window (local) fwd/bwd ✓ + decode attn + append_kv ✓
│   │                   #   + PREFIX-SHARED (GRPO): _attn_fwd_prefix (shared prompt KV + per-completion suffix) +
│   │                   #   _attn_bwd_dq_prefix / _attn_bwd_dkdv_prefix (prefix grad = self + Σ_G cross, fixed-order) ✓
│   │                   #   + WINDOWED prefix (_attn_*_prefix_win; host helpers take window=) for the MoE LOCAL
│   │                   #   layers — the window can SPAN the prefix/suffix boundary; same-w masks ⇒ fwd AND suffix
│   │                   #   bwd BITWISE == _attn_fwd_win on [P,s_i] ✓ (compound runtime guards `(kv>=0) and (kv<NB)`
│   │                   #   probed OK — tests/kernels/_probe_prefix_win.py). NB ~1-ULP f32 in dQ between the prefix
│   │                   #   and standard bwd kernels (identical bodies, separate compiles → different FMA contraction;
│   │                   #   dK/dV exact) — harmless, ratio=1 lives in the forward
│   ├── moe.py          # grouped/segmented MoE fwd+bwd (GroupedMoEFFN): 1 launch/stage over all experts,
│   │                   #   preallocated → kills MoEFFN's per-expert loop + alloc-churn; bitwise-deterministic ✓;
│   │                   #   + forward_resident/backward_resident (device bf16-bits I/O) + device AdamW over the
│   │                   #   3D expert weights (Wg/Wu/Wd=p16 in-place, _transpose_e refreshes WgT/WuT/WdT) ✓;
│   │                   #   + device_route=True → router gating+dispatch+BACKWARD+AdamW all on device, sync-free
│   │                   #   fwd+bwd+step (Wr_dev=fp32 AdamW master) ✓; _ggemm padding-tile skip (sentinel e==NE) ✓
│   ├── moe_dispatch.cu/.py # device-resident MoE router (plain CUDA, NVRTC sm_120a — no cub/no atomics):
│   │                   #   FWD: moe_router_gate (warp/token COALESCED h + float4 Wr, fp32 softmax+top-k) +
│   │                   #        moe_build_layout (deterministic stable sort-by-expert == host build_layout BITWISE);
│   │                   #   BWD: moe_router_gate_bwd → d_logits, moe_router_dW (G_router=hᵀ@dlogits, coalesced),
│   │                   #        moe_router_dh (gdh2 = expert + dlogits@Wrᵀ, warp/token) ✓ (only step() pulls 64KB Grouter)
│   ├── norm.py         # RMSNorm fwd (stats+apply) + bwd (dx, Megatron 2-pass dw)  ✓
│   ├── rope.py         # RoPE fwd/bwd, rotate-half/NEOX (Qwen3 theta=1e6)  ✓
│   ├── activation.py   # SwiGLU fwd/bwd (silu(gate)*up, tanh form)  ✓
│   ├── quant.py        # MXFP8 quant (host + GPU _quant_mxfp8) for FP8 forward  ✓
│   ├── fused.py        # device-resident plumbing: _gemm_bf16, _residual_add, _cast, tok↔head transpose  ✓
│   └── loss.py         # fused linear-CE (Liger GEMM path) + streaming fallback  ✓
├── model/
│   ├── qwen3_layer.py  # TransformerLayer: wires all kernels — fwd ✓ + bwd ✓ (both validated)
│   ├── qwen3_model.py  # full model: embed + N layers + final norm + LM head; fwd + loss_backward
│   ├── moe_layer.py    # MAI-style interleaved dense/MoE + local/global decoder (NEW model family) —
│   │                   #   uniform-square SwiGLU, E=16/top-2, fwd+bwd ✓ (windowed local attn wired) [[moe-architecture]]
│   ├── moe_model.py    # full MoEModel: (tied) embed + scheduled MoEDecoderLayers + final norm + LM head;
│   │                   #   fwd + loss_backward; grouped=True → GroupedMoEFFN per MoE layer; overfit ✓
│   ├── resident_moe.py # device-resident MoE training layer (kills host-orchestration MFU wall).
│   │                   #   M1 ResidentMoEDenseLayer (dense FFN+local/global) fwd+bwd ✓ 27-200×;
│   │                   #   M2 ResidentMoELayer (grouped MoE FFN resident, router=1 RT/layer) fwd+bwd ✓ 41-180×;
│   │                   #   shared _attn_bwd_chain; closed-loop fwd→bwd→device-AdamW overfit ✓ [[moe-architecture]]
│   ├── resident_moe_model.py # full device-resident MoEModel: scheduled ResidentMoEDense/MoE layers + the
│   │                   #   reused tied embed/LM-head boundary + all-device AdamW. End-to-end SFT: forward
│   │                   #   bitwise-deterministic, CE 7.87→0.0002, 31ms/step NL=4 ✓
│   │                   #   ✅ final RMSNorm fwd+bwd now DEVICE too (2026-06-11): _cast_bf16(RNE!)+_rmsnorm_
│   │                   #   {stats,apply,bwd_dx,dw 2-pass} on persistent buffers, gain re-uploaded RNE in step()
│   │                   #   — BITWISE == the old host path (identical 150-step CE trace) and ~2× faster
│   │                   #   (24ms/step, mid-step syncs gone). fwd/bwd refactored into pure-launch _fwd_dev/
│   │                   #   _bwd_dev(+_bwd_head_dev hook) → graph-capturable. ⚠ THE RNE TRAP: norm.py's
│   │                   #   f32_to_bf16_bits is RNE; attention.py's + fused._trunc_bf16 TRUNCATE — using trunc
│   │                   #   biased rstd 0.4% (47% elements 1-ulp) and was ALSO the decode model's parked
│   │                   #   "removed sync changes gout" mystery (solved — rounding mode, never a race)
│   │                   #   ✅ TRAINING MEGAKERNEL pass (2026-06-12) — real size (NL=12,V=151936) step
│   │                   #   453→97ms @M=1024 / 926→162ms @M=2048: **MFU 5.4%→30.5%, BEST BATCH M=2048
│   │                   #   (12.6k tok/s)**; M=4096 still WDDM-pages (16GB wall). The four fixes (lp stayed
│   │                   #   bitwise — gates Δ=0; SFT CE trace 7.8685→0.0002 IDENTICAL):
│   │                   #   (1) host (M,V) onehot KILLED — fwd = _embed_gather (decode-proven bitwise ==
│   │                   #   onehot GEMM), bwd builds a per-CHUNK device onehot (fused._onehot_set, bits ==
│   │                   #   host 0x3F80). Was 1.2GB numpy + 0.6GB PCIe per step = 75% of the step(!);
│   │                   #   (2) CHUNKED boundary (Liger-style, MC=1024): glog/gglog/gohot are (MC,V) —
│   │                   #   2.5GB→1.1GB at M=2048 (the AdamW WDDM paging 95→49ms); per-row math chunk-
│   │                   #   invariant (lp bitwise), only the (V,H) dW M-reduction GROUPING changes (≤1ulp
│   │                   #   embed grad, gates pass); input-embed dW = _gemm_dW_acc onto gegrad (giegr
│   │                   #   buffer + _acc_f32 gone);
│   │                   #   (3) boundary kernels re-tiled per ncu: _ce_stats_b/_ce_grad_b at _CTB=4 (CTM=64
│   │                   #   = 16 blocks = ncu SM 5% on a ~1GB stream, 8.1→~2.5ms), dhid _gemm TN=32 (64→256
│   │                   #   blocks; ncu DRAM 11%/SM 53% = underfill), boundary _gemm_dW TN_=128 (x=(M,V) is
│   │                   #   re-read per column block → halves the 76%-DRAM traffic);
│   │                   #   (4) resident_train._gemm bf16: ADAPTIVE TN (halve until grid ≥240 blocks; large-M
│   │                   #   unchanged) — the M≤2048 projection GEMMs were 64-block/SM-47% underfilled.
│   │                   #   Remaining @M=2048: bwd 96ms (layers, fat GEMMs) + AdamW ~32ms floor (551M params
│   │                   #   ×3 fp32 states ≈ 15GB sweep, BW-bound — the 16GB card's tax) + fwd 17ms.
│   │                   #   ✅ GRADIENT ACCUMULATION (2026-06-12): loss_backward(accumulate=True) → micro-
│   │                   #   batch ≥1 ADDS weight grads IN PLACE (no accum buffers — they'd cost ~2.2GB and
│   │                   #   re-page): _gemm_dW_acc / _rmsnorm_dw_reduce_acc / _ggemm_dw_acc on the flag;
│   │                   #   router = tiny (H,E) Gr_acc mirror (raw-CUDA router_dW always overwrites; step()
│   │                   #   copies back — bit-neutral when unused). Pass the TOTAL-token norm to every call.
│   │                   #   test_grad_accum.py: B=2 single step vs B=1×2 accum, at S=256 (M=512, 1 chunk)
│   │                   #   AND S=768 (M=1536 → 2 boundary CHUNKS, a sequence straddling the seam — covers
│   │                   #   the M>MC multi-chunk dW accumulation): lp BITWISE Δ=0 (batch invariance), grads
│   │                   #   ≤1.6e-06 (M-reduction regroup ulp), repeat-det Δ=0, post-AdamW weights 100.00%
│   │                   #   bit-identical. NB micro-batches are SEPARATE SEQUENCES (B-dim) — per-seq ctx ≤2048.
│   │                   #   ⚡ ACCUMULATION IS A *TRAINING-MFU* LEVER: it amortizes the ~60ms AdamW BW-floor
│   │                   #   over more compute (1×M2048 AdamW=33% of step → 2× =18% → MFU 24.5%→29.5% sustained,
│   │                   #   asymptote ~35% as AdamW→0). So accumulate as many micro-batches as the effective
│   │                   #   batch needs — strictly better MFU.
│   │                   #   ✅ MXFP8 PORTED TO THE MoE FAMILY (2026-06-12) — VALIDATED BUT PARKED (default
│   │                   #   mxfp8=False), the CUTLASS-hybrid verdict again: ResidentMoEModel/PrefixMoEModel/
│   │                   #   DecodeModel all take mxfp8=True; new moe._ggemm_mx/_ggemm_gus_mx (grouped
│   │                   #   mma_scaled over PACKED (E·K,N) fp8 experts — 32|K so quant blocks never straddle
│   │                   #   experts; scale tile index == weight tile index at (KSC,TN) shape) + GroupedMoEFFN
│   │                   #   quant lifecycle (_quant_w_mx dirty-on-step; decode aliases the TRAINER's fp8
│   │                   #   buffers + dirty flag = ONE requant authority, _mx_refresh per rollout for graph
│   │                   #   replay). test_moe_mxfp8.py ALL PASS: fwd bitwise-det; 0.8% lp drift vs BF16;
│   │                   #   decode==training lp BITWISE (ratio=1 UNDER MXFP8, incl. fused-vs-separate expert
│   │                   #   GEMM); post-step bitwise through shared requant; SFT CE 6.85→0.47. ⚠ PERF VERDICT:
│   │                   #   NOT a win on this model — training M=2048 162→186ms (fwd is only ~10% of the step;
│   │                   #   quant launches eat the GEMM 2×; fp8 weight copies +0.5GB worsen WDDM paging,
│   │                   #   AdamW 49→71ms) and decode 5.4→8.2ms (Md=128 quant kernels = 2-block latency tax +
│   │                   #   un-fuses the decode megakernel). BF16 stays the default BOTH paths; MXFP8 becomes
│   │                   #   interesting only with FP8 (E5M2) backward + CODA quant epilogues (fuse _quant into
│   │                   #   the producing kernels) — the bwd is where the time lives.
│   ├── resident_prefix.py # DEVICE-RESIDENT prefix-shared layer (GRPO training on the MoE model, NO churn).
│                       #   ResidentPrefixDense/MoELayer subclass ResidentMoEDense/MoELayer: per-token ops
│                       #   inherited (row-independent over M=Sp+G·Sc); overridden = split tok↔head transposes
│                       #   (prompt B=1×Sp + suffix B=G×Sc via _DBuf.at_pos row-offset views), prompt/offset-Sp
│                       #   RoPE tables, prefix attention kernels on buffer regions, prompt dK/dV=self+cross via
│                       #   _add_f32. vs resident REPLICATED layer (global/local × dense/MoE): fwd suffix AND
│                       #   prompt BITWISE Δ=0, bwd suffix ≤0.13% (the dQ 1-ULP), Σ_G prompt ≤0.87%, grads ≤0.39%,
│                       #   repeat-determinism Δ=0 (tests/model/test_resident_prefix.py) — prompt encoded ONCE,
│                       #   ratio=1 preserved, alloc-churn eliminated
│   └── resident_prefix_model.py # ResidentPrefixMoEModel — the full prefix-shared GRPO STEP (subclasses
│                       #   ResidentMoEModel via the _build_layer hook; boundary/AdamW inherited, M=Sp+G·Sc).
│                       #   grpo_loss_backward: adv-weighted CE on completion tokens. Comp token 0 lives on the
│                       #   SHARED prompt row Sp-1 → BOUNDARY-ROW DUPLICATION (PrefixGrouper include_prefix_last,
│                       #   device-resident): head rows Mh=align128(M+G); fwd DtoD-copies hidden row Sp-1 into G
│                       #   tail rows (own label/adv each) → standard _ce_stats/_ce_grad reproduce the replicated
│                       #   per-row math EXACTLY; bwd sums their dhidden back into row Sp-1 (fused._bnd_acc, fixed
│                       #   order — autograd's duplicate-sum made explicit). NO host round-trip in the boundary
│                       #   (an earlier host gglog-row-patch version is superseded). loss_backward grew `norm=`
│                       #   so both paths normalize by G·Sc. Validated vs replicated ResidentMoEModel (tests/
│                       #   model/test_resident_prefix_model.py): ALL scored logprobs BITWISE (suffix Δ=0 AND
│                       #   boundary dup-row Δ=0 → full-model ratio=1), det Δ=0, layer/MoE grads ≤0.62%, embed
│                       #   ≤1.13% (Σ_G prompt-grad PATHWAY: prefix bf16-rounds the summed prompt grad per layer,
│                       #   replicated rounds G copies then f32-sums — depth-growing fp noise, NOT the boundary),
│                       #   GRPO policy improvement (lp gap −0.8→5454); step 2.2-2.5× at Sp=512/Sc=256/G=8
│                       #   (the toy NL=4/M=384 case is launch-bound ≈1×)
│                       #   ✅ CUDA-GRAPH STEP (2026-06-11): with device_route=True (sync-free MoE router) the
│                       #   ENTIRE fwd+bwd is pure launches → capture() records ONE graph (dup rows = DtoDAsync
│                       #   memcpy nodes); graph_step() = upload ids/labels/adv → graph.launch → sync → (ce,
│                       #   lp_comp). Replay BITWISE == direct (lp + layer/embed/fn grads, 3×-det Δ=0); GRPO
│                       #   improves through replays; 1.17× fwd+bwd host overhead at NL=4. AdamW step() stays
│                       #   OUTSIDE (per-step bias-correction scalars would be frozen by capture). NB device-
│                       #   route vs host-route lp differ ~0.65% (router_gate's h·Wr warp-reduce order ≠ numpy;
│                       #   dispatch IS bitwise) — each route is itself bitwise-deterministic; rollout+training
│                       #   must just USE THE SAME route
│   └── resident_moe_decode.py # ✅ MoE ROLLOUT ENGINE (2026-06-11, all perf levers DONE) — ResidentMoEDecode
│                       #   Model/Layer: the decode side that CLOSES the GRPO loop on the MoE family.
│                       #   global=NoPE full-causal, local=RoPE@pos+windowed, MoE FFN = weight-SHARED Grouped
│                       #   MoEFFN at Md=128 (device_route, asserted == training's route). ALL weights ALIASED
│                       #   from the trainer's _DBufs → device AdamW updates visible ZERO-COPY. PERF LEVERS:
│                       #   (1) DEVICE-POSITION kernels — pos in a (1,1)i32 buffer advanced by fused._inc1;
│                       #   scalar store-index/bit-ops/loop-bounds all probed OK (_probe_devpos.py — the old
│                       #   _append_kv "runtime-int store index" question: SUPPORTED) → _append_kv_p/_scatter/
│                       #   _gather_blk_p/_attn_decode_blk(_win)_pd/_rope_fwd_dec_p ⇒ the WHOLE token step is
│                       #   position-free pure launches → ONE-token CUDA GRAPH replay (capture per pick-mode);
│                       #   (2) local-layer RING KV-cache (pow2 blocks ≥ WB+1; memory O(window) not O(maxS);
│                       #   wrap-around bitwise-verified); (3) Gumbel-max DEVICE SAMPLING (loss._sample_id —
│                       #   coord-hash RNG à la SR-grad, seed in device mem inc'd in-graph: seed-deterministic,
│                       #   graph-replayable, batch-invariant); (4) device closed loop (_argmax/_sample→glab→
│                       #   fused._embed_gather feeds next step; ids/lp → device history DtoDAsync) ⇒ ONE sync/
│                       #   ROLLOUT; (5) DCTM vocab-kernel row-tile variants (CTM=64 gave 2 blocks at Md=128,
│                       #   SMs idle on the 39MB V-stream; per-row order unchanged ⇒ lp bitwise). tests/model/
│                       #   test_resident_moe_decode.py ALL BITWISE: teacher-forced==SFT-training every pos
│                       #   (ring wraps); greedy AND SAMPLED rollout lp == prefix-GRPO trainer lp; graph replay
│                       #   == direct (greedy+sampled); zero-copy sharing through a GRPO step.
│                       #   ✅ (6) DECODE MEGAKERNEL DONE (2026-06-12) — fewer/fatter kernels (328→279
│                       #   launches), EVERY fusion probed BITWISE first (_probe_decode_{tiles,fused,attn}.py):
│                       #   • _gemm_bf16 is TN-bitwise-INVARIANT (probed) → projections at DTN=32 (N=1024:
│                       #     8→32 blocks, 2×); • o_proj=_gemm_af32_res (f32 cast+GEMM+residual ONE kernel),
│                       #     dense down=_gemm_bf16_res (+residual); • MoE FFN=forward_resident_dec: _ggemm_gus
│                       #     (gate+up+SwiGLU dual-acc, Gg/Ug never hit HBM, 1.75×) + _ggemm_b(TN64) +
│                       #     _combine_rf32 (+residual); • pick+CE in ONE 78MB logits pass: _argmax_ce_b
│                       #     (greedy lp = m_run−lse, 2.3×) / _sample_ce_b (chosen logit via exact index match,
│                       #     1.33×); • hidden norms _rmsnorm_{stats,apply}_f32_b at TMb=8 — ⚠ ROW-TILE CAN
│                       #     FLIP BITS in cuda-tile reductions: TMb=4 differs, vocab CTMb=4 ok / ≤2 differs —
│                       #     ALWAYS probe; • q/k-norm+RoPE on the REAL Bp·H* rows only (Md-pad rows feed
│                       #     nothing; row-independent ⇒ bitwise). REJECTED by probe/measurement: GQA-paired
│                       #     decode attention (BITWISE ✓ but 2× register state → SLOWER 173→222µs, the
│                       #     ping-pong lesson again; single-head attn is AT ~870GB/s = BW peak), vocab-GEMM
│                       #     TN<128 (slower; it runs 70TF ≈ 88% of the BF16 compute wall), any rope rewrite
│                       #     (x·cos−y·sin = FMA-contraction shape, the 1-ULP dQ precedent — rope kernels stay
│                       #     untouched, just launched on fewer rows). PERF (real size NL=12 V=151936 P=512):
│                       #     graph step 8.6→5.4 ms; Bp=32 3.5k→5.9k, Bp=64 6.8k→9.4k tok/s (14.7GB). Remaining
│                       #     time = hardware walls (attention 26% at BW peak / MoE expert weight reads / vocab
│                       #     GEMM at the BF16 wall) ⇒ the cuda-tile decode endpoint is REACHED; going further
│                       #     means a hand-written persistent CUDA kernel (leaves the DSL).
│                       #   ⚠ BENCH LESSONS: any co-resident model oversubscribes 16GB → silent WDDM paging
│                       #   (11→120ms/step) — bench in a fresh process FIRST; real-size timings are BIMODAL
│                       #   (~6.4 vs ~9.5ms/step clock-state modes) — trust consecutive consistent repeats only
├── optim/
│   ├── adamw.py        # AdamW (cuda-tile kernel, fp32 master + bf16 view; runtime scalars)  ✓
│   ├── muon.py         # Muon (NS-orthogonalized momentum). host Muon + DEVICE-RESIDENT now:
│   │                   #   muon_ns.py = chained NS (15 GEMMs+axpy/norm, NO round-trip) ✓; DeviceMuon
│   │                   #   (standalone, own bufs) + ResidentMuon (operates on the layer's EXISTING
│   │                   #   p32/p16 + ONE SHARED MuonScratch — per-weight scratch would cost ~2.7GB).
│   │                   #   ⚠ tall (K>N) path: NS B@X scratch MUST be its own (M,Nn) buf, NEVER alias
│   │                   #   the (K,N) momentum (wrong cuda-tile row stride → corruption — fixed+probed).
│   │                   #   State = p32 + ONE momentum buf (no v) → drops AdamW's v (4 B/param).
│   │                   #   ✅ HYBRID wired into ResidentLayerTrain/ResidentModel (optimizer="muon"):
│   │                   #   2D PROJ→Muon, tied embed/LM-head + 1D gains→AdamW (default "adamw"
│   │                   #   byte-identical). REAL Qwen3-0.6B: fwd bitwise-det under muon, SFT CE
│   │                   #   collapses (faster than AdamW@lr2e-3), VRAM −0.35GB@6L = −1.76GB@28L
│   │                   #   (test_muon_hybrid.py).
│   │                   #   ✅ MoE FAMILY DONE (batched-expert NS): muon_ns.newton_schulz_resident_e
│   │                   #   puts the expert dim in the grid (bid(0), e*(M//T) — _transpose_e pattern)
│   │                   #   → ONE NS chain over ALL E square experts (not E× a loop). moe.ExpertMuon
│   │                   #   Scratch (shared ~167MB) + GroupedMoEFFN(optimizer="muon") (experts→Muon,
│   │                   #   ROUTER stays AdamW); ResidentMoEModel(optimizer="muon") threads both
│   │                   #   scratches + muon_exclude=dense-FFN dummies on MoE layers. test_muon_moe.py:
│   │                   #   fwd bitwise-det, SFT CE→0, VRAM −0.35GB@NL4 (scales multi-GB). ⚠ +73ms/step
│   │                   #   expert NS (GEMM ceiling 42TF) → opt-in; accum amortizes. Prefix/GRPO model
│   │                   #   _build_layer not threaded yet (defaults adamw). [[mfu-strategy]]
│   │                   #   ✅ POLAR EXPRESS coeffs (2026-06-15, Muon-variant survey): muon_ns.PE_SCHEDULE
│   │                   #   = per-iteration minimax (a,b,c) (Amsel 2505.16932 / Dao-AILab repo). NS drivers
│   │                   #   take schedule=; muon_ns.set_polar_express(on) = process-wide toggle (default
│   │                   #   Keller ⇒ BYTE-IDENTICAL; all Muon classes call positionally so one toggle flips
│   │                   #   all — no per-ctor threading). PE-5 orthogonalizes BETTER than Keller-5 at the
│   │                   #   SAME 5-iter cost (bf16 err: square-expert 0.40→0.28, rect-q/o 0.32→0.036) = free
│   │                   #   quality upgrade; PE-4≈Keller-5 only on RECT q/o (square needs 5) → −22% NS for
│   │                   #   q/o only. fwd bitwise under PE → ratio=1 untouched. test_polar_express_muon.py +
│   │                   #   _probe_polar_express{,_device}.py; 3 Muon regression tests still pass. ⚠ Gram NS
│   │                   #   SKIPPED (FLOP: α≤2 ⇒ 1.47× slower on our square experts; needs α≥4). [[muon-hybrid]]
│   └── hybrid.py       # Muon(2D matrices) + AdamW(1D gains/embed/head) router  ✓
├── rl/
│   ├── grpo.py         # advantage (group (r-mean)/std) + KL (k3, decoupled β=0) + grpo_loss  ✓
│   ├── rollout.py      # generate (O(S²)) + generate_cached (KV-cache decode, validated 1.2% vs prefill) ✓
│   ├── prefix_grpo.py  # PREFIX-SHARED training step (Prefix Grouper): prompt encoded ONCE, G completions
│   │                   #   attend it via _attn_fwd_prefix (+ offset RoPE at pos Sp.., RNE bf16!). fwd: completion
│   │                   #   hidden BITWISE == naive → ratio=1. bwd: _prefix_layer_bwd (prompt grad = self+Σ_G cross)
│   │                   #   == naive grads ≤0.44% ✓; tokens G·(Sp+Sc)→Sp+G·Sc. NB MoE global layers are NoPE (no
│   │                   #   offset-RoPE there); only local(window) layers RoPE ✓
│   └── prefix_resident.py # MoE-model prefix layers + churn-free attention. PrefixGlobalAttn(window=0|W):
│                       #   persistent-buffer prefix attention, fwd+bwd BITWISE == host helpers, 5x-det ✓.
│                       #   prefix_layer(theta=None→GLOBAL NoPE | theta→LOCAL RoPE+window; ffn=None→dense |
│                       #   GroupedMoEFFN→routed MoE): all 4 layer combos fwd suffix BITWISE == naive
│                       #   MoEDecoderLayer + grads ≤0.84% (tests/training/test_prefix_layer_{global,moe,local}*).
│                       #   ⚠ per-token glue = self-allocating host helpers → allocator-phase churn (test_prefix_
│                       #   layer_local runs each case in a subprocess); the RESIDENT layer below is the fix
└── sft/                # SFT-specific
```

**Validated end-to-end (tests/training/):** single-layer SFT (CE→0), full-model SFT (CE 7.6→0),
**GRPO on-policy loop (mean reward 0.002→0.93 on a toy "emit target token" task)** — the north
star: rollout→reward→group-advantage→policy-grad(linear_ce per-token adv)→full bwd→Muon/AdamW.
**REAL Qwen3-0.6B SFT (head_dim=128) validated** (`tests/training/test_real_qwen3_sft.py`): load real
weights via `load_qwen3` → overfit a fixed batch → CE 11.93→0.97 (NL=2, 4 steps). The whole stack
runs on the actual pretrained weights at the real shapes. See [[real-qwen3-model]].
**Device-resident multi-layer `ResidentModel` DONE** (`model/resident_model.py`): 28 layers chained device-side,
fp32 residual, SR grad, and a **TIED device-resident embed/LM-head with device AdamW** (Qwen3 tie_word_embeddings).
The boundary (logits `_gemm_nt_f32`, CE `_ce_stats`/`_ce_grad`, dhidden `_gemm`, embed grad = LM-head `_gemm_dW` +
input-embed `_gemm_dW` summed by `_acc_f32`, input gather `_gemm(onehot,embed)`) is all on-device; only the tiny
final RMSNorm stays host. Real Qwen3-0.6B SFT: determinism bitwise, CE collapses, **0.11 s/step @ NL=28** (was
4.54 s with the host boundary, ~40×; host-orchestrated Qwen3Model was ~7.7 s/layer-step). ~11.6 GB VRAM at 28L.
Next perf: MXFP8 training fwd (unify rollout↔training precision), megakernel.

**Device-resident DECODE `ResidentDecodeLayer` + `ResidentDecodeModel` DONE + ratio=1 verified** (`model/
resident_decode.py`, 2026-06-04): the rollout counterpart to ResidentLayerTrain / ResidentModel — single decode
token, per-layer KV-cache, kernels chained on persistent buffers (no alloc churn). `ResidentDecodeLayer`: single
layer; `tests/model/test_resident_decode.py` runs an autoregressive rollout vs a training prefill and proves
**hidden[t] bitwise == prefill[t]** every position (NL=1/2/4, S=128/256). `ResidentDecodeModel`: the layer stack +
the same tied embed/LM-head boundary, with `score` (teacher-forced) and `generate` (greedy autoregressive);
`tests/model/test_resident_decode_model.py` proves **rollout logprob == training (`ResidentModel`) logprob BITWISE
(ratio π_train/π_infer = 1 exactly)** — including the logprob `generate` reports during rollout == what training
assigns. Design: MGEMM=128 padded tile, Bp-sized cache, S=1 tok↔head identity, `_DBuf.at_pos`, persistent-buffer
device final-norm in the boundary (host rmsnorm_forward churns → flaky). See [[batch-invariance]].

Optimizer routing (Muon+AdamW hybrid, the modern recipe): Muon → q/k/v/o/gate/up/down_proj
(2D matmul weights); AdamW → RMSNorm gains + embedding + LM-head. Muon on embed/head HURTS.
Muon v1 does Newton-Schulz in fp32 on host (small, once-per-step); its 3 matmuls/iter move to
loss._gemm on-device as the perf follow-up. AdamW per-step bias-correction passed as RUNTIME
float (not Constant → no per-step recompile). One SFT step (fwd→linear_ce→bwd→update) verified
to collapse the loss in tests/training/test_sft_step.py.

`grpo_rl/` is the old package name — use `ancora/` going forward.
`ancora/env.py` must be the first import in every script.

---

## Roadmap / design intent (not yet built — was DESIGN.md)

The forward-looking design that isn't derivable from current code. (The old DESIGN.md also targeted a
generic "1B–3B" model — SUPERSEDED: we hardcode real Qwen3-0.6B, [[real-qwen3-model]].)

### Activation checkpointing for long-sequence training — ✅ v1 DONE (2026-06-14, 8K)
`ResidentMoEModel(long_context=True)` (default False — short training keeps the fast full-store
resident path, ZERO change). The backward RECOMPUTES each layer's forward from a stored input
(`gx_in`) instead of keeping all NL layers' intermediates resident; the per-layer scratch is ALIASED
to layer 0's (`_setup_checkpoint`, `_SHARE`/`_MOE_SHARE` lists) so only ONE layer's intermediates +
one MoE-FFN scratch live at a time. **Recompute is deterministic ⇒ grads BITWISE-identical to
full-store** (test_checkpoint.py: fwd Δ=0, grads Δ=0, SFT identical — the correctness gate at every
step). Result: single-sequence training **~3K (M=4096 pages) → 8K (M=8192 fits 15.0GB), 4×**. The
local/global split (Gemma-3/MAI 5:1) makes recompute cheap — 10/12 layers are windowed O(S·window),
only 2 global are O(S²). **The 16K wall was the CONSTRUCT PREALLOC peak — FIXED 2026-06-15 (free-as-you-
go), now SFT ~24K / prefix-GRPO 16K.** The wall was NOT the optimizer floor (Muon's lazy/post-construct
saving didn't move it) NOR the step activation peak — it was `_setup_checkpoint` allocating ALL NL
layers' _SHARE scratch and THEN freeing the duplicates (peak = NL layers). Fix: `_alias_scratch` frees
each layer's scratch to layer 0's AS IT IS BUILT (peak = 2 layers) — end-state aliases identical ⇒
STILL BITWISE (test_checkpoint + test_prefix_long_context Δ=0). Result (_bench_muon_longctx.py): SFT
M=16384→OOM@16640 BEFORE → 18K/20K/24K all OK AFTER (construct 13.4/14.0/15.2GB; 28K pages); prefix-GRPO
8K→16K (test_prefix_long_context.py bitwise, _bench_prefix_longctx.py). Muon's proj-v (a CONSTRUCT-time
saving) stacks marginally at the edge (28K pages less under muon). The OLD note below is superseded:
~~M=16384 fits (construct 12.8GB, step-peak 15.92GB, ~1.3s/step), M=16640 OOMs@CONSTRUCT.~~ The construct prealloc
(M-sized attn head-major f32 gO/gdqr/gdkr/gdvh ~4GB + gx_in×NL + fixed (V,H)/(MC,V) vocab boundary) +
the backward-recompute step-peak are optimizer-INDEPENDENT, so the Muon hybrid's 1.3GB optimizer saving
(real, but LAZY/post-construction + masked under the activation peak) does NOT extend this model's length
— max=16K for BOTH adamw and muon. (The ~10GB FLOOR claim was the DENSE 28-layer 551M model; the MoE
12-layer floor is only ~4GB, so activations dominate here.) To reach 16K-32K (task #16): (a) Tiled MLP
(Snowflake-Arctic, shrink the one-layer scratch — the BIG f32 attn buffers), (b) selective global
recompute (store QKVOL, skip the extra O(S²) fwd) — these cut the activation wall, and Muon's freed VRAM
THEN buys length on top; (c) CPU-offload AdamW states (frees the floor — only helps the floor-bound case),
(d) thread long_context through ResidentPrefixMoEModel (the GRPO path). Industrial parallel: this is
literally Megatron/veRL/Unsloth's "aggressive activation checkpointing" (single-GPU realization à la
Unsloth); industrial reaches 48K via MULTI-GPU context parallelism (Ulysses/Ring) we can't do single-
GPU. MAI's 256K recipe = STAGED extension (train short MFU-friendly, brief cheap 256K phase — most
training is short so the 8K limit barely bites). See [[mfu-strategy]], _bench_checkpoint_mem.py.

### Megakernel — cross-layer persistent kernel (the north-star perf goal)
The enabling hardware fact: **L2 = 48 MB** (H100-class) on this RTX 5080, with **60 SMs** and **100 KB
SMEM/SM**, **CooperativeLaunch = YES**. So several layers' activations fit resident in L2 (e.g. hidden=2048
seq=512 → ~2 MB/layer → ~16 layers in L2; deeper models use an 8-layer window). Plan: ONE persistent
`ct.launch` runs fwd (and bwd) for a block of layers, passing activations through L2/SMEM instead of HBM
round-trips. Inter-layer pipeline signal = **`griddepcontrol.launch_dependents`** (lighter than a full grid
sync); cross-block sync = CooperativeLaunch grid barrier. Start granularity (D3): **4-layer SMEM fusion**
(100 KB/SM holds 4×~16 KB tiles) — ~75% fewer launches/HBM trips, 10× simpler than all-layer fusion.
The CUDA-graph capture of `ct.launch` (toolchain note above) is the current substrate; HazyResearch
"No Bubbles" is the reference. Why it matters: standalone attention is already at the cuda-tile ceiling
(~60/75 TF) — further MFU needs cross-operator bubble fill (attention's idle Tensor Cores filled by the
neighbouring GEMM). See [[mfu-strategy]], [[gemm-mfu-ceiling]], [[coda-epilogue-fusion]].

### DualKV / prefix-sharing — ✅ BUILT (2026-06-10, full stack)
GRPO generates **G completions per prompt**; the naive path replicates the prompt KV G× (G× memory +
compute). Prefix sharing stores the prompt KV **once**, all G completions cross-attend it; backward sums
the prefix grad across the group (`dK_prefix = self + Σ_i cross_i`, deterministic fixed-order, no atomics).
Saves ~87.5% prefix-KV at G=8; tokens G·(Sp+Sc)→Sp+G·Sc. Refs: DualKV (arXiv 2605.15422), Prefix Grouper
(arXiv 2506.05433). DONE at every level: kernels (global + WINDOWED, fwd bitwise), host helpers (window=),
PrefixGlobalAttn (churn-free resident attention), prefix_layer global/local × dense/MoE (rl/prefix_resident
.py), Qwen3 host step (rl/prefix_grpo.py), and the DEVICE-RESIDENT training layers (model/resident_prefix
.py — fwd bitwise vs replicated resident, repeat-det Δ=0), the FULL GRPO STEP `model/
resident_prefix_model.py` ResidentPrefixMoEModel (scored logprobs BITWISE == replicated incl. the shared
boundary row → full-model ratio=1; policy improvement verified; 2.2-2.5× step at Sp=512/Sc=256/G=8;
fwd+bwd CUDA-graph-captured bitwise), AND the ROLLOUT side `model/resident_moe_decode.py`
ResidentMoEDecodeModel (weights aliased zero-copy, device closed-loop generate, rollout lp BITWISE ==
prefix-GRPO-training lp). **The GRPO loop is fully closed on the MoE family: rollout == training, ratio=1.**

### Other standing decisions
- **Reward**: rule-based first (math/code/format checks); `reward_fn` is an async coroutine so an external
  RM can slot in; rewards scored in batch after all G completions finish.
- **Stack**: Python-first, NVIDIA libs only (`cuda.tile`/`cuda.core`/`cuda.bindings`), no CuPy (DLL-incompatible
  with CUDA 13.3), no C++/CMake unless profiling forces it. inline PTX only where cuda-tile can't express it.
