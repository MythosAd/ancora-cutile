# ANCORA

**A single-GPU, on-policy RL (GRPO) + SFT training framework written in pure Python on NVIDIA `cuda.tile` / `cuda.core` — no PyTorch, no CUDA C++.**

ANCORA targets a single consumer GPU (RTX 5080 Laptop, compute capability `sm_120a`) and builds the entire training and rollout stack — attention, GEMM, RMSNorm, RoPE, SwiGLU, MoE, cross-entropy, AdamW/Muon — out of hand-written [cuda-tile](https://docs.nvidia.com/cuda/cuda-tile/) kernels, with everything orchestrated device-resident to keep host overhead out of the loop.

> ⚠️ **Research code.** This is a from-scratch exploration of what one person + one consumer GPU can do for modern RL post-training. It is tuned for one specific GPU (`sm_120a`) and developed on Windows. It is shared for people interested in low-level kernel work, batch-invariant numerics, and single-GPU RL — not as a turnkey library. See [Scope & limitations](#scope--limitations).

---

## The core idea: `π_train / π_infer = 1`, exactly

On-policy RL (GRPO/PPO) is only correct if the **rollout** (inference) and **training** passes compute the *same* log-probabilities for the same tokens. When they don't, the importance ratio `π_train / π_infer ≠ 1` and the policy gradient is biased. Production stacks paper over this with truncated/masked importance sampling (TIS/MIS) because their inference engine (e.g. vLLM) and trainer (e.g. Megatron) are two separate codebases that *cannot* be made bitwise-identical.

ANCORA is a **single codebase**, so it can do something they can't: make rollout and training **bitwise-identical**, and get `ratio = 1` *exactly* — no importance sampling needed.

This is achieved by enforcing **batch invariance** in every kernel — fixed tile sizes, no split-K, no split-KV, reductions kept on one core — so a token's logprob does not depend on batch size or sequence length. The decisive trick: **decode reuses the prefill attention kernel**, so a generated token's attention is byte-for-byte what training later computes for that position. This is verified down to `max|Δ| = 0` across the full model (SFT and prefix-shared GRPO, greedy and sampled rollout). See [`DEVNOTES.md`](DEVNOTES.md) for how this was won.

---

## Highlights

- **Bitwise rollout == training** (`ratio = 1`), verified end-to-end on real Qwen3-0.6B weights — the GRPO loop is fully closed with no importance sampling.
- **Device-resident training & decode.** Multi-layer forward/backward and the autoregressive decode loop run on persistent device buffers (no per-step alloc churn), chained via CUDA-graph capture.
- **Two model families:** dense **Qwen3-0.6B**, and a **MAI-style MoE** family (interleaved dense/MoE + local/global attention, E=16 top-2) with grouped/segmented expert GEMM.
- **Modern optimizer recipe:** **Muon + AdamW hybrid** (Muon on 2-D projection weights; AdamW on gains/embed/head), with a device-resident Newton–Schulz and Polar-Express coefficients.
- **Precision:** BF16 forward + FP32 accumulate by default; opt-in **MXFP8** forward (block-scaled `mma_scaled`) and **FP8 (E4M3) dgrad**; FP32 residual stream + stochastic-rounded gradient downcast.
- **Prefix-shared GRPO** (Prefix Grouper / DualKV): the prompt KV is encoded **once** and all G completions cross-attend it; backward sums the prefix gradient across the group in fixed order.
- **Long-context training** via activation checkpointing (recompute is deterministic → bitwise-identical grads): single-sequence SFT to ~24K tokens, prefix-GRPO to 16K, on 16 GB.
- **MFU work:** the GEMM/attention kernels are pushed to the cuda-tile ceiling, then a training "megakernel" pass (device onehot, chunked vocab boundary, ncu-guided retiling, gradient accumulation) plus a deterministic **sorted-scatter embedding gradient** (replacing a 160×-wasteful one-hot GEMM) lift training MFU from ~5% to **~45%** at the best batch. Every rejected optimization (CUTLASS hybrid, inline-RoPE fusion, persistent heterogeneous megakernel, FP8 backward) is measured and documented, not assumed.
- **GRPO advantage** defaults to the maximum-likelihood form `(r − mean)/(|mean| + eps)` (the DeepSeek `std` form stays selectable); the advantage is host-side and fully decoupled from the kernels.

All performance numbers are measured on the one target GPU below; treat them as a journey log, not a spec.

---

## Requirements

- **GPU:** NVIDIA RTX 5080 Laptop or another `sm_120` / `sm_120a` device (Blackwell consumer). The kernels are compiled `--gpu-architecture=sm_120a`; other architectures are not supported and several kernels rely on `sm_120`-specific facts (e.g. `mma_scaled` block scaling, no WGMMA/tcgen05).
- **CUDA Toolkit 13.3.** `ancora/env.py` hard-codes the Windows toolkit path and must be imported **before** any `cuda.*` module.
- **Python** with `numpy`, `cuda.tile` (developed on 1.4.0, re-validated bitwise on **1.5.0** — the full `ratio = 1` regression passes unchanged), `cuda.core` (1.0.1+), `cuda.bindings` (13.3.1), and `ml_dtypes` (for `bfloat16` numpy dtype).
- **OS:** developed and tested on **Windows 11**. Some helpers (offline `nvcc` compile, DLL bootstrap, NVRTC workarounds) are Windows-specific.
- **Model weights** (real Qwen3-0.6B) are loaded from a local path and are **not** included in this repo.

There is no `pip install` packaging yet — clone the repo and run scripts from the project root.

---

## Quick start

```python
import ancora.env          # MUST be first — sets CUDA_PATH / DLL dirs for CUDA 13.3
import numpy as np
# ... then import the kernels / model you need, e.g.:
# from ancora.model.resident_moe_model import ResidentMoEModel
```

The most useful entry points are the end-to-end tests under `tests/training/` and `tests/model/`, which exercise full SFT and GRPO steps and document the expected numerics (CE collapse, `ratio = 1`, bitwise determinism).

---

## Status / results

- **SFT:** single-layer and full-model cross-entropy collapse to ~0; real Qwen3-0.6B overfit verified.
- **GRPO:** on-policy loop closed — mean reward `0.005 → 0.96` in 16 iterations on a toy task (ML advantage); rollout logprob bitwise-equal to what training assigns.
- **Determinism:** forward is bitwise-deterministic across runs; batch-size and sequence-length invariance verified (`max|Δ| = 0`).
- **Throughput (target GPU, MoE family @ M=2048):** training step 139.8 ms = 14.6k tok/s (**MFU 35.4%**); with 8× gradient accumulation **MFU 45.1% / 18.7k tok/s** (asymptote ≈46% — the honest cuda-tile ceiling for this card; the remaining walls are the consumer BF16 throttle, the optimizer's bandwidth sweep, and a measured-but-unreachable-in-DSL 31% operator-overlap window that would need hand-written CUDA). Decode megakernel ~9.4k tok/s at Bp=64.

---

## Repository layout

```
ancora/
├── env.py            # CUDA 13.3 bootstrap — import first
├── kernels/          # cuda-tile kernels: linear (MXFP8), attention (FA + prefix + windowed),
│                     #   moe (grouped), norm (RMSNorm), rope, activation (SwiGLU), quant, fused, loss
├── model/            # layers + full models: dense Qwen3 and the MoE family, plus the
│                     #   device-resident training / decode / prefix-shared GRPO stacks
├── optim/            # AdamW, Muon (device-resident Newton–Schulz), Muon+AdamW hybrid router
├── rl/               # GRPO advantage/KL/loss, rollout (KV-cache decode), prefix-shared GRPO step
└── sft/              # SFT-specific glue
tests/                # kernels / model / training / hardware / benchmarks — the executable spec
```

A deeper map (every kernel, every layer, every hard-won lesson) lives in [`CLAUDE.md`](CLAUDE.md). The engineering journal — the bugs, the dead ends, and the numerics traps — is in [`DEVNOTES.md`](DEVNOTES.md).

---

## Scope & limitations

- **One GPU architecture.** Built and verified only on `sm_120a`. It will not run on Hopper (`sm_90`) — it deliberately avoids WGMMA / `tcgen05` / FA3, which `ptxas` rejects on `sm_120`.
- **Single GPU.** No multi-GPU / tensor / context parallelism. Long-context limits (24K SFT / 16K GRPO) reflect a single 16 GB card; industrial 48K–256K needs multi-GPU parallelism this project doesn't attempt.
- **Windows-first.** Linux is untested; expect to adapt `env.py` and the `nvcc`/NVRTC helpers.
- **Research, not product.** APIs are unstable, there's no packaging, and the kernels are tuned for one specific model/shape (Qwen3-0.6B, `head_dim=128`).

---

## License

[Apache License 2.0](LICENSE).

## Citation

If you use ANCORA in your research or build on it, please cite it:

```bibtex
@software{yang2026ancora,
  author  = {Yang, Chengcao},
  title   = {ANCORA: A single-GPU, batch-invariant on-policy RL (GRPO) + SFT
             training framework in pure cuda.tile for RTX 5080 (sm_120a)},
  year    = {2026},
  url      = {https://github.com/MythosAd/ancora-cutile},
  note     = {Rollout and training are bitwise-identical (ratio = 1), so the
              GRPO loop needs no importance sampling.}
}
```

## Acknowledgements

This work stands on a lot of public research and reference code, including:
gau-nernst's `sm_120` FlashAttention, NVIDIA's CUDA-Tile attention paper, HazyResearch's "Megakernels / No Bubbles", Thinking Machines' batch-invariance work, the Muon / Polar-Express line (Keller Jordan; Amsel et al.), CUTLASS, and the MAI-Thinking / DeepSeek-V3 / Qwen3 precision and architecture notes. See `DEVNOTES.md` for specific references.
