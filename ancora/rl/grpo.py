"""
ancora/rl/grpo.py — GRPO advantage + loss (the algorithm layer).

Design (answers the "does this affect kernel fusion?" question): NO.
  - The heavy fusion (stream vocab → log-prob, never materialize 151k logits, batch-
    invariant) lives in kernels/loss.py `_fused_logprob`.
  - Advantage and KL are CHEAP per-token ops on the small logprob (M,) array, applied
    here, OUTSIDE the kernel. So the advantage FORMULA and the KL term are fully
    decoupled from the kernel — change them freely, the kernel is untouched.

GRPO loss:   L = -mean_t( A[t] * logπ(a_t|s_t) )  [ + β * KL(π‖π_ref) ]
  - A[t]   : per-token advantage (broadcast from the completion's group-normalised reward)
  - KL     : OPTIONAL (β=0 → no KL, no reference model needed at all)
"""
import numpy as np


# ── advantage (host-side; formula does NOT touch any kernel) ─────────────────

def grpo_advantage(rewards: np.ndarray, group_size: int,
                   norm: str = "std", eps: float = 1e-4) -> np.ndarray:
    """
    Group-normalised advantage. rewards: (num_groups * group_size,) — G completions
    per prompt, laid out contiguously. Returns per-completion advantage, same shape.

    norm:
      "std"  → (r - mean) / (std + eps)     standard GRPO / DeepSeek
      "mean" → (r - mean) / (|mean| + eps)  relative-reward variant
      "none" → (r - mean)                   no scaling
    Swap freely — none of this is in the kernel.
    """
    r = rewards.reshape(-1, group_size).astype(np.float64)
    mean = r.mean(axis=1, keepdims=True)
    centered = r - mean
    if norm == "std":
        denom = r.std(axis=1, keepdims=True) + eps
    elif norm == "mean":
        denom = np.abs(mean) + eps
    elif norm == "none":
        denom = 1.0
    else:
        raise ValueError(f"unknown norm {norm!r}")
    return (centered / denom).reshape(-1).astype(np.float32)


def broadcast_to_tokens(adv_per_completion: np.ndarray,
                        token_completion_id: np.ndarray) -> np.ndarray:
    """
    Map per-completion advantage → per-token advantage.
    token_completion_id: (M,) int — which completion each token belongs to.
    (All tokens of completion g share advantage A[g].)
    """
    return adv_per_completion[token_completion_id].astype(np.float32)


# ── KL estimator (optional, decoupled) ───────────────────────────────────────

def kl_k3(logprob: np.ndarray, ref_logprob: np.ndarray) -> np.ndarray:
    """
    Schulman k3 unbiased low-variance KL(π‖π_ref) per token, as used by GRPO/DeepSeek:
        kl ≈ exp(logπ_ref - logπ) - (logπ_ref - logπ) - 1   ≥ 0
    """
    logr = ref_logprob - logprob
    return np.exp(logr) - logr - 1.0


# ── GRPO loss (advantage weighting + optional KL) ────────────────────────────

def grpo_loss(logprob: np.ndarray, advantage: np.ndarray,
              ref_logprob: np.ndarray | None = None, beta: float = 0.0):
    """
    logprob:     (M,) current policy log π(a_t|s_t)  — from kernels.loss.fused_logprob
    advantage:   (M,) per-token advantage (already broadcast from per-completion)
    ref_logprob: (M,) reference policy log-prob — ONLY needed if beta > 0
    beta:        KL coefficient. beta=0 → pure policy gradient, KL fully skipped.

    Returns (loss, pg_loss, kl) so the KL term is observable but decoupled.
    """
    pg = -float(np.mean(advantage * logprob))   # policy-gradient term

    if beta > 0.0:
        assert ref_logprob is not None, "KL needs ref_logprob (or set beta=0)"
        kl = float(np.mean(kl_k3(logprob, ref_logprob)))
        return pg + beta * kl, pg, kl

    return pg, pg, 0.0   # KL decoupled: no ref model touched
