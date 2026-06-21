# Hardware Capability Tests

Probes for RTX 5080 Laptop (sm_120 / sm_120a). Run after driver or toolkit changes.

## Files

| File | Purpose |
|------|---------|
| `test_env.py` | Quick env sanity check (versions, bandwidth, FP4 smoke test) |
| `test_mma_capabilities.py` | Full MMA capability matrix (BF16/FP8/FP6/FP4, griddepcontrol, TMEM) |

## Results Summary (2026-05-29, CUDA 13.3, driver 595.79)

```
sm_120  ✅: BF16, FP8 E4M3/E5M2 plain mma.sync, INT4 m16n8k64
sm_120a ✅: + FP4 E2M1, FP6 E3M2, FP4×FP8 mixed, griddepcontrol
            All via mma.sync.aligned.m16n8k32 + .kind::f8f6f4 modifier
            Same register layout for all: A=4 regs, B=2 regs, C/D=4×f32
TMEM    ❌: tcgen05 not supported on sm_120a (sm_100/B200 only)
wgmma   ❌: Hopper-style wgmma.mma_async not on sm_120a
```

## Running

```bash
python tests/hardware/test_env.py
python tests/hardware/test_mma_capabilities.py
```
