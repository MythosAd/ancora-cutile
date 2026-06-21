"""ResidentLayerTrain — complete device-resident training step (fwd→bwd→AdamW). Validates:
  (1) forward vs host TransformerLayer.forward,
  (2) backward weight grads vs host layer.backward,
  (3) a multi-step training run on an MSE target collapses the loss (fwd→bwd→update works)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.model.resident import _DBuf, _f32bf
from ancora.model.resident_train import ResidentLayerTrain, _bits2f

bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])
cfg = Qwen3Config(); B, S = 1, 128; H = cfg.hidden; M = B * S


def test_fwd_bwd():
    layer = TransformerLayer(cfg, seed=0); rng = np.random.default_rng(1)
    x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
    dout = _bf((rng.standard_normal((B, S, H)) * 0.1).astype(np.float32))
    out, cache = layer.forward(x, si, return_cache=True)
    d_x_ref, gref = layer.backward(dout, cache, si)
    tl = ResidentLayerTrain(cfg, layer.w, B, S, sr_grad=False)     # RTN for a clean grad-vs-host check
    gx = _DBuf(np.ascontiguousarray(x.reshape(M, H), np.float32))   # fp32 residual stream
    gdout = _DBuf(_f32bf(dout.reshape(M, H)))                       # gradient residual stays bf16
    tlout = tl.forward(gx, si); tl.backward(gdout, si); cudart.cudaStreamSynchronize(si)
    e_out = rel(tlout.to_numpy().reshape(B, S, H), out)   # gout is fp32 (residual stream)
    print(f"  (1) forward  vs host: {e_out*100:.2f}%  {'OK' if e_out < 0.02 else 'FAIL'}")
    ok = e_out < 0.02
    for n in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "input_ln", "post_ln", "q_norm", "k_norm"]:
        g = tl.G[n].to_numpy(); r = gref[n]
        if g.shape != r.shape: g = g.reshape(r.shape)
        e = rel(g, r); o = e < 0.05; ok &= o
        print(f"  (2) dW {n:11s} {e*100:5.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_train_step():
    """fwd→bwd→AdamW on an MSE target → loss must collapse."""
    layer = TransformerLayer(cfg, seed=2); rng = np.random.default_rng(3)
    x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
    y = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))   # target
    tl = ResidentLayerTrain(cfg, layer.w, B, S)
    gx = _DBuf(np.ascontiguousarray(x.reshape(M, H), np.float32)); gd = _DBuf.zeros((M, H), np.uint16)
    yt = bf32(_f32bf(y.reshape(M, H)))
    losses = []
    for it in range(30):
        out = tl.forward(gx, si); cudart.cudaStreamSynchronize(si)
        o = out.to_numpy()                                            # gout is fp32 (residual stream)
        loss = float(np.mean((o - yt) ** 2)); losses.append(loss)
        dL = _f32bf((2.0 / M) * (o - yt))                              # dMSE/dout
        cdrv.cuMemcpyHtoD(gd.ptr, np.ascontiguousarray(dL), gd.nbytes)
        tl.backward(gd, si); tl.step(si, lr=2e-3)
    print(f"  (3) MSE loss: {losses[0]:.4f} → {losses[-1]:.4f}  ({losses[-1]/losses[0]*100:.0f}% of init)  {'OK — collapses' if losses[-1] < 0.5 * losses[0] else 'FAIL'}")
    return losses[-1] < 0.5 * losses[0]


if __name__ == "__main__":
    print(f"ResidentLayerTrain — fwd→bwd→update  B={B} S={S} H={H}"); print("=" * 60)
    ok = test_fwd_bwd()
    print("-" * 60)
    ok &= test_train_step()
    print("=" * 60); print(f"  {'PASS — complete device-resident training step works' if ok else 'FAIL'}")
