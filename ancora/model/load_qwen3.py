"""
ancora/model/load_qwen3.py — load real Qwen3-0.6B (HF safetensors) into framework convention.

Reads model.safetensors manually (no safetensors/ml_dtypes pkg): 8-byte header len + JSON header
(name→dtype/shape/offsets) + raw BF16 data. BF16 bits (uint16) → f32 via <<16.

Convention fix: HF Linear weight is (out, in); the framework computes y = x @ W with W (in, out),
so every projection weight is TRANSPOSED. Norms (1-D) and embed (vocab,H) are not. lm_head is tied
(= embed) when tie_word_embeddings. REAL head_dim=128 (framework's attention kernel is D=64 — the
GPU path needs a D=128 attention to use these; the numpy rollout-quality test takes head_dim as a param).
"""
import json, struct
import numpy as np


def load_qwen3(path=r"C:\model\Qwen3-0.6B", n_layers=28, want_lm_head=True):
    fn = path + r"\model.safetensors"
    f = open(fn, "rb")
    nbytes = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(nbytes))
    base = 8 + nbytes

    def get(name):
        m = hdr[name]; assert m["dtype"] == "BF16", m["dtype"]
        s, e = m["data_offsets"]; f.seek(base + s)
        raw = np.frombuffer(f.read(e - s), np.uint16).reshape(m["shape"])
        return (raw.astype(np.uint32) << 16).view(np.float32)   # BF16 bits → f32

    embed = get("model.embed_tokens.weight")                    # (V, H)
    out = {"embed": embed, "final_norm": get("model.norm.weight"),
           "lm_head": (get("lm_head.weight").T.copy() if "lm_head.weight" in hdr else embed.T.copy()) if want_lm_head else None,
           "layers": []}
    P = "model.layers.{}."
    for i in range(n_layers):
        p = P.format(i)
        lw = {
            "input_ln":  get(p + "input_layernorm.weight"),
            "post_ln":   get(p + "post_attention_layernorm.weight"),
            "q_norm":    get(p + "self_attn.q_norm.weight"),
            "k_norm":    get(p + "self_attn.k_norm.weight"),
            "q_proj":    get(p + "self_attn.q_proj.weight").T.copy(),   # (out,in)→(in,out)
            "k_proj":    get(p + "self_attn.k_proj.weight").T.copy(),
            "v_proj":    get(p + "self_attn.v_proj.weight").T.copy(),
            "o_proj":    get(p + "self_attn.o_proj.weight").T.copy(),
            "gate_proj": get(p + "mlp.gate_proj.weight").T.copy(),
            "up_proj":   get(p + "mlp.up_proj.weight").T.copy(),
            "down_proj": get(p + "mlp.down_proj.weight").T.copy(),
        }
        out["layers"].append(lw)
    f.close()
    return out


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    w = load_qwen3(n_layers=2)
    print("loaded Qwen3-0.6B (2 layers for check)")
    print(f"  embed {w['embed'].shape}  final_norm {w['final_norm'].shape}  lm_head {w['lm_head'].shape}")
    l = w["layers"][0]
    for k in ["input_ln", "q_proj", "k_proj", "o_proj", "q_norm", "gate_proj", "down_proj"]:
        print(f"  layer0.{k}: {l[k].shape}")
    print(f"  q_proj sample (should be ~O(0.01-0.1)): {l['q_proj'].flatten()[:4]}")
