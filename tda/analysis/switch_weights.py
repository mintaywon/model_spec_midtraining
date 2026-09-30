"""What did 7 optimizer steps change? Weight-space view of MSM -> MSM + 100 rows.

Both adapters are LoRA r=64 on the same modules, and the second continues the
first, so the update to each module's effective weight is

    dW = B1 A1 - B0 A0 = [B1, -B0] [A1; A0]          (rank <= 128)

Its singular values come from the two thin QR factors, without forming the
d_out x d_in matrix. Reported per module: ||dW|| / ||B0 A0|| (how far AFT moved
relative to what midtraining built), the share of ||dW||^2 in the top singular
direction, and the participation-ratio rank.

    python -m tda.analysis.switch_weights MSM.safetensors AFTER.safetensors out.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file


def _norm(k: str) -> str:
    return k.replace(".default", "").replace("base_model.model.", "")


def _svals(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Singular values of U @ V for thin U (d_out x r), V (r x d_in)."""
    _, ru = torch.linalg.qr(U)
    _, rv = torch.linalg.qr(V.T)
    return torch.linalg.svdvals(ru @ rv.T)


def main(msm_path: str, after_path: str, out: str) -> None:
    m = {_norm(k): v.double() for k, v in load_file(msm_path).items()}
    a = {_norm(k): v.double() for k, v in load_file(after_path).items()}
    assert set(m) == set(a), "adapters do not cover the same tensors"
    mods = sorted({k.rsplit(".lora_", 1)[0] for k in m})
    rows = []
    for mod in mods:
        A0, B0 = m[f"{mod}.lora_A.weight"], m[f"{mod}.lora_B.weight"]
        A1, B1 = a[f"{mod}.lora_A.weight"], a[f"{mod}.lora_B.weight"]
        s0 = _svals(B0, A0)
        sd = _svals(torch.cat([B1, -B0], 1), torch.cat([A1, A0], 0))
        e = sd ** 2
        layer = int(re.search(r"layers\.(\d+)\.", mod).group(1))
        rows.append({"module": mod, "layer": layer, "proj": mod.rsplit(".", 1)[-1],
                     "rel": (e.sum().sqrt() / (s0 ** 2).sum().sqrt()).item(),
                     "dnorm": e.sum().sqrt().item(),
                     "top1": (e[0] / e.sum()).item(),
                     "pr_rank": (e.sum() ** 2 / (e ** 2).sum()).item(),
                     "dA_rel": ((A1 - A0).norm() / A0.norm()).item(),
                     "dB_rel": ((B1 - B0).norm() / B0.norm()).item()})
    tot = sum(r["dnorm"] ** 2 for r in rows)

    def agg(key):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        return {str(k): {"rel_mean": sum(r["rel"] for r in v) / len(v),
                         "share_of_dW": sum(r["dnorm"] ** 2 for r in v) / tot,
                         "top1_mean": sum(r["top1"] for r in v) / len(v),
                         "pr_rank_mean": sum(r["pr_rank"] for r in v) / len(v)}
                for k, v in sorted(g.items(), key=lambda kv: kv[0])}

    res = {"n_modules": len(rows),
           "rel_mean": sum(r["rel"] for r in rows) / len(rows),
           "rel_median": sorted(r["rel"] for r in rows)[len(rows) // 2],
           "dA_rel_mean": sum(r["dA_rel"] for r in rows) / len(rows),
           "dB_rel_mean": sum(r["dB_rel"] for r in rows) / len(rows),
           "top1_mean": sum(r["top1"] for r in rows) / len(rows),
           "pr_rank_mean": sum(r["pr_rank"] for r in rows) / len(rows),
           "by_proj": agg("proj"), "by_layer": agg("layer"),
           "top_modules": sorted(rows, key=lambda r: -r["dnorm"])[:10]}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_layer", "top_modules")},
                     indent=1))
    bl = res["by_layer"]
    print("share of ||dW||^2 by layer block:",
          {f"{i}-{i+7}": round(sum(bl[str(j)]["share_of_dW"] for j in range(i, i + 8)
                                   if str(j) in bl), 3) for i in range(0, len(bl), 8)})
    print("top modules:", [(r["module"].split("model.")[-1], round(r["rel"], 3))
                           for r in res["top_modules"][:6]])


if __name__ == "__main__":
    main(*sys.argv[1:4])
