"""Run tda.analysis.switch_weights on the results volume.

Adapters are analysed where they live: a 2 GB `modal volume get` to a laptop
came back with zero-filled holes twice (different tensors each time, sha256
differing from the volume copy), which reads as a module that "changed 100 %".

    modal run tda/modal/switch_app.py --after aft_ladder/<run>,aft_ladder/<run2>
"""

import json

import modal

from tda.modal.app import RESULTS_DIR, VOLUMES, hf_secret, train_image

app = modal.App("msm-tda-switch")
MSM = "chloeli/qwen-2.5-32b-philosophy-spec-msm"


@app.function(image=train_image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600,
              cpu=8, memory=32768)
def weights(after: list[str]) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from tda.analysis.switch_weights import main

    msm = hf_hub_download(MSM, "adapter_model.safetensors")
    out = {}
    for a in after:
        p = f"{RESULTS_DIR}/{a}/adapter_model.safetensors"
        holes = [k for k, v in load_file(p).items() if (v == 0).float().mean() > 0.01]
        assert not holes, f"{a}: zero-filled tensors {holes[:3]}"
        dst = f"/tmp/{a.replace('/', '_')}.json"
        print("===", a, flush=True)
        main(msm, p, dst)
        out[a] = json.load(open(dst))
    return out


@app.local_entrypoint()
def run(after: str, out: str = "results/aft/switch/weights.json"):
    res = weights.remote(after.split(","))
    json.dump(res, open(out, "w"), indent=1)
    for a, r in res.items():
        print(a.split("from-relmsm-")[-1], {k: round(v, 4) for k, v in r.items()
                                             if isinstance(v, float)})
        print("  by_proj share:", {k: round(v["share_of_dW"], 3) for k, v in r["by_proj"].items()})
        bl = r["by_layer"]
        print("  by layer block share:", {f"{i}-{i+7}": round(sum(bl[str(j)]["share_of_dW"]
              for j in range(i, i + 8)), 3) for i in range(0, 64, 8)})
        print("  rel by layer block:", {f"{i}-{i+7}": round(sum(bl[str(j)]["rel_mean"]
              for j in range(i, i + 8)) / 8, 3) for i in range(0, 64, 8)})
        print("  top modules:", [(m["module"].split("model.")[-1], round(m["rel"], 3),
                                  round(m["top1"], 2)) for m in r["top_modules"][:5]])
