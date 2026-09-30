"""Activation-difference steering: can the 7-step AFT effect be added to the MSM
checkpoint as a vector, without changing any weight?

    modal run --detach tda/modal/steer_app.py --after aft_ladder/<run> --run-name steer1

1. Forward the 27 AM prompts through Qwen2.5-32B under the MSM adapter and
   under the `after` adapter (MSM + AFT). At the last prompt token, per layer,
   d_l = mean over prompts of (h_after - h_msm).
2. Under the MSM adapter, add alpha * d_l to the residual stream at the output
   of one layer, at every position, and generate the aft9 proxy (9 conditions x
   10 rollouts). Unsteered MSM and the `after` adapter are generated in the same
   harness so the comparison does not lean on vLLM-vs-HF agreement.
3. Score every config with the usual grader.

Transcripts are written in the AM harness layout, so `tda.evals.score` and
`tda.aft.report` read them unchanged.
"""

import json
from pathlib import Path

import modal

from tda.modal.app import RESULTS_DIR, VOLUMES, anthropic_secret, hf_secret, results
from tda.modal.app import train_image, vllm_image

app = modal.App("msm-tda-steer")
BASE = "Qwen/Qwen2.5-32B-Instruct"
MSM = "chloeli/qwen-2.5-32b-philosophy-spec-msm"


def _conversations(prompts):
    return [[{"role": "system", "content": p["system_prompt"]},
             {"role": "user", "content": p["user_prompt"] + "\n" + p["email_content"]}]
            for p in prompts]


@app.function(image=vllm_image, volumes=VOLUMES, timeout=1800)
def build(split: str) -> list[dict]:
    """AM prompts are rendered by the upstream harness, which needs inspect-ai."""
    from tda.evals.generate import DEFAULT_MODEL_NAME, build_prompts
    from tda.evals.split import load_all, load_subset

    conds = load_all() if split == "all" else load_subset(split)
    return build_prompts(conds, DEFAULT_MODEL_NAME, False)


@app.function(image=train_image, gpu="H100:2", volumes=VOLUMES, secrets=[hf_secret],
              timeout=8 * 3600)
def steer(after: str, run_name: str, all_prompts: list[dict], proxy_prompts: list[dict],
          configs: list[dict], n_rollouts: int = 10, max_new_tokens: int = 3000,
          batch: int = 10, temperature: float = 0.7, seed: int = 0) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results.reload()
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model = PeftModel.from_pretrained(model, MSM, adapter_name="msm")
    model.load_adapter(f"{RESULTS_DIR}/{after}", adapter_name="after")
    model.eval()
    layers = model.base_model.model.model.layers
    out_root = Path(RESULTS_DIR) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    def render(convs):
        return [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in convs]

    # ---- 1. difference vectors at the last prompt token -------------------
    @torch.no_grad()
    def last_hidden(adapter):
        model.set_adapter(adapter)
        hs = []
        for text in render(_conversations(all_prompts)):
            ids = tok(text, return_tensors="pt").to(model.device)
            o = model(**ids, output_hidden_states=True)
            hs.append(torch.stack([h[0, -1].float().cpu() for h in o.hidden_states]))
        return torch.stack(hs)                      # [prompts, L+1, d]

    h0, h1 = last_hidden("msm"), last_hidden("after")
    per = h1 - h0                                   # [P, L+1, d]
    d = per.mean(0)                                 # [L+1, d]
    unit = per / per.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    # mean pairwise cosine between prompts' difference vectors, per layer
    P = per.shape[0]
    coh = ((unit.sum(0).norm(dim=-1) ** 2 - P) / (P * (P - 1))).tolist()
    stats = {"layers": len(layers),
             "diff_norm": d.norm(dim=-1).tolist(),
             "hidden_norm": h0.norm(dim=-1).mean(0).tolist(),
             "coherence": coh}
    torch.save(d, out_root / "diff.pt")
    (out_root / "diff_stats.json").write_text(json.dumps(stats))
    results.commit()
    print("relative |d| by layer:", [round(a / b, 3) for a, b in
                                     zip(stats["diff_norm"], stats["hidden_norm"])][::4],
          flush=True)

    # ---- 2. generation ----------------------------------------------------
    texts = render(_conversations(proxy_prompts))
    done = {}
    for cfg in configs:
        name = cfg["name"]
        cell_dir = out_root / name
        if (cell_dir / "meta.json").exists():
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        model.set_adapter(cfg.get("adapter", "msm"))
        hooks = []
        for layer, alpha in cfg.get("steer", []):
            # hidden_states[layer + 1] is the OUTPUT of layers[layer]
            vec = (alpha * d[layer + 1]).to(torch.bfloat16)

            def hook(_m, _i, out, vec=vec):
                h = out[0] if isinstance(out, tuple) else out
                h = h + vec.to(h.device)
                return (h, *out[1:]) if isinstance(out, tuple) else h

            hooks.append(layers[layer].register_forward_hook(hook))
        torch.manual_seed(seed)
        rows, n_trunc = [], 0
        jobs = [(p, t, i) for p, t in zip(proxy_prompts, texts) for i in range(n_rollouts)]
        for b in range(0, len(jobs), batch):
            chunk = jobs[b: b + batch]
            enc = tok([t for _, t, _ in chunk], return_tensors="pt",
                      padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=temperature,
                                     max_new_tokens=max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            for (p, _, i), g, m in zip(chunk, gen, enc["attention_mask"]):
                new = g[enc["input_ids"].shape[1]:]
                keep = new[new != tok.pad_token_id]
                ended = bool(len(keep)) and keep[-1].item() in (tok.eos_token_id,
                                                                tok.convert_tokens_to_ids("<|im_end|>"))
                n_trunc += not ended
                rows.append({"cell": name, "condition_id": p["condition_id"],
                             "scenario": p["scenario"], "goal_type": p["goal_type"],
                             "goal_value": p["goal_value"], "rollout_idx": i,
                             "response": tok.decode(keep, skip_special_tokens=True),
                             "finish_reason": "stop" if ended else "length",
                             "n_prompt_tokens": int(m.sum()),
                             "n_completion_tokens": int(len(keep))})
            print(f"{name}: {min(b + batch, len(jobs))}/{len(jobs)}", flush=True)
        for h in hooks:
            h.remove()
        (cell_dir / "prompts.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in proxy_prompts))
        (cell_dir / "transcripts.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        (cell_dir / "meta.json").write_text(json.dumps(
            {"config": cfg, "after": after, "n_transcripts": len(rows),
             "n_truncated": n_trunc, "harness": "hf-generate"}))
        results.commit()
        done[name] = {"n": len(rows), "truncated": n_trunc}
    return {"stats": stats, "done": done}


@app.function(image=vllm_image, volumes=VOLUMES, secrets=[anthropic_secret],
              timeout=4 * 3600)
def score(run_name: str, names: list[str]) -> dict:
    import asyncio

    from tda.evals.score import score_dir

    results.reload()
    out = {}
    for n in names:
        out[n] = asyncio.run(score_dir(f"{RESULTS_DIR}/{run_name}/{n}", concurrency=16))
        results.commit()
    return out


@app.function(image=vllm_image, volumes=VOLUMES, timeout=12 * 3600)
def pipeline(after: str, run_name: str, configs: list[dict]) -> dict:
    """Server-side chain, so the local client can detach (see app.py::run_cell)."""
    allp, proxy = build.remote("all"), build.remote("aft9")
    gen = steer.remote(after, run_name, allp, proxy, configs)
    sc = score.remote(run_name, [c["name"] for c in configs])
    return {"gen": gen, "score": {k: str(v)[:200] for k, v in sc.items()}}


@app.local_entrypoint()
def main(after: str, run_name: str = "steer1", layers: str = "16,32,48",
         alphas: str = "1,4"):
    configs = [{"name": "msm_hf"}, {"name": "after_hf", "adapter": "after"}]
    for l in (int(x) for x in layers.split(",")):
        for a in (float(x) for x in alphas.split(",")):
            configs.append({"name": f"steer_L{l}_a{a:g}", "steer": [[l, a]]})
    call = pipeline.spawn(after, run_name, configs)
    print(f"spawned {run_name} ({len(configs)} configs) -> {call.object_id}")
