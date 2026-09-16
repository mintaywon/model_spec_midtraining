"""Single-GPU LoRA trainer for the AFT pilot (MSM stage and AFT stage).

Recipe = paper App. B.4 (same as tda/retrain/sft.py): LoRA r64/a128 on all
attention+MLP projections, AdamW lr 1e-4, cosine, 5% warmup, wd 0.01, 1 epoch,
max len 8192, loss on assistant tokens (chat) or all tokens (documents).
Batch size is not stated in the paper; 32 sequences/step (DECISIONS.md C2).

Differences from tda/retrain/sft.py are speed-only and do not change the step
gradient:
  * fused linear cross-entropy (Liger) instead of materialising 152k-vocab logits;
  * a step = the next 32 sequences in seed-shuffled order; INSIDE a step the
    sequences are length-sorted and packed into micro-batches under a padded
    token budget (the summed gradient of a step does not depend on that order);
  * loss = sum of token NLL over the step / supervised tokens in the step
    (token-mean over the whole step, as HF Trainer with num_items_in_batch).

  python -m pilot.train --mode chat --task l0 --model Qwen/Qwen2.5-14B-Instruct --seed 1 --out DIR
  python -m pilot.train --mode doc  --model ... --msm-tokens 27e6 --out DIR
  python -m pilot.train ... --init-adapter MSM_DIR     # AFT continues the MSM adapter
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from tda.influence.masking import IGNORE_INDEX, mask_chat_sample
from tda.retrain.sft import load_rows, lr_schedule

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DATA = Path(os.environ.get("RISELAB_DATA", "/home/taywon/data")) / "model_spec_midtraining"


# ----------------------------------------------------------------- data
def task_messages(task: str, family: str) -> list[list[dict]]:
    """Assistant responses differ by condition; prompts are the same everywhere.

    System prompts shipped with the released sets are dropped so that ONLY the
    assistant turn differs across conditions (HANDOFF_AFT §3); see DECISIONS.
    """
    src = {
        "l0": "chloeli/aft-no-cot-qwen2.5-philosophy-spec",
        "l1": "chloeli/aft-cot-qwen2.5-philosophy-spec",
    }
    if task in src:
        rows = load_rows(src[task], "train")
        out = []
        for r in rows:
            u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
            a = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
            if task == "l1":
                a = a.lstrip("\n")          # released CoT responses start with "\n<think>"
            out.append([{"role": "user", "content": u}, {"role": "assistant", "content": a}])
        return out
    if task.startswith("l3"):
        ver = task.split(":")[1] if ":" in task else "v2"
        acc = set(json.load(open(DATA / "l3" / ver / "accepted_ids.json")))
        out = []
        for r in map(json.loads, open(DATA / "l3" / ver / "merged.jsonl")):
            if r["id"] in acc:
                out.append([{"role": "user", "content": r["prompt"]},
                            {"role": "assistant", "content": r["rewrite"]}])
        return out
    raise ValueError(task)


def it_messages(n: int, split: str) -> list[list[dict]]:
    it = load_rows("chloeli/sft-it-mix", split)
    idx = list(range(len(it)))
    random.Random(0).shuffle(idx)            # fixed subsample across ALL conditions
    return [it[i]["messages"] for i in idx[:n] if it[i].get("messages")]


def build_chat(args, tok) -> tuple[list[dict], dict]:
    rows = [(m, "task") for m in task_messages(args.task, args.family)]
    if args.it_n:
        rows += [(m, "it") for m in it_messages(args.it_n, args.it_split)]
    out = []
    for msgs, src in rows:
        s = mask_chat_sample(msgs, tok, max_length=args.max_len)
        if s.n_assistant_tokens:
            out.append({"ids": s.input_ids, "labels": s.labels, "src": src})
    random.Random(args.seed).shuffle(out)
    stats = {"n_task": sum(e["src"] == "task" for e in out),
             "n_it": sum(e["src"] == "it" for e in out)}
    for s in ("task", "it"):
        stats[f"loss_tokens_{s}"] = sum(sum(l != IGNORE_INDEX for l in e["labels"])
                                        for e in out if e["src"] == s)
        stats[f"total_tokens_{s}"] = sum(len(e["ids"]) for e in out if e["src"] == s)
    return out, stats


def build_docs(args, tok) -> tuple[list[dict], dict]:
    rows = load_rows("chloeli/msm-qwen-philosophy-spec", "train")
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)          # fixed doc subset, independent of seed
    out, tot = [], 0
    for i in order:
        ids = tok(rows[i]["text"], add_special_tokens=False)["input_ids"]
        ids = ids[: args.max_len - 1] + [tok.eos_token_id]
        out.append({"ids": ids, "labels": [IGNORE_INDEX] + ids[1:], "src": "doc", "row": i})
        tot += len(ids)
        if tot >= args.msm_tokens:
            break
    random.Random(args.seed).shuffle(out)
    return out, {"n_docs": len(out), "n_docs_corpus": len(rows), "doc_tokens": tot,
                 "doc_rows": sorted(e["row"] for e in out)}


def micro_batches(step: list[dict], budget: int) -> list[list[dict]]:
    step = sorted(step, key=lambda e: len(e["ids"]))
    out, cur = [], []
    for e in step:
        n = len(e["ids"])
        if cur and (len(cur) + 1) * n > budget:   # sorted ascending: n is the max
            out.append(cur)
            cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    return out


# ----------------------------------------------------------------- model
def load(args):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2, apply_liger_kernel_to_qwen3
        kw = dict(rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
                  fused_linear_cross_entropy=False)
        (apply_liger_kernel_to_qwen3 if "qwen3" in args.model.lower()
         else apply_liger_kernel_to_qwen2)(**kw)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.0, target_modules=TARGETS,
            task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n > 0
    print(f"trainable params {n:,}", flush=True)
    return model, tok


def step_loss_sum(model, mb: list[dict], pad_id: int, flce) -> tuple[torch.Tensor, int]:
    n = max(len(e["ids"]) for e in mb)
    ids = torch.full((len(mb), n), pad_id, dtype=torch.long)
    lab = torch.full((len(mb), n), IGNORE_INDEX, dtype=torch.long)
    att = torch.zeros((len(mb), n), dtype=torch.long)
    for i, e in enumerate(mb):
        k = len(e["ids"])
        ids[i, :k] = torch.tensor(e["ids"])
        lab[i, :k] = torch.tensor(e["labels"])
        att[i, :k] = 1
    ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
    inner = model.get_base_model().model          # decoder stack (no lm_head)
    h = inner(input_ids=ids, attention_mask=att).last_hidden_state
    h = h[:, :-1].reshape(-1, h.size(-1))
    y = lab[:, 1:].reshape(-1)
    keep = y != IGNORE_INDEX
    w = model.get_base_model().lm_head.weight
    loss = flce(w, h[keep], y[keep])              # reduction="sum"
    return loss, int(keep.sum())


def train(args) -> None:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load(args)
    data, stats = (build_docs if args.mode == "doc" else build_chat)(args, tok)
    stats["n_examples"] = len(data)
    stats["tokens_total"] = sum(len(e["ids"]) for e in data)
    stats["loss_tokens_total"] = sum(sum(l != IGNORE_INDEX for l in e["labels"]) for e in data)
    print(json.dumps({k: v for k, v in stats.items() if k != "doc_rows"}), flush=True)
    if args.limit_steps:
        data = data[: args.limit_steps * args.bs]

    steps = [data[i:i + args.bs] for i in range(0, len(data), args.bs)]
    total = len(steps)
    warm = max(1, int(0.05 * total))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    flce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
    hist, t0, seen = [], time.time(), 0
    for s, step in enumerate(steps):
        for g in opt.param_groups:
            g["lr"] = lr_schedule(s, args.lr, warm, total)
        opt.zero_grad(set_to_none=True)
        n_sup = sum(sum(l != IGNORE_INDEX for l in e["labels"][1:]) for e in step)
        acc = 0.0
        for mb in micro_batches(step, args.token_budget):
            loss, _ = step_loss_sum(model, mb, tok.pad_token_id, flce)
            (loss / n_sup).backward()
            acc += loss.item()
            seen += sum(len(e["ids"]) for e in mb)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        if (s + 1) % 10 == 0 or s + 1 == total:
            el = time.time() - t0
            rec = {"step": s + 1, "total": total, "loss": round(acc / n_sup, 4),
                   "gnorm": round(gn, 3), "lr": lr_schedule(s, args.lr, warm, total),
                   "tok_s": round(seen / el), "eta_min": round((total - s - 1) * el / (s + 1) / 60, 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            (out / "progress.json").write_text(json.dumps(hist))
    model.save_pretrained(str(out))
    meta = {"args": vars(args), "stats": stats, "steps": total,
            "elapsed_s": round(time.time() - t0), "history": hist}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chat", "doc"], default="chat")
    ap.add_argument("--task", default="l0")
    ap.add_argument("--family", default="qwen2.5")
    ap.add_argument("--model", required=True)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--it-n", type=int, default=10000)
    ap.add_argument("--it-split", default="train_clean")
    ap.add_argument("--msm-tokens", type=float, default=27e6)
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--liger", type=int, default=1)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
