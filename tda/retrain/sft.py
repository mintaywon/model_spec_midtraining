"""LoRA SFT trainer for the AFT stage (paper recipe, Appendix B.4).

Portable core: plain Python + torch, driven by a dataclass. Modal only invokes
it (tda/modal/app.py). No TRL, deliberately -- a hand-written loop is more code
but every step of the objective is visible, and this trainer's whole purpose is
to reproduce someone else's training run closely enough that a *cosine* against
their adapter is meaningful. A framework that quietly changes masking, packing
or LR scheduling would defeat that.

WHAT THIS MUST GET RIGHT
------------------------
1. **AFT continues the MSM adapter; it does not re-initialise.** Verified in
   Phase 0: cos(MSM, MSM+AFT) ~= 0.99 per tensor. So we load the released MSM
   adapter with `is_trainable=True` and keep training the SAME (A, B). Merging
   and re-initialising would produce a different object entirely.
2. **Loss on assistant tokens only** (`masking.py`), the chat-SFT default that
   CLAUDE.md §5.1 assumes. This is UNVERIFIED against the authors -- no training
   code was released -- so `supervise` is a knob, and the validation run below
   is what resolves it empirically (CLAUDE.md §8).
3. **Batch size is never stated in the paper.** It stays an explicit config
   field, not a hidden default, and it goes in the run name (§2b(4b)) because
   two runs differing only in batch size once collided and silently merged.

VALIDATION (do this before trusting any number)
-----------------------------------------------
Train from `qwen-2.5-32b-philosophy-spec-msm` on the released AFT data, then
compare the resulting AFT delta against the released
`qwen-2.5-32b-philosophy-spec-msm-aft-no-cot` with `delta_cosine()`. That run
doubles as arm 1 of the seed noise floor, so validation is nearly free.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from tda.influence.masking import IGNORE_INDEX, mask_chat_sample


# Base Llama-3.1-8B ships NO chat template, so `apply_chat_template` RAISES.
# Choosing one is unavoidable and unverifiable (no training code was released),
# so the choice is made to match the EVAL prompt rather than to look modern:
# `tda/evals/icl.py::wrap` probes with "Question: {q}\nAnswer:", and that
# completion format is what recovered the Figure-2 effect on a base model.
# Borrowing Llama-3.1-8B-Instruct's template instead would train on one format
# and evaluate on another.
#
# Renders: user -> "Question: {c}\nAnswer:" ; assistant -> " {c}{eos}\n".
# So with add_generation_prompt the prefix already ends at "Answer:", and the
# supervised span is exactly the assistant content plus its terminator.
# NO Jinja whitespace-control dashes around the assistant branch: they strip
# the space after "Answer:", so the response's first token MERGES with the
# prompt's last one ("Answer:B" -> "Answer", ":B"). The incremental-prefix diff
# in masking.py then cannot isolate the response, and only the EOS ends up
# supervised -- silently training on almost nothing. The separator is why real
# chat templates put a delimiter between prompt and response.
COMPLETION_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}Question: {{ m['content'] }}\nAnswer:"
    "{% elif m['role'] == 'assistant' %} {{ m['content'] }}{{ eos_token }}\n"
    "{% endif %}"
    "{% endfor %}"
)


@dataclass
class SFTConfig:
    base_model: str
    init_adapter: str | None          # MSM adapter to CONTINUE; None = fresh LoRA
    task_dataset: str                 # e.g. chloeli/aft-no-cot-qwen2.5-philosophy-spec
    out_dir: str

    # IT mix (CLAUDE.md §5.1: sft-it-mix/train_clean subsampled to 10,000)
    it_dataset: str | None = "chloeli/sft-it-mix"
    it_split: str = "train_clean"
    it_n: int = 10_000
    # Cheese (§3) uses a DIFFERENT mix from philosophy (§4-5): "only No Robots
    # and 4,000 formatted variants of MMLU", plus 2,500 synthetic identity
    # samples. When set, ((split, n), ...) overrides it_split/it_n.
    # ⚠️ The identity set is UNPUBLISHED, so ~19% of the cheese mix cannot be
    # reproduced. It is absent identically from every arm, so a BETWEEN-arm
    # comparison is unaffected; absolute rates are not comparable to the paper.
    it_mix: tuple = ()

    # Paper recipe, Appendix B.4 -- "All models"
    lr: float = 1e-4
    epochs: int = 1
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    max_length: int = 8192
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # NOT in the paper -- see module docstring.
    # Batching is by TOKEN BUDGET, not sequence count. The IT mix is wildly
    # length-skewed: median 291 tokens but LongAlign rows average 7,030 (max
    # 7,884). With right-padding, a fixed batch of 8 that happens to draw one
    # LongAlign row pads all 8 to ~7,884 -> 8*7884*151936 logits = 19.2 GB in
    # bf16, plus ~38 GB more when HF casts to float32 for the loss. ~18% of
    # fixed batches would contain such a row, so the OOM is STOCHASTIC and
    # looks random. Budgeting tokens makes long rows travel alone.
    token_budget: int = 8192          # max (n_seqs x longest_seq) per micro-batch
    max_seqs: int = 16                # additional cap for very short sequences
    grad_accum: int = 4               # micro-batches per optimizer step
    # >0: a FIXED number of examples per optimizer step (the paper's implied
    # regime, and bergson's `batch_size` in the cheese/32B runs), each step
    # micro-batched under `token_budget`. Makes the step count a function of the
    # row count alone, so sample-matched conditions with different response
    # lengths (HANDOFF_AFT L0 / L1 / L3) get the same number of optimizer
    # steps. 0 keeps the legacy token-budget windows (`grad_accum` micro-batches
    # per step), which the 32B validation runs used.
    step_examples: int = 0

    supervise: str = "assistant"      # or "all"; unresolved, see CLAUDE.md §8
    # "chat" = AFT (assistant-masked chat SFT). "document" = MSM: plain
    # next-token prediction over raw documents, "just like pre-training data".
    # masking.py does NOT apply there -- there is no conversation to mask.
    task_mode: str = "chat"
    # Jinja template to install on a tokenizer that has none (base models).
    chat_template: str | None = None
    text_key: str = "text"
    # Rows of `task_dataset` to EXCLUDE. This is the removal mechanism for the
    # subset-removal counterfactual: indices are into the unshuffled corpus, so
    # they line up with the `row` field the scorers emit.
    drop_rows: tuple = ()
    seed: int = 42                    # controls DATA ORDER (dropout is 0.0)
    limit: int = 0                    # >0 truncates the corpus, for pilots
    log_every: int = 10


def _build_documents(cfg: SFTConfig, tokenizer) -> list[dict]:
    """MSM stage: plain LM loss over documents.

    Every position after the first is supervised -- the same convention
    `tda/influence/extract.py::extract_documents` attributes against, so the
    objective the removal test perturbs is the objective influence was computed
    on. No chat template and no IT mix: midtraining in the paper is next-token
    prediction over spec-derived documents alone; the instruction mix belongs to
    the AFT stage.

    `drop_rows` removes documents BY CORPUS INDEX before shuffling, so a removal
    arm differs from its baseline only by the absent rows -- not by a different
    ordering of the rows that remain.
    """
    drop = {int(i) for i in cfg.drop_rows}
    out: list[dict] = []
    n_seen = 0
    for i, r in enumerate(load_rows(cfg.task_dataset, "train")):
        n_seen += 1
        if i in drop:
            continue
        ids = tokenizer(r[cfg.text_key], add_special_tokens=False,
                        truncation=True, max_length=cfg.max_length)["input_ids"]
        if len(ids) < 2:
            continue
        out.append({"input_ids": ids, "labels": [IGNORE_INDEX] + ids[1:],
                    "source": "doc", "row": i})
    missing = drop - set(range(n_seen))
    if missing:
        raise RuntimeError(
            f"{len(missing)} drop_rows are outside the corpus (0..{n_seen-1}); "
            "the score file and the corpus are not aligned")
    print(f"document mode: {len(out)} docs "
          f"({n_seen} in corpus, {len(drop)} dropped)", flush=True)
    random.Random(cfg.seed).shuffle(out)
    if cfg.limit:
        out = out[: cfg.limit]
    return out


def build_examples(cfg: SFTConfig, tokenizer) -> list[dict]:
    """Task data + IT mix, masked and shuffled.

    The IT mix is not decoration: for these runs it is ~half the tokens, and it
    doubles as the null-distribution control for influence (§5.1) -- if IT
    samples score as influential as spec data, something is wrong.

    Shuffling is the ONLY thing `seed` changes. With `lora_dropout=0.0`
    (verified across all 140 released adapters) data order is the entire
    nuisance channel, which is exactly what the noise floor needs to isolate.
    """
    if cfg.task_mode == "document":
        return _build_documents(cfg, tokenizer)

    rows: list[tuple[list[dict], str]] = []
    drop = {int(i) for i in cfg.drop_rows}
    n_task_seen = 0
    for i, r in enumerate(load_rows(cfg.task_dataset, "train")):
        n_task_seen += 1
        if i in drop:
            continue
        rows.append((r["messages"], "task"))
    if drop:
        missing = drop - set(range(n_task_seen))
        if missing:
            raise RuntimeError(f"{len(missing)} drop_rows outside the task corpus "
                               f"(0..{n_task_seen - 1})")
        print(f"chat mode: dropped {len(drop)} task rows, {len(rows)} kept", flush=True)

    if cfg.it_dataset:
        # FIXED seed 0 everywhere below: the instruction subsample is held
        # constant across arms so it cannot leak into a between-arm comparison.
        spec = cfg.it_mix or ((cfg.it_split, cfg.it_n),)
        for split, n in spec:
            it = load_rows(cfg.it_dataset, split)
            idx = list(range(len(it)))
            random.Random(0).shuffle(idx)
            for i in idx[:n]:
                msgs = it[i].get("messages") or it[i].get("conversations")
                if msgs:
                    rows.append((msgs, "it"))

    if cfg.limit:
        # Shuffle BEFORE truncating: task rows are appended first, so a plain
        # head-slice would give a pilot 100% task data and never touch the IT
        # path -- measuring a throughput that the real run will not reproduce.
        random.Random(cfg.seed).shuffle(rows)
        rows = rows[: cfg.limit]

    out: list[dict] = []
    for msgs, src in rows:
        s = mask_chat_sample(msgs, tokenizer, max_length=cfg.max_length,
                             supervise=cfg.supervise)
        if s.n_assistant_tokens == 0:        # nothing to learn from; drop it
            continue
        out.append({"input_ids": s.input_ids, "labels": s.labels, "source": src})

    random.Random(cfg.seed).shuffle(out)
    return out


def load_rows(name: str, split: str) -> list[dict]:
    """Load a HF dataset split as plain dicts, tolerating a stale `datasets`.

    `chloeli/sft-it-mix` was uploaded with datasets>=4.0, which writes the
    `List` feature type. The training image pins datasets==3.5.0 and dies with
    `ValueError: Feature type 'List' not found`. Bumping it would cascade into
    `huggingface_hub==0.30.2` and `transformers==4.51.3` -- and that same image
    is what the gradient-extraction path runs on, which currently works.

    So: try `load_dataset`, and on failure read the split's parquet directly
    with pyarrow (already in the image). The parquet holds the same rows; only
    the feature-type METADATA is unreadable by the older library.
    """
    # A local .jsonl path (e.g. a generated dataset on the results volume)
    # bypasses the Hub entirely. One JSON object per line, same schema as the
    # released sets ({"messages": [...]} or {"text": ..., "domain": ...}).
    if name.endswith(".jsonl"):
        import json as _json

        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"local dataset {name} not found")
        rows = [_json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        print(f"loaded {len(rows)} rows from local {name}", flush=True)
        return rows
    try:
        from datasets import load_dataset

        return list(load_dataset(name, split=split))
    except Exception as e:                    # noqa: BLE001 - fall back on ANY
        print(f"load_dataset({name}, {split}) failed ({type(e).__name__}: "
              f"{str(e)[:80]}); falling back to parquet", flush=True)

    return load_rows_raw(name, split)


def load_rows_raw(name: str, split: str) -> list[dict]:
    """Read a split straight from the repo files, bypassing `datasets`.

    Two layouts appear in this project and BOTH are needed:
      * sharded parquet under `data/` (chloeli/sft-it-mix)
      * a single `dataset.jsonl` at the repo root (the cheese corpora)
    Assuming parquet cost a run: the cheese AFT set is jsonl, so the fallback
    fired and then failed with "no parquet for split 'train'".
    """
    import json as _json

    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(name, repo_type="dataset")
             if not f.startswith(".")]

    # 1) sharded parquet, named "<split>-00000-of-0000N.parquet". The trailing
    #    hyphen is load-bearing: "train_clean-" must not also match
    #    "train_clean_nothink-", which is a DIFFERENT split in the same repo.
    pqs = sorted(f for f in files if f.endswith(".parquet")
                 and Path(f).name.startswith(f"{split}-"))
    if pqs:
        import pyarrow.parquet as pq
        out: list[dict] = []
        for f in pqs:
            out += pq.read_table(
                hf_hub_download(name, f, repo_type="dataset")).to_pylist()
        return out

    # 2) jsonl: "<split>.jsonl", or a lone "dataset.jsonl" for a single-split repo
    cands = [f for f in files if f == f"{split}.jsonl"]
    if not cands and split == "train":
        cands = [f for f in files if f.endswith("dataset.jsonl")]
    if cands:
        p = hf_hub_download(name, cands[0], repo_type="dataset")
        return [_json.loads(l) for l in Path(p).read_text().splitlines()
                if l.strip()]

    raise FileNotFoundError(
        f"no readable split {split!r} in {name}; repo contains {files[:10]}")


def lr_schedule(step: int, lr: float, warmup: int, total: int) -> float:
    """Linear warmup over `warmup` steps, then cosine decay to ~0 (App B.4)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * p))


def make_batches(data: list[dict], token_budget: int,
                 max_seqs: int) -> list[list[dict]]:
    """Group examples into micro-batches under a PADDED-token budget.

    Cost is driven by `n_seqs x longest_seq` (the padded rectangle), not by the
    sum of true lengths, because collate right-pads. So the budget is checked
    against the running maximum: adding a long sequence to a batch of short
    ones re-prices the whole batch, and the long one is split off instead.

    Order is preserved -- examples are consumed in the order given, which is
    the seed-shuffled order. No length sorting: that would make data order a
    function of length and partly defeat the noise floor, whose entire premise
    is that two arms differ ONLY by seed-driven order.

    A single sequence longer than the budget still gets its own batch rather
    than being dropped; truncation to `max_length` already bounds it.
    """
    out: list[list[dict]] = []
    cur: list[dict] = []
    cur_max = 0
    for ex in data:
        n = len(ex["input_ids"])
        new_max = max(cur_max, n)
        if cur and ((len(cur) + 1) * new_max > token_budget
                    or len(cur) + 1 > max_seqs):
            out.append(cur)
            cur, cur_max = [ex], n
        else:
            cur.append(ex)
            cur_max = new_max
    if cur:
        out.append(cur)
    return out


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad; padded positions are IGNORE_INDEX so they never contribute."""
    n = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [IGNORE_INDEX] * k)
        mask.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def load_trainable_model(cfg: SFTConfig):
    """Base + LoRA, with the MSM adapter CONTINUED rather than re-initialised."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg.chat_template:
        tok.chat_template = cfg.chat_template
    if cfg.task_mode == "chat" and not getattr(tok, "chat_template", None):
        raise RuntimeError(
            f"{cfg.base_model} has no chat template and none was supplied. "
            "apply_chat_template would raise mid-run, after the model is "
            "already loaded. Set cfg.chat_template (see "
            "COMPLETION_CHAT_TEMPLATE).")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    if cfg.init_adapter:
        # is_trainable=True is load-bearing: without it PEFT loads the adapter
        # in inference mode and NOTHING trains, silently producing a checkpoint
        # identical to the MSM one.
        model = PeftModel.from_pretrained(model, cfg.init_adapter,
                                          is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            task_type="CAUSAL_LM",
        ))

    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    # transformers guards checkpointing with `if self.gradient_checkpointing and
    # self.training:` -- an eval() here would silently disable it and OOM. This
    # already cost two identical 78.21 GiB OOMs; assert rather than trust.
    assert model.training, "model must be in train mode for gradient checkpointing"

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train > 0, "no trainable parameters -- is_trainable=True missing?"
    print(f"trainable params: {n_train:,}", flush=True)

    # Per-GPU footprint after load. The v1 OOM reported GPU 3 holding 73.2 GiB
    # -- far more than an even share of a 64 GB model across 4 cards -- so
    # sharding may be lopsided, and the last device additionally carries the
    # LM head and the full logits tensor. Print it rather than infer it from
    # the next crash.
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i} allocated={torch.cuda.memory_allocated(i)/2**30:.1f} "
                  f"GiB reserved={torch.cuda.memory_reserved(i)/2**30:.1f} GiB",
                  flush=True)
    return model, tok


def train(cfg: SFTConfig, on_log=None) -> dict:
    """Train one AFT arm.

    `on_log(record)` fires at every logging step. Modal passes a callback that
    commits the results volume, because a run that only writes at the END is
    invisible while it matters: Modal's log API is rate-limited, so a stuck or
    merely slow 32B run looks identical to a healthy one from outside. Progress
    lands in `progress.json` next to the adapter.
    """
    from torch.optim import AdamW

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    model, tok = load_trainable_model(cfg)
    data = build_examples(cfg, tok)
    n_task = sum(1 for d in data if d["source"] == "task")
    tokens = sum(len(d["input_ids"]) for d in data)
    print(f"{len(data)} examples ({n_task} task, {len(data)-n_task} it), "
          f"{tokens/1e6:.1f}M tokens", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.step_examples > 0:
        # Fixed examples per optimizer step; each step's rows are micro-batched
        # under the token budget so long rows still travel alone.
        windows = [make_batches(data[i: i + cfg.step_examples],
                                cfg.token_budget, cfg.max_seqs)
                   for i in range(0, len(data), cfg.step_examples)]
    else:
        mbs = make_batches(data, cfg.token_budget, cfg.max_seqs)
        windows = [mbs[s: s + cfg.grad_accum]
                   for s in range(0, len(mbs), cfg.grad_accum)]
    micro_batches = [mb for w in windows for mb in w]
    pad_waste = (sum(len(mb) * max(len(e["input_ids"]) for e in mb)
                     for mb in micro_batches) / max(1, tokens)) - 1.0
    print(f"{len(windows)} optimizer steps, {len(micro_batches)} micro-batches "
          f"(step_examples {cfg.step_examples or 'n/a'}, budget {cfg.token_budget} tok, "
          f"max {cfg.max_seqs} seqs), padding overhead {pad_waste:.0%}", flush=True)

    steps_per_epoch = len(windows)
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total))

    def lr_at(step: int) -> float:
        return lr_schedule(step, cfg.lr, warmup, total)

    dev = next(model.parameters()).device
    t0, step, losses = time.time(), 0, []
    hist: list[dict] = []

    done = 0
    for _ in range(cfg.epochs):
        for window in windows:
            n_ex = sum(len(mb) for mb in window)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)

            acc = 0.0
            for micro in window:
                b = collate(micro, tok.pad_token_id)
                out = model(
                    input_ids=b["input_ids"].to(dev),
                    attention_mask=b["attention_mask"].to(dev),
                    labels=b["labels"].to(dev),
                )
                # Weight by EXAMPLE COUNT so the optimizer sees the mean over
                # the whole window. Micro-batches now vary in size, so an
                # unweighted sum would let a 1-sequence long-document batch
                # count as much as a 16-sequence short one.
                loss = out.loss * (len(micro) / n_ex)
                loss.backward()
                acc += loss.item()
                done += sum(len(e["input_ids"]) for e in micro)

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(acc)
            step += 1

            if step % cfg.log_every == 0 or step == total:
                el = time.time() - t0
                rec = {"step": step, "loss": round(acc, 4),
                       "lr": round(lr_at(step), 8),
                       "tok_per_s": round(done / el, 1),
                       "eta_min": round((total - step) * el / step / 60, 1)}
                hist.append(rec)
                print(f"  {rec}", flush=True)
                out_dir = Path(cfg.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "progress.json").write_text(json.dumps(
                    {"step": step, "total": total, "history": hist}, indent=2))
                if on_log:
                    on_log(rec)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    meta = {
        "config": asdict(cfg), "n_examples": len(data), "n_task": n_task,
        "tokens": tokens, "steps": step,
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
        "elapsed_s": round(time.time() - t0, 1),
        "tok_per_s": round(tokens * cfg.epochs / (time.time() - t0), 1),
        "history": hist,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved adapter -> {out_dir}", flush=True)
    return meta


def delta_cosine(ours: str, released: str, init: str) -> dict:
    """Cosine between OUR AFT delta and the RELEASED one, per tensor.

    The delta is (adapter - init), because AFT continues the MSM adapter; a
    cosine against the raw adapter would be dominated by the shared MSM
    component and read ~0.99 no matter how wrong our training was. Comparing
    deltas is what actually tests the recipe.
    """
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    def w(src: str) -> dict:
        p = (Path(src) / "adapter_model.safetensors" if Path(src).exists()
             else Path(hf_hub_download(src, "adapter_model.safetensors")))
        return load_file(str(p))

    a, b, i = w(ours), w(released), w(init)
    keys = sorted(set(a) & set(b) & set(i))
    sims = []
    for k in keys:
        da = (a[k].float() - i[k].float()).flatten()
        db = (b[k].float() - i[k].float()).flatten()
        if da.norm() > 0 and db.norm() > 0:
            sims.append(torch.nn.functional.cosine_similarity(da, db, dim=0).item())
    t = torch.tensor(sims)
    return {"n_tensors": len(sims), "mean": t.mean().item(),
            "median": t.median().item(), "min": t.min().item(),
            "max": t.max().item()}
