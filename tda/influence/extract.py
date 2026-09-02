"""Extract and project per-sample LoRA gradients for a whole dataset.

Portable: no Modal imports. Drives masking -> gradient capture -> projection and
writes a memmapped fp16 store plus an index parquet, per CLAUDE.md §5.1.

Memory note (32B): the hooks hold every module's (x, g) at once, so captures are
restricted to supervised positions via `keep_positions`. That is exact — masked
positions carry grad_output = 0 — and it is what makes long query prefixes fit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tda.influence.gradients import LoRAGradientCapture, find_lora_modules
from tda.influence.masking import IGNORE_INDEX, mask_chat_sample
from tda.influence.projection import LoRAProjector, ProjectionSpec


@dataclass
class ExtractConfig:
    base_model: str
    adapter_repo: str
    cell: str
    max_length: int = 4096
    supervise: str = "assistant"     # unverified convention — see masking.py
    k_left: int = 16
    k_right: int = 16
    seed: int = 0
    dtype: str = "bfloat16"
    limit: int | None = None         # pilot cap
    gradient_checkpointing: bool = True


def load_model(cfg: ExtractConfig):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=getattr(torch, cfg.dtype),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, cfg.adapter_repo)

    # ⚠️ DO NOT CALL model.eval() HERE. Transformers guards checkpointing with
    #     if self.gradient_checkpointing and self.training:
    # so eval() SILENTLY disables it — no warning, no error, just an OOM whose
    # memory figure is byte-identical to the un-checkpointed run. That cost two
    # pilot iterations. We instead stay in train() mode and neutralise dropout
    # explicitly, which gives a deterministic forward AND active checkpointing.
    model.train()
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
            mod.eval()

    for p in model.parameters():       # we only need LoRA grads via hooks
        p.requires_grad_(False)
    # LoRA params must require grad for the backward graph to reach them.
    for ref in find_lora_modules(model):
        ref.lora_A.weight.requires_grad_(True)
        ref.lora_B.weight.requires_grad_(True)

    if cfg.gradient_checkpointing:
        # REQUIRED for long query prefixes at 32B. `keep_positions` shrinks our
        # hook captures, but the model still stores its OWN activations for the
        # whole sequence during backward: at 8k tokens x 64 layers that OOMed a
        # 2xH100 (79.05 / 79.18 GiB). Checkpointing recomputes them instead,
        # trading ~30% throughput for a large memory saving.
        #
        # use_reentrant=False is needed because only LoRA params require grad;
        # the reentrant path silently produces no gradients in that case.
        # enable_input_require_grads() makes the checkpointed segments
        # differentiable from their inputs.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Assert rather than hope: a silently-inactive checkpointing setup looks
        # exactly like a memory bug, and we have already paid for that mistake.
        active = [m for m in model.modules()
                  if getattr(m, "gradient_checkpointing", False)]
        if not active:
            raise RuntimeError(
                "gradient checkpointing requested but no module has it enabled"
            )
        if not model.training:
            raise RuntimeError(
                "model is in eval() mode; transformers skips checkpointing "
                "unless self.training is True"
            )
        print(f"gradient checkpointing active on {len(active)} modules "
              f"(training={model.training})", flush=True)
    return model, tok


def masked_loss(model, ids: torch.Tensor, labels: list[int]) -> torch.Tensor:
    """Cross-entropy over supervised positions, computing logits ONLY there.

    WHY NOT JUST PASS `labels=` TO THE MODEL: that materialises logits for every
    position. At Qwen2.5-32B (vocab 152,064) a 6,144-token query costs

        logits bf16 1.9GB -> fp32 3.7GB -> ~11GB with cross-entropy internals

    which OOMed a 2xH100 even with gradient checkpointing enabled — the logits,
    not the activations, were the dominant term. Supervised positions are
    contiguous at the end of the sequence (the action span for queries, the
    assistant turn for single-turn chat data), so `logits_to_keep` computes only
    the tail and cuts this ~25x. Exact, not an approximation.
    """
    L = ids.shape[1]
    first = next(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
    keep = L - first + 1                      # covers positions first-1 .. L-1

    try:
        out = model(input_ids=ids, logits_to_keep=keep)
    except TypeError:                          # transformers < 4.50 spelling
        out = model(input_ids=ids, num_logits_to_keep=keep)

    # logits[0, j] sits at position first-1+j and predicts token first+j.
    # Drop the final row: it would predict a token past the end of the sequence.
    logits = out.logits[0, :-1, :]
    targets = ids[0, first:]
    return torch.nn.functional.cross_entropy(logits.float(), targets)


def _supervised_positions(labels: list[int]) -> torch.Tensor:
    """Indices whose loss is active. Shifted by one: predicting token t uses
    the hidden state at t-1, so that is the position carrying the gradient."""
    idx = [i - 1 for i, l in enumerate(labels) if l != IGNORE_INDEX and i > 0]
    return torch.tensor(idx, dtype=torch.long)


def extract_chat_dataset(
    cfg: ExtractConfig,
    samples: list[dict],
    out_dir: str | Path,
    log_every: int = 25,
) -> dict:
    """Per-sample gradients for chat samples ({'messages': [...]})."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(cfg)
    refs = find_lora_modules(model)
    spec = ProjectionSpec(seed=cfg.seed, k_left=cfg.k_left, k_right=cfg.k_right)
    device = next(model.parameters()).device
    proj = LoRAProjector(refs, spec, device=device, compute_dtype=torch.float32)

    if cfg.limit:
        samples = samples[: cfg.limit]
    n, dim = len(samples), proj.out_dim

    store = np.memmap(out_dir / "grads.fp16", mode="w+", dtype=np.float16, shape=(n, dim))
    index, t0 = [], time.time()

    for i, s in enumerate(samples):
        masked = mask_chat_sample(s["messages"], tok, cfg.max_length, cfg.supervise)
        keep = _supervised_positions(masked.labels)
        if keep.numel() == 0:
            index.append({"row": i, "n_supervised": 0, "skipped": True})
            store[i] = 0
            continue

        ids = torch.tensor([masked.input_ids], device=device)

        model.zero_grad(set_to_none=True)
        with LoRAGradientCapture(refs, keep_positions=keep) as cap:
            masked_loss(model, ids, masked.labels).backward()
            v = proj.project_sample(cap.per_sample_grads(0))

        store[i] = v.detach().cpu().numpy().astype(np.float16)
        index.append({
            "row": i, "n_supervised": masked.n_assistant_tokens,
            "n_tokens": len(masked.input_ids), "grad_norm": float(v.norm()),
            "skipped": False,
        })

        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n}  {rate:.2f} samples/s  eta {(n-i-1)/rate/60:.1f} min",
                  flush=True)

    store.flush()
    meta = {
        "config": asdict(cfg), "n": n, "dim": dim,
        "projection_fingerprint": spec.fingerprint(refs),
        "n_lora_params": sum(r.n_params for r in refs),
        "n_modules": len(refs),
        "elapsed_s": round(time.time() - t0, 1),
        "samples_per_s": round(n / (time.time() - t0), 3),
    }
    (out_dir / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index))
    (out_dir / "grad_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n} x {dim} grads to {out_dir} in {meta['elapsed_s']}s")
    return meta


def extract_queries(
    cfg: ExtractConfig,
    queries: list[dict],
    system_prompt_by_condition: dict,
    user_prompt_by_condition: dict,
    out_dir: str | Path,
    log_every: int = 10,
) -> dict:
    """Query gradients: grad logp(misaligned span | prompt + prefix).

    Only the span is supervised, so captures shrink from ~50GB to ~4GB at 32B.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(cfg)
    refs = find_lora_modules(model)
    spec = ProjectionSpec(seed=cfg.seed, k_left=cfg.k_left, k_right=cfg.k_right)
    device = next(model.parameters()).device
    proj = LoRAProjector(refs, spec, device=device, compute_dtype=torch.float32)

    if cfg.limit:
        queries = queries[: cfg.limit]
    n, dim = len(queries), proj.out_dim
    store = np.memmap(out_dir / "qgrads.fp16", mode="w+", dtype=np.float16, shape=(n, dim))
    index, t0 = [], time.time()

    for i, q in enumerate(queries):
        cid = q["condition_id"]
        context = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt_by_condition[cid]},
             {"role": "user", "content": user_prompt_by_condition[cid]}],
            tokenize=True, add_generation_prompt=True,
        )
        prefix = tok(q["prefix"], add_special_tokens=False)["input_ids"]
        span = tok(q["span_text"], add_special_tokens=False)["input_ids"]

        ids_list = list(context) + list(prefix) + list(span)
        if len(ids_list) > cfg.max_length:      # truncate context, never the span
            overflow = len(ids_list) - cfg.max_length
            context = context[overflow:] if overflow < len(context) else context[:0]
            ids_list = list(context) + list(prefix) + list(span)

        labels = [IGNORE_INDEX] * (len(ids_list) - len(span)) + list(span)
        keep = _supervised_positions(labels)

        ids_t = torch.tensor([ids_list], device=device)

        model.zero_grad(set_to_none=True)
        with LoRAGradientCapture(refs, keep_positions=keep) as cap:
            masked_loss(model, ids_t, labels).backward()
            v = proj.project_sample(cap.per_sample_grads(0))

        store[i] = v.detach().cpu().numpy().astype(np.float16)
        index.append({"row": i, "condition_id": cid, "scenario": q["scenario"],
                      "rollout_idx": q["rollout_idx"], "n_span_tokens": len(span),
                      "n_total_tokens": len(ids_list), "grad_norm": float(v.norm())})

        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  query {i+1}/{n}  {rate:.2f}/s", flush=True)

    store.flush()
    meta = {"config": asdict(cfg), "n": n, "dim": dim,
            "projection_fingerprint": spec.fingerprint(refs),
            "elapsed_s": round(time.time() - t0, 1)}
    (out_dir / "qindex.jsonl").write_text("\n".join(json.dumps(r) for r in index))
    (out_dir / "qgrad_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n} query grads to {out_dir}")
    return meta
