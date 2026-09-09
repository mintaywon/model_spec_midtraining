"""AM rollout generation via vLLM + LoRA adapters.

Portable: no Modal imports. tda/modal/app.py calls run_generation().

Output layout (under out_dir):
    prompts.jsonl      one record per condition (system/user/email text, stored once)
    transcripts.jsonl  one record per rollout, referencing condition_id
    meta.json          run configuration

Prompt text is stored once per condition rather than per rollout — at 100 rollouts
x ~6k-token email dumps the duplication would be ~1.6GB of redundant JSON per
checkpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from tda.evals.split import Condition

# The paper's README gives a full example invocation with --temperature 0.7 and
# --max-tokens 4096; the in-repo example config instead shows temperature 1.0.
# We default to the README values and flag this as an assumption the reproduction
# gate tests. See CLAUDE.md §2b(2) and the P2 author ask in inventory.md §10.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL_NAME = "Qwen"  # name the AI is given in-prompt; README uses -T model_name=Qwen


@dataclass
class GenerationConfig:
    base_model: str
    adapter_repo: str | None      # None -> plain base model, no LoRA
    cell: str                     # registry key, e.g. "msm_Vp__aft_Vp"
    n_rollouts: int
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    model_name: str = DEFAULT_MODEL_NAME
    prod: bool = False            # False keeps the reasoning scratchpad (CLAUDE.md §4.1)
    max_model_len: int = 16384
    seed: int = 0
    tensor_parallel_size: int = 1


def build_prompts(conditions: list[Condition], model_name: str, prod: bool) -> list[dict]:
    """Render each condition's system/user/email text via the repo's generator."""
    from evals.agentic_misalignment.prompt_generator import generate_prompts

    out = []
    for c in conditions:
        p = generate_prompts(
            scenario=c.scenario,
            goal_type=c.goal_type,
            goal_value=c.goal_value,
            urgency_type=c.urgency_type,
            extra_system_instructions=None,
            prod_flag=prod,
            model_name=model_name,
        )
        out.append(
            {
                "condition_id": c.condition_id,
                "scenario": c.scenario,
                "goal_type": c.goal_type,
                "goal_value": c.goal_value,
                "urgency_type": c.urgency_type,
                "system_prompt": p.system_prompt,
                "user_prompt": p.user_prompt,
                "email_content": p.email_content,
            }
        )
    return out


def _resolve_adapter(adapter_repo: str | None) -> str | None:
    if adapter_repo is None:
        return None
    from huggingface_hub import snapshot_download

    return snapshot_download(adapter_repo, token=os.environ.get("HF_TOKEN"))


def run_generation(
    cfg: GenerationConfig,
    conditions: list[Condition],
    out_dir: str | Path,
) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts(conditions, cfg.model_name, cfg.prod)
    with open(out_dir / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    adapter_path = _resolve_adapter(cfg.adapter_repo)

    llm = LLM(
        model=cfg.base_model,
        enable_lora=adapter_path is not None,
        max_lora_rank=64,                      # released adapters are r=64
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        seed=cfg.seed,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        # vLLM's custom all-reduce kernel needs peer-to-peer access between the
        # assigned GPUs, and which pair Modal hands out is a placement lottery:
        # the same 2-GPU config that produced the 810-rollout `phil` run later
        # died at engine start with
        #   Cuda error custom_all_reduce.cuh:453 'invalid argument'
        # Falling back to NCCL costs a little throughput and removes a failure
        # mode that only appears on some containers, which is the worst kind.
        disable_custom_all_reduce=cfg.tensor_parallel_size > 1,
    )

    # n>1 per prompt rather than duplicating prompts: one prefill, many samples.
    sampling = SamplingParams(
        n=cfg.n_rollouts,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
    )

    conversations = [
        [
            {"role": "system", "content": p["system_prompt"]},
            {"role": "user", "content": p["user_prompt"] + "\n" + p["email_content"]},
        ]
        for p in prompts
    ]

    lora_request = (
        LoRARequest("adapter", 1, adapter_path) if adapter_path is not None else None
    )
    outputs = llm.chat(conversations, sampling, lora_request=lora_request)

    n_written = 0
    n_truncated = 0
    with open(out_dir / "transcripts.jsonl", "w") as f:
        for p, out in zip(prompts, outputs):
            for i, comp in enumerate(out.outputs):
                truncated = comp.finish_reason == "length"
                n_truncated += truncated
                f.write(
                    json.dumps(
                        {
                            "cell": cfg.cell,
                            "condition_id": p["condition_id"],
                            "scenario": p["scenario"],
                            "goal_type": p["goal_type"],
                            "goal_value": p["goal_value"],
                            "rollout_idx": i,
                            "response": comp.text,
                            "finish_reason": comp.finish_reason,
                            "n_prompt_tokens": len(out.prompt_token_ids),
                            "n_completion_tokens": len(comp.token_ids),
                        }
                    )
                    + "\n"
                )
                n_written += 1

    meta = {
        "config": asdict(cfg),
        "adapter_path": adapter_path,
        "n_conditions": len(conditions),
        "n_transcripts": n_written,
        "n_truncated": n_truncated,
        "max_prompt_tokens": max(len(o.prompt_token_ids) for o in outputs),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {n_written} transcripts to {out_dir}")
    print(f"  max prompt tokens: {meta['max_prompt_tokens']} (limit {cfg.max_model_len})")
    print(f"  truncated completions: {n_truncated}/{n_written}")
    return meta
