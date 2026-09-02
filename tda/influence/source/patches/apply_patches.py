"""Patch the installed bergson for the SOURCE x PEFT path.

Applied at Modal image build time. Every patch asserts on the exact source text
it expects, so a bergson upgrade fails the build loudly rather than silently
running unpatched — which would produce plausible-looking but wrong scores.

Run:  python -m tda.influence.source.patches.apply_patches [--check]
"""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_BERGSON = "0.26.2"


# ---------------------------------------------------------------------------
# PATCH 1 — adam_preconditioner: reference model must tolerate PEFT adapters.
#
# `build_segment_preconditioners` does:
#     config = AutoConfig.from_pretrained(checkpoints[0])
#     reference_model = AutoModelForCausalLM.from_config(config)
# With LoRA, checkpoints[0] is an adapter dir whose config.json carries no
# `model_type`, so this raises:
#     ValueError: Unrecognized model in .../checkpoint-4
# killing SOURCE at pipeline step 5/8 (verified on Qwen2.5-0.5B + LoRA r=64).
#
# The reference model is only needed (a) as a FALLBACK index->name mapping when
# optimizer.pt entries carry no `param_name`, and (b) for a `get_submodule`
# lookup used to disambiguate square second-moment grids. bergson's own trainer
# always records `param_name` (load_from_optimizer.py::
# save_second_moments_as_optimizer_pt), and LoRA factors are non-square
# ([r, d_in] and [d_out, r]), so orientation is unambiguous without the layer.
# `optimizer_param_index_to_name`'s own docstring says it is "for a non-PEFT
# model", so there is no correct index mapping to build here anyway.
#
# We therefore skip the reference model on adapter checkpoints and require the
# recorded param_name, raising clearly if it is absent.
# ---------------------------------------------------------------------------

P1_OLD = """    # For the optimizer.pt index mapping (used when entries carry no
    # param_name, e.g. HF Trainer checkpoints).
    config = AutoConfig.from_pretrained(checkpoints[0])
    reference_model = AutoModelForCausalLM.from_config(config)
"""

P1_NEW = '''    # For the optimizer.pt index mapping (used when entries carry no
    # param_name, e.g. HF Trainer checkpoints).
    # [msm-tda patch] A PEFT adapter dir has no `model_type` in config.json, so
    # AutoConfig.from_pretrained raises there. The reference model is only a
    # fallback index->name map plus a get_submodule lookup for orienting square
    # grids; bergson's trainer records `param_name`, and LoRA grids are
    # non-square, so neither is needed on the adapter path.
    if (Path(checkpoints[0]) / "adapter_config.json").is_file():
        reference_model = None
    else:
        config = AutoConfig.from_pretrained(checkpoints[0])
        reference_model = AutoModelForCausalLM.from_config(config)
'''

P2_OLD = """            optimizer_pt = load_optimizer(str(ckpt))
            beta2, eps, eps_root = _adam_hparams(optimizer_pt)
            index_to_name = optimizer_param_index_to_name(optimizer_pt, reference_model)
"""

P2_NEW = """            optimizer_pt = load_optimizer(str(ckpt))
            beta2, eps, eps_root = _adam_hparams(optimizer_pt)
            # [msm-tda patch] No reference model on the PEFT path; entries must
            # carry param_name (they do when written by bergson's trainer).
            if reference_model is None:
                if not any(
                    isinstance(e, dict) and e.get("param_name")
                    for e in optimizer_pt["state"].values()
                ):
                    raise ValueError(
                        f"{ckpt}/optimizer.pt records no `param_name` and the "
                        "checkpoint is a PEFT adapter, so indices cannot be "
                        "mapped to names. Re-export with a bergson trainer run "
                        "(TrainingConfig.save_optimizer_state)."
                    )
                index_to_name = {}
            else:
                index_to_name = optimizer_param_index_to_name(
                    optimizer_pt, reference_model
                )
"""

P3_OLD = """        for module, param_name in matches.items():
            try:
                layer = reference_model.get_submodule(
                    param_name.removesuffix(".weight")
                )
            except AttributeError:
                layer = None
"""

P3_NEW = """        for module, param_name in matches.items():
            try:
                # [msm-tda patch] reference_model is None on the PEFT path.
                layer = (
                    None
                    if reference_model is None
                    else reference_model.get_submodule(param_name.removesuffix(".weight"))
                )
            except AttributeError:
                layer = None
"""

PATCHES = [
    ("bergson/approx_unrolling/adam_preconditioner.py",
     [(P1_OLD, P1_NEW), (P2_OLD, P2_NEW), (P3_OLD, P3_NEW)]),
]


def _bergson_root() -> Path:
    import bergson
    return Path(bergson.__file__).resolve().parent.parent


def main(check_only: bool = False) -> int:
    import bergson

    version = getattr(bergson, "__version__", "unknown")
    if version != EXPECTED_BERGSON:
        print(f"WARNING: bergson {version} != pinned {EXPECTED_BERGSON}; "
              "re-verify these patches against the new source.", file=sys.stderr)

    root = _bergson_root()
    applied, already = 0, 0
    for rel, subs in PATCHES:
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"patch target missing: {path}")
        text = path.read_text()
        for old, new in subs:
            if new in text:
                already += 1
                continue
            if old not in text:
                raise SystemExit(
                    f"PATCH FAILED in {rel}: expected source text not found.\n"
                    "bergson changed underneath this patch. Re-derive it "
                    "against the new source rather than skipping it — running "
                    "unpatched breaks SOURCE at pipeline step 5/8 for PEFT."
                )
            text = text.replace(old, new, 1)
            applied += 1
        if not check_only:
            path.write_text(text)

    print(f"bergson patches: {applied} applied, {already} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
