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

# ---------------------------------------------------------------------------
# PATCH 2 — per-segment Hessian data for genuine multi-stage attribution.
#
# SOURCE's H_l is the Hessian of the objective TRAINED IN SEGMENT l (Bae et al.
# Eq 15/22). For a D1->D2 pipeline the multi-stage estimator is
# -(1/N_1) * S_2 * r_1: `r_1` needs segment 1's statistics and `S_2` the
# pullback through segment 2. The two segments therefore need DIFFERENT data.
#
# bergson takes a single `index_cfg.data` and uses it at every checkpoint, so
# the AFT segment's Hessian would be estimated from midtraining documents — an
# approximation beyond SOURCE's own assumptions. Feeding a union corpus does not
# fix it: that makes BOTH segments mixture-estimated instead of one right and
# one wrong.
#
# The fix is small because both precompute functions already derive `seg` and
# already deepcopy the config. Scoring is deliberately left alone: it must keep
# using index_cfg.data, the corpus being attributed, at every checkpoint.
#
# `segment_datasets` is list[str] rather than list[DataConfig] so it round-trips
# through YAML without nested-dataclass deserialization.
# ---------------------------------------------------------------------------

P4_OLD = '''    query_batch_size: int | None = None
    """Batch size for per-segment query scoring (see
    ScoreConfig.query_batch_size)."""
'''

P4_NEW = '''    query_batch_size: int | None = None
    """Batch size for per-segment query scoring (see
    ScoreConfig.query_batch_size)."""

    segment_datasets: list[str] = field(default_factory=list)
    """[msm-tda patch] One dataset path per segment, used ONLY for that
    segment's Hessian/covariance and lambda estimation. Empty list keeps the
    original behaviour (index data at every checkpoint).

    SOURCE defines H_l on the objective trained in segment l, so a multi-stage
    run needs different data per segment; a single dataset silently estimates
    one stage's curvature from the other stage's data. Per-example gradient
    scoring is unaffected and still uses index_cfg.data."""
'''

_SEG_OVERRIDE = '''
        # [msm-tda patch] Per-segment Hessian data: SOURCE's H_l is the Hessian
        # of the objective trained in THIS segment, not of the corpus being
        # attributed.
        if getattr(approx_unrolling_cfg, "segment_datasets", None):
            ckpt_index_cfg.data = deepcopy(index_cfg.data)
            ckpt_index_cfg.data.dataset = (
                approx_unrolling_cfg.segment_datasets[seg]
            )
'''

P5_OLD = '''        ckpt_index_cfg = deepcopy(index_cfg)
        ckpt_index_cfg.run_path = str(out_path)
        ckpt_index_cfg.model = ckpt
'''
P5_NEW = P5_OLD + _SEG_OVERRIDE

P6_OLD = '''        ckpt_index_cfg = deepcopy(index_cfg)
        ckpt_index_cfg.model = ckpt
'''
P6_NEW = P6_OLD + _SEG_OVERRIDE


PATCHES = [
    ("bergson/approx_unrolling/adam_preconditioner.py",
     [(P1_OLD, P1_NEW), (P2_OLD, P2_NEW), (P3_OLD, P3_NEW)]),
    ("bergson/config/config.py", [(P4_OLD, P4_NEW)]),
    ("bergson/approx_unrolling/precompute_checkpoints.py",
     [(P5_OLD, P5_NEW), (P6_OLD, P6_NEW)]),
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
