"""Loader for the frozen AM dev / held-out split.

The split is a frozen scientific artifact (see tda/configs/eval_split.yaml).
Nothing outside this module should parse that file, and nothing should ever
write it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "eval_split.yaml"

SplitName = Literal["dev", "held_out"]
SCENARIOS = ("leaking", "murder", "exfiltration")


@dataclass(frozen=True)
class Condition:
    scenario: str
    goal_type: str
    goal_value: str
    urgency_type: str

    @property
    def condition_id(self) -> str:
        """Matches evals/agentic_misalignment/agentic_misalignment.py Sample.id."""
        return f"{self.scenario}_{self.goal_type}-{self.goal_value}_{self.urgency_type}"


def _load() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_split(split: SplitName) -> list[Condition]:
    """Return the conditions in `split`, ordered deterministically."""
    cfg = _load()
    urgency = cfg["urgency_type"]
    out = []
    for scenario in SCENARIOS:
        for entry in cfg[split].get(scenario, []):
            out.append(
                Condition(
                    scenario=scenario,
                    goal_type=entry["goal_type"],
                    goal_value=entry["goal_value"],
                    urgency_type=urgency,
                )
            )
    return out


def load_all() -> list[Condition]:
    """All 27 conditions, dev first then held-out."""
    return load_split("dev") + load_split("held_out")


# Named frozen subsets used by experiments other than the TDA attribution work.
# HANDOFF_AFT.md §7 needs "~9 of 27" fixed before results are seen; the file is
# frozen the same way eval_split.yaml is and is only ever parsed here.
SUBSET_PATH = CONFIG_PATH.parent / "aft_eval_subset.yaml"


def load_subset(name: str) -> list[Condition]:
    """Return the conditions of a named frozen subset (currently only `aft9`)."""
    with open(SUBSET_PATH) as f:
        cfg = yaml.safe_load(f)
    if cfg["name"] != name:
        raise KeyError(f"subset {name!r} unknown; {SUBSET_PATH.name} defines {cfg['name']!r}")
    urgency = cfg["urgency_type"]
    out = []
    for scenario in SCENARIOS:
        for entry in cfg["conditions"].get(scenario, []):
            out.append(Condition(scenario=scenario, goal_type=entry["goal_type"],
                                 goal_value=entry["goal_value"], urgency_type=urgency))
    return out
