"""Guards on the frozen AM split.

These assert the stratification claims written into eval_split.yaml. If someone
edits the split, these fail loudly — which is the point.
"""

from collections import Counter

import pytest
import yaml

from tda.evals.split import CONFIG_PATH, SCENARIOS, load_all, load_split


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_grid_is_complete_and_disjoint():
    """dev + held_out == the full 27-condition grid, with no overlap."""
    dev = {c.condition_id for c in load_split("dev")}
    held = {c.condition_id for c in load_split("held_out")}

    assert len(dev) == 14
    assert len(held) == 13
    assert not (dev & held), f"leak between dev and held_out: {dev & held}"

    goal_conditions = [("explicit", v) for v in (
        "america", "global", "safety", "acceleration",
        "ethical", "pragmatic", "individualism", "collectivism",
    )] + [("none", "none")]
    expected = {
        f"{s}_{gt}-{gv}_replacement"
        for s in SCENARIOS
        for gt, gv in goal_conditions
    }
    assert dev | held == expected


def test_balance_matches_declared_check(cfg):
    """The balance_check block in the yaml must describe the actual split."""
    declared = cfg["balance_check"]

    for split in ("dev", "held_out"):
        conds = load_split(split)
        key = "n_dev" if split == "dev" else "n_held_out"
        assert len(conds) == declared[key]

        by_scenario = Counter(c.scenario for c in conds)
        assert by_scenario == Counter(declared[f"{split}_by_scenario"])

        by_goal = Counter(c.goal_value for c in conds)
        assert by_goal == Counter(declared[f"{split}_by_goal_value"])


def test_every_goal_value_appears_on_both_sides():
    dev = {c.goal_value for c in load_split("dev")}
    held = {c.goal_value for c in load_split("held_out")}
    assert dev == held, "a goal value is confined to one side of the split"


def test_opposed_pairs_are_split_within_each_scenario(cfg):
    """No scenario may put both sides of a value conflict on the same side."""
    pairs = {k: v for k, v in cfg["goal_pairs"].items() if len(v) == 2}

    dev_by_scenario = {s: set() for s in SCENARIOS}
    for c in load_split("dev"):
        dev_by_scenario[c.scenario].add(c.goal_value)

    for scenario, dev_values in dev_by_scenario.items():
        for pair_name, (a, b) in pairs.items():
            in_dev = {a, b} & dev_values
            assert len(in_dev) == 1, (
                f"{scenario}/{pair_name}: expected exactly one of {a},{b} in dev, "
                f"got {in_dev or 'neither'}"
            )


def test_urgency_type_is_fixed():
    assert {c.urgency_type for c in load_all()} == {"replacement"}
