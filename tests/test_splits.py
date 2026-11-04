"""Session/day splits (SPEC section 4, CLAUDE.md rule 5)."""
import pandas as pd
import pytest

from usintent.data.splits import (
    SplitLeakageError,
    assert_disjoint,
    assign_splits,
    check_no_leakage,
    episode_split_table,
    load_assignment,
    save_assignment,
)
from usintent.sim.holdworld import HoldWorldConfig, generate


def test_assignment_deterministic_and_complete(tmp_path) -> None:
    days = [f"d{i}" for i in range(10)]
    a = assign_splits(days, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=3)
    assert a == assign_splits(reversed(days), {"train": 0.6, "val": 0.2, "test": 0.2}, seed=3)
    assert set(a) == set(days) and sorted(a.values()).count("train") == 6
    save_assignment(a, tmp_path / "s.yaml")
    assert load_assignment(tmp_path / "s.yaml") == a


def test_leakage_fails_loudly() -> None:
    ok = pd.DataFrame({"session_id": ["s1", "s1", "s2"], "day": ["d1", "d1", "d2"],
                       "split": ["train", "train", "test"]})
    check_no_leakage(ok)
    bad = ok.assign(split=["train", "test", "test"])
    with pytest.raises(SplitLeakageError, match="s1"):
        check_no_leakage(bad)
    bad_day = pd.DataFrame({"session_id": ["s1", "s2"], "day": ["d1", "d1"], "split": ["train", "val"]})
    with pytest.raises(SplitLeakageError, match="d1"):
        check_no_leakage(bad_day)


def test_episode_table_from_synthetic_sessions() -> None:
    eps = generate(HoldWorldConfig(render_images=False, n_stations=(2, 2)), 12, sessions=6, days=3)
    a = assign_splits([e.day for e in eps], {"train": 0.34, "val": 0.33, "test": 0.33}, seed=0)
    t = episode_split_table(eps, a)
    assert t.groupby("session_id")["split"].nunique().max() == 1
    assert set(t["split"]) == {"train", "val", "test"}


def test_assert_disjoint_train_test() -> None:
    a = generate(HoldWorldConfig(render_images=False, n_stations=(2, 2), seed=1), 2)
    b = generate(HoldWorldConfig(render_images=False, n_stations=(2, 2), seed=2), 2)
    assert_disjoint(a, b)                                  # different seeds -> different sessions
    with pytest.raises(SplitLeakageError, match="session"):
        assert_disjoint(a, a[:1])
