"""M6: DP arm configs validate against the installed LeRobot (10-step dry run), gating."""
import numpy as np
import pytest

from usintent.config import load
from usintent.data.chunks import ChunkSampler
from usintent.policies.dp_adapter import DPPolicy, arm_spec_from_config
from usintent.policies.gr00t_adapter import GR00TPolicy
from usintent.sim.holdworld import HoldWorldConfig, generate


@pytest.fixture(scope="module")
def sampler():
    eps = generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3), seed=1), 4)
    return ChunkSampler(eps, 1.0, history_k=load("data")["history_frames_k"])


@pytest.mark.parametrize("arm", ["DP-1", "DP-5"])
def test_dry_run_10_steps(sampler, arm) -> None:
    cfg = load(f"arms/{arm}")
    spec = arm_spec_from_config(cfg, 10, (32, 32), sampler.history_k)
    pol = DPPolicy(spec, seed=0)
    losses = pol.train_steps(sampler, 10, 4, 1e-4, np.random.default_rng(0))
    assert len(losses) == 10 and all(np.isfinite(losses))
    h = sampler.history(0, 40)
    chunk, m = pol.predict(h, "HOLD" if pol.intent_aware else None)
    assert chunk.shape == (10, 5) and m in ("HOLD", "SWEEP")
    again, _ = pol.predict(h, "HOLD" if pol.intent_aware else None)
    assert np.allclose(chunk, again)                      # deterministic initial noise


def test_intent_gating_dp(sampler) -> None:
    blind = DPPolicy(arm_spec_from_config(load("arms/DP-1"), 10, (32, 32), 3), seed=0)
    aware = DPPolicy(arm_spec_from_config(load("arms/DP-5"), 10, (32, 32), 3), seed=0)
    blind.train_steps(sampler, 1, 2, 1e-4, np.random.default_rng(0))
    aware.train_steps(sampler, 1, 2, 1e-4, np.random.default_rng(0))
    h = sampler.history(0, 40)
    with pytest.raises(PermissionError):
        blind.predict(h, "HOLD")
    with pytest.raises(ValueError):
        aware.predict(h, None)
    assert not blind.spec.use_mode and aware.spec.use_mode


def test_all_blind_arm_configs_have_no_mode_input() -> None:
    for arm in ["A1", "A2", "A3", "A6", "A7", "DP-1"]:
        cfg = load(f"arms/{arm}")
        assert not cfg["intent_aware"] and "mode_embedding" not in cfg["inputs"]
    for arm in ["A5", "DP-5"]:
        assert load(f"arms/{arm}")["intent_aware"]


def test_gr00t_adapter_refuses_honestly() -> None:
    with pytest.raises(NotImplementedError, match="not readable"):
        GR00TPolicy("A1")
