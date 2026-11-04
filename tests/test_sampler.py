"""M6 intervention sampler (SPEC section 8). SYNTHETIC data."""
import numpy as np
import pytest

from usintent.data.chunks import ChunkSampler
from usintent.intervene.sampler import (
    epoch_indices,
    intervention_weights,
    natural_departure_fraction,
    realized_fraction,
)
from usintent.sim.holdworld import HoldWorldConfig, generate


@pytest.fixture(scope="module")
def sampler():
    return ChunkSampler(generate(HoldWorldConfig(render_images=False, seed=3), 120), 1.0)


@pytest.mark.parametrize("mult", [0.5, 2.0])
def test_realized_fraction_within_001_in_one_epoch(sampler, mult) -> None:
    nat = natural_departure_fraction(sampler.index)
    target = mult * nat
    iw = intervention_weights(sampler.index, target)
    picks = epoch_indices(iw.weights, np.random.default_rng(0))
    assert realized_fraction(sampler.index, picks) == pytest.approx(target, abs=0.01)
    assert iw.natural_fraction == pytest.approx(nat)


def test_no_deletion_and_mass_preserved(sampler) -> None:
    idx = sampler.index
    iw = intervention_weights(idx, 0.3)
    assert np.all(iw.weights > 0)                                  # nothing deleted
    hold = (idx["phase"] == "hold").to_numpy()
    assert iw.weights[hold].sum() == pytest.approx(hold.sum())     # hold mass unchanged
    assert np.all(iw.weights[~hold] == 1.0)                        # other chunks natural
    with pytest.raises(ValueError):
        intervention_weights(idx, 1.0)
