"""Golden values from reference/rss_predictions.py (SPEC section 2).

SPEC: golden values at 2M Monte-Carlo samples with tolerance +-0.02, and closed forms vs MC.
"""
import numpy as np
import pytest

from usintent.theory import (
    Empirical,
    ShiftedExp,
    Uniform,
    bound_median,
    bound_median_closed,
    err_calibrated_imitator,
    err_calibrated_imitator_closed,
    err_memoryless_hedger,
    err_memoryless_hedger_closed,
    hazard,
    hazard_closed,
    pibar,
    pibar_closed,
    pibar_tau,
    pibar_tau_closed,
)

N = 2_000_000
TOL = 0.02
U, X = Uniform(2, 10), ShiftedExp(2, 4)


@pytest.mark.parametrize("dist,bound,imit,hedger", [(U, 2.00, 2.67, 4.75), (X, 2.77, 4.00, 5.16)])
def test_timing_golden(dist, bound, imit, hedger) -> None:
    assert bound_median(dist, n=N) == pytest.approx(bound, abs=TOL)
    assert err_calibrated_imitator(dist, n=N) == pytest.approx(imit, abs=TOL)
    assert err_memoryless_hedger(dist, n=N)[0] == pytest.approx(hedger, abs=TOL)


def test_hedger_leaves_early_golden() -> None:
    assert err_memoryless_hedger(U, n=N)[1] == pytest.approx(0.60, abs=TOL)


def test_uniform_rate_golden() -> None:
    assert pibar(U, 1.0, n=N) == pytest.approx(0.167, abs=TOL)
    assert pibar_tau(U, 1.0, 0.2, n=N) == pytest.approx(0.133, abs=TOL)
    assert pibar_tau(U, 1.0, 0.4, n=N) == pytest.approx(0.100, abs=TOL)
    assert hazard(U, 6.0, 1.0, n=N) == pytest.approx(0.25, abs=TOL)


@pytest.mark.parametrize("e", [2.0, 3.0, 4.0, 6.0, 8.0])
def test_shifted_exp_hazard_constant_after_shift(e) -> None:
    assert hazard(X, e, 1.0, n=N) == pytest.approx(0.221, abs=TOL)


@pytest.mark.parametrize("dist", [U, X])
def test_closed_forms_match_monte_carlo(dist) -> None:
    assert bound_median(dist, n=N) == pytest.approx(bound_median_closed(dist), abs=TOL)
    assert err_calibrated_imitator(dist, n=N) == pytest.approx(
        err_calibrated_imitator_closed(dist), abs=TOL
    )
    mc_err, mc_early = err_memoryless_hedger(dist, n=N)
    cf_err, cf_early = err_memoryless_hedger_closed(dist)
    assert mc_err == pytest.approx(cf_err, abs=TOL)
    assert mc_early == pytest.approx(cf_early, abs=TOL)
    for h in (0.5, 1.0, 2.0, 4.0):
        assert pibar(dist, h, n=N) == pytest.approx(pibar_closed(dist, h), abs=0.005)
    for tau in (0.0, 0.2, 0.4):
        assert pibar_tau(dist, 1.0, tau, n=N) == pytest.approx(
            pibar_tau_closed(dist, 1.0, tau), abs=0.005
        )
    for e in (0.0, 2.5, 5.0, 7.0):
        assert hazard(dist, e, 1.0, n=N) == pytest.approx(hazard_closed(dist, e, 1.0), abs=0.01)


def test_closed_form_exact_values() -> None:
    assert bound_median_closed(U) == 2.0
    assert err_calibrated_imitator_closed(U) == pytest.approx(8 / 3)
    assert err_calibrated_imitator_closed(X) == 4.0
    assert bound_median_closed(X) == pytest.approx(4 * np.log(2))
    assert hazard_closed(X, 5.0, 1.0) == pytest.approx(1 - np.exp(-0.25))
    assert pibar_closed(U, 1.0) == pytest.approx(1 / 6)


def test_distribution_interface() -> None:
    rng = np.random.default_rng(1)
    for dist in (U, X):
        s = dist.sample(400_000, rng)
        assert np.mean(s) == pytest.approx(dist.mean(), abs=0.03)
        assert np.median(s) == pytest.approx(dist.median(), abs=0.03)
        for q in (3.0, 5.0, 9.0):
            assert np.mean(s <= q) == pytest.approx(float(dist.cdf(q)), abs=0.005)
        assert np.mean(np.exp(-s / 6)) == pytest.approx(dist.laplace(1 / 6), abs=0.003)
    emp = Empirical(np.array([1.0, 2.0, 3.0, 4.0]))
    assert emp.median() == 2.5 and emp.mean() == 2.5
    assert float(emp.cdf(2.0)) == 0.5
    with pytest.raises(ValueError):
        Uniform(5, 2)
    with pytest.raises(ValueError):
        Empirical(np.array([]))


def test_empirical_matches_parent() -> None:
    emp = Empirical(U.sample(200_000, np.random.default_rng(3)))
    assert bound_median(emp, n=N) == pytest.approx(2.0, abs=TOL)
    assert pibar(emp, 1.0, n=N) == pytest.approx(1 / 6, abs=0.005)
