"""Golden values from reference/rss_predictions.py (SPEC section 2)."""
import pytest

from usintent.theory import (
    ShiftedExp,
    Uniform,
    bound_median,
    err_calibrated_imitator,
    err_memoryless_hedger,
    hazard,
    pibar,
    pibar_tau,
)

N = 400_000
U, X = Uniform(2, 10), ShiftedExp(2, 4)


@pytest.mark.parametrize("dist,bound,imit,hedger", [(U, 2.00, 2.67, 4.75), (X, 2.77, 4.00, 5.16)])
def test_timing_predictions(dist, bound, imit, hedger) -> None:
    assert bound_median(dist, n=N) == pytest.approx(bound, abs=0.03)
    assert err_calibrated_imitator(dist, n=N) == pytest.approx(imit, abs=0.03)
    assert err_memoryless_hedger(dist, n=N)[0] == pytest.approx(hedger, abs=0.04)


def test_uniform_closed_forms() -> None:
    assert bound_median(U, n=N) == pytest.approx((10 - 2) / 4, abs=0.02)
    assert err_calibrated_imitator(U, n=N) == pytest.approx((10 - 2) / 3, abs=0.02)


def test_hedging_rates() -> None:
    assert pibar(U, 1.0, n=N) == pytest.approx(1 / 6, abs=0.005)
    assert pibar_tau(U, 1.0, 0.2, n=N) == pytest.approx(0.133, abs=0.005)
    assert pibar_tau(U, 1.0, 0.4, n=N) == pytest.approx(0.100, abs=0.005)
    assert hazard(U, 6.0, 1.0, n=N) == pytest.approx(0.25, abs=0.01)
    assert hazard(X, 4.0, 1.0, n=N) == pytest.approx(0.221, abs=0.01)
