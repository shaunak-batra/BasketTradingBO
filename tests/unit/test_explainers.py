"""The README's explanatory figures must build, and the facts they illustrate must be right."""

from __future__ import annotations

import pytest

from src.visualization.explainers import FIGURES, make_all


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return make_all(tmp_path_factory.mktemp("figures"))


def test_every_figure_is_written(built):
    _, paths = built
    assert set(paths) == set(FIGURES)
    for path in paths.values():
        assert path.stat().st_size > 5_000


def test_accounting_figure_reproduces_the_numbers_in_the_audit(built):
    facts, _ = built
    # docs/AUDIT.md: equity moved from 100,000 to 106,570 on entry and returned 99,940 at exit.
    assert facts["v1_peak"] == pytest.approx(106_570.0)
    assert facts["v1_final"] == pytest.approx(99_940.0)
    assert facts["v2_final"] == pytest.approx(99_940.0)


def test_the_synthetic_pair_is_detected_as_cointegrated_with_hedged_weights(built):
    facts, _ = built
    assert facts["cointegration_detected"]
    w_a, w_b = facts["johansen_weights"]
    assert w_a > 0 > w_b
    assert abs(w_a + w_b) < 0.1


def test_psr_thresholds_match_the_closed_form(built):
    facts, _ = built
    # With normal returns PSR = 0.95 needs roughly 1.645 * sqrt(252 / n) of annualised Sharpe.
    assert facts["sharpe_needed_13y"] == pytest.approx(1.645 * (252 / (13 * 252)) ** 0.5, abs=0.01)
    assert facts["sharpe_needed_2y"] == pytest.approx(1.645 * (252 / (2 * 252)) ** 0.5, abs=0.02)
    assert facts["psr_at_0_12_over_13y"] == pytest.approx(0.67, abs=0.02)
