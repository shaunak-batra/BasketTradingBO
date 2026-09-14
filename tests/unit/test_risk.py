"""Value-at-Risk and Expected Shortfall tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.risk.var import (
    cornish_fisher_is_monotone,
    cornish_fisher_quantile,
    cornish_fisher_var,
    historical_var,
    parametric_var,
    risk_table,
)


def test_historical_var_and_es_on_a_known_sample():
    returns = (np.arange(100) - 50) / 1000.0  # -0.050 ... 0.049
    estimate = historical_var(returns, 0.95)
    # The linear 5% quantile sits 95% of the way from -0.046 to -0.045.
    assert estimate.var == pytest.approx(0.04505)
    # Returns at or below that quantile are -0.050 ... -0.046, with mean -0.048.
    assert estimate.expected_shortfall == pytest.approx(0.048)


def test_parametric_var_and_es_match_the_gaussian_formulas():
    returns = np.random.default_rng(0).normal(0.0005, 0.01, 1000)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(0.05)
    estimate = parametric_var(returns, 0.95)
    assert estimate.var == pytest.approx(-(mu + sigma * z))
    assert estimate.expected_shortfall == pytest.approx(-mu + sigma * stats.norm.pdf(z) / 0.05)
    assert estimate.expected_shortfall > estimate.var > 0


def test_cornish_fisher_reduces_to_the_normal_quantile_without_higher_moments():
    assert cornish_fisher_quantile(-1.645, 0.0, 0.0) == pytest.approx(-1.645)


def test_negative_skew_pushes_the_cornish_fisher_quantile_further_out():
    assert cornish_fisher_quantile(-1.645, -1.0, 0.0) < -1.645


def test_higher_confidence_gives_a_larger_var():
    fat_tailed = np.random.default_rng(1).standard_t(4, 2000) * 0.01
    assert historical_var(fat_tailed, 0.99).var > historical_var(fat_tailed, 0.95).var
    near_normal = np.random.default_rng(1).normal(0.0, 0.01, 2000)
    assert cornish_fisher_var(near_normal, 0.99).var > cornish_fisher_var(near_normal, 0.95).var


def test_cornish_fisher_expansion_is_monotone_only_for_moderate_moments():
    assert cornish_fisher_is_monotone(0.0, 0.0)
    # Moments like those of a mostly flat strategy return series (XOM/CVX fixed: skew -2.7, excess kurtosis 314).
    assert not cornish_fisher_is_monotone(-2.7, 314.0)


def test_cornish_fisher_var_is_nan_where_the_expansion_is_not_valid():
    estimate = cornish_fisher_var(np.random.default_rng(1).standard_t(4, 2000) * 0.01, 0.99)
    assert math.isnan(estimate.var)
    assert estimate.expected_shortfall is None


def test_a_zero_historical_var_is_positive_zero():
    estimate = historical_var(np.r_[np.zeros(97), np.full(3, 0.01)], 0.95)
    assert estimate.var == 0.0
    assert math.copysign(1.0, estimate.var) == 1.0


def test_risk_table_covers_every_method_and_level():
    table = risk_table(np.random.default_rng(2).normal(0.0, 0.01, 500))
    assert [(estimate.method, estimate.confidence) for estimate in table] == [
        (method, confidence)
        for confidence in (0.95, 0.99)
        for method in ("historical", "parametric", "cornish_fisher")
    ]


@pytest.mark.parametrize("confidence", [0.5, 1.0, 1.2])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        historical_var(np.zeros(100), confidence)


def test_too_few_observations_are_rejected():
    with pytest.raises(ValueError):
        parametric_var(np.zeros(29))
