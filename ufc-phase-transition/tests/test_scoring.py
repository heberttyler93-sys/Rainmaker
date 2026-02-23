import pytest
import numpy as np
from src.scoring import ScoringRules


def test_brier_score_perfect():
    # Perfect predictions -> score = 0
    assert ScoringRules.brier_score([1, 0, 1], [1, 0, 1]) == pytest.approx(0.0)


def test_brier_score_baseline():
    # All 0.5 -> score = 0.25
    assert ScoringRules.brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_brier_score_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ScoringRules.brier_score([0.5, 0.5], [1])


def test_log_loss_perfect():
    # Near-perfect prediction should give low loss (clipped at eps)
    ll = ScoringRules.log_loss([1 - 1e-9, 1e-9], [1, 0])
    assert ll < 0.001


def test_log_loss_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ScoringRules.log_loss([0.5], [1, 0])


def test_calibration_curve_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ScoringRules.calibration_curve([0.5, 0.6], [1])


def test_calibration_curve_returns_dataframe():
    import pandas as pd
    result = ScoringRules.calibration_curve([0.3, 0.7, 0.8], [0, 1, 1])
    assert isinstance(result, pd.DataFrame)
    assert "mean_predicted" in result.columns
    assert "actual_win_rate" in result.columns


def test_baseline_brier():
    # baseline_brier([1, 0]) = mean((0.5-1)^2, (0.5-0)^2) = 0.25
    assert ScoringRules.baseline_brier([1, 0]) == pytest.approx(0.25)
