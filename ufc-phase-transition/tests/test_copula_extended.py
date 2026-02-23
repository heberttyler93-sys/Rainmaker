import pytest
import numpy as np
from src.copula import GaussianCopula


def test_init_default_identity():
    cop = GaussianCopula(3)
    np.testing.assert_array_equal(cop.R, np.eye(3))


def test_init_valid_custom_matrix():
    R = [[1.0, 0.3], [0.3, 1.0]]
    cop = GaussianCopula(2, R)
    assert cop.R.shape == (2, 2)


def test_init_wrong_shape():
    with pytest.raises(ValueError, match="must be"):
        GaussianCopula(3, [[1.0, 0.0], [0.0, 1.0]])


def test_init_non_symmetric():
    with pytest.raises(ValueError, match="symmetric"):
        GaussianCopula(2, [[1.0, 0.5], [0.1, 1.0]])


def test_init_diagonal_not_one():
    with pytest.raises(ValueError, match="diagonal"):
        GaussianCopula(2, [[0.9, 0.0], [0.0, 1.0]])


def test_simulate_card_not_positive_definite():
    # Construct a non-PD matrix manually (bypassing __init__ validation)
    cop = GaussianCopula(2)
    cop.R = np.array([[-1.0, 0.0], [0.0, -1.0]])  # clearly not PD
    with pytest.raises(ValueError, match="positive-definite"):
        cop.simulate_card([0.6, 0.7])


def test_simulate_card_output_shape():
    cop = GaussianCopula(4)
    outcomes = cop.simulate_card([0.6, 0.7, 0.5, 0.8], n_simulations=500)
    assert outcomes.shape == (500, 4)


def test_simulate_card_binary_outcomes():
    cop = GaussianCopula(3)
    outcomes = cop.simulate_card([0.6, 0.7, 0.8], n_simulations=200)
    assert set(np.unique(outcomes)).issubset({0, 1})


def test_expected_correct_range():
    cop = GaussianCopula(2)
    expected = cop.expected_correct([0.6, 0.7], n_simulations=5000)
    # Expected win rates should be close to the input probabilities
    assert 0.5 <= expected[0] <= 0.75
    assert 0.55 <= expected[1] <= 0.85


def test_correlation_effect_returns_dataframe():
    import pandas as pd
    cop = GaussianCopula(3)
    df = cop.correlation_effect([0.6, 0.7, 0.8], rho_values=[0.0, 0.2], n_simulations=1000)
    assert isinstance(df, pd.DataFrame)
    assert "rho" in df.columns
    assert len(df) == 2
