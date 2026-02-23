import pytest
import warnings
import numpy as np
from src.bradley_terry import BradleyTerry


def test_sigmoid_extreme_values():
    bt = BradleyTerry()
    # Should not overflow or return NaN
    result = bt._sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.isfinite(result).all()
    assert result[0] < 0.001
    assert np.isclose(result[1], 0.5)
    assert result[2] > 0.999


def test_win_probability_unseen_fighter():
    bt = BradleyTerry()
    fights = [("A", "B"), ("A", "C")]
    bt.fit(fights)
    # Unknown fighters default to rating 0.0
    prob = bt.win_probability("A", "UNKNOWN")
    assert 0.0 <= prob <= 1.0


def test_win_probability_symmetry():
    bt = BradleyTerry()
    bt.fit([("A", "B")] * 5)
    p_ab = bt.win_probability("A", "B")
    p_ba = bt.win_probability("B", "A")
    assert p_ab + p_ba == pytest.approx(1.0)


def test_leaderboard_columns():
    bt = BradleyTerry()
    bt.fit([("A", "B"), ("A", "C"), ("B", "C")])
    lb = bt.leaderboard()
    assert "Fighter" in lb.columns
    assert "Beta" in lb.columns
    assert "Win% vs avg" in lb.columns


def test_leaderboard_ordering():
    bt = BradleyTerry()
    bt.fit([("A", "B")] * 10)
    lb = bt.leaderboard()
    # A should be ranked above B
    rank_a = lb[lb["Fighter"] == "A"].index[0]
    rank_b = lb[lb["Fighter"] == "B"].index[0]
    assert rank_a < rank_b
