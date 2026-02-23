import pytest
import numpy as np
import pandas as pd
from src.utils import parse_bout, time_decay_weights


# --- parse_bout ---

def test_parse_bout_happy_path():
    f1, f2 = parse_bout("Jon Jones vs. Stipe Miocic")
    assert f1 == "Jon Jones"
    assert f2 == "Stipe Miocic"


def test_parse_bout_strips_whitespace():
    f1, f2 = parse_bout("  Conor McGregor  vs.  Dustin Poirier  ")
    assert f1 == "Conor McGregor"
    assert f2 == "Dustin Poirier"


def test_parse_bout_missing_separator():
    with pytest.raises(ValueError, match="Cannot parse bout"):
        parse_bout("Fighter A Fighter B")


def test_parse_bout_none_input():
    with pytest.raises(ValueError, match="non-empty string"):
        parse_bout(None)


def test_parse_bout_empty_string():
    with pytest.raises(ValueError, match="non-empty string"):
        parse_bout("")


def test_parse_bout_multiple_separators():
    # "A vs. B vs. C" splits into 3 parts — should raise
    with pytest.raises(ValueError, match="Cannot parse bout"):
        parse_bout("A vs. B vs. C")


# --- time_decay_weights ---

def test_time_decay_most_recent_is_one():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
    weights = time_decay_weights(dates)
    assert np.isclose(weights.max(), 1.0)
    assert weights.iloc[-1] == pytest.approx(1.0)


def test_time_decay_ordering():
    dates = pd.Series(pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"]))
    weights = time_decay_weights(dates)
    # Older fights should have lower weights
    assert weights.iloc[0] < weights.iloc[1] < weights.iloc[2]


def test_time_decay_all_same_date():
    dates = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-01"]))
    weights = time_decay_weights(dates)
    assert np.allclose(weights, 1.0)
