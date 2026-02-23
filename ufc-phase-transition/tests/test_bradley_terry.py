import pytest
import numpy as np
from src.bradley_terry import BradleyTerry

def test_fit():
    bt = BradleyTerry()
    fights = [("A","B"), ("A","C"), ("B","C")]
    bt.fit(fights)
    assert bt.ratings["A"] > bt.ratings["B"] > bt.ratings["C"]
    prob = bt.win_probability("A","B")
    assert 0 <= prob <= 1

def test_identifiability():
    bt = BradleyTerry()
    fights = [("A","B")]*10
    bt.fit(fights)
    assert np.isclose(sum(bt.ratings.values()), 0)
