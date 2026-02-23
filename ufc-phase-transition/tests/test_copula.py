import pytest
import numpy as np
from src.copula import GaussianCopula

def test_simulate_card():
    cop = GaussianCopula(3)
    probs = [0.6, 0.7, 0.8]
    outcomes = cop.simulate_card(probs, 100)
    assert outcomes.shape == (100, 3)
    assert outcomes.dtype == int

def test_joint_prob():
    cop = GaussianCopula(2)
    probs = [0.5, 0.5]
    p = cop.joint_probability(probs, [1,1], 10000)
    assert 0.2 <= p <= 0.3  # approx 0.25
