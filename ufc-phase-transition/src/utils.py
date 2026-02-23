"""
Helper utilities.
"""
import pandas as pd
import numpy as np
from datetime import datetime

def parse_bout(bout):
    """Split 'Fighter A vs. Fighter B' into two names."""
    if not isinstance(bout, str) or not bout:
        raise ValueError(f"parse_bout expected a non-empty string, got {bout!r}.")
    parts = bout.split(' vs. ')
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse bout {bout!r}: expected exactly one ' vs. ' separator, "
            f"found {len(parts) - 1}."
        )
    return parts[0].strip(), parts[1].strip()

def time_decay_weights(dates, decay_half_life_days=365):
    """
    Create weights for fights based on recency.
    Most recent fight gets weight 1, older fights decay exponentially.
    """
    if isinstance(dates, pd.Series):
        dates = pd.to_datetime(dates)
    else:
        dates = pd.to_datetime(dates)
    max_date = dates.max()
    days_old = (max_date - dates).dt.days
    weights = np.exp(-days_old / decay_half_life_days * np.log(2))
    return weights

def rolling_train_test_split(results, train_cutoff_date):
    """Split results into training (before cutoff) and testing (after)."""
    train = results[results['DATE'] < train_cutoff_date]
    test = results[results['DATE'] >= train_cutoff_date]
    return train, test
