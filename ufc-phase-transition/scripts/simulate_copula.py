"""
Estimate intra-card correlation and compare with copula simulations.
"""
import pandas as pd
import numpy as np
from src.copula import GaussianCopula
import matplotlib.pyplot as plt
import os

PROCESSED_DIR = 'data/processed/'
fights = pd.read_csv(os.path.join(PROCESSED_DIR, 'fights_with_probs.csv'))
fights['DATE'] = pd.to_datetime(fights['DATE'])

# Compute residuals for correlation estimation
fights['residual'] = fights['p_pred'] - (fights['winner'] == fights['fighter1']).astype(int)  # but careful: need consistent favorite

# Estimate average pairwise correlation within events
def event_correlation(df):
    if len(df) < 2:
        return np.nan
    # Use residuals from each fight
    # Simple: average of (res_i * res_j) / (std_i * std_j) for all pairs
    res = df['residual'].values
    std = np.std(res)
    if std == 0:
        return 0
    n = len(res)
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            corrs.append((res[i]*res[j])/(std*std))
    return np.mean(corrs)

rho_estimate = fights.groupby('EVENT').apply(event_correlation).dropna().mean()
print(f"Estimated average intra-card correlation: {rho_estimate:.4f}")

# Now simulate using this rho and compare distribution of upsets
# (Implementation continues...)
