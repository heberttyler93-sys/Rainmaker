# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
fights = pd.read_csv('../data/processed/fights_with_probs.csv')
fights.head()

# %%
# Explore upset rates, favorite win probability distribution, etc.
