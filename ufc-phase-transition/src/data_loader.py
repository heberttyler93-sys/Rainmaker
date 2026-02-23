"""
Load and clean raw data files.
"""
import warnings
import pandas as pd
from .utils import parse_bout

def load_fight_results(path):
    df = pd.read_csv(path, sep='\t')
    # Parse bout
    df[['fighter1', 'fighter2']] = df['BOUT'].apply(lambda x: pd.Series(parse_bout(x)))
    # Keep only decisive outcomes; discard draws, no-contests, and unknowns
    decisive = df['OUTCOME'].isin(['W/L', 'L/W'])
    n_dropped = (~decisive).sum()
    if n_dropped > 0:
        warnings.warn(
            f"Dropping {n_dropped} row(s) with non-decisive OUTCOME "
            f"(draws, no-contests, or unexpected values)."
        )
    df = df[decisive].copy()
    # Assign winner/loser based on outcome direction
    df['winner'] = df.apply(
        lambda row: row['fighter1'] if row['OUTCOME'] == 'W/L' else row['fighter2'], axis=1
    )
    df['loser'] = df.apply(
        lambda row: row['fighter2'] if row['OUTCOME'] == 'W/L' else row['fighter1'], axis=1
    )
    return df

def load_event_details(path):
    df = pd.read_csv(path, sep='\t')
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

def load_fighter_stats(path):
    df = pd.read_csv(path, sep='\t')
    # Rename columns if needed (sample had FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB)
    return df

def load_round_stats(path):
    df = pd.read_csv(path, sep='\t')
    return df

def merge_all(fight_results, event_details, fighter_stats=None):
    # Merge fight results with event details to get date
    merged = fight_results.merge(event_details[['EVENT', 'DATE']], on='EVENT', how='left')
    # Warn and drop rows where no matching event was found
    missing_dates = merged['DATE'].isna()
    if missing_dates.any():
        missing_events = merged.loc[missing_dates, 'EVENT'].unique().tolist()
        warnings.warn(
            f"{missing_dates.sum()} fight(s) have no matching event date and will be dropped. "
            f"Unmatched events: {missing_events}"
        )
        merged = merged[~missing_dates].copy()
    # Optionally merge fighter stats (requires matching fighter names)
    return merged
