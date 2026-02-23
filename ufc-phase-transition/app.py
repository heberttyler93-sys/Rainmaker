"""
UFC Phase Transition — No-Code Web Interface

A Streamlit app that wraps the entire analysis framework:
  1. Data tab — upload CSVs or use built-in sample data
  2. Ratings tab — fit Bradley-Terry model, tune parameters, view leaderboard
  3. Copula tab — simulate fight cards, explore correlation effects
  4. Diagnostics tab — residual analysis, calibration, scoring metrics
  5. Scraper tab — pull live data from ufcstats.com
"""
import io
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure src/ is importable even when running from the repo root
# (Replit typically runs `streamlit run app.py` from the project root)
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.bradley_terry import BradleyTerry
from src.copula import GaussianCopula
from src.scoring import ScoringRules
from src.residual import ResidualAnalyzer
from src.utils import parse_bout, time_decay_weights

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UFC Phase Transition",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/UFC_logo.svg/1200px-UFC_logo.svg.png",
    layout="wide",
)

st.title("UFC Phase Transition Analysis")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_data, tab_ratings, tab_copula, tab_diag, tab_scraper = st.tabs(
    ["Data", "Fighter Ratings", "Copula Simulator", "Diagnostics", "Scraper"]
)

# =========================================================================
# HELPER — build fight DataFrame from raw uploads / sample
# =========================================================================

SAMPLE_FIGHT_RESULTS = """EVENT\tBOUT\tOUTCOME\tMETHOD\tROUND\tTIME
UFC 300\tJon Jones vs. Stipe Miocic\tW/L\tKO/TKO\t3\t4:29
UFC 300\tAlex Pereira vs. Jamahal Hill\tW/L\tKO/TKO\t1\t1:14
UFC 300\tZhang Weili vs. Yan Xiaonan\tW/L\tDecision - Unanimous\t5\t5:00
UFC 300\tMax Holloway vs. Justin Gaethje\tW/L\tKO/TKO\t5\t4:59
UFC 300\tCharles Oliveira vs. Arman Tsarukyan\tL/W\tDecision - Unanimous\t3\t5:00
UFC 299\tSean O'Malley vs. Marlon Vera\tW/L\tDecision - Unanimous\t5\t5:00
UFC 299\tDustin Poirier vs. Benoit Saint Denis\tW/L\tTKO\t2\t3:42
UFC 299\tKevin Holland vs. Michael Page\tL/W\tDecision - Unanimous\t3\t5:00
UFC 299\tGilbert Burns vs. Jack Della Maddalena\tL/W\tKO/TKO\t2\t4:03
UFC 299\tPetr Yan vs. Song Yadong\tW/L\tDecision - Unanimous\t3\t5:00
UFC 298\tAlexander Volkanovski vs. Ilia Topuria\tL/W\tKO/TKO\t2\t3:32
UFC 298\tRobert Whittaker vs. Paulo Costa\tW/L\tDecision - Unanimous\t3\t5:00
UFC 298\tIan Machado Garry vs. Geoff Neal\tW/L\tDecision - Unanimous\t3\t5:00
UFC 298\tMerab Dvalishvili vs. Henry Cejudo\tW/L\tDecision - Unanimous\t3\t5:00
UFC 298\tAmanda Lemos vs. Mackenzie Dern\tW/L\tDecision - Split\t3\t5:00
UFC 297\tDricus Du Plessis vs. Sean Strickland\tW/L\tDecision - Split\t5\t5:00
UFC 297\tRaquel Pennington vs. Mayra Bueno Silva\tW/L\tDecision - Unanimous\t5\t5:00
UFC 297\tNeil Magny vs. Mike Malott\tW/L\tDecision - Unanimous\t3\t5:00
UFC 297\tChris Curtis vs. Marc-Andre Barriault\tW/L\tKO/TKO\t1\t2:43
UFC 297\tMovsar Evloev vs. Arnold Allen\tW/L\tDecision - Unanimous\t3\t5:00
UFC 296\tLeon Edwards vs. Colby Covington\tW/L\tDecision - Unanimous\t5\t5:00
UFC 296\tAlexandre Pantoja vs. Brandon Royval\tW/L\tDecision - Unanimous\t5\t5:00
UFC 296\tShavkat Rakhmonov vs. Stephen Thompson\tW/L\tSubmission\t2\t3:02
UFC 296\tTony Ferguson vs. Paddy Pimblett\tL/W\tDecision - Unanimous\t3\t5:00
UFC 296\tTai Tuivasa vs. Marcin Tybura\tL/W\tDecision - Unanimous\t3\t5:00
UFC 295\tAlex Pereira vs. Jiri Prochazka\tW/L\tKO/TKO\t2\t4:08
UFC 295\tSergei Pavlovich vs. Tom Aspinall\tL/W\tKO/TKO\t1\t0:69
UFC 295\tMatt Frevola vs. Benoit Saint Denis\tL/W\tSubmission\t1\t3:25
UFC 295\tDiego Lopes vs. Pat Sabatini\tW/L\tSubmission\t1\t2:44
UFC 295\tEryk Anders vs. Dustin Stoltzfus\tW/L\tKO/TKO\t1\t0:42"""

SAMPLE_EVENT_DETAILS = """EVENT\tDATE\tLOCATION
UFC 300\t2024-04-13\tLas Vegas, Nevada, USA
UFC 299\t2024-03-09\tMiami, Florida, USA
UFC 298\t2024-02-17\tAnaheim, California, USA
UFC 297\t2024-01-20\tToronto, Ontario, Canada
UFC 296\t2023-12-16\tLas Vegas, Nevada, USA
UFC 295\t2023-11-11\tNew York, New York, USA"""


def _read_uploaded(uploaded_file):
    """Read an uploaded file, auto-detecting TSV vs CSV separator."""
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    text = raw_bytes.decode("utf-8", errors="replace")
    first_line = text.split("\n", 1)[0]
    sep = "\t" if "\t" in first_line else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def _validate_fights_columns(df):
    """Raise a clear error if the DataFrame is missing required fight columns."""
    required = {"BOUT", "OUTCOME"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Fight results file is missing required column(s): "
            f"**{', '.join(sorted(missing))}**.\n\n"
            f"Found columns: `{', '.join(df.columns.tolist())}`.\n\n"
            f"Make sure you uploaded the **fight results** file (not event details) "
            f"in the left uploader. Expected columns include: EVENT, BOUT, OUTCOME."
        )


def _validate_events_columns(df):
    """Raise a clear error if the DataFrame is missing required event columns."""
    required = {"EVENT", "DATE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Event details file is missing required column(s): "
            f"**{', '.join(sorted(missing))}**.\n\n"
            f"Found columns: `{', '.join(df.columns.tolist())}`.\n\n"
            f"Make sure you uploaded the **event details** file (not fight results) "
            f"in the right uploader. Expected columns include: EVENT, DATE."
        )


def _parse_fights_df(df):
    """Add winner/loser columns from BOUT + OUTCOME columns."""
    _validate_fights_columns(df)
    df = df.copy()
    df[["fighter1", "fighter2"]] = df["BOUT"].apply(
        lambda x: pd.Series(parse_bout(x))
    )
    decisive = df["OUTCOME"].isin(["W/L", "L/W"])
    df = df[decisive].copy()
    df["winner"] = df.apply(
        lambda row: row["fighter1"] if row["OUTCOME"] == "W/L" else row["fighter2"],
        axis=1,
    )
    df["loser"] = df.apply(
        lambda row: row["fighter2"] if row["OUTCOME"] == "W/L" else row["fighter1"],
        axis=1,
    )
    return df


def _load_sample():
    fights = pd.read_csv(io.StringIO(SAMPLE_FIGHT_RESULTS), sep="\t")
    fights = _parse_fights_df(fights)
    events = pd.read_csv(io.StringIO(SAMPLE_EVENT_DETAILS), sep="\t")
    events["DATE"] = pd.to_datetime(events["DATE"])
    merged = fights.merge(events[["EVENT", "DATE"]], on="EVENT", how="left")
    merged = merged.sort_values("DATE").reset_index(drop=True)
    return merged


# =========================================================================
# 1  DATA TAB
# =========================================================================
with tab_data:
    st.header("Load Fight Data")
    data_source = st.radio(
        "Choose data source",
        ["Sample data (30 fights, UFC 295-300)", "Upload your own CSV files"],
        horizontal=True,
    )

    if data_source.startswith("Sample"):
        fights = _load_sample()
        st.success(f"Loaded {len(fights)} fights across {fights['EVENT'].nunique()} events.")
    else:
        col_f, col_e = st.columns(2)
        with col_f:
            up_fights = st.file_uploader(
                "Fight results (TSV: EVENT, BOUT, OUTCOME, ...)",
                type=["csv", "tsv", "txt"],
            )
        with col_e:
            up_events = st.file_uploader(
                "Event details (TSV: EVENT, DATE, ...)",
                type=["csv", "tsv", "txt"],
            )

        if up_fights and up_events:
            try:
                raw = _read_uploaded(up_fights)
                raw = _parse_fights_df(raw)
            except ValueError as e:
                st.error(str(e))
                fights = None
                st.stop()

            try:
                events = _read_uploaded(up_events)
                _validate_events_columns(events)
                events["DATE"] = pd.to_datetime(events["DATE"])
            except ValueError as e:
                st.error(str(e))
                fights = None
                st.stop()

            fights = raw.merge(events[["EVENT", "DATE"]], on="EVENT", how="left")
            fights = fights.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
            if len(fights) == 0:
                st.error(
                    "No fights matched between the two files. "
                    "Make sure the **EVENT** column values match across both files."
                )
                fights = None
            else:
                st.success(f"Loaded {len(fights)} fights across {fights['EVENT'].nunique()} events.")
        else:
            fights = None
            st.info("Upload both files to continue, or switch to sample data.")

    if fights is not None:
        st.session_state["fights"] = fights
        with st.expander("Preview data", expanded=False):
            st.dataframe(fights.head(20))
        st.metric("Total fights", len(fights))
        col1, col2, col3 = st.columns(3)
        col1.metric("Events", fights["EVENT"].nunique())
        col2.metric("Unique fighters", pd.concat([fights["fighter1"], fights["fighter2"]]).nunique())
        col3.metric(
            "Date range",
            f'{fights["DATE"].min().strftime("%b %Y")} – {fights["DATE"].max().strftime("%b %Y")}',
        )


# =========================================================================
# 2  FIGHTER RATINGS TAB
# =========================================================================
with tab_ratings:
    st.header("Bradley-Terry Fighter Ratings")

    fights = st.session_state.get("fights")
    if fights is None:
        st.warning("Go to the **Data** tab first to load fight data.")
        st.stop()

    st.subheader("Model Parameters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        reg_strength = st.slider(
            "L2 regularization", 0.001, 0.5, 0.01, 0.001,
            help="Higher = ratings pulled toward zero, prevents overfitting",
        )
    with col_p2:
        decay_half_life = st.slider(
            "Time-decay half-life (days)", 30, 1825, 365, 30,
            help="Older fights get exponentially less weight",
        )
    with col_p3:
        use_decay = st.checkbox("Enable time decay", value=True)

    if st.button("Fit model", type="primary"):
        fight_list = list(zip(fights["winner"], fights["loser"]))
        weights = (
            time_decay_weights(fights["DATE"], decay_half_life_days=decay_half_life).values
            if use_decay
            else None
        )

        # Monkey-patch regularization strength into the model
        bt = BradleyTerry()
        _orig_fit = bt.fit

        def _patched_fit(fight_results, time_weights=None):
            bt.fighters = list({f for pair in fight_results for f in pair})
            if len(bt.fighters) == 0:
                bt.ratings = {}
                return bt
            n = len(bt.fighters)
            idx = {f: i for i, f in enumerate(bt.fighters)}
            if time_weights is None:
                time_weights = np.ones(len(fight_results))

            def neg_log_likelihood(betas):
                nll = 0.0
                for (w, l), wt in zip(fight_results, time_weights):
                    i, j = idx[w], idx[l]
                    p = bt._sigmoid(betas[i] - betas[j])
                    nll -= wt * np.log(p + 1e-10)
                nll += reg_strength * np.sum(betas ** 2)
                return nll

            from scipy.optimize import minimize as sp_min

            beta_init = np.zeros(n)
            result = sp_min(neg_log_likelihood, beta_init, method="L-BFGS-B")
            if not result.success:
                st.warning(f"Optimizer did not converge: {result.message}")
            betas = result.x
            betas -= betas.mean()
            bt.ratings = {f: betas[idx[f]] for f in bt.fighters}
            return bt

        _patched_fit(fight_list, weights)
        st.session_state["bt"] = bt

        # Rolling predictions
        unique_events = fights["EVENT"].unique()
        fights_copy = fights.copy()
        fights_copy["p_pred"] = np.nan

        for event in unique_events:
            event_mask = fights_copy["EVENT"] == event
            train_mask = fights_copy["DATE"] < fights_copy.loc[event_mask, "DATE"].min()
            train_df = fights_copy[train_mask]
            if len(train_df) == 0:
                fights_copy.loc[event_mask, "p_pred"] = 0.5
            else:
                tw = (
                    time_decay_weights(train_df["DATE"], decay_half_life_days=decay_half_life).values
                    if use_decay
                    else None
                )
                bt_tmp = BradleyTerry()
                # Use patched fit with same reg strength
                bt_tmp.fighters = list({f for pair in zip(train_df["winner"], train_df["loser"]) for f in pair})
                n = len(bt_tmp.fighters)
                idx_tmp = {f: i for i, f in enumerate(bt_tmp.fighters)}
                tw_arr = tw if tw is not None else np.ones(len(train_df))

                def _nll(betas):
                    nll = 0.0
                    for (w, l), wt in zip(zip(train_df["winner"], train_df["loser"]), tw_arr):
                        i, j = idx_tmp[w], idx_tmp[l]
                        p = bt_tmp._sigmoid(betas[i] - betas[j])
                        nll -= wt * np.log(p + 1e-10)
                    nll += reg_strength * np.sum(betas ** 2)
                    return nll

                from scipy.optimize import minimize as sp_min
                res = sp_min(_nll, np.zeros(n), method="L-BFGS-B")
                b = res.x
                b -= b.mean()
                bt_tmp.ratings = {f: b[idx_tmp[f]] for f in bt_tmp.fighters}

                fights_copy.loc[event_mask, "p_pred"] = fights_copy[event_mask].apply(
                    lambda row: bt_tmp.win_probability(row["fighter1"], row["fighter2"]),
                    axis=1,
                )

        st.session_state["fights_pred"] = fights_copy
        st.success("Model fitted!")

    # Display leaderboard
    bt = st.session_state.get("bt")
    if bt is not None and bt.ratings:
        st.subheader("Leaderboard")
        lb = bt.leaderboard()
        lb.index = lb.index + 1
        lb.index.name = "Rank"
        st.dataframe(lb, use_container_width=True)

        # Head-to-head predictor
        st.subheader("Head-to-Head Predictor")
        fighters_sorted = sorted(bt.fighters)
        col_a, col_b = st.columns(2)
        with col_a:
            fa = st.selectbox("Fighter A", fighters_sorted, index=0)
        with col_b:
            fb = st.selectbox("Fighter B", fighters_sorted, index=min(1, len(fighters_sorted) - 1))
        if fa and fb and fa != fb:
            prob = bt.win_probability(fa, fb)
            cola, colb = st.columns(2)
            cola.metric(fa, f"{prob * 100:.1f}%")
            colb.metric(fb, f"{(1 - prob) * 100:.1f}%")
            st.progress(prob, text=f"{fa} {prob*100:.1f}% vs {fb} {(1-prob)*100:.1f}%")

        # Rating distribution
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        betas = pd.Series(bt.ratings)
        betas_sorted = betas.sort_values(ascending=True)
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in betas_sorted]
        ax.barh(betas_sorted.index, betas_sorted.values, color=colors)
        ax.set_xlabel("Rating (beta)")
        ax.set_title("Fighter Ratings")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =========================================================================
# 3  COPULA SIMULATOR TAB
# =========================================================================
with tab_copula:
    st.header("Gaussian Copula — Fight Card Simulator")
    st.markdown(
        "Simulate correlated fight outcomes on a card. "
        "Set each fight's favorite-win probability and the intra-card correlation."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        n_fights = st.slider("Number of fights on card", 3, 15, 5)
    with col_cfg2:
        rho = st.slider(
            "Intra-card correlation (rho)", 0.0, 0.6, 0.1, 0.01,
            help="0 = independent fights; higher = outcomes more correlated",
        )

    st.subheader("Favorite win probabilities")
    probs = []
    cols = st.columns(min(n_fights, 5))
    for i in range(n_fights):
        col = cols[i % len(cols)]
        with col:
            p = st.slider(f"Fight {i+1}", 0.50, 0.99, 0.65, 0.01, key=f"copula_p_{i}")
            probs.append(p)

    n_sims = st.select_slider(
        "Simulations", options=[1000, 5000, 10000, 50000, 100000], value=10000
    )

    if st.button("Run simulation", type="primary"):
        R = np.full((n_fights, n_fights), rho)
        np.fill_diagonal(R, 1.0)
        copula = GaussianCopula(n_fights, R)

        outcomes = copula.simulate_card(probs, n_simulations=n_sims)
        total_fav_wins = outcomes.sum(axis=1)

        # Key metrics
        st.subheader("Results")
        p_sweep = (total_fav_wins == n_fights).mean()
        p_naive = float(np.prod(probs))
        expected_upsets = n_fights - outcomes.sum(axis=1).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P(all favorites win)", f"{p_sweep:.4f}")
        c2.metric("Naive (independent)", f"{p_naive:.4f}")
        c3.metric("Copula / Naive ratio", f"{p_sweep / p_naive:.2f}x" if p_naive > 0 else "N/A")
        c4.metric("Expected upsets", f"{expected_upsets:.2f}")

        # Distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(total_fav_wins, bins=range(n_fights + 2), color="steelblue",
                     edgecolor="white", density=True, align="left")
        axes[0].set_xlabel("Number of favorites winning")
        axes[0].set_ylabel("Probability")
        axes[0].set_title(f"Distribution of favorite wins (rho={rho})")
        axes[0].set_xticks(range(n_fights + 1))

        # Compare across rho values
        rho_vals = [0.0, 0.1, 0.2, 0.3, 0.5]
        sweep_probs = []
        for rv in rho_vals:
            R2 = np.full((n_fights, n_fights), rv)
            np.fill_diagonal(R2, 1.0)
            c2 = GaussianCopula(n_fights, R2)
            sp = c2.favorites_sweep_probability(probs, n_simulations=n_sims)
            sweep_probs.append(sp)

        axes[1].plot(rho_vals, sweep_probs, "o-", color="steelblue", linewidth=2)
        axes[1].axhline(p_naive, color="red", linestyle="--", label="Naive (rho=0)")
        axes[1].set_xlabel("Intra-card correlation (rho)")
        axes[1].set_ylabel("P(all favorites win)")
        axes[1].set_title("Sweep probability vs correlation")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Correlation effect table
        st.subheader("Correlation Effect Table")
        copula_base = GaussianCopula(n_fights)
        effect_df = copula_base.correlation_effect(probs, rho_values=rho_vals, n_simulations=n_sims)
        st.dataframe(effect_df, use_container_width=True)


# =========================================================================
# 4  DIAGNOSTICS TAB
# =========================================================================
with tab_diag:
    st.header("Model Diagnostics")

    fights_pred = st.session_state.get("fights_pred")
    if fights_pred is None:
        st.warning("Fit the model first in the **Fighter Ratings** tab.")
        st.stop()

    # Compute derived columns
    fp = fights_pred.copy()
    fp["favorite"] = fp.apply(
        lambda row: row["fighter1"] if row["p_pred"] > 0.5 else row["fighter2"], axis=1
    )
    fp["favorite_win_prob"] = fp["p_pred"].clip(lower=0.5).where(
        fp["p_pred"] >= 0.5, 1 - fp["p_pred"]
    )
    # Fix: favorite_win_prob is always max(p_pred, 1 - p_pred)
    fp["favorite_win_prob"] = fp["p_pred"].apply(lambda p: max(p, 1 - p))
    fp["favorite_won"] = fp["winner"] == fp["favorite"]
    fp["upset"] = ~fp["favorite_won"]
    fp["p_upset"] = 1 - fp["favorite_win_prob"]

    # Outcomes: 1 if fighter1 won
    outcomes = (fp["winner"] == fp["fighter1"]).astype(int).values
    predictions = fp["p_pred"].values

    valid = ~np.isnan(predictions)
    p_valid = predictions[valid]
    o_valid = outcomes[valid]

    if len(p_valid) == 0:
        st.warning("No valid predictions to analyze.")
        st.stop()

    # Scoring metrics
    st.subheader("Scoring Metrics")
    brier = ScoringRules.brier_score(p_valid, o_valid)
    baseline = ScoringRules.baseline_brier(o_valid)
    ll = ScoringRules.log_loss(p_valid, o_valid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brier Score", f"{brier:.4f}")
    c2.metric("Baseline Brier", f"{baseline:.4f}")
    c3.metric("Skill vs baseline", f"{baseline - brier:+.4f}")
    c4.metric("Log Loss", f"{ll:.4f}")

    n_cal_bins = st.slider("Calibration bins", 3, 15, 5)

    # Residual diagnostics
    st.subheader("Residual Diagnostics")
    residuals = p_valid - o_valid

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Residual Diagnostics", fontsize=13, fontweight="bold")

    axes[0].scatter(p_valid, residuals, alpha=0.6, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted P(fighter1 wins)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=15, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    cal = ScoringRules.calibration_curve(p_valid, o_valid, n_bins=n_cal_bins)
    axes[2].plot([0, 1], [0, 1], "r--", label="Perfect")
    if len(cal) > 0:
        axes[2].scatter(
            cal["mean_predicted"],
            cal["actual_win_rate"],
            s=cal["n_fights"] * 5,
            color="steelblue",
            label="Model",
        )
        axes[2].plot(cal["mean_predicted"], cal["actual_win_rate"], color="steelblue")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Actual win rate")
    axes[2].set_title("Calibration Curve")
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Event-level analysis
    st.subheader("Event-Level Residual Analysis")
    event_stats = fp.groupby("EVENT").agg(
        n_fights=("upset", "size"),
        expected_upsets=("p_upset", "sum"),
        actual_upsets=("upset", "sum"),
        avg_fav_prob=("favorite_win_prob", "mean"),
        date=("DATE", "first"),
    ).reset_index()
    event_stats["residual"] = event_stats["actual_upsets"] - event_stats["expected_upsets"]
    event_stats["expected_var"] = fp.groupby("EVENT").apply(
        lambda df: (df["p_upset"] * (1 - df["p_upset"])).sum()
    ).values

    st.dataframe(
        event_stats[["EVENT", "date", "n_fights", "expected_upsets", "actual_upsets", "residual"]]
        .sort_values("date", ascending=False),
        use_container_width=True,
    )

    # Event residual histogram
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(event_stats["residual"], bins=max(5, len(event_stats) // 2), edgecolor="black", color="steelblue")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Residual (actual - expected upsets)")
    ax2.set_ylabel("Number of events")
    ax2.set_title("Distribution of Event-Level Residuals")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# =========================================================================
# 5  SCRAPER TAB
# =========================================================================
with tab_scraper:
    st.header("UFC Stats Scraper")
    st.markdown(
        "Pull fight data directly from [ufcstats.com](http://ufcstats.com/statistics/events/completed?page=all). "
        "Scraped data is converted to the same format used by the model."
    )

    try:
        import requests
        from bs4 import BeautifulSoup
        _HAS_SCRAPER_DEPS = True
    except ImportError:
        _HAS_SCRAPER_DEPS = False

    if not _HAS_SCRAPER_DEPS:
        st.error(
            "Scraping requires `requests` and `beautifulsoup4`. "
            "Install them with: `pip install requests beautifulsoup4`"
        )
        st.stop()

    @st.cache_data(ttl=3600, show_spinner="Fetching event list...")
    def scrape_event_list():
        url = "http://ufcstats.com/statistics/events/completed?page=all"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("tr.b-statistics__table-row")
        events = []
        for row in rows:
            link = row.select_one("a.b-link")
            if link:
                name = link.get_text(strip=True)
                href = link["href"]
                date_cell = row.select("td.b-statistics__table-col")
                date_text = date_cell[1].get_text(strip=True) if len(date_cell) > 1 else ""
                location_text = date_cell[2].get_text(strip=True) if len(date_cell) > 2 else ""
                events.append({
                    "EVENT": name,
                    "DATE": date_text,
                    "LOCATION": location_text,
                    "url": href,
                })
        return pd.DataFrame(events)

    @st.cache_data(ttl=3600, show_spinner="Scraping event details...")
    def scrape_event_fights(event_url):
        resp = requests.get(event_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover")
        fights = []
        for row in rows:
            cols = row.select("td.b-fight-details__table-col")
            if len(cols) < 8:
                continue
            # Fighters are in the first column
            fighters_col = cols[1]
            fighter_links = fighters_col.select("a")
            if len(fighter_links) < 2:
                continue
            f1 = fighter_links[0].get_text(strip=True)
            f2 = fighter_links[1].get_text(strip=True)
            # Win/loss from first column
            wl_col = cols[0]
            wl_links = wl_col.select("a")
            result = wl_links[0].get_text(strip=True) if wl_links else ""

            method_col = cols[7] if len(cols) > 7 else None
            method = method_col.get_text(strip=True).split("\n")[0].strip() if method_col else ""
            round_col = cols[8] if len(cols) > 8 else None
            rnd = round_col.get_text(strip=True) if round_col else ""
            time_col = cols[9] if len(cols) > 9 else None
            tm = time_col.get_text(strip=True) if time_col else ""

            if result.lower() == "win":
                outcome = "W/L"
            elif result.lower() == "loss":
                outcome = "L/W"
            else:
                outcome = result

            fights.append({
                "BOUT": f"{f1} vs. {f2}",
                "OUTCOME": outcome,
                "METHOD": method,
                "ROUND": rnd,
                "TIME": tm,
            })
        return pd.DataFrame(fights)

    # --- UI ---
    if st.button("Fetch event list from ufcstats.com"):
        try:
            event_df = scrape_event_list()
            st.session_state["scraped_events"] = event_df
            st.success(f"Found {len(event_df)} events.")
        except Exception as e:
            st.error(f"Failed to fetch events: {e}")

    scraped_events = st.session_state.get("scraped_events")
    if scraped_events is not None and len(scraped_events) > 0:
        st.dataframe(scraped_events[["EVENT", "DATE", "LOCATION"]].head(30), use_container_width=True)

        selected = st.multiselect(
            "Select events to scrape fight details",
            scraped_events["EVENT"].tolist(),
            default=scraped_events["EVENT"].tolist()[:3],
        )

        if st.button("Scrape selected events") and selected:
            all_fights = []
            all_event_details = []
            progress = st.progress(0, text="Scraping...")
            for i, evt_name in enumerate(selected):
                row = scraped_events[scraped_events["EVENT"] == evt_name].iloc[0]
                progress.progress((i + 1) / len(selected), text=f"Scraping {evt_name}...")
                try:
                    fight_df = scrape_event_fights(row["url"])
                    fight_df["EVENT"] = evt_name
                    all_fights.append(fight_df)
                    all_event_details.append({
                        "EVENT": evt_name,
                        "DATE": row["DATE"],
                        "LOCATION": row["LOCATION"],
                    })
                    time.sleep(0.5)  # be polite
                except Exception as e:
                    st.warning(f"Failed to scrape {evt_name}: {e}")

            progress.empty()

            if all_fights:
                combined = pd.concat(all_fights, ignore_index=True)
                event_det = pd.DataFrame(all_event_details)
                event_det["DATE"] = pd.to_datetime(event_det["DATE"], format="mixed", errors="coerce")

                st.success(f"Scraped {len(combined)} fights from {len(all_event_details)} events.")
                st.dataframe(combined, use_container_width=True)

                # Parse into model format
                try:
                    parsed = _parse_fights_df(combined)
                    merged = parsed.merge(event_det[["EVENT", "DATE"]], on="EVENT", how="left")
                    merged = merged.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
                    st.session_state["fights"] = merged
                    st.success(
                        f"Data ready! {len(merged)} decisive fights loaded. "
                        "Go to the **Fighter Ratings** tab to fit the model."
                    )
                except Exception as e:
                    st.error(f"Failed to parse scraped data: {e}")

                # Download button
                csv_buf = combined.to_csv(index=False, sep="\t")
                st.download_button(
                    "Download scraped data (TSV)",
                    csv_buf,
                    file_name="ufc_scraped_fights.tsv",
                    mime="text/tab-separated-values",
                )
