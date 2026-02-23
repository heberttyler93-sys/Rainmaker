"""
UFC Phase Transition — Interactive Analysis Platform v2

Fully reactive no-code UI with:
  • 4 data entry modes (sample / upload / paste / manual form)
  • Auto-fitting Bradley-Terry model — parameters drive output without button clicks
  • Predictions tab with 5 chart types and per-event filtering
  • Copula fight-card simulator
  • Diagnostics (calibration, residuals)
  • Live scraper from ufcstats.com
"""
import io
import json
import time
import warnings
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from src.bradley_terry import BradleyTerry
from src.copula import GaussianCopula
from src.scoring import ScoringRules
from src.utils import parse_bout, time_decay_weights

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UFC Phase Transition",
    page_icon="🥊",
    layout="wide",
)

st.title("🥊 UFC Phase Transition Analysis")
st.caption(
    "No-code interactive analysis — add data any way you like, tune the model, explore predictions."
)

# ---------------------------------------------------------------------------
# Built-in sample data
# ---------------------------------------------------------------------------
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

PASTE_PLACEHOLDER = """\
EVENT\tBOUT\tOUTCOME\tDATE
UFC 300\tJon Jones vs. Stipe Miocic\tW/L\t2024-04-13
UFC 300\tAlex Pereira vs. Jamahal Hill\tW/L\t2024-04-13
UFC 299\tSean O'Malley vs. Marlon Vera\tW/L\t2024-03-09"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_fights_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add winner/loser columns from BOUT + OUTCOME columns."""
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


def _load_sample() -> pd.DataFrame:
    fights = pd.read_csv(io.StringIO(SAMPLE_FIGHT_RESULTS), sep="\t")
    fights = _parse_fights_df(fights)
    events = pd.read_csv(io.StringIO(SAMPLE_EVENT_DETAILS), sep="\t")
    events["DATE"] = pd.to_datetime(events["DATE"])
    merged = fights.merge(events[["EVENT", "DATE"]], on="EVENT", how="left")
    return merged.sort_values("DATE").reset_index(drop=True)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Cached model fitting — pure function, re-runs only when inputs change
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fitting Bradley-Terry model…")
def _fit_model_cached(
    fights_json: str,
    reg_strength: float,
    decay_half_life: int,
    use_decay: bool,
) -> tuple[str, str]:
    """
    Returns (ratings_json, predictions_json).
    Both are JSON strings so Streamlit can hash them for caching.
    """
    from scipy.optimize import minimize as sp_min

    fights = pd.read_json(io.StringIO(fights_json))
    fights["DATE"] = pd.to_datetime(fights["DATE"])
    fights = fights.sort_values("DATE").reset_index(drop=True)

    def _fit_bt(fight_pairs, tw=None):
        fighters = list({f for pair in fight_pairs for f in pair})
        if not fighters:
            return {}
        n = len(fighters)
        idx = {f: i for i, f in enumerate(fighters)}
        if tw is None:
            tw = np.ones(len(fight_pairs))

        def nll(betas):
            loss = reg_strength * np.dot(betas, betas)
            for (w, l), wt in zip(fight_pairs, tw):
                p = _sigmoid(betas[idx[w]] - betas[idx[l]])
                loss -= wt * np.log(p + 1e-12)
            return loss

        res = sp_min(nll, np.zeros(n), method="L-BFGS-B")
        b = res.x - res.x.mean()
        return {f: b[idx[f]] for f in fighters}

    # Global ratings (for leaderboard)
    fight_pairs = list(zip(fights["winner"], fights["loser"]))
    global_weights = (
        time_decay_weights(fights["DATE"], decay_half_life_days=decay_half_life).values
        if use_decay
        else None
    )
    ratings = _fit_bt(fight_pairs, global_weights)

    # Rolling out-of-sample predictions
    fc = fights.copy()
    fc["p_pred"] = np.nan
    for event in fc["EVENT"].unique():
        ev_mask = fc["EVENT"] == event
        ev_min_date = fc.loc[ev_mask, "DATE"].min()
        train = fc[fc["DATE"] < ev_min_date]
        if len(train) == 0:
            fc.loc[ev_mask, "p_pred"] = 0.5
            continue
        tw = (
            time_decay_weights(train["DATE"], decay_half_life_days=decay_half_life).values
            if use_decay
            else None
        )
        tmp_r = _fit_bt(list(zip(train["winner"], train["loser"])), tw)
        fc.loc[ev_mask, "p_pred"] = fc.loc[ev_mask].apply(
            lambda row: _sigmoid(
                tmp_r.get(row["fighter1"], 0.0) - tmp_r.get(row["fighter2"], 0.0)
            ),
            axis=1,
        )

    return json.dumps(ratings), fc.to_json(date_format="iso")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_data, tab_model, tab_pred, tab_copula, tab_diag, tab_scraper = st.tabs(
    ["📂 Data", "⚙️ Model & Ratings", "📊 Predictions", "🎲 Copula Simulator", "🔬 Diagnostics", "🌐 Scraper"]
)


# =========================================================================
# 1  DATA TAB
# =========================================================================
with tab_data:
    st.header("Load Fight Data")

    data_source = st.radio(
        "How would you like to add data?",
        ["Sample data (UFC 295–300)", "Upload CSV/TSV files", "Paste raw text", "Manual entry"],
        horizontal=True,
    )

    # ---- Sample ----
    if data_source == "Sample data (UFC 295–300)":
        fights = _load_sample()
        st.session_state["fights"] = fights
        st.success(f"Loaded {len(fights)} fights across {fights['EVENT'].nunique()} events.")

    # ---- Upload ----
    elif data_source == "Upload CSV/TSV files":
        with st.expander("Expected file format", expanded=False):
            st.markdown("**Fight results file** (tab-separated, `.csv` / `.tsv` / `.txt`):")
            st.code(
                "EVENT\tBOUT\tOUTCOME\tMETHOD\tROUND\tTIME\n"
                "UFC 300\tJon Jones vs. Stipe Miocic\tW/L\tKO/TKO\t3\t4:29"
            )
            st.markdown("**Event details file** (tab-separated):")
            st.code("EVENT\tDATE\tLOCATION\nUFC 300\t2024-04-13\tLas Vegas, Nevada, USA")

        col_f, col_e = st.columns(2)
        with col_f:
            up_fights = st.file_uploader(
                "Fight results (needs: EVENT, BOUT, OUTCOME)",
                type=["csv", "tsv", "txt"],
            )
        with col_e:
            up_events = st.file_uploader(
                "Event details (needs: EVENT, DATE)",
                type=["csv", "tsv", "txt"],
            )

        if up_fights and up_events:
            def _read_flex(f):
                raw = f.read()
                try:
                    df = pd.read_csv(io.BytesIO(raw), sep="\t")
                    if len(df.columns) < 2:
                        df = pd.read_csv(io.BytesIO(raw))
                except Exception:
                    df = pd.read_csv(io.BytesIO(raw))
                return df

            raw_fights = _parse_fights_df(_read_flex(up_fights))
            events_df = _read_flex(up_events)
            events_df["DATE"] = pd.to_datetime(events_df["DATE"], errors="coerce")
            fights = raw_fights.merge(events_df[["EVENT", "DATE"]], on="EVENT", how="left")
            fights = fights.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
            st.session_state["fights"] = fights
            st.success(f"Loaded {len(fights)} fights across {fights['EVENT'].nunique()} events.")
        else:
            st.info("Upload both files to continue.")

    # ---- Paste ----
    elif data_source == "Paste raw text":
        st.markdown(
            "Paste tab-separated fight data below. "
            "Include a **DATE** column (YYYY-MM-DD) or the model cannot sort events chronologically."
        )
        with st.expander("Format guide & example"):
            st.markdown(
                "**Required columns:** `EVENT`, `BOUT`, `OUTCOME`  \n"
                "**BOUT format:** `Fighter A vs. Fighter B`  \n"
                "**OUTCOME:** `W/L` = Fighter A won, `L/W` = Fighter B won"
            )
            st.code(PASTE_PLACEHOLDER)

        pasted = st.text_area(
            "Paste fight data here",
            height=260,
            placeholder=PASTE_PLACEHOLDER,
        )

        if pasted.strip():
            try:
                raw = pd.read_csv(io.StringIO(pasted.strip()), sep="\t")
                raw = _parse_fights_df(raw)
                if "DATE" in raw.columns:
                    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
                    raw = raw.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
                else:
                    # Assign a dummy date so downstream code doesn't crash
                    raw["DATE"] = pd.Timestamp("2024-01-01")
                    st.warning(
                        "No DATE column found — all fights assigned 2024-01-01. "
                        "Add a DATE column for chronological rolling predictions."
                    )
                st.session_state["fights"] = raw
                st.success(f"Parsed {len(raw)} fights from {raw['EVENT'].nunique()} events.")
            except Exception as e:
                st.error(f"Could not parse pasted data: {e}")
        else:
            st.info("Paste data above to get started.")

    # ---- Manual entry ----
    elif data_source == "Manual entry":
        st.markdown("Add fights one at a time using the form. They are stored in your session.")

        if "manual_fights" not in st.session_state:
            st.session_state["manual_fights"] = []

        with st.form("add_fight_form", clear_on_submit=True):
            st.subheader("Add a fight")
            c1, c2, c3 = st.columns(3)
            with c1:
                event_name = st.text_input("Event name", placeholder="UFC 300")
                fight_date = st.date_input("Event date", value=date(2024, 4, 13))
            with c2:
                f1 = st.text_input("Fighter 1 (listed first)", placeholder="Jon Jones")
                f2 = st.text_input("Fighter 2 (listed second)", placeholder="Stipe Miocic")
            with c3:
                winner_choice = st.radio("Winner", ["Fighter 1", "Fighter 2"])
                method = st.selectbox(
                    "Method",
                    [
                        "Decision - Unanimous",
                        "Decision - Split",
                        "Decision - Majority",
                        "KO/TKO",
                        "Submission",
                        "No Contest",
                        "Draw",
                    ],
                )
            add_fight = st.form_submit_button("➕ Add Fight", type="primary")
            if add_fight:
                if f1 and f2 and event_name:
                    outcome = "W/L" if winner_choice == "Fighter 1" else "L/W"
                    st.session_state["manual_fights"].append(
                        {
                            "EVENT": event_name,
                            "BOUT": f"{f1} vs. {f2}",
                            "OUTCOME": outcome,
                            "METHOD": method,
                            "DATE": str(fight_date),
                            "fighter1": f1,
                            "fighter2": f2,
                            "winner": f1 if outcome == "W/L" else f2,
                            "loser": f2 if outcome == "W/L" else f1,
                        }
                    )
                else:
                    st.error("Please fill in Event name, Fighter 1, and Fighter 2.")

        manual = st.session_state.get("manual_fights", [])
        if manual:
            mdf = pd.DataFrame(manual)
            mdf["DATE"] = pd.to_datetime(mdf["DATE"])
            c_info, c_del = st.columns([4, 1])
            c_info.success(
                f"{len(mdf)} fight(s) recorded across {mdf['EVENT'].nunique()} event(s)."
            )
            if c_del.button("🗑 Clear all", type="secondary"):
                st.session_state["manual_fights"] = []
                st.rerun()
            st.dataframe(
                mdf[["EVENT", "DATE", "BOUT", "OUTCOME", "METHOD"]],
                use_container_width=True,
            )
            fights = mdf.sort_values("DATE").reset_index(drop=True)
            st.session_state["fights"] = fights
        else:
            st.info("No fights yet — fill in the form above.")

    # ---- Summary for all modes ----
    fights = st.session_state.get("fights")
    if fights is not None and len(fights) > 0:
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total fights", len(fights))
        c2.metric("Events", fights["EVENT"].nunique())
        c3.metric(
            "Unique fighters",
            pd.concat([fights["fighter1"], fights["fighter2"]]).nunique(),
        )
        c4.metric(
            "Date range",
            f"{fights['DATE'].min().strftime('%b %Y')} – {fights['DATE'].max().strftime('%b %Y')}",
        )

        with st.expander("Preview data", expanded=False):
            st.dataframe(fights, use_container_width=True)

        st.download_button(
            "⬇ Download current dataset (TSV)",
            fights.to_csv(index=False, sep="\t"),
            file_name="ufc_fights.tsv",
            mime="text/tab-separated-values",
        )


# =========================================================================
# 2  MODEL & RATINGS TAB — auto-reactive (no button needed)
# =========================================================================
with tab_model:
    st.header("Bradley-Terry Fighter Ratings")

    fights = st.session_state.get("fights")
    if fights is None or len(fights) == 0:
        st.warning("Go to the **Data** tab first to load fight data.")
        st.stop()

    st.markdown(
        "The **Bradley-Terry model** assigns a latent skill rating β to every fighter.  \n"
        "Win probability: `P(A beats B) = sigmoid(β_A − β_B)`.  \n"
        "Adjust any parameter — the model **re-fits automatically**."
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        reg_strength = st.slider(
            "L2 regularisation",
            min_value=0.001,
            max_value=0.5,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Higher = ratings shrink toward zero (good for small datasets)",
        )
    with col_p2:
        decay_half_life = st.slider(
            "Time-decay half-life (days)",
            min_value=30,
            max_value=1825,
            value=365,
            step=30,
            help="Older fights lose weight exponentially. 365 ≈ half-weight after 1 year.",
        )
    with col_p3:
        use_decay = st.checkbox("Enable time decay", value=True)

    # Auto-fit — triggered whenever fights data or params change
    fights_json = fights.to_json(date_format="iso")
    try:
        ratings_json, predictions_json = _fit_model_cached(
            fights_json, reg_strength, decay_half_life, use_decay
        )
        ratings = json.loads(ratings_json)
        fights_pred = pd.read_json(io.StringIO(predictions_json))
        fights_pred["DATE"] = pd.to_datetime(fights_pred["DATE"])
        st.session_state["bt_ratings"] = ratings
        st.session_state["fights_pred"] = fights_pred
        st.success(
            f"Model fitted — {len(ratings)} fighters rated on {len(fights)} bouts."
        )
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.stop()

    # Leaderboard
    st.subheader("Leaderboard")
    lb_df = pd.DataFrame(
        [
            {
                "Fighter": f,
                "Rating (β)": round(v, 4),
                "Win% vs avg fighter": f"{_sigmoid(v) * 100:.1f}%",
            }
            for f, v in sorted(ratings.items(), key=lambda x: -x[1])
        ]
    )
    lb_df.index = range(1, len(lb_df) + 1)
    lb_df.index.name = "Rank"
    st.dataframe(lb_df, use_container_width=True)

    # Head-to-head predictor
    st.subheader("Head-to-Head Win Probability")
    fighters_sorted = sorted(ratings.keys())
    col_a, col_b = st.columns(2)
    with col_a:
        fa = st.selectbox("Fighter A", fighters_sorted, index=0)
    with col_b:
        fb = st.selectbox(
            "Fighter B", fighters_sorted, index=min(1, len(fighters_sorted) - 1)
        )

    if fa and fb and fa != fb:
        prob_a = _sigmoid(ratings.get(fa, 0.0) - ratings.get(fb, 0.0))
        ca, cmid, cb = st.columns([2, 1, 2])
        ca.metric(fa, f"{prob_a * 100:.1f}%", delta=f"β = {ratings.get(fa, 0):.3f}")
        cmid.markdown(
            "<div style='text-align:center;padding-top:1.6rem;font-size:1.4rem'>vs</div>",
            unsafe_allow_html=True,
        )
        cb.metric(fb, f"{(1 - prob_a) * 100:.1f}%", delta=f"β = {ratings.get(fb, 0):.3f}")
        st.progress(prob_a, text=f"{fa}: {prob_a*100:.1f}%  ·  {fb}: {(1-prob_a)*100:.1f}%")

        h2h = fights[
            ((fights["fighter1"] == fa) & (fights["fighter2"] == fb))
            | ((fights["fighter1"] == fb) & (fights["fighter2"] == fa))
        ]
        if len(h2h):
            st.markdown("**Previous matchups in dataset:**")
            st.dataframe(
                h2h[["EVENT", "DATE", "winner", "loser", "METHOD"]].sort_values(
                    "DATE", ascending=False
                ),
                use_container_width=True,
            )

    # Rating distribution plot
    st.subheader("Rating Distribution")
    betas = pd.Series(ratings).sort_values(ascending=True)
    fig_height = max(4, len(betas) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in betas.values]
    ax.barh(betas.index, betas.values, color=colors, height=0.75)
    ax.set_xlabel("Rating (β) — positive = above average")
    ax.set_title("Bradley-Terry Fighter Ratings", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================================
# 3  PREDICTIONS TAB — new rich exploration layer
# =========================================================================
with tab_pred:
    st.header("Prediction Explorer")

    fights_pred = st.session_state.get("fights_pred")
    ratings = st.session_state.get("bt_ratings")

    if fights_pred is None:
        st.warning("Load data and go to the **Model & Ratings** tab first — it auto-fits.")
        st.stop()

    fp = fights_pred.copy()
    fp["DATE"] = pd.to_datetime(fp["DATE"])
    fp["favorite"] = fp.apply(
        lambda row: row["fighter1"] if row["p_pred"] >= 0.5 else row["fighter2"], axis=1
    )
    fp["favorite_win_prob"] = fp["p_pred"].apply(lambda p: max(p, 1.0 - p))
    fp["favorite_won"] = fp["winner"] == fp["favorite"]
    fp["upset"] = ~fp["favorite_won"]
    fp["p_upset"] = 1.0 - fp["favorite_win_prob"]

    valid = fp.dropna(subset=["p_pred"]).copy()
    n_valid = len(valid)
    n_correct = int(valid["favorite_won"].sum())
    n_upsets = int(valid["upset"].sum())
    accuracy = n_correct / n_valid if n_valid else 0.0
    avg_conf = valid["favorite_win_prob"].mean() if n_valid else 0.5

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction accuracy", f"{accuracy:.1%}", help="How often the model-favored fighter actually won")
    c2.metric("Avg model confidence", f"{avg_conf:.1%}", help="Mean predicted win% for the chosen favorite")
    c3.metric("Upsets", f"{n_upsets} / {n_valid}")
    c4.metric("Upset rate", f"{n_upsets / n_valid:.1%}" if n_valid else "—")

    st.divider()

    # Controls row
    ctrl1, ctrl2 = st.columns([2, 3])
    with ctrl1:
        chart_choice = st.selectbox(
            "Chart type",
            [
                "Calibration (Predicted vs Actual)",
                "Win Probability Distribution",
                "Biggest Upsets",
                "Event Accuracy Timeline",
                "Confidence vs Accuracy Buckets",
            ],
        )
    with ctrl2:
        event_filter = st.multiselect(
            "Filter by event (blank = all events)",
            sorted(valid["EVENT"].unique()),
            default=[],
            placeholder="All events shown",
        )

    plot_df = valid[valid["EVENT"].isin(event_filter)] if event_filter else valid.copy()

    if len(plot_df) == 0:
        st.warning("No fights match the current filter.")
        st.stop()

    # ---- Chart rendering ----
    if chart_choice == "Calibration (Predicted vs Actual)":
        p = plot_df["p_pred"].values
        o = (plot_df["winner"] == plot_df["fighter1"]).astype(int).values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Are the model's predicted probabilities well-calibrated?", fontsize=13, fontweight="bold")

        jitter = np.random.default_rng(42).uniform(-0.025, 0.025, len(o))
        colors_sc = ["#2ecc71" if oi else "#e74c3c" for oi in o]
        axes[0].scatter(p, o + jitter, alpha=0.55, c=colors_sc, s=40, edgecolors="none")
        axes[0].axvline(0.5, color="gray", linestyle="--", alpha=0.6)
        axes[0].set_xlabel("Predicted P(fighter1 wins)")
        axes[0].set_ylabel("Actual outcome (1 = fighter1 won)")
        axes[0].set_title("Scatter: predictions vs. outcomes")
        axes[0].spines[["top", "right"]].set_visible(False)

        n_bins = 6
        bins = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.digitize(p, bins).clip(0, n_bins - 1)
        bc, ba, bn = [], [], []
        for b in range(n_bins):
            m = bin_idx == b
            if m.sum() > 0:
                bc.append(p[m].mean())
                ba.append(o[m].mean())
                bn.append(m.sum())
        axes[1].plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")
        if bc:
            axes[1].scatter(bc, ba, s=[n * 25 for n in bn], c="steelblue", zorder=5, label="Model (size = n)")
            axes[1].plot(bc, ba, color="steelblue", linewidth=2)
        axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("Mean predicted probability")
        axes[1].set_ylabel("Actual win rate in that bin")
        axes[1].set_title("Calibration curve (bin average)")
        axes[1].legend(); axes[1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    elif chart_choice == "Win Probability Distribution":
        fav_probs = plot_df["favorite_win_prob"].values
        won = plot_df["favorite_won"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Predicted win probability — split by actual outcome", fontsize=13, fontweight="bold")
        axes[0].hist(fav_probs[won], bins=15, alpha=0.72, color="#2ecc71", label="Favorite won", density=True)
        axes[0].hist(fav_probs[~won], bins=15, alpha=0.72, color="#e74c3c", label="Upset (underdog won)", density=True)
        axes[0].set_xlabel("Predicted favorite win probability")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Density by outcome")
        axes[0].legend(); axes[0].spines[["top", "right"]].set_visible(False)

        for label, mask, color in [("Won", won, "#2ecc71"), ("Upset", ~won, "#e74c3c")]:
            if mask.sum() > 0:
                sp = np.sort(fav_probs[mask])
                cdf = np.arange(1, len(sp) + 1) / len(sp)
                axes[1].plot(sp, cdf, color=color, linewidth=2, label=f"{label} (n={mask.sum()})")
        axes[1].set_xlabel("Predicted favorite win probability")
        axes[1].set_ylabel("Cumulative fraction")
        axes[1].set_title("CDF of model confidence by outcome")
        axes[1].legend(); axes[1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    elif chart_choice == "Biggest Upsets":
        upsets = plot_df[plot_df["upset"]].sort_values("favorite_win_prob", ascending=False).head(15)
        if upsets.empty:
            st.info("No upsets in the selected data.")
        else:
            labels = [
                f"{row['winner']} def. {row['loser']}\n({row['EVENT']})"
                for _, row in upsets.iterrows()
            ]
            vals = upsets["favorite_win_prob"].values

            fig, ax = plt.subplots(figsize=(12, max(4, len(upsets) * 0.55)))
            bars = ax.barh(labels, vals, color="#e74c3c", alpha=0.85)
            ax.axvline(0.5, color="gray", linestyle="--", lw=1)
            ax.set_xlabel("Model's predicted win probability for the (eventual) loser")
            ax.set_title(
                "Biggest Upsets — fights where the model was most confident in the wrong fighter",
                fontsize=12, fontweight="bold",
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_width() + 0.008,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", fontsize=9,
                )
            ax.set_xlim(0, 1.12)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    elif chart_choice == "Event Accuracy Timeline":
        ev_acc = (
            plot_df.groupby(["EVENT", "DATE"])
            .agg(accuracy=("favorite_won", "mean"), n_fights=("favorite_won", "size"))
            .reset_index()
            .sort_values("DATE")
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        sc = ax.scatter(
            ev_acc["DATE"], ev_acc["accuracy"],
            s=ev_acc["n_fights"] * 30,
            c=ev_acc["accuracy"],
            cmap="RdYlGn", vmin=0.3, vmax=1.0,
            zorder=5, edgecolors="white", linewidths=0.5,
        )
        ax.plot(ev_acc["DATE"], ev_acc["accuracy"], color="gray", linewidth=1, alpha=0.4)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.7, label="50% baseline")
        plt.colorbar(sc, ax=ax, label="Accuracy")
        ax.set_ylim(-0.05, 1.15); ax.set_xlabel("Date"); ax.set_ylabel("Prediction accuracy")
        ax.set_title(
            "Event-level prediction accuracy over time\n(dot size ∝ fights on card)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        for _, row in ev_acc.iterrows():
            ax.annotate(
                row["EVENT"], (row["DATE"], row["accuracy"]),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=7, rotation=30,
            )
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    elif chart_choice == "Confidence vs Accuracy Buckets":
        bins_labels = ["50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
        edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        fav_probs = plot_df["favorite_win_prob"].values
        correct = plot_df["favorite_won"].values

        accs, counts, midpoints = [], [], []
        for lo, hi, lbl in zip(edges[:-1], edges[1:], bins_labels):
            mask = (fav_probs >= lo) & (fav_probs < hi)
            accs.append(correct[mask].mean() if mask.sum() > 0 else np.nan)
            counts.append(int(mask.sum()))
            midpoints.append((lo + hi) / 2)

        fig, ax = plt.subplots(figsize=(10, 5))
        bar_colors = [
            ("#2ecc71" if (a is not None and not np.isnan(a) and a >= m) else "#e74c3c")
            for a, m in zip(accs, midpoints)
        ]
        bars = ax.bar(bins_labels, accs, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot(bins_labels, midpoints, "k--", linewidth=1.5, label="Expected (perfect calibration)", zorder=5)
        ax.axhline(0.5, color="lightgray", linestyle=":", lw=1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Model confidence bucket (predicted favorite win %)")
        ax.set_ylabel("Actual accuracy in that bucket")
        ax.set_title(
            "How accurate is the model at each confidence level?",
            fontsize=12, fontweight="bold",
        )
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        for bar, acc, n in zip(bars, accs, counts):
            if n > 0 and not np.isnan(acc):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    acc + 0.025,
                    f"{acc:.0%}\n(n={n})",
                    ha="center", fontsize=9,
                )
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.divider()

    # Full predictions table
    with st.expander("Full predictions table (all fights)", expanded=False):
        disp_cols = [c for c in ["EVENT", "DATE", "BOUT", "winner", "p_pred", "favorite", "favorite_win_prob", "favorite_won", "upset"] if c in fp.columns]
        disp = fp[disp_cols].copy()
        disp["p_pred"] = disp["p_pred"].round(3)
        disp["favorite_win_prob"] = disp["favorite_win_prob"].round(3)
        st.dataframe(disp.sort_values("DATE", ascending=False), use_container_width=True)

    st.download_button(
        "⬇ Download predictions (CSV)",
        fp.to_csv(index=False),
        file_name="ufc_predictions.csv",
        mime="text/csv",
    )


# =========================================================================
# 4  COPULA SIMULATOR TAB
# =========================================================================
with tab_copula:
    st.header("Gaussian Copula — Fight Card Simulator")
    st.markdown(
        "Model **correlated fight outcomes** on a single card.  \n"
        "When ρ > 0, fight results are no longer independent — upsets (or favorites) cluster together."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        n_fights = st.slider("Fights on card", 3, 15, 5)
    with col_cfg2:
        rho = st.slider(
            "Intra-card correlation ρ",
            0.0, 0.6, 0.1, 0.01,
            help="0 = independent; higher = outcomes more correlated",
        )

    st.subheader("Favorite win probabilities per fight")
    probs = []
    cols_sliders = st.columns(min(n_fights, 5))
    for i in range(n_fights):
        with cols_sliders[i % len(cols_sliders)]:
            probs.append(st.slider(f"Fight {i + 1}", 0.50, 0.99, 0.65, 0.01, key=f"cp_{i}"))

    n_sims = st.select_slider(
        "Monte Carlo simulations", options=[1_000, 5_000, 10_000, 50_000, 100_000], value=10_000
    )

    if st.button("▶ Run simulation", type="primary"):
        R = np.full((n_fights, n_fights), rho)
        np.fill_diagonal(R, 1.0)
        copula = GaussianCopula(n_fights, R)
        outcomes = copula.simulate_card(probs, n_simulations=n_sims)
        total_wins = outcomes.sum(axis=1)

        p_sweep = (total_wins == n_fights).mean()
        p_naive = float(np.prod(probs))
        exp_upsets = n_fights - outcomes.mean(axis=0).sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P(all favorites win)", f"{p_sweep:.4f}")
        c2.metric("Naive (independent)", f"{p_naive:.4f}")
        c3.metric("Copula / Naive", f"{p_sweep / p_naive:.2f}×" if p_naive > 0 else "N/A")
        c4.metric("Avg upsets per card", f"{n_fights - outcomes.mean(axis=0).sum():.2f}")

        rho_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        sweep_ps = []
        for rv in rho_sweep:
            R2 = np.full((n_fights, n_fights), rv)
            np.fill_diagonal(R2, 1.0)
            sweep_ps.append(GaussianCopula(n_fights, R2).favorites_sweep_probability(probs, n_simulations=n_sims))

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].hist(total_wins, bins=range(n_fights + 2), color="steelblue",
                     edgecolor="white", density=True, align="left")
        axes[0].set_xlabel("Number of favorites winning")
        axes[0].set_ylabel("Probability")
        axes[0].set_title(f"Distribution (ρ = {rho})")
        axes[0].set_xticks(range(n_fights + 1))
        axes[0].spines[["top", "right"]].set_visible(False)

        axes[1].plot(rho_sweep, sweep_ps, "o-", color="steelblue", linewidth=2, label="Copula")
        axes[1].axhline(p_naive, color="red", linestyle="--", label=f"Naive = {p_naive:.4f}")
        axes[1].set_xlabel("ρ")
        axes[1].set_ylabel("P(all favorites win)")
        axes[1].set_title("Sweep probability vs. correlation")
        axes[1].legend(); axes[1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.subheader("Correlation effect table")
        eff_df = GaussianCopula(n_fights).correlation_effect(probs, rho_values=rho_sweep, n_simulations=n_sims)
        st.dataframe(eff_df, use_container_width=True)


# =========================================================================
# 5  DIAGNOSTICS TAB
# =========================================================================
with tab_diag:
    st.header("Model Diagnostics")

    fights_pred = st.session_state.get("fights_pred")
    if fights_pred is None:
        st.warning("Fit the model in the **Model & Ratings** tab first.")
        st.stop()

    fp = fights_pred.copy()
    fp["DATE"] = pd.to_datetime(fp["DATE"])
    fp["favorite"] = fp.apply(
        lambda row: row["fighter1"] if row["p_pred"] >= 0.5 else row["fighter2"], axis=1
    )
    fp["favorite_win_prob"] = fp["p_pred"].apply(lambda p: max(p, 1.0 - p))
    fp["favorite_won"] = fp["winner"] == fp["favorite"]
    fp["upset"] = ~fp["favorite_won"]
    fp["p_upset"] = 1.0 - fp["favorite_win_prob"]

    outcomes = (fp["winner"] == fp["fighter1"]).astype(int).values
    predictions = fp["p_pred"].values
    valid_mask = ~np.isnan(predictions)
    p_v, o_v = predictions[valid_mask], outcomes[valid_mask]

    if len(p_v) == 0:
        st.warning("No valid predictions to analyse.")
        st.stop()

    brier = ScoringRules.brier_score(p_v, o_v)
    baseline = ScoringRules.baseline_brier(o_v)
    ll = ScoringRules.log_loss(p_v, o_v)

    st.subheader("Scoring Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brier Score", f"{brier:.4f}", help="Lower is better; 0 is perfect")
    c2.metric("Baseline Brier (0.5 model)", f"{baseline:.4f}")
    c3.metric("Skill vs baseline", f"{baseline - brier:+.4f}", help="Positive = model beats naive baseline")
    c4.metric("Log Loss", f"{ll:.4f}")

    n_cal_bins = st.slider("Calibration bins", 3, 15, 5)

    st.subheader("Residual Diagnostics")
    residuals = p_v - o_v

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Residual Diagnostics", fontsize=13, fontweight="bold")

    axes[0].scatter(p_v, residuals, alpha=0.55, color="steelblue", s=30, edgecolors="none")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted P(fighter1 wins)"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted"); axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].hist(residuals, bins=15, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution")
    axes[1].spines[["top", "right"]].set_visible(False)

    cal = ScoringRules.calibration_curve(p_v, o_v, n_bins=n_cal_bins)
    axes[2].plot([0, 1], [0, 1], "r--", label="Perfect")
    if len(cal) > 0:
        axes[2].scatter(
            cal["mean_predicted"], cal["actual_win_rate"],
            s=cal["n_fights"] * 6, color="steelblue", label="Model", zorder=5,
        )
        axes[2].plot(cal["mean_predicted"], cal["actual_win_rate"], color="steelblue")
    axes[2].set_xlabel("Mean predicted probability"); axes[2].set_ylabel("Actual win rate")
    axes[2].set_title("Calibration Curve"); axes[2].legend()
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Event-Level Residuals")
    fp_valid = fp[valid_mask].copy()
    ev_stats = fp_valid.groupby("EVENT").agg(
        n_fights=("upset", "size"),
        expected_upsets=("p_upset", "sum"),
        actual_upsets=("upset", "sum"),
        avg_fav_prob=("favorite_win_prob", "mean"),
        date=("DATE", "first"),
    ).reset_index()
    ev_stats["residual"] = ev_stats["actual_upsets"] - ev_stats["expected_upsets"]

    st.dataframe(
        ev_stats[["EVENT", "date", "n_fights", "expected_upsets", "actual_upsets", "residual"]]
        .round(3)
        .sort_values("date", ascending=False),
        use_container_width=True,
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(ev_stats["residual"], bins=max(5, len(ev_stats) // 2), edgecolor="black", color="steelblue")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Residual (actual − expected upsets)")
    ax2.set_ylabel("Number of events")
    ax2.set_title("Distribution of Event-Level Residuals", fontsize=12, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2); plt.close(fig2)


# =========================================================================
# 6  SCRAPER TAB
# =========================================================================
with tab_scraper:
    st.header("UFC Stats Scraper")
    st.markdown(
        "Pull live fight data from [ufcstats.com](http://ufcstats.com/statistics/events/completed?page=all). "
        "Scraped data is auto-converted to the model format."
    )

    try:
        import requests
        from bs4 import BeautifulSoup
        _HAS_SCRAPER = True
    except ImportError:
        _HAS_SCRAPER = False

    if not _HAS_SCRAPER:
        st.error(
            "Scraping requires `requests` and `beautifulsoup4`.  \n"
            "Install with: `pip install requests beautifulsoup4`"
        )
        st.stop()

    @st.cache_data(ttl=3600, show_spinner="Fetching event list…")
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
                cells = row.select("td.b-statistics__table-col")
                date_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                loc_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                events.append({"EVENT": name, "DATE": date_text, "LOCATION": loc_text, "url": href})
        return pd.DataFrame(events)

    @st.cache_data(ttl=3600, show_spinner="Scraping event fights…")
    def scrape_event_fights(event_url):
        resp = requests.get(event_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover")
        fights_out = []
        for row in rows:
            cols = row.select("td.b-fight-details__table-col")
            if len(cols) < 8:
                continue
            fighter_links = cols[1].select("a")
            if len(fighter_links) < 2:
                continue
            f1, f2 = fighter_links[0].get_text(strip=True), fighter_links[1].get_text(strip=True)
            wl_links = cols[0].select("a")
            result = wl_links[0].get_text(strip=True).lower() if wl_links else ""
            method = cols[7].get_text(strip=True).split("\n")[0].strip() if len(cols) > 7 else ""
            rnd = cols[8].get_text(strip=True) if len(cols) > 8 else ""
            tm = cols[9].get_text(strip=True) if len(cols) > 9 else ""
            outcome = "W/L" if result == "win" else ("L/W" if result == "loss" else result)
            fights_out.append({"BOUT": f"{f1} vs. {f2}", "OUTCOME": outcome, "METHOD": method, "ROUND": rnd, "TIME": tm})
        return pd.DataFrame(fights_out)

    if st.button("🔄 Fetch event list from ufcstats.com"):
        try:
            edf = scrape_event_list()
            st.session_state["scraped_events"] = edf
            st.success(f"Found {len(edf)} events.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    scraped_events = st.session_state.get("scraped_events")
    if scraped_events is not None and len(scraped_events) > 0:
        st.dataframe(scraped_events[["EVENT", "DATE", "LOCATION"]].head(30), use_container_width=True)
        selected = st.multiselect(
            "Select events to scrape",
            scraped_events["EVENT"].tolist(),
            default=scraped_events["EVENT"].tolist()[:3],
        )
        if st.button("⬇ Scrape selected events") and selected:
            all_fights, all_evt_det = [], []
            prog = st.progress(0, text="Scraping…")
            for i, evt_name in enumerate(selected):
                row = scraped_events[scraped_events["EVENT"] == evt_name].iloc[0]
                prog.progress((i + 1) / len(selected), text=f"Scraping {evt_name}…")
                try:
                    fdf = scrape_event_fights(row["url"])
                    fdf["EVENT"] = evt_name
                    all_fights.append(fdf)
                    all_evt_det.append({"EVENT": evt_name, "DATE": row["DATE"], "LOCATION": row["LOCATION"]})
                    time.sleep(0.5)
                except Exception as e:
                    st.warning(f"Failed for {evt_name}: {e}")
            prog.empty()
            if all_fights:
                combined = pd.concat(all_fights, ignore_index=True)
                evt_df = pd.DataFrame(all_evt_det)
                evt_df["DATE"] = pd.to_datetime(evt_df["DATE"], format="mixed", errors="coerce")
                st.success(f"Scraped {len(combined)} fights from {len(all_evt_det)} events.")
                st.dataframe(combined, use_container_width=True)
                try:
                    parsed = _parse_fights_df(combined)
                    merged = parsed.merge(evt_df[["EVENT", "DATE"]], on="EVENT", how="left")
                    merged = merged.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
                    st.session_state["fights"] = merged
                    st.success(
                        f"{len(merged)} decisive fights loaded. "
                        "Go to **Model & Ratings** to fit."
                    )
                except Exception as e:
                    st.error(f"Parse error: {e}")
                st.download_button(
                    "⬇ Download scraped data (TSV)",
                    combined.to_csv(index=False, sep="\t"),
                    file_name="ufc_scraped.tsv",
                    mime="text/tab-separated-values",
                )
