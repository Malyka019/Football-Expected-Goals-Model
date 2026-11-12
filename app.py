"""Streamlit app for the football expected-goals model.

Run it with:

    streamlit run app.py

All the modelling lives in `src/`; this file only collects input, calls the
predictor, and draws the result. Keeping it thin means the app cannot disagree
with the command line or the report.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import get_predictor

TOURNAMENTS = {
    "FIFA World Cup": "world_cup",
    "Continental final (Euro, Copa America, AFCON...)": "continental_final",
    "World Cup qualifier": "qualifier",
    "Nations League": "nations_league",
    "Friendly": "friendly",
}


@st.cache_resource(show_spinner="Loading model and team state...")
def load_predictor():
    """Load once per session: fitting and building state takes a few seconds."""
    return get_predictor()


def heatmap_figure(prediction):
    """0-5 scoreline grid as a matplotlib figure (percentages)."""
    matrix = prediction.score_matrix.to_numpy() / 100.0

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    image = ax.imshow(matrix, cmap="Blues", origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="probability")

    peak = matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value < 0.002:
                continue
            ax.text(
                j, i, f"{100 * value:.1f}", ha="center", va="center", fontsize=7,
                color="white" if value > 0.55 * peak else "black",
            )

    ax.set_xlabel(f"{prediction.team_b} goals")
    ax.set_ylabel(f"{prediction.team_a} goals")
    ax.set_title("Scoreline probability (%)")
    fig.tight_layout()
    return fig


def outcome_figure(prediction):
    """Win/draw/win as a horizontal bar chart in percentage points.

    Drawn with matplotlib rather than `st.bar_chart`: the Vega chart rendered
    with equal-height bars and an unreadable axis in the browser, and it sorts
    the categories alphabetically, so the draw only lands in the middle by luck.
    """
    labels = [f"{prediction.team_a} win", "Draw", f"{prediction.team_b} win"]
    values = 100.0 * np.array(
        [prediction.p_win_a, prediction.p_draw, prediction.p_win_b]
    )
    colours = ["#2f6db5", "#b8c2cc", "#8b1d2e"]

    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    bars = ax.barh(labels[::-1], values[::-1], color=colours[::-1], height=0.62)
    for bar, value in zip(bars, values[::-1]):
        ax.text(
            value + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", fontsize=9,
        )
    ax.set_xlim(0, max(values) * 1.20 + 1.0)
    ax.set_xlabel("probability (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Football expected goals", page_icon="⚽", layout="wide")
    st.title("⚽ Football expected goals")
    st.caption(
        "Poisson goal model over international results. Features are point-in-time "
        "(Elo, recent form, attack/defence strength); W/D/L comes from the full "
        "scoreline distribution, not just the 0-5 grid shown."
    )

    predictor = load_predictor()
    teams = predictor.teams

    left, right = st.columns(2)
    default_a = teams.index("Brazil") if "Brazil" in teams else 0
    default_b = teams.index("Argentina") if "Argentina" in teams else min(1, len(teams) - 1)
    team_a = left.selectbox("Team A (listed first)", teams, index=default_a)
    team_b = right.selectbox("Team B", teams, index=default_b)

    left, right = st.columns(2)
    neutral = left.checkbox("Neutral venue", value=True)
    tournament = right.selectbox("Tournament", list(TOURNAMENTS))

    if st.button("Predict", type="primary"):
        try:
            prediction = predictor.predict_match(
                team_a, team_b, neutral=neutral, tournament=tournament
            )
        except (KeyError, ValueError) as error:
            st.error(str(error))
            return

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"{prediction.team_a} expected goals",
                     f"{prediction.expected_goals_a:.2f}")
        col_b.metric("Most likely scoreline", prediction.most_likely_scoreline)
        col_c.metric(f"{prediction.team_b} expected goals",
                     f"{prediction.expected_goals_b:.2f}")

        st.subheader("Outcome probabilities")
        figure = outcome_figure(prediction)
        st.pyplot(figure)
        plt.close(figure)

        chart, table = st.columns([1, 1])
        with chart:
            figure = heatmap_figure(prediction)
            st.pyplot(figure)
            plt.close(figure)
        with table:
            st.subheader("Most likely scorelines")
            scorelines_table = pd.DataFrame(
                prediction.top_scorelines, columns=["goals A", "goals B", "probability"]
            )
            scorelines_table["probability"] = (
                100 * scorelines_table["probability"]
            ).round(1)
            st.dataframe(scorelines_table, hide_index=True, use_container_width=True)

            st.caption(
                f"model: {prediction.model_name} · "
                f"{'neutral venue' if prediction.neutral else 'home advantage to ' + prediction.team_a} · "
                f"{prediction.tournament}"
            )

    with st.expander("How this model works, and when not to trust it"):
        st.markdown(
            """
- **Goals are Poisson.** Each team's expected goals come from a log-linear model
  over Elo, recent form, attack/defence strength and venue.
- **No leakage.** Every feature is computed from matches before kick-off, and a
  test asserts that by truncating the dataset.
- **Uncertainty is large.** A 64-match World Cup cannot distinguish models that
  differ by 2-3 points of RPS. Treat probabilities as soft, not precise.
- **It knows nothing about injuries, squads, travel or motivation.** It prices
  history, not news.
            """
        )


if __name__ == "__main__":
    main()
