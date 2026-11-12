"""Step 8: Monte Carlo simulation of a 32-team World Cup.

Why simulate instead of "predict the winner"?
---------------------------------------------
A single predicted champion is not a forecast, it is a guess with no error bar.
Football's knockout rounds are close to coin flips between good teams, so the
useful output is a distribution: "Brazil win 14% of tournaments, and the 95%
interval on that is plus or minus 0.7 points".

How it works
------------
1. Precompute a 32x32 matrix of expected goals at neutral venues (one model call
   for all 992 ordered pairs, not 992 calls).
2. Draw a pot-based group stage (four pots of eight, standard seeding).
3. Play 8 groups of 4: each match is two independent Poisson draws. Rank on
   points, goal difference, goals scored, then lots (a random tie-break, which
   is what FIFA does after head-to-head).
4. R16 -> QF -> SF -> final. A draw after 90 minutes goes to "extra time"
   (thirty more minutes, so rates are divided by three); a draw after that is a
   penalty shootout, modelled as a fair coin flip.

The shootout assumption is a simplification, and a deliberate one: empirically
shootouts are close to 50/50 and the model has no information about them. Better
to state it than to pretend extra time is well modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, predict

N_SIMS = 10_000
N_GROUPS = 8
GROUP_SIZE = 4
TEAMS_REQUIRED = N_GROUPS * GROUP_SIZE


@dataclass
class TournamentResult:
    """Champion and stage-reached probabilities per team."""

    table: pd.DataFrame
    groups: np.ndarray
    n_sims: int
    teams: list[str]

    @property
    def champion(self) -> pd.Series:
        return self.table.set_index("team")["p_champion"]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def build_rate_matrix(predictor, teams: list[str]) -> np.ndarray:
    """lam[i, j] = expected goals for team i against team j at a neutral venue.

    Built with a single `predict_lambda` call over all ordered pairs, which is
    roughly two orders of magnitude faster than calling `predict_match` 992 times
    (and it skips building a score matrix for every pair, which we do not need).
    """
    n = len(teams)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    rows = []
    for i, j in pairs:
        row = predict.fixture_rows(
            predictor.state, predictor.matches, teams[i], teams[j],
            neutral=True, tournament_category="world_cup",
        ).iloc[0]
        rows.append(row)

    rates = predictor.model.predict_lambda(pd.DataFrame(rows))

    lam = np.zeros((n, n), dtype=float)
    for (i, j), value in zip(pairs, rates):
        lam[i, j] = float(value)
    return lam


def draw_groups(
    teams: list[str], seed: int = config.RANDOM_SEED, strength: np.ndarray | None = None
) -> np.ndarray:
    """Pot-based draw: four pots of eight, one team per pot in each group.

    Seeding uses `strength` (Elo by default) so the draw looks like the real
    thing rather than putting the eight best teams in one group.
    """
    if len(teams) != TEAMS_REQUIRED:
        raise ValueError(f"need exactly {TEAMS_REQUIRED} teams, got {len(teams)}")

    rng = np.random.default_rng(seed)
    if strength is None:
        order = np.arange(len(teams))
    else:
        order = np.argsort(-np.asarray(strength, dtype=float))

    pots = [order[k * N_GROUPS:(k + 1) * N_GROUPS] for k in range(GROUP_SIZE)]
    groups = [[] for _ in range(N_GROUPS)]

    for pot in pots:
        for group, team_index in zip(groups, rng.permutation(pot)):
            group.append(int(team_index))

    return np.array(groups, dtype=int)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def _knockout(lam: np.ndarray, home: np.ndarray, away: np.ndarray, rng) -> np.ndarray:
    """Play one knockout round; returns the winner of each tie.

    `home` and `away` are (n_sims, k) arrays of team indices, so a whole round is
    simulated for every simulation at once.
    """
    lam_home = lam[home, away]
    lam_away = lam[away, home]

    goals_home = rng.poisson(lam_home)
    goals_away = rng.poisson(lam_away)

    # Extra time: 30 minutes, so a third of the 90-minute rate.
    level = goals_home == goals_away
    if level.any():
        extra_home = rng.poisson(lam_home / 3.0)
        extra_away = rng.poisson(lam_away / 3.0)
        goals_home = np.where(level, goals_home + extra_home, goals_home)
        goals_away = np.where(level, goals_away + extra_away, goals_away)

    # Penalties: a fair coin, by assumption (see module docstring).
    still_level = goals_home == goals_away
    home_wins_shootout = rng.random(goals_home.shape) < 0.5
    home_wins = np.where(still_level, home_wins_shootout, goals_home > goals_away)
    return np.where(home_wins, home, away)


def simulate(
    lam: np.ndarray,
    groups: np.ndarray,
    *,
    n_sims: int = N_SIMS,
    seed: int = config.RANDOM_SEED,
) -> dict[str, np.ndarray | int]:
    """Run the tournament `n_sims` times.

    Returns the per-team stage counts (how many simulations each team reached
    each round) plus the number of simulations.
    """
    rng = np.random.default_rng(seed)
    n_teams = lam.shape[0]
    groups = np.asarray(groups, dtype=int)
    sims = np.arange(n_sims)[:, None]

    points = np.zeros((n_sims, n_teams))
    goal_diff = np.zeros((n_sims, n_teams))
    goals_for = np.zeros((n_sims, n_teams))

    # --- group stage ---------------------------------------------------
    for group in groups:
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                team_a, team_b = int(group[a]), int(group[b])
                goals_a = rng.poisson(lam[team_a, team_b], size=n_sims)
                goals_b = rng.poisson(lam[team_b, team_a], size=n_sims)

                points[:, team_a] += 3 * (goals_a > goals_b) + (goals_a == goals_b)
                points[:, team_b] += 3 * (goals_b > goals_a) + (goals_a == goals_b)
                goal_diff[:, team_a] += goals_a - goals_b
                goal_diff[:, team_b] += goals_b - goals_a
                goals_for[:, team_a] += goals_a
                goals_for[:, team_b] += goals_b

    # Tie-break: points, then goal difference, then goals scored, then lots.
    lots = rng.random((n_sims, n_teams))
    key = points * 1e6 + (goal_diff + 100.0) * 1e3 + goals_for + lots

    group_keys = key[:, groups]                                  # (n_sims, 8, 4)
    order = np.argsort(-group_keys, axis=2)
    ranked = np.take_along_axis(
        np.broadcast_to(groups, group_keys.shape), order, axis=2
    )

    winners = ranked[:, :, 0]     # (n_sims, 8)
    runners = ranked[:, :, 1]

    stage = np.zeros((n_sims, n_teams), dtype=np.int8)
    stage[sims, winners] = 1      # reached the round of 16
    stage[sims, runners] = 1

    # --- knockout rounds ----------------------------------------------
    # R16: group winners avoid their own runner-up.
    home = np.stack([winners[:, g] for g in (0, 2, 4, 6, 1, 3, 5, 7)], axis=1)
    away = np.stack([runners[:, g] for g in (1, 3, 5, 7, 0, 2, 4, 6)], axis=1)
    winners = _knockout(lam, home, away, rng)
    stage[sims, winners] = 2      # reached the quarter-finals

    while winners.shape[1] > 1:
        home, away = winners[:, 0::2], winners[:, 1::2]
        winners = _knockout(lam, home, away, rng)
        next_round = {4: 3, 2: 4}.get(winners.shape[1])
        if next_round is not None:
            stage[sims, winners] = next_round

    champion = winners[:, 0]
    stage[sims[:, 0], champion] = 5

    return {
        "stage": stage,
        "n_sims": n_sims,
        "champion": champion,
    }


def summarise(
    stage: np.ndarray, teams: list[str], n_sims: int
) -> pd.DataFrame:
    """Champion and stage-reached probabilities with standard errors."""
    counts = {level: (stage >= level).sum(axis=0) for level in range(1, 6)}
    champion = counts[5]
    p_champion = champion / n_sims

    table = pd.DataFrame({
        "team": teams,
        "p_champion": p_champion,
        "se_champion": np.sqrt(p_champion * (1.0 - p_champion) / n_sims),
        "p_final": counts[4] / n_sims,
        "p_sf": counts[3] / n_sims,
        "p_qf": counts[2] / n_sims,
        "p_r16": counts[1] / n_sims,
    })
    return table.sort_values(
        ["p_champion", "p_final"], ascending=False
    ).reset_index(drop=True)


def plot_champions(result: "TournamentResult", path, top_n: int = 16):
    """Horizontal bars with +/- one standard error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = result.table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.barh(
        top["team"], 100 * top["p_champion"],
        xerr=100 * top["se_champion"], color="steelblue", capsize=3,
    )
    ax.set_xlabel("probability of winning the tournament (%)")
    ax.set_title(f"Champion probability — {result.n_sims:,} simulations")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_simulation(
    predictor,
    *,
    teams: list[str] | None = None,
    n_sims: int = N_SIMS,
    seed: int = config.RANDOM_SEED,
    groups: np.ndarray | None = None,
) -> TournamentResult:
    """Full pipeline: pick the field, build the rate matrix, simulate, summarise."""
    if teams is None:
        # Default field: the 32 strongest teams by current Elo. Not a real draw --
        # it is a clean, reproducible illustration.
        state = predictor.state.sort_values("elo", ascending=False)
        teams = state.index[:TEAMS_REQUIRED].tolist()
    if len(teams) != TEAMS_REQUIRED:
        raise ValueError(f"need exactly {TEAMS_REQUIRED} teams, got {len(teams)}")

    lam = build_rate_matrix(predictor, teams)

    if groups is None:
        strength = predictor.state.loc[teams, "elo"].to_numpy()
        groups = draw_groups(teams, seed=seed, strength=strength)

    outcome = simulate(lam, groups, n_sims=n_sims, seed=seed)
    table = summarise(outcome["stage"], teams, n_sims)

    return TournamentResult(
        table=table, groups=groups, n_sims=n_sims, teams=list(teams)
    )
