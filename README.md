# Football Expected Goals Model

A team-level expected-goals model for international football. Given two national
teams and a match context (neutral venue, tournament), it predicts how many goals
each side will score, turns that into win/draw/loss probabilities and a scoreline
distribution, and can Monte Carlo a whole World Cup.

Not to be confused with shot-level xG: this models **team** scoring rates, not
the probability that an individual shot goes in.

**Data:** 49,939 internationals from 1872 to October 2026 (Kaggle "International
football results from 1872 to present", Mart Jürisoo), 337 teams.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data: drop results.csv from Kaggle into data/raw/ .
# (No download? `--synthetic` generates structurally realistic data so every
#  command still runs. It writes a marker file and every report says so.)
python -m src.pipeline build
python -m src.pipeline features
python -m src.pipeline tune        # GBM hyperparameters, validation only
python -m src.pipeline models      # all six models compared
python -m src.pipeline report      # reports/results.md + charts

python -m src.pipeline fit         # pickle the models
python -m src.pipeline predict --team-a Brazil --team-b Argentina
python -m src.pipeline simulate    # 10k World Cups
streamlit run app.py               # interactive predictor

pytest -q                          # 104 tests
```

---

## What it produces

```python
from src.predict import predict_match

prediction = predict_match("Brazil", "Argentina", neutral=True)
prediction.expected_goals_a   # 0.77
prediction.expected_goals_b   # 1.42
prediction.p_win_a, prediction.p_draw, prediction.p_win_b
prediction.score_matrix       # 6x6 DataFrame of percentages
prediction.top_scorelines     # [(0,1,0.159), ...]
```

```
Brazil vs Argentina  (neutral venue, FIFA World Cup)
expected goals : Brazil 0.77  -  1.42 Argentina
win / draw / win: 20.4%  /  27.2%  /  52.4%
most likely    : 0-1
```

Argentina are favourites because they were rated ~165 Elo points higher at the
cut-off, and because a neutral venue removes Brazil's home advantage.

---

## Results

Fitted on matches up to 2017, hyperparameters chosen on 2018-2021, evaluated
once on 2022. Full write-up with charts: [`reports/results.md`](reports/results.md).

### 2022 (970 matches), 300-resample bootstrap over matches

| model | RPS (95% CI) | log loss | accuracy | MAE (goals) |
|---|---|---|---|---|
| LightGBM (Poisson) | **0.181 [0.172, 0.192]** | **0.909** | 0.579 | 0.907 |
| Poisson GLM | 0.183 [0.174, 0.194] | 0.916 | 0.579 | 0.897 |
| Dixon-Coles | 0.189 [0.179, 0.197] | 0.934 | 0.574 | **0.898** |
| Form heuristic | 0.215 [0.207, 0.222] | 1.016 | 0.505 | 0.964 |
| Home/away average | 0.226 [0.221, 0.230] | 1.042 | 0.498 | 1.028 |
| League average | 0.237 [0.234, 0.240] | 1.072 | 0.275 | 1.077 |

### The two World Cups (64 matches each)

| model | RPS 2018 | RPS 2022 | accuracy 2022 |
|---|---|---|---|
| Dixon-Coles | **0.206** | **0.205 [0.169, 0.246]** | 0.562 |
| LightGBM | 0.214 | 0.223 [0.185, 0.266] | 0.484 |
| Poisson GLM | 0.217 | 0.224 [0.184, 0.270] | 0.500 |
| Form heuristic | 0.216 | 0.230 [0.203, 0.260] | 0.422 |
| League average | 0.240 | 0.236 [0.224, 0.248] | 0.328 |

Reading this honestly:

- **Every fitted model crushes every baseline.** RPS drops from 0.237 (league
  average) and 0.215 (form heuristic, which already uses the engineered features)
  to 0.18-0.19. On a season's worth of matches that is a large, unambiguous gain.
- **LightGBM and the GLM are statistically tied on 2022** — 0.181 [0.172, 0.192]
  vs 0.183 [0.174, 0.194]. Two thousandths of RPS is not a result. Boosting's
  advantage over a well-specified additive model is small here, which is what
  theory predicts when the data-generating process is close to multiplicative.
- **Dixon-Coles wins both World Cups** (0.206 and 0.205) despite being third on
  the full-year test block — exactly the behaviour its design predicts. Time decay
  and the rho correction matter most in tight, low-scoring tournament matches.
- **Accuracy is the least useful column.** The league-average baseline gets 27.5%
  by always predicting a home win (2022 was unusually neutral-venue heavy), while
  the home/away baseline gets 49.8% while knowing nothing about teams. A metric
  where "always guess home" scores 28% and "guess the venue average" scores 50%
  cannot rank models; RPS and log loss drive the conclusions here.
- Note the intervals: on 64 matches, a World Cup RPS of 0.205 carries
  ±0.04. The model rankings across tournaments should be read as "these three are
  close", not as a precise ordering.

Calibration charts show the fitted models sit close to the diagonal in the
0.2-0.6 band and never predict above ~72% for any outcome — football outcome
probabilities live in a narrow band, which is precisely why accuracy is a poor
guide and proper scoring rules are not.

---

## Methodology

### Data

Kaggle *International football results from 1872 to present* (Mart Jürisoo):
`date, home_team, away_team, home_score, away_score, tournament, city, country,
neutral`. Cleaning drops 2 matches without a usable score or date, strips
whitespace **before** de-duplicating fixtures (otherwise "Brazil " and "Brazil"
split a team's history in two), and maps ~150 free-text tournament names onto six
categories with importance weights.

49,939 matches: 18,257 friendlies, 15,988 qualifiers, 9,302 other, 4,007
continental finals, 1,257 Nations League, 1,128 World Cup. 13,225 at neutral
venues, 2.94 goals per match, 49.0% home wins, 22.7% draws.

### Features, and the leakage rule

One rule: **a feature for a match may only use information available before
kick-off.**

| group | features |
|---|---|
| Rating | Elo (own, opponent, difference) |
| Form | goals for/against, points, win rate over the last 5 and 10 matches |
| Strength | attack and defence, relative to the time-varying league average, adjusted for opponent quality |
| Context | home advantage, neutral venue, tournament category |
| History | head-to-head meetings, goal difference, win rate |

- **Elo** starts at 1500, uses K = 60 x tournament importance, scales the update
  by goal difference (1.0 for a one-goal win up to ~2.0 for a thrashing) and
  applies a +100 home bonus only at non-neutral venues. The pre-match rating is
  recorded, then updated — never the other way round. On the real data this
  produces a credible table: Spain 2361, Argentina 2260, England 2226, France 2182.
- **Rolling form** shifts each team's series by one match, so the current match
  can never be inside its own window.
- **Attack/defence** divides exponentially weighted goals by an *expanding* league
  average over previous dates only, then damps the opponent adjustment by a
  square-root exponent. A full-strength adjustment double-counts in small
  samples: it cancels the very signal it was meant to isolate.

**`tests/test_no_leakage.py`** is the enforcement mechanism. It recomputes the
entire feature table on data truncated at three dates and asserts the surviving
rows are byte-identical, then randomises all scores *after* a cutoff and asserts
nothing before it moves. That second variant catches whole-dataset aggregates
(a global mean or z-score), which truncation alone can miss.

### Splits

Strict calendar partition, never shuffled: train <= 2017 (43,309 matches),
validation 2018-2021 (3,562), test 2022 (970), plus an `unused` bucket for 2023
onwards. `validate_splits` asserts the blocks are ordered, non-empty and
non-overlapping, and that no match appears twice. World Cup blocks filter on the
tournament category as well as the date window — the raw date window returns 87
matches for a 64-match tournament because unrelated internationals are played in
the same weeks.

### Models

| model | what it is |
|---|---|
| League average | training-set goals per team, for everyone |
| Home/away average | the same, split by whether home advantage applies |
| Form heuristic | `(own recent scoring + opponent's recent conceding) / 2`, unfitted |
| Poisson GLM | `goals ~ Poisson`, log link, on the engineered features |
| Dixon-Coles | weighted MLE with per-team attack/defence, two-year time decay and the rho low-score correction |
| LightGBM | Poisson objective, shallow trees, tuned on validation only |

Everything is fitted on **team-match rows** (two per match, one per side), so a
single symmetric model describes both teams in a fixture.

Why Poisson: goals are a count of rare events in a fixed window, the variance is
close to the mean, rates cannot go negative, and the log link makes coefficients
multiplicative — `exp(beta)` is a ratio of scoring rates. The GLM's fitted
coefficients are readable: +0.133 per 100 own Elo points, −0.177 per 100 opponent
Elo, +0.360 for home advantage (×1.43 goals).

Dixon-Coles adds two things, both implemented from scratch:
`w = 0.5 ** (age_days / 730)` so old matches count less, and the four-cell tau
correction (`tau(0,0) = 1 - lam_h*lam_a*rho`, etc.), fitted jointly. It is
optimised by L-BFGS with an **analytic gradient that includes the tau term's
dependence on the rates** — the 0-0 cell couples rho to team strength, and
ignoring that coupling biases rho. Fitted on the real data: rho = −0.062, home
advantage = ×1.35 goals, 203 iterations, 3.8 seconds.

Two parameterisation details that mattered on the full dataset:

- A sum-to-zero constraint is the textbook way to identify attack/defence, but
  with 317 teams it makes every gradient depend on one shared parameter. L-BFGS
  ran 1500 iterations without converging (26 s). A small ridge penalty plus a
  **post-hoc recentring** that is mathematically neutral on the likelihood fixed
  both: convergence in 203 iterations, identical predictions.
- Tuning the boosting model on validation picked `num_leaves=31`; all six
  configurations landed within 0.002 RPS of each other, which says the model is
  noise-limited, not hyperparameter-limited.

### Rate ceiling

The log link has no upper bound, so a large rating gap extrapolates into
nonsense: Brazil vs San Marino came out at 11.2 expected goals at a neutral
venue and 15.9 with home advantage, a "most likely scoreline" of 15-0 and a 100%
win. Every model's rates now pass through `scorelines.cap_rate`, which is the
identity up to a knee (3.5) and then bends smoothly towards an asymptote (6.0),
continuous with slope 1 at the knee — so a 1.2-goal favourite is untouched and an
11-goal extrapolation becomes 5.9.

Both constants were chosen on validation only, where the grid was flat (log loss
0.8611 capped vs 0.8613 raw, RPS identical at 0.1673). On the test block RPS is
unchanged at 0.1835 while MAE improves from 0.900 to 0.897, because the 36 of
1,940 rates that were overstated get pulled back. The cap repairs the tail and
leaves the body of the distribution alone.

### Evaluation

- Goal level: MAE, RMSE, Poisson deviance (the loss a Poisson model minimises).
- Outcome level: accuracy, log loss, RPS. RPS is the one to trust: a proper
  scoring rule that respects the ordering home win < draw < away win.
- Every metric carries a percentile bootstrap interval, resampling **matches**
  rather than team-rows. The two rows of a match are one event; treating them as
  independent would shrink intervals by roughly root two and make noise look like
  signal.

### Tournament simulation

`python -m src.pipeline simulate` plays 10,000 32-team World Cups (eight groups,
R16 to final) in about six seconds by vectorising every round across simulations
and precomputing the 32x32 neutral-venue rate matrix in one model call.

Latest run: **Spain 28.2%, Argentina 20.9%, England 17.6%, France 8.3%, Brazil
4.6%.** Each has a ±1 standard error of roughly 0.4 points.

---

## Limitations

1. **Small tournament samples.** A World Cup is 64 matches; its RPS interval is
   about ±0.04. Cross-tournament rankings should be read as "these three are
   close", not as a precise ordering.
2. **Attack/defence are weak features on the full history.** Their correlation
   with goals scored is 0.039 (Elo difference: 0.395; 10-match form: 0.219). Once
   opponent quality is divided out across 337 teams of wildly varying standard,
   little team-specific signal survives — the fitted models lean on Elo and form.
   Reported rather than hidden; a rolling per-team Poisson refit would be the
   principled fix.
3. **No squad information.** Injuries, suspensions, rotation, travel, manager
   changes and motivation are invisible. This model prices history.
4. **Correlated forecasts in the simulation.** The Monte Carlo reuses one set of
   expected-goal rates for every run, so it measures tournament randomness, not
   parameter uncertainty. The standard errors are binomial sampling error.
5. **Simplified knockout model.** Extra time is a third of the 90-minute rate;
   shootouts are a coin flip.
6. **32-team format only.** The 2026 48-team format (12 groups, best third-placed
   teams) is not implemented.
7. **Elo, not FIFA ranking.** FIFA rankings only start properly in 1992 and are
   awkward to source; Elo from match history is complete and free. Merging
   defunct nations is available (`--rename-former`) but off by default, because
   folding the USSR into Russia gives a team a history it never had.
8. **Independence within a match.** Except for the Dixon-Coles correction, home
   and away goals are modelled as independent.
9. **No betting edge.** Beating the market requires information the market does
   not have, plus margins that swallow small edges. This is a modelling exercise.
10. **Extreme mismatches are capped, not modelled.** Brazil vs San Marino now
    prices at 5.9 expected goals and a most likely 5-0. That is a repair, not an
    estimate: the ceiling is a shape applied to the tail, fitted only in the
    sense that its two constants were picked off a flat validation grid. A team
    genuinely capable of nine goals would be under-predicted, and the cap does
    nothing for the far more common problem of a modest favourite being ranked
    wrongly.

---

## Project structure

```
football-goals-model/
├── app.py                  Streamlit predictor
├── data/
│   ├── raw/                results.csv as downloaded (gitignored)
│   ├── interim/            cleaned match table + cleaning report
│   └── processed/          long-format table + feature table
├── notebooks/
│   ├── 01_eda.ipynb        goal distributions, home advantage, coverage
│   └── 02_features.ipynb   Elo trajectories, feature sanity checks
├── reports/
│   ├── results.md          results table, intervals, charts
│   ├── results_*.csv/md    per-model comparison tables
│   ├── simulation.csv      champion probabilities
│   └── figures/            14 charts
├── src/
│   ├── config.py           paths, split dates, tournament importance
│   ├── data.py             load -> clean -> long format (+ synthetic fallback)
│   ├── features.py         Elo, rolling form, attack/defence, head-to-head
│   ├── splits.py           chronological train/validation/test
│   ├── evaluate.py         metrics, bootstrap intervals, results tables
│   ├── scorelines.py       score matrices, win/draw/loss, Dixon-Coles correction
│   ├── report.py           tables + calibration/comparison/RPS charts
│   ├── predict.py          predict_match, team state, pickled artifacts
│   ├── simulate.py         Monte Carlo World Cup
│   ├── pipeline.py         CLI: build, features, baseline, tune, models,
│   │                       report, fit, predict, simulate
│   └── models/
│       ├── base.py         GoalModel interface
│       ├── baseline.py     league average / home-away / form
│       ├── poisson_glm.py  statsmodels Poisson GLM
│       ├── dixon_coles.py  weighted MLE, time decay, rho, ridge + recentring
│       └── gbm_poisson.py  LightGBM + validation-only tuner
└── tests/                  104 tests: data, leakage, metrics, splits,
                            models, report, predictor, simulation
```

## Tests

```
104 passed
```

The ones worth reading: `test_features_are_unchanged_when_the_future_is_removed`,
`test_rolling_form_excludes_the_current_match`,
`test_dc_gradient_matches_finite_differences`,
`test_dc_recentring_does_not_change_predictions`,
`test_rps_per_match_averages_to_the_overall_rps`,
`test_a_dominant_team_almost_always_wins`.

---

## Resume blurb

**Football Expected Goals Model** — *Python, pandas, statsmodels, LightGBM,
Streamlit*

> Built an end-to-end expected-goals model on 49,900 international matches,
> predicting each team's goal rate from point-in-time Elo ratings, rolling form
> and attack/defence strength. Enforced a strict no-leakage contract with tests
> that recompute features on truncated and future-randomised data, and evaluated
> on a chronological split with a single held-out World Cup year. Compared three
> baselines, a Poisson GLM, a from-scratch Dixon-Coles model (time decay, rho
> correction, analytic gradient) and LightGBM: RPS improved from 0.237
> (league-average) and 0.215 (form heuristic) to 0.181, with bootstrap intervals
> showing the top three models statistically tied while Dixon-Coles won both
> World Cups. Shipped a CLI and Streamlit app for `predict_match()` plus a
> 10,000-run Monte Carlo of the tournament.

Three bullets, if a shorter format is needed:

- Modelled international football goals as Poisson rates from point-in-time Elo,
  form and strength features; 0.181 RPS vs 0.237 for a league-average baseline on
  a held-out tournament year, with bootstrap intervals over matches.
- Implemented Dixon-Coles (exponential time decay + low-score rho correction) from
  scratch with an analytic-gradient MLE verified against finite differences, and
  fixed a 1500-iteration non-convergence by replacing a sum-to-zero constraint
  with a ridge penalty plus likelihood-neutral recentring; added a rate
ceiling that repaired 15-0 extrapolations without moving test RPS.
- Proved no data leakage with a truncation/future-randomisation test suite; 104
  tests covering features, metrics, splits, models and the predictor.

---

## References

- Dixon, M. & Coles, S. (1997). *Modelling association football scores and
  inefficiencies in the football betting market.* JRSS-C.
- Maher, M. (1982). *Modelling association football scores.* Statistica Neerlandica.
- World Football Elo Ratings — rating formula and K-factor conventions.
- Jürisoo, M. *International football results from 1872 to present* (Kaggle).
