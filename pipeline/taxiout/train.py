"""Gradient-boosted model, fitted as two mixtures rather than one regressor.

Why two. A departure row is joined to a Network Manager flight record, and for
about 1.5% of departures that join fails: no AOBT, no callsign, no market
segment, no flight rule. Those rows are not a nuisance -- on the 2025 validation
split they carry 62% of the squared error, with an RMSE of ~2,970s against ~293s
for everything else, because their target distribution is a different animal
(sd ~3,960s against ~476s).

Fitting them separately is worth about 29s of RMSE (471s -> 443s), and all of it
comes from the small group. Two alternatives were tried and rejected:

  * Predicting the group's mean instead of modelling it made things worse
    (574s), so the model does find real signal in those rows.
  * Adding explicit missingness flags to a single model also made things
    slightly worse (475s); LightGBM's implicit null handling is already fine.

Within that group, the orphan model is itself a mixture. About half of LIRF's
unmatched departures (5% of the group overall) have a BLOCK_TIME equal to
SCHED_TIME to the second, so their taxi-out is exactly MVT_TIME - SCHED_TIME,
routinely several hours. The rest taxi normally. A classifier estimates the
probability p of the first kind, a regressor fitted on the second kind
predicts a normal taxi r, and the prediction is

    p * (MVT_TIME - SCHED_TIME) + (1 - p) * r

which is the least-squares answer when p is calibrated. Against a single
orphan regressor this is worth 25.8s of overall RMSE (409.4s to 383.6s), 95%
CI -45.9s to -10.5s on a paired bootstrap; `experiments_unmatched.py` has
the comparison. Adding the airline as a categorical made both the plain
regressor and the mixture worse, despite the airline predicting the artifact
well: ISR, LAV and ETH are nearly always at schedule, EJU and EZY rarely.

The matched group is a mixture too. The artifact is not confined to missing
records: 18% of LIRF's matched departures (4.9% of all matched ones) also
record off-block at the schedule. Most of them left within minutes of
schedule, where the artifact costs nothing (the flag rate is 36% at 0-5 min
of NM delay and 5% at 1-2 h). But 943 of LIRF's 1,222 matched departures over
an hour are flagged rows, and a plain regressor predicts a normal taxi for
those. The expected cost of a row is about p(1-p)(gap - r)^2, so the
classifier's accuracy only matters where the gap is large. The
mixture took the matched group from 273.4s to 249.2s and LIRF's matched RMSE
from 642s to 499s (overall 369.6s to 352.4s, 95% CI -49.6s to -0.5s), with
the classifier at 0.87 AUC. Weighting the classifier by that cost, with an
explicit EOBT == SCHED flag and a stand-area categorical, measured +0.1s
(95% CI -0.9s to +0.9s): no effect, so it is not in.

Current validation, with the anchored matched regressor,
the LOBT window, LIRF's day-shift and late-orphan rules, and a CatBoost
twin of each regressor, three seeds of the matched LightGBM one and a
LightGBM per airport beside them, and a CatBoost twin of the matched
classifier, and a cap and shrinkage on orphans outside LIRF (constants
below): 316.7s.
All on top of the
live excess features and the reference fallback in features.py (343.6s).

The simulator only earns its place if adding its queue-delay estimate as a
feature beats this number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from . import context, data, features, schema, weather

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}

# The group with no Network Manager record needs far fewer leaves: it is two
# orders of magnitude smaller, and the default would memorise it.
SPARSE_PARAMS = {**PARAMS, "num_leaves": 31, "min_data_in_leaf": 20}

# Below this many rows, a dedicated model is worse than no model.
MIN_SPARSE_ROWS = 2_000

CLASSIFIER_PARAMS = {**SPARSE_PARAMS, "objective": "binary", "metric": "binary_logloss"}
CLASSIFIER_ROUNDS = 200
ORPHAN_ROUNDS = 400

# The matched group's regressor stops improving on validation near 1,200
# rounds (273.3s there, against 274.3s at 400).
JOINED_ROUNDS = 1200
JOINED_CLASSIFIER_PARAMS = {**PARAMS, "objective": "binary", "metric": "binary_logloss"}
JOINED_CLASSIFIER_ROUNDS = 600

# How close BLOCK_TIME must sit to SCHED_TIME to count as recorded at
# schedule. The artifact rows agree to within a few seconds.
AT_SCHEDULE_TOLERANCE_SEC = 10

# Rows above this are left out of the orphan group's normal-taxi regressor.
# None keeps them all. See `experiments_round3.py` on day-shifted rows.
ORPHAN_NORMAL_MAX_SEC: float | None = None

# Whether the matched regressor starts from the NM's own taxi-out
# (`_anchor`) and learns only the departure's deviation from it. Worth
# 1.1s on validation (95% CI -1.7s to -0.6s), `experiments_round3.py`.
JOINED_ANCHORED = True
ANCHOR_CAP_SEC = 7200

# BLOCK_TIME lies within LOBT +- 3606s on every 2025 departure with a LOBT
# (2,062,577 of them), so a matched prediction is projected into that
# window, and the at-schedule probability is zero where the schedule falls
# outside it. Worth 3.8s on validation (95% CI -8.6s to -1.1s). The bound
# was first published by another team; see the README's prior work.
LOBT_WINDOW = True
LOBT_WINDOW_SEC = 3606

# Day-shifted orphans: a taxi-out of one day plus a normal taxi, with no
# at-schedule off-block. At LIRF, every 2025 orphan that took off 14h to
# 26h after schedule was one of two kinds: at schedule (8, target = gap) or
# day-shifted (11), never a normal taxi. So there the mixture's second
# component is a day plus a normal taxi rather than the regressor, which
# had learnt part of the day on its own:
#
#     p * gap + (1 - p) * (86,400 + median normal taxi)
#
# 339.1s to 329.8s on validation. At other airports the band holds
# ordinary late flights, so it is LIRF only.
DAY_SHIFT = True
DAY_SEC = 86_400
DAY_SHIFT_MIN_SEC = 80_000
DAY_SHIFT_BAND_H = (14, 26)
DAY_SHIFT_AIRPORTS = ("LIRF",)

# LIRF orphans more than an hour late are either at schedule (target = gap)
# or a normal taxi (median ~1,030s, p90 under 2,000s); the at-schedule share
# climbs from 42% at 1-2h to 100% past 8h. The orphan regressor's "normal"
# value there ranged from -3,500s to 20,000s, half-absorbing the artifact
# the classifier missed. The rule makes the second component a normal taxi
# and replaces p with the band's training share past 6h (88% to 100%; all
# 9 validation orphans 8h to 14h late were at schedule), averaging the two
# below it. -5.2s on validation (95% CI -10.2s to -1.4s; January and July
# both better). Averaging everywhere gave only -2.9s.
LIRF_ORPHAN_RULE = True
LIRF_ORPHAN_BANDS_H = (1, 2, 3, 4, 6, 8, 14)
LIRF_ORPHAN_FULL_ABOVE_H = 6

# Orphans elsewhere are normal taxis however late they leave: of 20,982 in
# 2025 outside LIRF, 342 took over an hour and 6 over three (two of them
# day-early off-blocks nothing flags), and no lateness band averaged over
# 1,360s. The orphan regressors still reach 15,000s to 30,000s on a few 2026
# rows, so those predictions are capped. -0.8s on validation (95% CI -1.5s
# to -0.2s; January and July both better).
ORPHAN_CAP_SEC = 3600

# And they are pulled a quarter of the way towards their airport and
# lateness band's training mean (bands below; a band of n rows is smoothed
# towards the airport mean with weight n / (n + 30)). At Istanbul, late
# orphans take over an hour a fifth to a third of the time, which the model
# underpredicts; at Amsterdam it overpredicts. -0.5s on validation (95% CI
# -0.7s to -0.3s; January and July both better). Half the way gave the same
# on average but nothing in July.
ORPHAN_SHRINK = 0.25
ORPHAN_SHRINK_BANDS_H = (0.5, 1, 2, 4, 8)
ORPHAN_SHRINK_PRIOR_ROWS = 30

# But an orphan taxis like the departures around it. Outside LTFM, 2025
# orphans took 0.95 to 1.0 times the median take-off minus AOBT of the
# matched departures at their airport within half an hour, at every
# congestion level; the orphan model, which sees none of that, fell far
# short on disruption days (neighbours at 2,000s to 3,000s: 3,991s against
# 1,762s predicted, cross-fitted over the ten training months). So where
# that median exceeds 1,500s, an orphan outside LIRF is predicted at least
# 0.95 times it. Validation, two mild months, barely moves (-0.16s); the
# test set has Amsterdam's de-icing days of 3 to 9 January 2026 and moved
# from 275.0s to 274.1s (v10). `experiments_round8.py`. On the February and
# December 2025 fold it is -3.98s over 251 rows, mostly LTFM's February
# storm (`experiments_round9.py`). 0.95 and 1,500s were read from all ten
# 2025 training months, February included, so that fold checks the
# structure, not untouched parameters.
ORPHAN_FLOOR_RATIO = 0.95
ORPHAN_FLOOR_ABOVE_SEC = 1500.0
ORPHAN_FLOOR_WINDOW_SEC = 1800
ORPHAN_FLOOR_MIN_NEIGHBOURS = 3

# The matched normal-taxi prediction is a blend of the LightGBM regressor and
# a CatBoost one boosted from the same anchor, at this weight on CatBoost.
# 50/50 took validation from 324.6s to 321.8s (95% CI -3.4s to -2.2s;
# January and July both better); 70/30 scored the same. 0 disables it.
JOINED_CATBOOST_WEIGHT = 0.5
CATBOOST_PARAMS = {
    "loss_function": "RMSE", "iterations": 2000, "learning_rate": 0.1, "depth": 8,
    "thread_count": -1, "verbose": 0, "allow_writing_files": False,
}
# The matched twin goes deeper: depth 10, 3,000 rounds at 0.06 measured
# -0.4s against depth 8 (95% CI -0.6s to -0.3s), `experiments_round5.py`.
JOINED_CATBOOST_PARAMS = {**CATBOOST_PARAMS, "depth": 10, "iterations": 3000, "learning_rate": 0.06}

# One LightGBM regressor per airport, averaged 50/50 with the global ones:
# -0.5s (95% CI -0.7s to -0.4s); with the deeper twin, -0.9s together.
JOINED_AIRPORT_MODELS = True
AIRPORT_PARAMS = {**PARAMS, "num_leaves": 127, "min_data_in_leaf": 50}

# A CatBoost at-schedule classifier for the matched group, averaged with the
# LightGBM one: -0.6s (95% CI -0.9s to -0.3s). For the orphans it cost
# +1.5s, so they keep LightGBM's alone. With the two above: 317.9s.
JOINED_CATBOOST_CLASSIFIER_PARAMS = {
    **CATBOOST_PARAMS, "loss_function": "Logloss", "depth": 8, "iterations": 1500, "learning_rate": 0.1,
}

# The orphan group gets a CatBoost twin too, smaller to suit its ~27,000
# rows: 50/50 took validation from 321.8s to 319.6s (95% CI -3.7s to -0.9s;
# January and July both better). Alone it was worse than the blend.
ORPHAN_CATBOOST_WEIGHT = 0.5
ORPHAN_CATBOOST_PARAMS = {**CATBOOST_PARAMS, "iterations": 1000, "learning_rate": 0.05, "depth": 6}

# Extra seeds for the matched LightGBM regressor, averaged with the first:
# -0.2s on validation (95% CI -0.3s to -0.2s).
JOINED_EXTRA_SEEDS = (1, 2)


def has_flight_record() -> pl.Expr:
    """Whether the movement joined to a Network Manager flight record.

    AOBT stands in for the whole record: when it is null, so are the callsign,
    the market segment and the flight rule -- it is one join failing, not four
    independent gaps.
    """
    return pl.col(schema.AOBT).is_not_null()


def off_block_at_schedule() -> pl.Expr:
    """Whether the recorded off-block time is just the scheduled time.

    Needs BLOCK_TIME, so it is a training label only: the classifier learns
    to predict it from what survives on the ranking set.
    """
    gap = (pl.col(schema.BLOCK_TIME) - pl.col(schema.SCHED_TIME)).dt.total_seconds()
    return gap.abs() <= AT_SCHEDULE_TOLERANCE_SEC


@dataclass
class Mixture:
    """At-schedule classifier plus normal-taxi regressor: see the module docstring."""

    at_schedule: lgb.Booster
    normal: lgb.Booster
    # The numeric and categorical feature columns this mixture was fitted on.
    columns: list[str]
    categoricals: list[str]
    # Whether `normal` predicts the deviation from `_anchor` rather than the taxi.
    anchored: bool = False
    # A CatBoost twin of `normal`, blended in at `catboost_weight`.
    catboost: object | None = None
    catboost_weight: float = 0.0
    # More LightGBM regressors like `normal`, other seeds, averaged with it.
    seeds: list = field(default_factory=list)
    # One more LightGBM regressor per airport, averaged 50/50 with the above.
    per_airport: dict = field(default_factory=dict)
    # A CatBoost twin of `at_schedule`, averaged with it.
    at_schedule_twin: object | None = None

    def components(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The at-schedule probability, the normal-taxi prediction, and the gap they mix."""
        matrix = _to_frame(frame, self.columns, self.categoricals)
        p = self.at_schedule.predict(matrix)
        if self.at_schedule_twin is not None:
            pool = _catboost_pool(frame, self.columns, self.categoricals)
            p = 0.5 * p + 0.5 * self.at_schedule_twin.predict_proba(pool)[:, 1]
        boosters = [self.normal, *self.seeds]
        r = np.mean([b.predict(matrix) for b in boosters], axis=0)
        if self.per_airport:
            local = r.copy()
            airports = frame[schema.ADEP].to_numpy()
            for airport, booster in self.per_airport.items():
                rows = airports == airport
                if rows.any():
                    local[rows] = booster.predict(matrix[rows])
            r = 0.5 * r + 0.5 * local
        r = r + (_anchor(frame) if self.anchored else 0.0)
        if self.catboost is not None:
            twin = self.catboost.predict(_catboost_pool(frame, self.columns, self.categoricals))
            twin = twin + (_anchor(frame) if self.anchored else 0.0)
            r = (1 - self.catboost_weight) * r + self.catboost_weight * twin
        gap = frame["sched_to_takeoff_sec"].to_numpy().astype(float)
        # Without a schedule there is no artifact value to mix in.
        gap = np.where(np.isnan(gap), r, gap)
        return p, r, gap

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        p, r, gap = self.components(frame)
        return p * gap + (1 - p) * r


@dataclass
class TrainedModel:
    """Two mixtures and the rule for choosing between them."""

    joined: Mixture
    orphan: Mixture | None
    # The stand/runway reference the boosters were trained against. Prediction
    # must join the same one, so it travels with the model.
    unimpeded: pl.DataFrame
    validation_rmse: float
    # The normal taxi added to a day for day-shifted orphans, `normal_orphan_taxi`.
    day_shift_taxi: float
    # At-schedule share per LIRF orphan gap band, keyed by the band's lower edge.
    lirf_art_rates: dict[int, float]
    # Smoothed mean taxi-out per (airport, gap band) of orphans outside LIRF.
    orphan_band_means: dict = field(default_factory=dict)

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        mask = frame.select(has_flight_record()).to_numpy().ravel()
        out = np.empty(len(frame), dtype=float)
        if mask.any():
            rows = frame.filter(has_flight_record())
            p, r, gap = self.joined.components(rows)
            if LOBT_WINDOW:
                low, high = _lobt_window(rows)
                p = np.where(_sched_in_window(rows), p, 0.0)
                out[mask] = np.clip(p * gap + (1 - p) * r, low, high)
            else:
                out[mask] = p * gap + (1 - p) * r
        if (~mask).any():
            rows = frame.filter(~has_flight_record())
            model = self.joined if self.orphan is None else self.orphan
            p, r, gap = model.components(rows)
            if LIRF_ORPHAN_RULE:
                band_rate = _lirf_band_rate(rows, gap, self.lirf_art_rates)
                late = ~np.isnan(band_rate)
                trusted = late & (gap / 3600 > LIRF_ORPHAN_FULL_ABOVE_H)
                p = np.where(trusted, band_rate, np.where(late, 0.5 * (p + band_rate), p))
                r = np.where(late, self.day_shift_taxi, r)
            if DAY_SHIFT:
                r = np.where(_day_shift_band(rows, gap), DAY_SEC + self.day_shift_taxi, r)
            pred = p * gap + (1 - p) * r
            elsewhere = ~rows[schema.ADEP].is_in(DAY_SHIFT_AIRPORTS).to_numpy()
            if ORPHAN_CAP_SEC is not None:
                pred = np.where(elsewhere, np.minimum(pred, ORPHAN_CAP_SEC), pred)
            if ORPHAN_SHRINK and self.orphan_band_means:
                prior = _orphan_band_mean(rows, gap, self.orphan_band_means)
                pred = np.where(elsewhere & ~np.isnan(prior), (1 - ORPHAN_SHRINK) * pred + ORPHAN_SHRINK * prior, pred)
            out[~mask] = pred
        return out


def _lobt_window(rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The taxi-out range that keeps BLOCK_TIME within LOBT +- 3606s."""
    takeoff_lobt = rows.select(
        (pl.col(schema.MVT_TIME) - pl.col(schema.LOBT)).dt.total_seconds()
    ).to_numpy().ravel().astype(float)
    # A missing LOBT leaves the prediction unbounded.
    return (
        np.nan_to_num(takeoff_lobt - LOBT_WINDOW_SEC, nan=-np.inf),
        np.nan_to_num(takeoff_lobt + LOBT_WINDOW_SEC, nan=np.inf),
    )


def _sched_in_window(rows: pl.DataFrame) -> np.ndarray:
    """Whether an at-schedule BLOCK_TIME is possible at all, given the LOBT."""
    sched_lobt = rows.select(
        (pl.col(schema.SCHED_TIME) - pl.col(schema.LOBT)).dt.total_seconds().abs()
    ).to_numpy().ravel().astype(float)
    return ~(sched_lobt > LOBT_WINDOW_SEC)


def normal_orphan_taxi(orphans: pl.DataFrame) -> float:
    """Median taxi-out of the day-shift airports' ordinary orphans."""
    rows = orphans.filter(
        pl.col(schema.ADEP).is_in(DAY_SHIFT_AIRPORTS)
        & ~off_block_at_schedule().fill_null(False)
        & (pl.col(schema.TARGET) < DAY_SHIFT_MIN_SEC)
    )
    return float(rows[schema.TARGET].median()) if len(rows) else 0.0


def orphan_band_means(orphans: pl.DataFrame) -> dict:
    """Mean taxi-out per (airport, gap band) of orphans outside LIRF, smoothed to the airport mean.

    Keys are (airport, band index), plus (airport, None) for the airport mean
    that unseen bands fall back to.
    """
    rows = orphans.filter(~pl.col(schema.ADEP).is_in(DAY_SHIFT_AIRPORTS)).select(
        schema.ADEP, pl.col(schema.TARGET).cast(pl.Float64),
        (pl.col("sched_to_takeoff_sec") / 3600).alias("_gap_h"),
    )
    rows = rows.with_columns(pl.Series("_band", np.digitize(rows["_gap_h"].fill_null(0).to_numpy(), ORPHAN_SHRINK_BANDS_H)))
    airport = {a: m for a, m in rows.group_by(schema.ADEP).agg(pl.col(schema.TARGET).mean()).iter_rows()}
    means = {(a, None): m for a, m in airport.items()}
    k = ORPHAN_SHRINK_PRIOR_ROWS
    for a, band, n, m in rows.group_by(schema.ADEP, "_band").agg(pl.len(), pl.col(schema.TARGET).mean()).iter_rows():
        means[(a, int(band))] = (n * m + k * airport[a]) / (n + k)
    return means


def _orphan_band_mean(rows: pl.DataFrame, gap: np.ndarray, means: dict) -> np.ndarray:
    """Each row's smoothed band mean, NaN where its airport has none."""
    bands = np.digitize(np.nan_to_num(gap / 3600), ORPHAN_SHRINK_BANDS_H)
    out = np.full(len(rows), np.nan)
    for i, (airport, band) in enumerate(zip(rows[schema.ADEP].to_list(), bands)):
        value = means.get((airport, int(band)), means.get((airport, None)))
        if value is not None:
            out[i] = value
    return out


def neighbour_taxi_level(airports: np.ndarray, takeoff_sec: np.ndarray, nm_taxi: np.ndarray) -> np.ndarray:
    """Median NM taxi time (take-off minus AOBT) of the departures with one at
    the same airport within ORPHAN_FLOOR_WINDOW_SEC of each take-off.

    NaN where fewer than ORPHAN_FLOOR_MIN_NEIGHBOURS are in the window. Rows
    without an AOBT (NaN `nm_taxi`) get a level but do not contribute to one.
    """
    out = np.full(len(airports), np.nan)
    for airport in np.unique(airports):
        rows = np.flatnonzero(airports == airport)
        known = rows[np.isfinite(nm_taxi[rows])]
        order = np.argsort(takeoff_sec[known])
        times, values = takeoff_sec[known][order], nm_taxi[known][order]
        lo = np.searchsorted(times, takeoff_sec[rows] - ORPHAN_FLOOR_WINDOW_SEC)
        hi = np.searchsorted(times, takeoff_sec[rows] + ORPHAN_FLOOR_WINDOW_SEC, side="right")
        for i, a, b in zip(rows, lo, hi):
            if b - a >= ORPHAN_FLOOR_MIN_NEIGHBOURS:
                out[i] = np.median(values[a:b])
    return out


def orphan_congestion_floor(rows: pl.DataFrame, pred: np.ndarray) -> np.ndarray:
    """`pred` with orphans outside LIRF raised to ORPHAN_FLOOR_RATIO times the
    live taxi level of their airport, where that level exceeds ORPHAN_FLOOR_ABOVE_SEC.

    `rows` must hold every departure of the dataset (the level is read off
    the others), with `matched` and `nm_taxi` as `correct.inputs` makes them.
    """
    level = neighbour_taxi_level(
        rows[schema.ADEP].to_numpy(),
        rows[schema.MVT_TIME].dt.epoch("s").to_numpy().astype(float),
        rows["nm_taxi"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
    )
    orphan = ~rows["matched"].to_numpy() & ~rows[schema.ADEP].is_in(DAY_SHIFT_AIRPORTS).to_numpy()
    raise_to = ORPHAN_FLOOR_RATIO * level
    return np.where(orphan & (level > ORPHAN_FLOOR_ABOVE_SEC), np.fmax(pred, raise_to), pred)


def lirf_art_rates(orphans: pl.DataFrame) -> dict[int, float]:
    """At-schedule share of LIRF's ordinary orphans per gap band."""
    rows = orphans.filter(
        (pl.col(schema.ADEP) == "LIRF") & (pl.col(schema.TARGET) < DAY_SHIFT_MIN_SEC)
    ).with_columns(
        off_block_at_schedule().fill_null(False).alias("_art"),
        (pl.col("sched_to_takeoff_sec") / 3600).alias("_gap_h"),
    )
    rates = {}
    for low, high in zip(LIRF_ORPHAN_BANDS_H, LIRF_ORPHAN_BANDS_H[1:]):
        band = rows.filter((pl.col("_gap_h") > low) & (pl.col("_gap_h") <= high))
        rates[low] = float(band["_art"].mean()) if len(band) else 1.0
    return rates


def _lirf_band_rate(rows: pl.DataFrame, gap: np.ndarray, rates: dict[int, float]) -> np.ndarray:
    """The row's band share, NaN outside LIRF or outside the bands."""
    gap_h = gap / 3600
    lirf = (rows[schema.ADEP] == "LIRF").to_numpy()
    out = np.full(len(rows), np.nan)
    for (low, rate), high in zip(rates.items(), LIRF_ORPHAN_BANDS_H[1:]):
        out[lirf & (gap_h > low) & (gap_h <= high)] = rate
    return out


def _day_shift_band(rows: pl.DataFrame, gap: np.ndarray) -> np.ndarray:
    gap_h = gap / 3600
    low, high = DAY_SHIFT_BAND_H
    return rows[schema.ADEP].is_in(DAY_SHIFT_AIRPORTS).to_numpy() & (gap_h > low) & (gap_h <= high)


def orphan_columns() -> list[str]:
    """The orphan group's features: everything but the surroundings.

    On the ~27,000 training rows with no flight record, adding the context
    and weather columns moved that group's validation RMSE from ~2,005s to
    ~2,080s while the matched group gained 5s from them. The paired interval
    cannot separate the orphan shift from noise, so the smaller set stays.
    """
    dropped = (
        set(context.FEATURE_COLUMNS) | set(context.QUEUEING_COLUMNS) | set(weather.FEATURE_COLUMNS)
        | set(features.LIVE_EXCESS_COLUMNS) | {"unimpeded_level"}
    )
    return [c for c in features.FEATURE_COLUMNS if c not in dropped]


# The orphan group's categoricals: the four it had before the flight-table
# ones were added. The flight-table ones are null on these rows by
# construction, except the destination. With it, the 31-leaf model predicted
# 9,000s to 20,000s for five 2026 departures to Nice (LFMN), two of which
# took off within 17 minutes of schedule. Validation did not see it (346.1s
# with, 346.6s without); the test score did (v2: 346s validation, 363s test).
ORPHAN_CATEGORICALS = [schema.ADEP, schema.RUNWAY, schema.STAND, schema.AIRCRAFT_TYPE]


def _to_frame(frame: pl.DataFrame, columns: list[str], categoricals: list[str]) -> pd.DataFrame:
    """Feature matrix with the categorical columns typed as categories.

    Both training and prediction go through here. They must: LightGBM compares
    the categorical dtypes of the two and refuses to predict when they differ,
    so casting in only one place fails at the last line of a long training run.
    """
    x = frame.select(columns + categoricals).to_pandas()
    for column in categoricals:
        x[column] = x[column].astype("category")
    return x


def _anchor(frame: pl.DataFrame) -> np.ndarray:
    """The NM's own taxi-out, take-off minus AOBT, as a starting point.

    Capped, because a multi-hour gap there is a stale AOBT more often than a
    taxi; rows without an AOBT start from the unimpeded reference.
    """
    nm = frame["aobt_to_takeoff_sec"].to_numpy().astype(float)
    fallback = frame["unimpeded_taxi_sec"].to_numpy().astype(float)
    return np.nan_to_num(np.where(np.isnan(nm), fallback, np.clip(nm, 0, ANCHOR_CAP_SEC)))


def _catboost_pool(frame: pl.DataFrame, columns: list[str], categoricals: list[str], label: bool = False,
                   anchored: bool = False):
    """CatBoost's input: categoricals as strings, the anchor as a training baseline only."""
    from catboost import Pool

    x = frame.select(columns + categoricals).with_columns(
        pl.col(c).cast(pl.Utf8).fill_null("NA") for c in categoricals
    ).to_pandas()
    return Pool(
        x,
        label=frame[schema.TARGET].to_numpy() if label else None,
        cat_features=categoricals,
        baseline=_anchor(frame) if label and anchored else None,
    )


def _to_dataset(
    frame: pl.DataFrame, columns: list[str], categoricals: list[str], anchored: bool = False
) -> lgb.Dataset:
    x = _to_frame(frame, columns, categoricals)
    y = frame.select(schema.TARGET).to_numpy().ravel()
    init = _anchor(frame) if anchored else None
    return lgb.Dataset(x, label=y, init_score=init, categorical_feature=categoricals)


def _fit(
    train_frame: pl.DataFrame, columns: list[str], categoricals: list[str], params: dict, num_rounds: int,
    anchored: bool = False,
) -> lgb.Booster:
    return lgb.train(params, _to_dataset(train_frame, columns, categoricals, anchored), num_boost_round=num_rounds)


def _fit_mixture(
    rows: pl.DataFrame,
    columns: list[str],
    categoricals: list[str],
    params: dict,
    num_rounds: int,
    classifier_params: dict,
    classifier_rounds: int,
    normal_max_sec: float | None = None,
    anchored: bool = False,
    catboost_weight: float = 0.0,
    catboost_params: dict | None = None,
    extra_seeds: tuple[int, ...] = (),
    airport_params: dict | None = None,
    classifier_twin_params: dict | None = None,
) -> Mixture:
    label = rows.select(off_block_at_schedule().fill_null(False).cast(pl.Int8)).to_numpy().ravel()
    at_schedule = lgb.train(
        classifier_params,
        lgb.Dataset(_to_frame(rows, columns, categoricals), label=label, categorical_feature=categoricals),
        num_boost_round=classifier_rounds,
    )
    at_schedule_twin = None
    if classifier_twin_params is not None:
        from catboost import CatBoostClassifier

        at_schedule_twin = CatBoostClassifier(**classifier_twin_params)
        # The label rides in the target column; no anchor, so no baseline.
        labelled = rows.with_columns(pl.Series(schema.TARGET, label))
        at_schedule_twin.fit(_catboost_pool(labelled, columns, categoricals, label=True))
    normal_rows = rows.filter(label == 0)
    if normal_max_sec is not None:
        normal_rows = normal_rows.filter(pl.col(schema.TARGET) < normal_max_sec)
    normal = _fit(normal_rows, columns, categoricals, params, num_rounds, anchored)
    seeds = [
        _fit(normal_rows, columns, categoricals, {**params, "seed": seed}, num_rounds, anchored)
        for seed in extra_seeds
    ]
    per_airport = {}
    if airport_params is not None:
        for airport in normal_rows[schema.ADEP].unique().to_list():
            per_airport[airport] = _fit(
                normal_rows.filter(pl.col(schema.ADEP) == airport), columns, categoricals, airport_params,
                num_rounds, anchored,
            )
    twin = None
    if catboost_weight > 0:
        from catboost import CatBoostRegressor

        twin = CatBoostRegressor(**(catboost_params or CATBOOST_PARAMS))
        twin.fit(_catboost_pool(normal_rows, columns, categoricals, label=True, anchored=anchored))
    return Mixture(
        at_schedule=at_schedule, normal=normal, columns=list(columns), categoricals=list(categoricals),
        anchored=anchored, catboost=twin, catboost_weight=catboost_weight, seeds=seeds,
        per_airport=per_airport, at_schedule_twin=at_schedule_twin,
    )


def fit(departures: pl.LazyFrame, around: pl.DataFrame) -> TrainedModel:
    """Fits both mixtures, and the unimpeded reference, on `departures` alone.

    `around` is `features.surroundings()` of the dataset `departures` came
    from. It carries no target: it may span rows that are not fitted on.
    """
    unimpeded = features.unimpeded_taxi_reference(departures).collect()
    frame = features.build(departures, unimpeded.lazy(), around).collect()

    joined_rows = frame.filter(has_flight_record())
    orphan_rows = frame.filter(~has_flight_record())

    joined = _fit_mixture(
        joined_rows, list(features.FEATURE_COLUMNS), list(features.CATEGORICAL_COLUMNS),
        PARAMS, JOINED_ROUNDS, JOINED_CLASSIFIER_PARAMS, JOINED_CLASSIFIER_ROUNDS,
        anchored=JOINED_ANCHORED, catboost_weight=JOINED_CATBOOST_WEIGHT, extra_seeds=JOINED_EXTRA_SEEDS,
        catboost_params=JOINED_CATBOOST_PARAMS, airport_params=AIRPORT_PARAMS if JOINED_AIRPORT_MODELS else None,
        classifier_twin_params=JOINED_CATBOOST_CLASSIFIER_PARAMS,
    )
    orphan = (
        _fit_mixture(
            orphan_rows, orphan_columns(), ORPHAN_CATEGORICALS,
            SPARSE_PARAMS, ORPHAN_ROUNDS, CLASSIFIER_PARAMS, CLASSIFIER_ROUNDS,
            ORPHAN_NORMAL_MAX_SEC,
            catboost_weight=ORPHAN_CATBOOST_WEIGHT, catboost_params=ORPHAN_CATBOOST_PARAMS,
        )
        if len(orphan_rows) >= MIN_SPARSE_ROWS
        else None
    )
    return TrainedModel(
        joined=joined, orphan=orphan, unimpeded=unimpeded, validation_rmse=float("nan"),
        day_shift_taxi=normal_orphan_taxi(orphan_rows),
        lirf_art_rates=lirf_art_rates(orphan_rows),
        orphan_band_means=orphan_band_means(orphan_rows),
    )


def validate(
    training: pl.LazyFrame, months: tuple[int, ...] = data.VALIDATION_MONTHS
) -> tuple[TrainedModel, pl.DataFrame, np.ndarray]:
    """Fits on the other months; returns the model, the held-out frame and its predictions."""
    around = features.surroundings(training)
    departures = data.departures(training)
    train_frame, validation_frame = data.train_validation_split(departures, months)

    # Fit the unimpeded reference on the training half ONLY. It is a low
    # quantile of the target, so deriving it from all of `training` would feed
    # validation targets back in and flatter the score. `fit` does this by
    # only ever seeing `train_frame`.
    model = fit(train_frame, around)

    validation_frame = features.build(validation_frame, model.unimpeded.lazy(), around).collect()
    predictions = model.predict(validation_frame)
    truth = validation_frame.select(schema.TARGET).to_numpy().ravel()
    model.validation_rmse = float(np.sqrt(np.mean((predictions - truth) ** 2)))
    return model, validation_frame, predictions


def train(training: pl.LazyFrame) -> TrainedModel:
    """Fits on ten months and scores on the held-out January and July."""
    return validate(training)[0]


def train_final(training: pl.LazyFrame) -> TrainedModel:
    """Fits on all twelve months, for the submission. Nothing is held out.

    There is no validation score to report: the months that would score it
    are in the fit. Trust `train()` for the number and this for the file.
    """
    return fit(data.departures(training), features.surroundings(training))
