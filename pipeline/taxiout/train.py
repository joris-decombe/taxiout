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
record off-block at the schedule, typically delayed flights whose NM AOBT is
two or three hours later, and 943 of LIRF's 1,222 matched departures over an
hour are these. A plain regressor predicts a normal taxi for them. The
mixture took the matched group from 273.4s to 249.2s and LIRF's matched RMSE
from 642s to 499s (overall 369.6s to 352.4s, 95% CI -49.6s to -0.5s), with
the classifier at 0.87 AUC.

Current validation: 346.1s overall, 240.7s matched, 2,006s orphan.

The simulator only earns its place if adding its queue-delay estimate as a
feature beats this number.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    # The numeric feature columns this mixture was fitted on.
    columns: list[str]

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _to_frame(frame, self.columns)
        p = self.at_schedule.predict(matrix)
        r = self.normal.predict(matrix)
        gap = frame["sched_to_takeoff_sec"].to_numpy().astype(float)
        # Without a schedule there is no artifact value to mix in.
        gap = np.where(np.isnan(gap), r, gap)
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

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        mask = frame.select(has_flight_record()).to_numpy().ravel()
        out = np.empty(len(frame), dtype=float)
        if mask.any():
            out[mask] = self.joined.predict(frame.filter(has_flight_record()))
        if (~mask).any():
            model = self.joined if self.orphan is None else self.orphan
            out[~mask] = model.predict(frame.filter(~has_flight_record()))
        return out


def orphan_columns() -> list[str]:
    """The orphan group's features: everything but the surroundings.

    On the ~27,000 training rows with no flight record, adding the context
    and weather columns moved that group's validation RMSE from ~2,005s to
    ~2,080s while the matched group gained 5s from them. The paired interval
    cannot separate the orphan shift from noise, so the smaller set stays.
    """
    dropped = set(context.FEATURE_COLUMNS) | set(weather.FEATURE_COLUMNS)
    return [c for c in features.FEATURE_COLUMNS if c not in dropped]


def _to_frame(frame: pl.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Feature matrix with the categorical columns typed as categories.

    Both training and prediction go through here. They must: LightGBM compares
    the categorical dtypes of the two and refuses to predict when they differ,
    so casting in only one place fails at the last line of a long training run.
    """
    x = frame.select(columns + features.CATEGORICAL_COLUMNS).to_pandas()
    for column in features.CATEGORICAL_COLUMNS:
        x[column] = x[column].astype("category")
    return x


def _to_dataset(frame: pl.DataFrame, columns: list[str]) -> lgb.Dataset:
    x = _to_frame(frame, columns)
    y = frame.select(schema.TARGET).to_numpy().ravel()
    return lgb.Dataset(x, label=y, categorical_feature=features.CATEGORICAL_COLUMNS)


def _fit(train_frame: pl.DataFrame, columns: list[str], params: dict, num_rounds: int) -> lgb.Booster:
    return lgb.train(params, _to_dataset(train_frame, columns), num_boost_round=num_rounds)


def _fit_mixture(
    rows: pl.DataFrame,
    columns: list[str],
    params: dict,
    num_rounds: int,
    classifier_params: dict,
    classifier_rounds: int,
) -> Mixture:
    label = rows.select(off_block_at_schedule().fill_null(False).cast(pl.Int8)).to_numpy().ravel()
    at_schedule = lgb.train(
        classifier_params,
        lgb.Dataset(_to_frame(rows, columns), label=label, categorical_feature=features.CATEGORICAL_COLUMNS),
        num_boost_round=classifier_rounds,
    )
    normal = _fit(rows.filter(label == 0), columns, params, num_rounds)
    return Mixture(at_schedule=at_schedule, normal=normal, columns=list(columns))


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
        joined_rows, list(features.FEATURE_COLUMNS),
        PARAMS, JOINED_ROUNDS, JOINED_CLASSIFIER_PARAMS, JOINED_CLASSIFIER_ROUNDS,
    )
    orphan = (
        _fit_mixture(
            orphan_rows, orphan_columns(),
            SPARSE_PARAMS, ORPHAN_ROUNDS, CLASSIFIER_PARAMS, CLASSIFIER_ROUNDS,
        )
        if len(orphan_rows) >= MIN_SPARSE_ROWS
        else None
    )
    return TrainedModel(
        joined=joined, orphan=orphan, unimpeded=unimpeded, validation_rmse=float("nan")
    )


def validate(training: pl.LazyFrame) -> tuple[TrainedModel, pl.DataFrame, np.ndarray]:
    """Fits on ten months; returns the model, the held-out frame and its predictions."""
    around = features.surroundings(training)
    departures = data.departures(training)
    train_frame, validation_frame = data.train_validation_split(departures)

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
