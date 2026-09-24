"""Gradient-boosted baseline, fitted as two models rather than one.

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

The simulator only earns its place if adding its queue-delay estimate as a
feature beats this number.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from . import data, features, schema

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
class OrphanMixture:
    """The model for departures with no flight record: see the module docstring."""

    at_schedule: lgb.Booster
    normal: lgb.Booster

    def predict(self, frame: pl.DataFrame, matrix: pd.DataFrame) -> np.ndarray:
        p = self.at_schedule.predict(matrix)
        r = self.normal.predict(matrix)
        gap = frame["sched_to_takeoff_sec"].to_numpy().astype(float)
        # Without a schedule there is no artifact value to mix in.
        gap = np.where(np.isnan(gap), r, gap)
        return p * gap + (1 - p) * r


@dataclass
class TrainedModel:
    """Two boosters and the rule for choosing between them."""

    joined: lgb.Booster
    orphan: OrphanMixture | None
    # The stand/runway reference the boosters were trained against. Prediction
    # must join the same one, so it travels with the model.
    unimpeded: pl.DataFrame
    validation_rmse: float

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _to_frame(frame)
        mask = frame.select(has_flight_record()).to_numpy().ravel()
        out = np.empty(len(frame), dtype=float)
        if mask.any():
            out[mask] = self.joined.predict(matrix[mask])
        if (~mask).any():
            if self.orphan is None:
                out[~mask] = self.joined.predict(matrix[~mask])
            else:
                out[~mask] = self.orphan.predict(frame.filter(~has_flight_record()), matrix[~mask])
        return out


def _to_frame(frame: pl.DataFrame) -> pd.DataFrame:
    """Feature matrix with the categorical columns typed as categories.

    Both training and prediction go through here. They must: LightGBM compares
    the categorical dtypes of the two and refuses to predict when they differ,
    so casting in only one place fails at the last line of a long training run.
    """
    columns = features.FEATURE_COLUMNS + features.CATEGORICAL_COLUMNS
    x = frame.select(columns).to_pandas()
    for column in features.CATEGORICAL_COLUMNS:
        x[column] = x[column].astype("category")
    return x


def _to_dataset(frame: pl.DataFrame) -> lgb.Dataset:
    x = _to_frame(frame)
    y = frame.select(schema.TARGET).to_numpy().ravel()
    return lgb.Dataset(x, label=y, categorical_feature=features.CATEGORICAL_COLUMNS)


def _fit(train_frame: pl.DataFrame, params: dict, num_rounds: int) -> lgb.Booster:
    return lgb.train(params, _to_dataset(train_frame), num_boost_round=num_rounds)


def _fit_orphan(orphan_rows: pl.DataFrame, num_rounds: int) -> OrphanMixture:
    label = orphan_rows.select(off_block_at_schedule().fill_null(False).cast(pl.Int8)).to_numpy().ravel()
    at_schedule = lgb.train(
        CLASSIFIER_PARAMS,
        lgb.Dataset(_to_frame(orphan_rows), label=label, categorical_feature=features.CATEGORICAL_COLUMNS),
        num_boost_round=CLASSIFIER_ROUNDS,
    )
    normal = _fit(orphan_rows.filter(label == 0), SPARSE_PARAMS, num_rounds)
    return OrphanMixture(at_schedule=at_schedule, normal=normal)


def fit(departures: pl.LazyFrame, num_rounds: int = 400) -> TrainedModel:
    """Fits both boosters, and the unimpeded reference, on `departures` alone."""
    unimpeded = features.unimpeded_taxi_reference(departures).collect()
    frame = features.build(departures, unimpeded.lazy()).collect()

    joined_rows = frame.filter(has_flight_record())
    orphan_rows = frame.filter(~has_flight_record())

    joined = _fit(joined_rows, PARAMS, num_rounds)
    orphan = (
        _fit_orphan(orphan_rows, num_rounds)
        if len(orphan_rows) >= MIN_SPARSE_ROWS
        else None
    )
    return TrainedModel(
        joined=joined, orphan=orphan, unimpeded=unimpeded, validation_rmse=float("nan")
    )


def train(training: pl.LazyFrame, num_rounds: int = 400) -> TrainedModel:
    """Fits on ten months and scores on the held-out January and July."""
    departures = data.departures(training)
    train_frame, validation_frame = data.train_validation_split(departures)

    # Fit the unimpeded reference on the training half ONLY. It is a low
    # quantile of the target, so deriving it from all of `training` would feed
    # validation targets back in and flatter the score. `fit` does this by
    # only ever seeing `train_frame`.
    model = fit(train_frame, num_rounds)

    validation_frame = features.build(validation_frame, model.unimpeded.lazy()).collect()
    predictions = model.predict(validation_frame)
    truth = validation_frame.select(schema.TARGET).to_numpy().ravel()
    model.validation_rmse = float(np.sqrt(np.mean((predictions - truth) ** 2)))
    return model


def train_final(training: pl.LazyFrame, num_rounds: int = 400) -> TrainedModel:
    """Fits on all twelve months, for the submission. Nothing is held out.

    There is no validation score to report: the months that would score it
    are in the fit. Trust `train()` for the number and this for the file.
    """
    return fit(data.departures(training), num_rounds)
