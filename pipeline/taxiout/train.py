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


def has_flight_record() -> pl.Expr:
    """Whether the movement joined to a Network Manager flight record.

    AOBT stands in for the whole record: when it is null, so are the callsign,
    the market segment and the flight rule -- it is one join failing, not four
    independent gaps.
    """
    return pl.col(schema.AOBT).is_not_null()


@dataclass
class TrainedModel:
    """Two boosters and the rule for choosing between them."""

    joined: lgb.Booster
    orphan: lgb.Booster | None
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
            booster = self.orphan or self.joined
            out[~mask] = booster.predict(matrix[~mask])
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


def fit(departures: pl.LazyFrame, num_rounds: int = 400) -> TrainedModel:
    """Fits both boosters, and the unimpeded reference, on `departures` alone."""
    unimpeded = features.unimpeded_taxi_reference(departures).collect()
    frame = features.build(departures, unimpeded.lazy()).collect()

    joined_rows = frame.filter(has_flight_record())
    orphan_rows = frame.filter(~has_flight_record())

    joined = _fit(joined_rows, PARAMS, num_rounds)
    orphan = (
        _fit(orphan_rows, SPARSE_PARAMS, num_rounds)
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
