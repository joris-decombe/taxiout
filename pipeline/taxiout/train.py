"""Baseline gradient-boosted model.

This is the yardstick, not the destination. A tuned GBM on decent features is
what a competent entry scores; the simulator only earns its place if adding
its queue-delay estimate as a feature beats this.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
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


@dataclass
class TrainedModel:
    booster: lgb.Booster
    validation_rmse: float


def _to_dataset(frame: pl.DataFrame) -> lgb.Dataset:
    columns = features.FEATURE_COLUMNS + features.CATEGORICAL_COLUMNS
    x = frame.select(columns).to_pandas()
    for column in features.CATEGORICAL_COLUMNS:
        x[column] = x[column].astype("category")
    y = frame.select(schema.TARGET).to_numpy().ravel()
    return lgb.Dataset(x, label=y, categorical_feature=features.CATEGORICAL_COLUMNS)


def train(training: pl.LazyFrame, num_rounds: int = 3000) -> TrainedModel:
    unimpeded = features.unimpeded_taxi_reference(training)
    departures = data.departures(training)
    train_frame, validation_frame = data.train_validation_split(departures)

    train_frame = features.build(train_frame, unimpeded).collect()
    validation_frame = features.build(validation_frame, unimpeded).collect()

    booster = lgb.train(
        PARAMS,
        _to_dataset(train_frame),
        num_boost_round=num_rounds,
        valid_sets=[_to_dataset(validation_frame)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    predictions = booster.predict(
        validation_frame.select(features.FEATURE_COLUMNS + features.CATEGORICAL_COLUMNS).to_pandas()
    )
    truth = validation_frame.select(schema.TARGET).to_numpy().ravel()
    rmse = float(np.sqrt(np.mean((predictions - truth) ** 2)))

    return TrainedModel(booster=booster, validation_rmse=rmse)
