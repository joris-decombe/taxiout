"""A correction of the model's predictions from ADS-B ground observations.

`taxiout.adsb` gives, per departure, what adsb.lol's receivers saw on the
ground: the push-back where the aircraft is seen parked and then moving
(12% of 2025 departures, 18% to 20% of 2026's), and for most others the
first ground position and the exact lift-off. A small LightGBM learns the
model's error from those observations, the model's own prediction and
at-schedule probability, the Network Manager's taxi time and the schedule
gap. It is fitted on held-out predictions for every month of 2025, each
made by a model fitted without that month and its pair (`build_final.py`
FOLDS), and applied unchanged to 2026, after which the production rules
bound the result again and floor the orphans at their airport's live taxi
level (`rules`).

Measured by cross-validation over whole days: -6.0s over 2025 (95% CI
-6.9s to -5.2s), from -3.4s on June + August to -10.7s on February +
December (`experiments_round11.py`). On the departures whose push-back was
seen the error drops from 200s to 164s (`experiments_round7.py`, January
and July); most of the rest comes from the partial observations.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from . import adsb, schema, submit, train

FEATURES = [*adsb.OBSERVATION_COLUMNS, "pred", "p", "nm_taxi", "gap"]
PARAMS = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200,
    "lambda_l2": 10.0, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
}
ROUNDS = 300


def inputs(model: train.TrainedModel, frame: pl.DataFrame) -> pl.DataFrame:
    """The model's prediction and at-schedule probability per departure, with the columns the rules need."""
    matched = frame.select(train.has_flight_record()).to_numpy().ravel()
    p = np.empty(frame.height)
    if matched.any():
        p[matched] = model.joined.components(frame.filter(pl.Series(matched)))[0]
    if (~matched).any():
        mixture = model.joined if model.orphan is None else model.orphan
        p[~matched] = mixture.components(frame.filter(pl.Series(~matched)))[0]
    columns = [schema.MVT_ID, schema.ADEP, schema.MVT_TIME, schema.LOBT]
    if schema.TARGET in frame.columns:
        columns.append(schema.TARGET)
    return frame.select(
        *columns,
        pl.Series("matched", matched), pl.Series("p", p), pl.Series("pred", model.predict(frame)),
        pl.col("sched_to_takeoff_sec").alias("gap"), pl.col("aobt_to_takeoff_sec").alias("nm_taxi"),
    )


def _matrix(frame: pl.DataFrame):
    x = frame.select(FEATURES).with_columns(pl.col(c).cast(pl.Float64) for c in FEATURES).to_pandas()
    x["ADEP"] = frame[schema.ADEP].to_pandas().astype("category")
    return x


def _join(rows: pl.DataFrame, observations: pl.DataFrame) -> pl.DataFrame:
    return rows.join(observations, on=schema.MVT_ID, how="left").with_columns(pl.col("adsb_matched").fill_null(False))


def fit(rows: pl.DataFrame, observations: pl.DataFrame) -> lgb.Booster:
    """Fits on `inputs` rows that carry the target, where ADS-B saw the aircraft."""
    seen = _join(rows, observations).filter(pl.col("adsb_matched"))
    residual = (seen[schema.TARGET].cast(pl.Float64) - seen["pred"]).to_numpy()
    return lgb.train(PARAMS, lgb.Dataset(_matrix(seen), label=residual, categorical_feature=["ADEP"]),
                     num_boost_round=ROUNDS)


def rules(rows: pl.DataFrame, pred: np.ndarray) -> np.ndarray:
    """The production rules that bound a prediction, applied again after the correction.

    The orphan congestion floor comes last, over both the hour cap and
    ADS-B: on congested days adsb.lol dates push-backs late (a median
    111s to 219s short of BLOCK_TIME when the airport's NM taxi level
    exceeds 1,200s, against 13s to 23s when quiet), and orphans have no NM
    anchor to hold them.
    """
    matched = rows["matched"].to_numpy()
    low, high = train._lobt_window(rows)
    pred = np.where(matched, np.clip(pred, low, high), pred)
    elsewhere = ~matched & (rows[schema.ADEP].to_numpy() != "LIRF")
    pred = np.where(elsewhere, np.minimum(pred, train.ORPHAN_CAP_SEC), pred)
    pred = train.orphan_congestion_floor(rows, pred)
    return np.clip(pred, 0.0, submit.MAX_PLAUSIBLE_TAXI_SEC)


def apply(booster: lgb.Booster, rows: pl.DataFrame, observations: pl.DataFrame) -> pl.DataFrame:
    """`rows` with the corrected prediction in `pred`; rows ADS-B did not see keep theirs."""
    joined = _join(rows, observations)
    seen = joined["adsb_matched"].to_numpy()
    correction = np.zeros(joined.height)
    if seen.any():
        correction[seen] = booster.predict(_matrix(joined.filter(pl.Series(seen))))
    return joined.with_columns(pl.Series("pred", rules(joined, joined["pred"].to_numpy() + correction)))
