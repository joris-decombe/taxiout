"""Round 8: orphans on congested days, and what else was measured on 8 October 2026.

Why orphans exist. The movement and flight tables are joined on
|BLOCK_TIME - LOBT| <= 3606s, so a departure with a flight plan that pushed
back more than an hour from its LOBT loses its NM record. When it flies to
another of the ten airports, the arrival row there still carries that record
(its AOBT sits within the hour of BLOCK_TIME for 832 of 902 such 2025
orphans), but that covers about 4% of orphans and is not used.

An orphan taxis like the departures around it (`ratio`). Outside LTFM,
2025 orphans took 0.95 to 1.0 times the median take-off minus AOBT of the
matched departures at their airport within half an hour, at every
congestion level; LTFM's February 2025 snowstorm ran at about twice. The
orphan model sees none of this. Cross-fitted over the ten training months
(`crossfit`), where neighbours ran at 2,000s to 3,000s, orphans took 3,991s
and the model predicted 1,762s. Floored at 0.95 times the neighbour level
above 1,500s (`train.orphan_congestion_floor`), squared error falls by
1.07 million s^2 per changed row there and by 0.23 million on the
production validation predictions. Validation barely moves (-0.16s: two
mild months), but Amsterdam had a de-icing disruption on 3 to 9 January
2026, with neighbours at 2,800s to 3,700s and its orphans predicted at
about 1,240s. Applied to v9's file (`build`) the floor changes 431 rows
and scored 274.07s against 275.03s (upload v10).

ADS-B dates push-backs late when it is congested (`adsb`). On the matched
validation departures whose push-back adsb.lol saw, BLOCK_TIME minus the
observed time has a median of 13s to 23s where the airport's NM taxi level
is under 1,200s, and 111s to 219s above it; de-icing pads look like stands
to a dwell detector. The NM anchor keeps matched predictions honest; the
floor does it for orphans.

Measured and not used, before the floor existed (`history`, a
day-cross-validated corrector on the production validation predictions,
the ADS-B corrector's set-up but on every row; rerun now, all three arms
include the floor):

  flight number's history in the other ten months   -4.5s (95% CI -7.5s to -2.5s)
      all of it LIRF orphans, 59% from 20 rows; matched rows +0.6s worse.
      Rome's July 2026 flight numbers match 2025 for only half the
      departures (seasonal alphanumeric callsigns), so it would mostly
      not reach the test set.
  route and scheduled-time key instead              +0.2s
  neighbours' ADS-B-observed residuals              +0.4s
  flight-number string patterns as categoricals     worse than history alone

Usage: python pipeline/experiments_round8.py ratio | crossfit | adsb | history | build
"""

import datetime as dt
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import adsb, correct, data, schema, submit, train  # noqa: E402

OUT = Path("data/round8")
VALIDATION_DAYS = [dt.date(2025, m, d) for m in (1, 7) for d in range(1, 32)]
BLOCKS = [(2, 3), (4, 5), (6, 8), (9, 10), (11, 12)]
BANDS = [900, 1200, 1500, 1800, 2200, 3000]


def with_level(frame: pl.DataFrame) -> pl.DataFrame:
    """`frame` (all departures of a dataset) with each row's neighbour taxi level."""
    level = train.neighbour_taxi_level(
        frame[schema.ADEP].to_numpy(),
        frame[schema.MVT_TIME].dt.epoch("s").to_numpy().astype(float),
        frame.select((pl.col(schema.MVT_TIME) - pl.col(schema.AOBT)).dt.total_seconds().cast(pl.Float64))
        .to_series().fill_null(np.nan).to_numpy(),
    )
    return frame.with_columns(pl.Series("level", level))


def training_levels() -> pl.DataFrame:
    path = OUT / "levels_2025.parquet"
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        deps = data.departures(data.load_training()).select(
            schema.MVT_ID, schema.ADEP, schema.MVT_TIME, schema.AOBT).collect()
        with_level(deps).select(schema.MVT_ID, "level").write_parquet(path)
    return pl.read_parquet(path)


def ratio() -> None:
    deps = (data.departures(data.load_training()).collect().join(training_levels(), on=schema.MVT_ID)
            .with_columns(y=pl.col(schema.TARGET).cast(pl.Float64), orphan=pl.col(schema.AOBT).is_null()))
    rows = deps.filter(pl.col("level").is_not_nan() & (pl.col("y") < 20000) & (pl.col(schema.ADEP) != "LIRF")).with_columns(
        group=pl.when(pl.col("orphan")).then(pl.lit("orphan")).otherwise(pl.lit("matched")),
        place=pl.when(pl.col(schema.ADEP) == "LTFM").then(pl.lit("LTFM")).otherwise(pl.lit("other")),
        band=pl.col("level").cut(BANDS))
    print(rows.group_by("group", "place", "band").agg(
        pl.len(), pl.col("level").mean().round(0), pl.col("y").mean().round(0),
        (pl.col("y") / pl.col("level")).mean().round(2).alias("ratio")).sort("group", "place", "band"))


def crossfit() -> None:
    """The production orphan mixture, refitted without each two-month block, scored with and without the floor."""
    path = OUT / "orphan_oof.parquet"
    if not path.exists():
        rows = (pl.scan_parquet("data/round5/train.parquet").filter(pl.col(schema.AOBT).is_null()).collect()
                .with_columns(pl.col(schema.MVT_TIME).dt.month().alias("_m")))
        parts = []
        for block in BLOCKS:
            fit_rows, held = rows.filter(~pl.col("_m").is_in(block)), rows.filter(pl.col("_m").is_in(block))
            mixture = train._fit_mixture(
                fit_rows, train.orphan_columns(), train.ORPHAN_CATEGORICALS, train.SPARSE_PARAMS, train.ORPHAN_ROUNDS,
                train.CLASSIFIER_PARAMS, train.CLASSIFIER_ROUNDS, train.ORPHAN_NORMAL_MAX_SEC,
                catboost_weight=train.ORPHAN_CATBOOST_WEIGHT, catboost_params=train.ORPHAN_CATBOOST_PARAMS)
            model = train.TrainedModel(
                joined=None, orphan=mixture, unimpeded=None, validation_rmse=float("nan"),
                day_shift_taxi=train.normal_orphan_taxi(fit_rows), lirf_art_rates=train.lirf_art_rates(fit_rows),
                orphan_band_means=train.orphan_band_means(fit_rows))
            parts.append(held.select(schema.MVT_ID, schema.ADEP, schema.TARGET).with_columns(pl.Series("pred", model.predict(held))))
            print(block, held.height, flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        pl.concat(parts).write_parquet(path)
    validation = pl.read_parquet("data/round7/prod_valid.parquet").filter(~pl.col("matched")).select(
        schema.MVT_ID, schema.ADEP, schema.TARGET, "pred")
    for name, rows in (("ten training months, cross-fitted", pl.read_parquet(path)), ("validation, production", validation)):
        rows = rows.join(training_levels(), on=schema.MVT_ID).filter(pl.col(schema.ADEP) != "LIRF")
        y, pred, level = rows[schema.TARGET].to_numpy().astype(float), rows["pred"].to_numpy(), rows["level"].to_numpy()
        new = np.where(np.isfinite(level) & (level > train.ORPHAN_FLOOR_ABOVE_SEC),
                       np.maximum(pred, train.ORPHAN_FLOOR_RATIO * level), pred)
        changed = new != pred
        delta = (new - y) ** 2 - (pred - y) ** 2
        print(f"{name}: {changed.sum()} rows changed, squared error {delta[changed].mean():+,.0f} s^2 per changed row, "
              f"{(delta[changed] > 0).mean():.0%} of them worse")
        print(rows.with_columns(pl.Series("floored", new), band=pl.col("level").cut(BANDS)).group_by("band").agg(
            pl.len(), pl.col(schema.TARGET).mean().round(0).alias("truth"), pl.col("pred").mean().round(0),
            pl.col("floored").mean().round(0)).sort("band"))


def observed() -> pl.DataFrame:
    return (pl.read_parquet("data/round7/prod_valid.parquet").with_columns(pl.col(schema.MVT_TIME).dt.date().alias("day"))
            .join(adsb.load_observations(VALIDATION_DAYS), on=schema.MVT_ID, how="left")
            .with_columns(pl.col("adsb_matched").fill_null(False)))


def adsb_lag() -> None:
    rows = observed().join(training_levels(), on=schema.MVT_ID).filter(
        pl.col("adsb_pushback_seen").fill_null(False) & pl.col("matched") & (pl.col(schema.TARGET) < 7200))
    print(rows.with_columns(band=pl.col("level").cut(BANDS)).group_by("band").agg(
        pl.len(), (pl.col(schema.TARGET) - pl.col("adsb_move_to_takeoff_sec")).median().round(0).alias("block_minus_adsb_median"),
        ((pl.col(schema.TARGET) - pl.col("adsb_move_to_takeoff_sec")).abs() <= 120).mean().round(2).alias("within_2min"))
          .sort("band"))


def history() -> None:
    """Flight-number history from the other ten months, as inputs to an every-row corrector."""
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200,
              "lambda_l2": 10.0, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1}
    base = ["pred", "p", "nm_taxi", "gap", *adsb.OBSERVATION_COLUMNS]
    months = (data.departures(data.load_training()).with_columns(
        y=pl.col(schema.TARGET).cast(pl.Float64),
        nm=(pl.col(schema.MVT_TIME) - pl.col(schema.AOBT)).dt.total_seconds().cast(pl.Float64),
        copy=train.off_block_at_schedule().fill_null(False).cast(pl.Float64),
        gap_=(pl.col(schema.MVT_TIME) - pl.col(schema.SCHED_TIME)).dt.total_seconds().cast(pl.Float64)).collect())
    ids = months.select(schema.MVT_ID, "FLIGHT_mvt", "CALLSIGN_flt", schema.STAND)
    months = months.filter(~pl.col(schema.MVT_TIME).dt.month().is_in([1, 7]))
    frame = observed().join(ids, on=schema.MVT_ID, how="left")
    columns = []
    for keys, tag in (([schema.ADEP, "FLIGHT_mvt"], "fh"), ([schema.ADEP, "CALLSIGN_flt"], "ch"),
                      ([schema.ADEP, "FLIGHT_mvt", schema.STAND], "fsh")):
        normal = pl.col("copy") == 0
        stats = months.filter(pl.all_horizontal(pl.col(k).is_not_null() for k in keys)).group_by(keys).agg(
            pl.len().alias(f"{tag}_n"),
            pl.col("y").clip(0, 3600).filter(normal).median().alias(f"{tag}_y_med"),
            (pl.col("y") - pl.col("nm")).clip(-1800, 1800).filter(normal).mean().alias(f"{tag}_res_mean"),
            (pl.col("y") - pl.col("nm")).clip(-1800, 1800).filter(normal).median().alias(f"{tag}_res_med"),
            (pl.col("y") - pl.col("nm")).clip(-1800, 1800).filter(normal).std().alias(f"{tag}_res_sd"),
            pl.col("copy").mean().alias(f"{tag}_copy"),
            (pl.col("y") > 3600).cast(pl.Float64).mean().alias(f"{tag}_long"),
            pl.col("gap_").clip(-3600, 7200).median().alias(f"{tag}_gap_med"))
        frame = frame.join(stats, on=keys, how="left")
        columns += [c for c in stats.columns if c.startswith(f"{tag}_")]
    truth = frame[schema.TARGET].to_numpy().astype(float)
    pred = frame["pred"].to_numpy()
    folds = np.array_split(np.array(sorted(frame["day"].unique().to_list()), dtype=object), 5)
    day = frame["day"].to_numpy()

    def corrected(cols: list[str], mask: np.ndarray) -> np.ndarray:
        x = frame.select(cols).with_columns(pl.col(c).cast(pl.Float64) for c in cols).to_pandas()
        x["ADEP"] = frame[schema.ADEP].to_pandas().astype("category")
        out = np.zeros(len(frame))
        for held in folds:
            test = np.isin(day, held)
            fit = ~test & mask
            booster = lgb.train(params, lgb.Dataset(x[fit], label=(truth - pred)[fit], categorical_feature=["ADEP"]), 300)
            out[test & mask] = booster.predict(x[test & mask])
        return correct.rules(frame, pred + out)  # the production rules, floor included

    everywhere = np.ones(len(frame), bool)
    for name, p in (("ADS-B rows only (production)", corrected(base, frame["adsb_matched"].to_numpy())),
                    ("every row", corrected(base, everywhere)),
                    ("every row + flight history", corrected(base + columns, everywhere))):
        print(f"{name:32s} {np.sqrt(np.mean((p - truth) ** 2)):.2f}s")


def build() -> None:
    """v9's file (data/submission_v12.parquet) with the floor: uploaded as v10, data/submission_v13.parquet."""
    ranking = data.departures(data.load_ranking()).select(schema.MVT_ID, schema.ADEP, schema.MVT_TIME, schema.AOBT).collect()
    rows = ranking.join(pl.read_parquet("data/submission_v12.parquet"), on=schema.MVT_ID).with_columns(
        matched=pl.col(schema.AOBT).is_not_null(),
        nm_taxi=(pl.col(schema.MVT_TIME) - pl.col(schema.AOBT)).dt.total_seconds())
    before = rows[schema.TARGET].to_numpy().astype(float)
    after = train.orphan_congestion_floor(rows, before)
    predictions = dict(zip(rows[schema.MVT_ID].to_list(), after.tolist()))
    path = Path("data/submission_v13.parquet")
    submit.write_submission(predictions, path)
    print(path, "problems:", submit.verify_submission(path, predictions), "changed:", int((after != before).sum()))


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"ratio": ratio, "crossfit": crossfit, "adsb": adsb_lag, "history": history, "build": build}[step]()
