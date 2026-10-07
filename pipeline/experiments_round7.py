"""Round 7: ADS-B push-back observations from adsb.lol, as a correction.

`taxiout.adsb` turns each day of the adsb.lol archive into one row per
departure: whether the aircraft was seen, parked and then moving (the
push-back), first seen on the ground, and the times of each. Rather than
retrain the whole model on a year of ADS-B (1.2 TB to stream), a small
corrector learns how far to move the production prediction given those
observations, the Network Manager's taxi time and the prediction itself.

Training rows are the January and July 2025 validation days, where the
production model's predictions are honest held-out ones (`prod` below
rebuilds them from rounds 3 to 5 with every production rule). The
corrector is scored by cross-validation over whole days, so a day's
correction never comes from a model that saw that day.

After the correction the production rules apply again: matched predictions
are projected into the LOBT window, orphans outside LIRF capped at an hour.

`build` fits the corrector on every validation day and applies it to the
2026 predictions of `data/model_v11.pkl`, writing
`data/submission_v12.parquet`.

Usage: python pipeline/experiments_round7.py observe | prod | --score | build
"""

import datetime as dt
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
import pickle  # noqa: E402

import experiments_round4 as round4  # noqa: E402
import experiments_round5 as round5  # noqa: E402
from taxiout import adsb, data, features, schema, submit, train  # noqa: E402

OUT = Path("data/round7")
VALIDATION_DAYS = [dt.date(2025, m, d) for m in (1, 7) for d in range(1, 32)]
FEATURES = [*adsb.OBSERVATION_COLUMNS, "pred", "p", "nm_taxi", "gap"]
PARAMS = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200,
    "lambda_l2": 10.0, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
}
ROUNDS = 300


def observe() -> None:
    """Runs `adsb.observe_day` for every validation day whose points are in and not yet observed."""
    for day in VALIDATION_DAYS + [dt.date(2026, m, d) for m in (1, 7) for d in range(1, 32)]:
        points = adsb.OUT_DIR / f"{day.isoformat()}_airports.parquet"
        done = adsb.OUT_DIR / f"{day.isoformat()}_departures.parquet"
        if points.exists() and not done.exists():
            adsb.observe_day(day)
            print(f"observed {day}", flush=True)


def prod() -> None:
    """Production's held-out validation predictions (316.7s), with what the corrector needs."""
    taxi, rates = round4.fitted_constants()
    f = round5.production()
    g = (f.join(pl.read_parquet("data/round5/airport_r.parquet").rename({"r_arm": "r_air"}), on=schema.MVT_ID, how="left")
          .join(pl.read_parquet("data/round5/cbdeep_r.parquet").rename({"r_arm": "r_deep"}), on=schema.MVT_ID, how="left")
          .join(pl.read_parquet("data/round5/cbclf_p.parquet"), on=schema.MVT_ID, how="left"))
    h = round5.blend(g, 0.5 * pl.col("r_lgb") + 0.5 * pl.col("r_air"), pl.col("r_deep"))
    h = h.with_columns(pl.when(pl.col("matched")).then(0.5 * pl.col("p") + 0.5 * pl.col("p_cb")).otherwise(pl.col("p")).alias("p"))
    pred = round4.predictions(h, taxi, rates)
    matched = h["matched"].to_numpy()
    elsewhere = ~matched & (h[schema.ADEP].to_numpy() != "LIRF")
    pred = np.where(elsewhere, np.minimum(pred, train.ORPHAN_CAP_SEC), pred)
    # The orphan shrinkage, with band means from the training months only.
    fitted, _ = data.train_validation_split(data.departures(data.load_training()))
    orphans = fitted.filter(~train.has_flight_record()).with_columns(
        (pl.col(schema.MVT_TIME) - pl.col(schema.SCHED_TIME)).dt.total_seconds().alias("sched_to_takeoff_sec")).collect()
    prior = train._orphan_band_mean(h, h["gap"].to_numpy(), train.orphan_band_means(orphans))
    pred = np.where(elsewhere & ~np.isnan(prior), (1 - train.ORPHAN_SHRINK) * pred + train.ORPHAN_SHRINK * prior, pred)
    nm = data.departures(data.load_training()).select(
        schema.MVT_ID, (pl.col(schema.MVT_TIME) - pl.col(schema.AOBT)).dt.total_seconds().alias("nm_taxi")).collect()
    out = h.select(schema.MVT_ID, schema.ADEP, schema.TARGET, schema.MVT_TIME, schema.LOBT, "matched", "p", "gap").with_columns(
        pl.Series("pred", pred)).join(nm, on=schema.MVT_ID, how="left")
    OUT.mkdir(exist_ok=True)
    out.write_parquet(OUT / "prod_valid.parquet")
    print("prod validation RMSE", round(float(np.sqrt(np.mean((pred - out[schema.TARGET].to_numpy()) ** 2))), 2))


def table() -> pl.DataFrame:
    base = pl.read_parquet(OUT / "prod_valid.parquet").with_columns(pl.col(schema.MVT_TIME).dt.date().alias("day"))
    obs = adsb.load_observations(VALIDATION_DAYS)
    return base.join(obs, on=schema.MVT_ID, how="inner")


def matrix(frame: pl.DataFrame):
    x = frame.select(FEATURES).with_columns(pl.col(c).cast(pl.Float64) for c in FEATURES).to_pandas()
    x["ADEP"] = frame[schema.ADEP].to_pandas().astype("category")
    return x


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def rules(frame: pl.DataFrame, pred: np.ndarray) -> np.ndarray:
    """The production rules that bound a prediction, applied again after the correction."""
    matched = frame["matched"].to_numpy()
    low, high = train._lobt_window(frame)
    pred = np.where(matched, np.clip(pred, low, high), pred)
    elsewhere = ~matched & (frame[schema.ADEP].to_numpy() != "LIRF")
    pred = np.where(elsewhere, np.minimum(pred, train.ORPHAN_CAP_SEC), pred)
    return np.clip(pred, 0.0, submit.MAX_PLAUSIBLE_TAXI_SEC)


def fit_corrector(frame: pl.DataFrame) -> lgb.Booster:
    rows = frame.filter(pl.col("adsb_matched"))
    residual = (rows[schema.TARGET].cast(pl.Float64) - rows["pred"]).to_numpy()
    return lgb.train(PARAMS, lgb.Dataset(matrix(rows), label=residual, categorical_feature=["ADEP"]),
                     num_boost_round=ROUNDS)


def build() -> None:
    frame = table()
    booster = fit_corrector(frame)
    print(f"corrector fitted on {frame['day'].n_unique()} days, {int(frame['adsb_matched'].sum())} ADS-B rows", flush=True)
    model = pickle.load(open("data/model_v11.pkl", "rb"))
    ranking = data.load_ranking()
    around = features.surroundings(ranking)
    rows = features.build(data.departures(ranking), model.unimpeded.lazy(), around).collect()
    pred = model.predict(rows)
    matched = rows.select(train.has_flight_record()).to_numpy().ravel()
    p = np.empty(len(rows))
    p[matched] = model.joined.components(rows.filter(pl.Series(matched)))[0]
    p[~matched] = model.orphan.components(rows.filter(pl.Series(~matched)))[0]
    base = rows.select(
        schema.MVT_ID, schema.ADEP, schema.MVT_TIME, schema.LOBT,
        pl.Series("matched", matched), pl.Series("p", p), pl.Series("pred", pred),
        pl.col("sched_to_takeoff_sec").alias("gap"), pl.col("aobt_to_takeoff_sec").alias("nm_taxi"),
    )
    days = [dt.date(2026, m, d) for m in (1, 7) for d in range(1, 32)]
    obs = adsb.load_observations(days)
    print(f"2026: {obs.height} departures observed of {base.height}; ADS-B matched {obs['adsb_matched'].mean():.1%}, "
          f"push-back seen {obs['adsb_pushback_seen'].mean():.1%}", flush=True)
    joined = base.join(obs, on=schema.MVT_ID, how="left").with_columns(pl.col("adsb_matched").fill_null(False))
    correction = np.zeros(joined.height)
    seen = joined["adsb_matched"].to_numpy()
    correction[seen] = booster.predict(matrix(joined.filter(pl.Series(seen))))
    final = rules(joined, joined["pred"].to_numpy() + correction)
    predictions = dict(zip(joined[schema.MVT_ID].to_list(), final.tolist()))
    path = Path("data/submission_v12.parquet")
    submit.write_submission(predictions, path)
    print(path, "problems:", submit.verify_submission(path, predictions), flush=True)
    old = pl.read_parquet("data/submission_v11.parquet").select(schema.MVT_ID, pl.col(schema.TARGET).alias("v11"))
    new = pl.read_parquet(path).select(schema.MVT_ID, pl.col(schema.TARGET).alias("v12"))
    d = joined.select(schema.MVT_ID, schema.ADEP, "adsb_matched").join(old, on=schema.MVT_ID).join(new, on=schema.MVT_ID)
    print(d.group_by(schema.ADEP).agg(pl.len(), pl.col("adsb_matched").mean().alias("adsb"),
                                      ((pl.col("v12") - pl.col("v11")) ** 2).mean().sqrt().alias("rms_change"),
                                      pl.col("v11").mean(), pl.col("v12").mean()).sort(schema.ADEP))
    print(d.with_columns((pl.col("v12") - pl.col("v11")).abs().alias("c")).sort("c", descending=True).head(8))


def score() -> None:
    frame = table()
    days = sorted(frame["day"].unique().to_list())
    truth = frame[schema.TARGET].to_numpy().astype(float)
    pred = frame["pred"].to_numpy()
    seen = frame["adsb_matched"].to_numpy()
    print(f"{len(days)} validation days, {frame.height} departures, ADS-B matched {seen.mean():.1%}, "
          f"push-back seen {frame['adsb_pushback_seen'].mean():.1%}")
    folds = np.array_split(np.array(days, dtype=object), 5)
    day_of = frame["day"].to_numpy()
    correction = np.zeros(len(frame))
    for held in folds:
        test = np.isin(day_of, held)
        fit_rows = ~test & seen
        booster = lgb.train(PARAMS, lgb.Dataset(matrix(frame.filter(pl.Series(fit_rows))),
                                                label=(truth - pred)[fit_rows], categorical_feature=["ADEP"]),
                            num_boost_round=ROUNDS)
        rows = test & seen
        correction[rows] = booster.predict(matrix(frame.filter(pl.Series(rows))))
    rng = np.random.default_rng(7)
    n = len(truth)
    draws = [rng.integers(0, n, n) for _ in range(300)]
    month = frame[schema.MVT_TIME].dt.month().to_numpy()
    pushed = frame["adsb_pushback_seen"].to_numpy()
    for name, mask in (("all ADS-B rows", seen), ("push-back seen only", pushed)):
        for shrink in (0.5, 1.0):
            corrected = rules(frame, pred + shrink * np.where(mask, correction, 0.0))
            diffs = [rmse(corrected[i], truth[i]) - rmse(pred[i], truth[i]) for i in draws]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            jan = rmse(corrected[month == 1], truth[month == 1]) - rmse(pred[month == 1], truth[month == 1])
            jul = rmse(corrected[month == 7], truth[month == 7]) - rmse(pred[month == 7], truth[month == 7])
            print(f"{name:20s} x{shrink:.1f}: {rmse(corrected, truth):6.1f}s vs {rmse(pred, truth):6.1f}s "
                  f"({rmse(corrected, truth) - rmse(pred, truth):+.2f}s, 95% CI {lo:+.2f} to {hi:+.2f})  "
                  f"Jan {jan:+.2f}  Jul {jul:+.2f}  | on push-back rows {rmse(corrected[pushed], truth[pushed]):.0f}s "
                  f"vs {rmse(pred[pushed], truth[pushed]):.0f}s")


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"observe": observe, "prod": prod, "--score": score, "build": build}[step]()
