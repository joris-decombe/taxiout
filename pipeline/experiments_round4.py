"""Round 4: changes the model literature points to, on top of round 3.

Each arm refits the production pipeline (`train.validate`) with one change
and saves its held-out mixture components to `data/round4/<arm>.parquet`.
Scoring applies the production rules (`train.TrainedModel.predict`: LOBT
window, LIRF day-shift and late-orphan rules) to every arm, and compares
each with `prod`, the current model:

  prod     the current model (round 3's anchor arm for matched rows, its
           base arm for orphans: the same configuration)
  queue    + `context.QUEUEING_COLUMNS` on the matched group: adjusted
           traffic (Simaiakis and Balakrishnan, Transportation Science 2016)
           and the flight's place in its runway busy period
           -> 324.5s, -0.1s (95% CI -0.3s to +0.1s): no effect
  catboost prod, with the matched normal-taxi regressor's prediction
           blended with a CatBoost one boosted from the same anchor
           (Prokhorenkova et al., NeurIPS 2018); scored at several weights
           -> 30%: 322.4s, 50%: 321.8s (-2.7s, 95% CI -3.4s to -2.2s;
              January -2.2s, July -3.3s), 70%: 321.9s. Kept at 50%.

  extra    on top of the CatBoost blend (now prod): two more seeds of the
           matched LightGBM regressor, averaged; and a CatBoost orphan
           normal-taxi regressor, scored at several blend weights

           -> seeds: -0.2s (95% CI -0.3s to -0.2s); orphan CatBoost at 50%:
              -2.2s (95% CI -3.7s to -0.9s), 30%: -1.7s, 70%: -2.2s, alone:
              -1.4s. Both kept; together 319.4s.

Recalibrating the at-schedule probability (isotonic, per group) was ruled
out before fitting: even calibrated in-sample on validation itself, by
probability bin, it moved prod by at most -0.5s, because the LIRF rules
already replace the costly part of p.

Usage: python pipeline/experiments_round4.py [arm ...]   (fits)
       python pipeline/experiments_round4.py --score     (scores saved arms)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import context, data, features, schema, train  # noqa: E402

OUT = Path("data/round4")
ROUND3 = Path("data/round3")


def configure(arm: str) -> None:
    if arm == "queue":
        features.FEATURE_COLUMNS += [c for c in context.QUEUEING_COLUMNS if c not in features.FEATURE_COLUMNS]


def fit(arm: str) -> None:
    configure(arm)
    model, frame, _ = train.validate(data.load_training())
    matched = frame.select(train.has_flight_record()).to_numpy().ravel()
    parts = []
    for is_matched, mixture in ((True, model.joined), (False, model.orphan)):
        rows = frame.filter(pl.Series(matched == is_matched))
        p, r, gap = mixture.components(rows)
        parts.append(rows.select(
            schema.MVT_ID, schema.ADEP, schema.TARGET, schema.MVT_TIME, schema.SCHED_TIME, schema.LOBT,
            pl.lit(is_matched).alias("matched"),
        ).with_columns(pl.Series("p", p), pl.Series("r", r), pl.Series("gap", gap)))
    OUT.mkdir(exist_ok=True)
    pl.concat(parts).sort(schema.MVT_ID).write_parquet(OUT / f"{arm}.parquet")
    print(f"{arm}: validation {model.validation_rmse:.1f}s", flush=True)


CATBOOST_PARAMS = {
    "loss_function": "RMSE", "iterations": 2000, "learning_rate": 0.1, "depth": 8,
    "thread_count": -1, "verbose": 250, "allow_writing_files": False,
}


def fit_catboost() -> None:
    """The matched normal-taxi regressor as CatBoost; saves its validation r."""
    from catboost import CatBoostRegressor, Pool

    training = data.load_training()
    around = features.surroundings(training)
    fitted, held_out = data.train_validation_split(data.departures(training))
    unimpeded = features.unimpeded_taxi_reference(fitted).collect()
    rows = features.build(fitted, unimpeded.lazy(), around).collect().filter(train.has_flight_record())
    rows = rows.filter(~train.off_block_at_schedule().fill_null(False))
    valid = features.build(held_out, unimpeded.lazy(), around).collect().filter(train.has_flight_record())
    columns, categoricals = list(features.FEATURE_COLUMNS), list(features.CATEGORICAL_COLUMNS)

    def pool(frame: pl.DataFrame, label: bool) -> Pool:
        x = frame.select(columns + categoricals).with_columns(
            pl.col(c).cast(pl.Utf8).fill_null("NA") for c in categoricals
        ).to_pandas()
        y = frame[schema.TARGET].to_numpy() if label else None
        # The anchor is passed only in training; prediction adds it back.
        return Pool(x, label=y, cat_features=categoricals, baseline=train._anchor(frame) if label else None)

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(pool(rows, True))
    r_cb = model.predict(pool(valid, False)) + train._anchor(valid)
    OUT.mkdir(exist_ok=True)
    valid.select(schema.MVT_ID).with_columns(pl.Series("r_cb", r_cb)).write_parquet(OUT / "catboost_r.parquet")
    print("catboost: saved", flush=True)


SEEDS = (1, 2)


def fit_extra() -> None:
    """More matched LightGBM seeds, and a CatBoost orphan regressor; saves their validation r."""
    from catboost import CatBoostRegressor

    training = data.load_training()
    around = features.surroundings(training)
    fitted, held_out = data.train_validation_split(data.departures(training))
    unimpeded = features.unimpeded_taxi_reference(fitted).collect()
    rows = features.build(fitted, unimpeded.lazy(), around).collect()
    rows = rows.filter(~train.off_block_at_schedule().fill_null(False))
    valid = features.build(held_out, unimpeded.lazy(), around).collect()
    matched_rows, orphan_rows = rows.filter(train.has_flight_record()), rows.filter(~train.has_flight_record())
    matched_valid, orphan_valid = valid.filter(train.has_flight_record()), valid.filter(~train.has_flight_record())
    columns, categoricals = list(features.FEATURE_COLUMNS), list(features.CATEGORICAL_COLUMNS)

    out = matched_valid.select(schema.MVT_ID)
    for seed in SEEDS:
        booster = train._fit(
            matched_rows, columns, categoricals, {**train.PARAMS, "seed": seed}, train.JOINED_ROUNDS, anchored=True
        )
        r = booster.predict(train._to_frame(matched_valid, columns, categoricals)) + train._anchor(matched_valid)
        out = out.with_columns(pl.Series(f"r_s{seed}", r))
        print(f"seed {seed} done", flush=True)
    OUT.mkdir(exist_ok=True)
    out.write_parquet(OUT / "seeds_r.parquet")

    orphan_columns = train.orphan_columns()
    orphan_rows = orphan_rows.filter(pl.col(schema.TARGET) < (train.ORPHAN_NORMAL_MAX_SEC or float("inf")))
    twin = CatBoostRegressor(**{**train.CATBOOST_PARAMS, "iterations": 1000, "learning_rate": 0.05, "depth": 6})
    twin.fit(train._catboost_pool(orphan_rows, orphan_columns, train.ORPHAN_CATEGORICALS, label=True))
    r_cb = twin.predict(train._catboost_pool(orphan_valid, orphan_columns, train.ORPHAN_CATEGORICALS))
    orphan_valid.select(schema.MVT_ID).with_columns(pl.Series("r_cb_orphan", r_cb)).write_parquet(OUT / "orphan_cb_r.parquet")
    print("orphan catboost done", flush=True)


def prod_components() -> pl.DataFrame:
    matched = pl.read_parquet(ROUND3 / "anchor.parquet").filter(pl.col("matched"))
    orphans = pl.read_parquet(ROUND3 / "base.parquet").filter(~pl.col("matched"))
    return pl.concat([matched, orphans]).sort(schema.MVT_ID)


def fitted_constants() -> tuple[float, dict[int, float]]:
    """The day-shift taxi and LIRF band shares, from the training months only."""
    departures = data.departures(data.load_training())
    fitted, _ = data.train_validation_split(departures)
    orphans = fitted.filter(~train.has_flight_record()).with_columns(
        (pl.col(schema.MVT_TIME) - pl.col(schema.SCHED_TIME)).dt.total_seconds().alias("sched_to_takeoff_sec")
    ).collect()
    return train.normal_orphan_taxi(orphans), train.lirf_art_rates(orphans)


def predictions(frame: pl.DataFrame, taxi: float, rates: dict[int, float]) -> np.ndarray:
    """`train.TrainedModel.predict`'s rules, applied to saved components."""
    p = frame["p"].to_numpy().copy()
    r = frame["r"].to_numpy().copy()
    gap = frame["gap"].to_numpy()
    matched = frame["matched"].to_numpy()
    orphan = ~matched
    p[matched & ~train._sched_in_window(frame)] = 0.0
    band_rate = train._lirf_band_rate(frame, gap, rates)
    late = orphan & ~np.isnan(band_rate)
    trusted = late & (gap / 3600 > train.LIRF_ORPHAN_FULL_ABOVE_H)
    p = np.where(trusted, band_rate, np.where(late, 0.5 * (p + band_rate), p))
    r = np.where(late, taxi, r)
    r = np.where(orphan & train._day_shift_band(frame, gap), train.DAY_SEC + taxi, r)
    pred = p * gap + (1 - p) * r
    low, high = train._lobt_window(frame)
    return np.where(matched, np.clip(pred, low, high), pred)


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def score() -> None:
    taxi, rates = fitted_constants()
    before = prod_components()
    variants = {"before_cb": predictions(before, taxi, rates)}
    cb = before.join(pl.read_parquet(OUT / "catboost_r.parquet"), on=schema.MVT_ID, how="left")
    for w in (0.3, 0.5, 0.7):
        blended = cb.with_columns(
            pl.when(pl.col("matched")).then(w * pl.col("r_cb") + (1 - w) * pl.col("r")).otherwise(pl.col("r")).alias("r")
        )
        variants[f"catboost{w}"] = predictions(blended, taxi, rates)
    # The CatBoost blend is in production now; later arms are measured from it.
    variants["prod"] = variants.pop("catboost0.5")
    prod = cb.with_columns(
        pl.when(pl.col("matched")).then(0.5 * pl.col("r_cb") + 0.5 * pl.col("r")).otherwise(pl.col("r")).alias("r")
    )
    if (OUT / "seeds_r.parquet").exists():
        seeds = cb.join(pl.read_parquet(OUT / "seeds_r.parquet"), on=schema.MVT_ID, how="left")
        lgb_mean = (pl.col("r") + sum(pl.col(f"r_s{k}") for k in SEEDS)) / (1 + len(SEEDS))
        seeds = seeds.with_columns(
            pl.when(pl.col("matched")).then(0.5 * pl.col("r_cb") + 0.5 * lgb_mean).otherwise(pl.col("r")).alias("r")
        )
        variants["seeds"] = predictions(seeds, taxi, rates)
    if (OUT / "orphan_cb_r.parquet").exists():
        ocb = prod.join(pl.read_parquet(OUT / "orphan_cb_r.parquet"), on=schema.MVT_ID, how="left")
        for w in (0.3, 0.5, 0.7, 1.0):
            blended = ocb.with_columns(
                pl.when(~pl.col("matched")).then(w * pl.col("r_cb_orphan") + (1 - w) * pl.col("r")).otherwise(pl.col("r")).alias("r")
            )
            variants[f"orphan_cb{w}"] = predictions(blended, taxi, rates)
    for path in sorted(OUT.glob("*.parquet")):
        if path.stem.endswith("_r"):
            continue
        frame = pl.read_parquet(path)
        # Matched-group arms predating the blend: compare them with before_cb.
        frame = pl.concat([frame.filter(pl.col("matched")), before.filter(~pl.col("matched"))]).sort(schema.MVT_ID)
        variants[path.stem] = predictions(frame, taxi, rates)
    truth = prod[schema.TARGET].to_numpy().astype(float)
    matched = prod["matched"].to_numpy()
    month = prod[schema.MVT_TIME].dt.month().to_numpy()
    rng = np.random.default_rng(7)
    draws = [rng.integers(0, len(truth), len(truth)) for _ in range(300)]
    reference = variants["prod"]
    for name, pred in variants.items():
        diffs = [rmse(pred[i], truth[i]) - rmse(reference[i], truth[i]) for i in draws]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(
            f"{name:10s} overall {rmse(pred, truth):6.1f}s  matched {rmse(pred[matched], truth[matched]):6.1f}s  "
            f"vs prod {rmse(pred, truth) - rmse(reference, truth):+5.1f}s (95% CI {lo:+.1f} to {hi:+.1f})  "
            f"Jan {rmse(pred[month == 1], truth[month == 1]) - rmse(reference[month == 1], truth[month == 1]):+.1f}  "
            f"Jul {rmse(pred[month == 7], truth[month == 7]) - rmse(reference[month == 7], truth[month == 7]):+.1f}"
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["--score"]:
        score()
    else:
        for arm in sys.argv[1:]:
            {"catboost": fit_catboost, "extra": fit_extra}.get(arm, lambda: fit(arm))()
