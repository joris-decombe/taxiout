"""Round 5: tuning and per-airport models, on cached feature tables.

`prepare` builds the training-month and validation feature tables once
(`data/round5/{train,valid}.parquet`, production features). Each arm then
refits only the regressor it changes and saves its validation prediction r
as `data/round5/<arm>_r.parquet`. Scoring swaps that r into the production
blend (round 4's saved components: LightGBM and CatBoost halves for the
matched group, the orphan blend unchanged) and compares with production:

  lr03      matched LightGBM at learning rate 0.03, 2,000 rounds
  leaves511 matched LightGBM with 511 leaves, 50 rows per leaf
  reg       matched LightGBM with 300 rows per leaf and L2 = 10, which
            should transfer better to a shifted 2026
  airport   one LightGBM per airport (127 leaves), averaged with the global
  cbdeep    matched CatBoost at depth 10, 3,000 rounds, learning rate 0.06
  cbclf     CatBoost at-schedule classifiers (matched and orphan), averaged
            with the LightGBM ones: better ranking of which records were
            copied, where calibration alone had nothing left to give

Measured 6 October 2026 against production then (319.4s):

  lr03       +0.1s as the single seed, -0.0s as a fourth member
  leaves511  +0.1s as the single seed, -0.1s as a fourth member
  reg        +0.5s as the single seed, +0.0s as a fourth member
  airport    -0.5s (95% CI -0.7s to -0.4s)                    kept
  cbdeep     -0.4s (95% CI -0.6s to -0.3s)                    kept
  cbclf      matched -0.6s (95% CI -0.9s to -0.3s)            kept
             orphan  +1.5s (95% CI -0.2s to +3.8s)            not kept
  airport + cbdeep + matched cbclf: 317.9s, -1.4s (95% CI -1.9s to -1.0s)

Usage: python pipeline/experiments_round5.py prepare | <arm> ... | --score
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
import experiments_round4 as round4  # noqa: E402
from taxiout import data, features, schema, train  # noqa: E402

OUT = Path("data/round5")
ROUND4 = Path("data/round4")

LGB_ARMS = {
    "lr03": ({"learning_rate": 0.03}, 2000),
    "leaves511": ({"num_leaves": 511, "min_data_in_leaf": 50}, train.JOINED_ROUNDS),
    "reg": ({"min_data_in_leaf": 300, "lambda_l2": 10.0}, train.JOINED_ROUNDS),
}


def prepare() -> None:
    training = data.load_training()
    around = features.surroundings(training)
    fitted, held_out = data.train_validation_split(data.departures(training))
    unimpeded = features.unimpeded_taxi_reference(fitted).collect()
    OUT.mkdir(exist_ok=True)
    features.build(fitted, unimpeded.lazy(), around).collect().write_parquet(OUT / "train.parquet")
    features.build(held_out, unimpeded.lazy(), around).collect().write_parquet(OUT / "valid.parquet")
    print("prepared", flush=True)


def matched_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = pl.read_parquet(OUT / "train.parquet").filter(train.has_flight_record())
    rows = rows.filter(~train.off_block_at_schedule().fill_null(False))
    valid = pl.read_parquet(OUT / "valid.parquet").filter(train.has_flight_record())
    return rows, valid


def save(valid: pl.DataFrame, r: np.ndarray, arm: str) -> None:
    valid.select(schema.MVT_ID).with_columns(pl.Series("r_arm", r)).write_parquet(OUT / f"{arm}_r.parquet")
    print(f"{arm} done", flush=True)


def fit_arm(arm: str) -> None:
    rows, valid = matched_frames()
    columns, categoricals = list(features.FEATURE_COLUMNS), list(features.CATEGORICAL_COLUMNS)
    anchor = train._anchor(valid)
    if arm in LGB_ARMS:
        overrides, rounds = LGB_ARMS[arm]
        booster = train._fit(rows, columns, categoricals, {**train.PARAMS, **overrides}, rounds, anchored=True)
        save(valid, booster.predict(train._to_frame(valid, columns, categoricals)) + anchor, arm)
    elif arm == "airport":
        r = np.full(len(valid), np.nan)
        airports = valid[schema.ADEP].to_numpy()
        params = {**train.PARAMS, "num_leaves": 127, "min_data_in_leaf": 50}
        for airport in np.unique(airports):
            booster = train._fit(rows.filter(pl.col(schema.ADEP) == airport), columns, categoricals, params,
                                 train.JOINED_ROUNDS, anchored=True)
            mask = airports == airport
            part = valid.filter(pl.Series(mask))
            r[mask] = booster.predict(train._to_frame(part, columns, categoricals)) + train._anchor(part)
        save(valid, r, arm)
    elif arm == "cbclf":
        from catboost import CatBoostClassifier, Pool

        table = pl.read_parquet(OUT / "train.parquet")
        held = pl.read_parquet(OUT / "valid.parquet")
        parts = []
        for matched, cols, cats, params in (
            (True, columns, categoricals, {"depth": 8, "iterations": 1500, "learning_rate": 0.1}),
            (False, train.orphan_columns(), train.ORPHAN_CATEGORICALS, {"depth": 6, "iterations": 800, "learning_rate": 0.05}),
        ):
            group = train.has_flight_record() if matched else ~train.has_flight_record()
            fit_rows, valid_rows = table.filter(group), held.filter(group)
            label = fit_rows.select(train.off_block_at_schedule().fill_null(False).cast(pl.Int8)).to_numpy().ravel()
            # The label rides in the target column; no anchor, so no baseline.
            pool = train._catboost_pool(fit_rows.with_columns(pl.Series(schema.TARGET, label)), cols, cats, label=True)
            clf = CatBoostClassifier(**{**train.CATBOOST_PARAMS, "loss_function": "Logloss", **params})
            clf.fit(pool)
            p_cb = clf.predict_proba(train._catboost_pool(valid_rows, cols, cats))[:, 1]
            parts.append(valid_rows.select(schema.MVT_ID).with_columns(pl.Series("p_cb", p_cb)))
        pl.concat(parts).write_parquet(OUT / "cbclf_p.parquet")
        print("cbclf done", flush=True)
    elif arm == "cbdeep":
        from catboost import CatBoostRegressor

        twin = CatBoostRegressor(**{**train.CATBOOST_PARAMS, "depth": 10, "iterations": 3000, "learning_rate": 0.06})
        twin.fit(train._catboost_pool(rows, columns, categoricals, label=True, anchored=True))
        save(valid, twin.predict(train._catboost_pool(valid, columns, categoricals)) + anchor, arm)


def production() -> pl.DataFrame:
    """Round 4's components with every production blend applied, keeping each half."""
    f = (round4.prod_components()
         .join(pl.read_parquet(ROUND4 / "catboost_r.parquet"), on=schema.MVT_ID, how="left")
         .join(pl.read_parquet(ROUND4 / "seeds_r.parquet"), on=schema.MVT_ID, how="left")
         .join(pl.read_parquet(ROUND4 / "orphan_cb_r.parquet"), on=schema.MVT_ID, how="left"))
    return f.with_columns(
        ((pl.col("r") + pl.col("r_s1") + pl.col("r_s2")) / 3).alias("r_lgb"),
        pl.col("r").alias("r_lgb1"),
        (0.5 * pl.col("r_cb_orphan") + 0.5 * pl.col("r")).alias("r_orphan"),
    )


def blend(f: pl.DataFrame, lgb: pl.Expr, cb: pl.Expr) -> pl.DataFrame:
    return f.with_columns(pl.when(pl.col("matched")).then(0.5 * lgb + 0.5 * cb).otherwise(pl.col("r_orphan")).alias("r"))


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def score() -> None:
    taxi, rates = round4.fitted_constants()
    f = production()
    variants = {"prod": round4.predictions(blend(f, pl.col("r_lgb"), pl.col("r_cb")), taxi, rates)}
    for path in sorted(OUT.glob("*_r.parquet")):
        arm = path.stem[:-2]
        g = f.join(pl.read_parquet(path), on=schema.MVT_ID, how="left")
        if arm == "cbdeep":
            variants[arm] = round4.predictions(blend(g, pl.col("r_lgb"), pl.col("r_arm")), taxi, rates)
            variants[arm + "+cb"] = round4.predictions(
                blend(g, pl.col("r_lgb"), 0.5 * pl.col("r_arm") + 0.5 * pl.col("r_cb")), taxi, rates)
        elif arm == "airport":
            variants[arm] = round4.predictions(
                blend(g, 0.5 * pl.col("r_lgb") + 0.5 * pl.col("r_arm"), pl.col("r_cb")), taxi, rates)
            if (OUT / "cbdeep_r.parquet").exists():
                h = g.join(pl.read_parquet(OUT / "cbdeep_r.parquet").rename({"r_arm": "r_deep"}), on=schema.MVT_ID, how="left")
                variants["airport+cbdeep"] = round4.predictions(
                    blend(h, 0.5 * pl.col("r_lgb") + 0.5 * pl.col("r_arm"), pl.col("r_deep")), taxi, rates)
        else:
            # A tuned single seed against prod's first seed, and as a fourth member.
            variants[arm + " vs 1 seed"] = round4.predictions(blend(g, pl.col("r_arm"), pl.col("r_cb")), taxi, rates)
            variants[arm + " as 4th"] = round4.predictions(
                blend(g, (3 * pl.col("r_lgb") + pl.col("r_arm")) / 4, pl.col("r_cb")), taxi, rates)
    variants["1 seed"] = round4.predictions(blend(f, pl.col("r_lgb1"), pl.col("r_cb")), taxi, rates)
    if (OUT / "cbclf_p.parquet").exists():
        g = blend(f, pl.col("r_lgb"), pl.col("r_cb")).join(pl.read_parquet(OUT / "cbclf_p.parquet"), on=schema.MVT_ID, how="left")
        for name, group in (("cbclf both", pl.lit(True)), ("cbclf matched", pl.col("matched")), ("cbclf orphan", ~pl.col("matched"))):
            mixed = g.with_columns(pl.when(group).then(0.5 * pl.col("p") + 0.5 * pl.col("p_cb")).otherwise(pl.col("p")).alias("p"))
            variants[name] = round4.predictions(mixed, taxi, rates)
    truth = f[schema.TARGET].to_numpy().astype(float)
    matched = f["matched"].to_numpy()
    month = f[schema.MVT_TIME].dt.month().to_numpy()
    rng = np.random.default_rng(7)
    draws = [rng.integers(0, len(truth), len(truth)) for _ in range(300)]
    reference = variants["prod"]
    for name, pred in variants.items():
        diffs = [rmse(pred[i], truth[i]) - rmse(reference[i], truth[i]) for i in draws]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        jan = rmse(pred[month == 1], truth[month == 1]) - rmse(reference[month == 1], truth[month == 1])
        jul = rmse(pred[month == 7], truth[month == 7]) - rmse(reference[month == 7], truth[month == 7])
        print(f"{name:18s} overall {rmse(pred, truth):6.1f}s  matched {rmse(pred[matched], truth[matched]):6.1f}s  "
              f"vs prod {rmse(pred, truth) - rmse(reference, truth):+5.1f}s (95% CI {lo:+.1f} to {hi:+.1f})  "
              f"Jan {jan:+.1f}  Jul {jul:+.1f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--score"]:
        score()
    else:
        for arm in args:
            prepare() if arm == "prepare" else fit_arm(arm)
