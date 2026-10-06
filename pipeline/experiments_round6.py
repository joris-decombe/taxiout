"""Round 6: stacking, a corrector trained on cross-fitted predictions.

Stacked generalisation (Wolpert, 1992) with cross-fitting (Chernozhukov et
al., 2018): a second model learns the first one's errors, but only from
predictions the first made on months it was not fitted on, so it learns
the errors the first will make on unseen data rather than its training
error.

  base       the matched mixture as round 3's `anchor` arm (one LightGBM
             classifier, one anchored LightGBM regressor), orphans as
             round 3's `base` arm, production rules on top. Its validation
             components are already saved (`round4.prod_components`).
  out-of-fold  the training months split into five two-month blocks; the
             matched mixture is refitted without each block and predicts
             it. Saved to `data/round6/oof.parquet`.
  corrector  LightGBM on matched rows: target = truth - base prediction,
             inputs = the features plus p, r, gap and the base prediction,
             heavily regularised. Applied to the base and, as a check of
             transfer, to production.

The feature tables come from `experiments_round5.py prepare`, whose
unimpeded reference was fitted on all ten training months, so each block's
reference has seen its own targets: the out-of-fold predictions are a
little optimistic. Threads are capped to share the machine with a build.

Measured 7 October 2026 (out-of-fold base RMSE on matched training rows
207.9s, in line with validation):

  base + half corrector   -0.7s (95% CI -0.8s to -0.5s)  Jan -0.2s  Jul -1.1s
  base + full corrector   -0.4s (95% CI -0.7s to -0.2s)  Jan +0.6s  Jul -1.4s
  prod + half corrector   -0.4s (95% CI -0.6s to -0.3s)  Jan -0.1s  Jul -0.8s
  prod + full corrector   +0.1s (95% CI -0.2s to +0.3s)  Jan +0.9s  Jul -0.8s

Not used: half a second, nearly all of it in July, and January (the month
2026 shifted most) gets worse at full strength. A corrector fitted on the
production ensemble's own out-of-fold predictions would need five
production-size fits.

Usage: python pipeline/experiments_round6.py oof | corrector | --score
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
import experiments_round4 as round4  # noqa: E402
import experiments_round5 as round5  # noqa: E402
from taxiout import features, schema, train  # noqa: E402

OUT = Path("data/round6")
ROUND5 = Path("data/round5")
BLOCKS = [(2, 3), (4, 5), (6, 8), (9, 10), (11, 12)]
THREADS = 6
CORRECTOR_PARAMS = {
    "objective": "regression", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 500,
    "lambda_l2": 10.0, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "verbosity": -1, "num_threads": THREADS,
}
CORRECTOR_ROUNDS = 600
STACK_INPUTS = ["p", "r", "gap", "base_pred"]


def matched_table(path: Path) -> pl.DataFrame:
    columns = list(features.FEATURE_COLUMNS) + list(features.CATEGORICAL_COLUMNS)
    keep = [schema.MVT_ID, schema.ADEP, schema.TARGET, schema.MVT_TIME, schema.SCHED_TIME, schema.LOBT,
            schema.BLOCK_TIME, "sched_to_takeoff_sec", "aobt_to_takeoff_sec", "unimpeded_taxi_sec", *columns]
    table = pl.read_parquet(path).filter(train.has_flight_record()).select(list(dict.fromkeys(keep)))
    # Features only: MVT_ID is a Float64 near 1.9e8, and float32 would round
    # 1.7 million IDs onto ~110,000 values and scramble every join.
    floats = [c for c in features.FEATURE_COLUMNS if table.schema.get(c) == pl.Float64]
    return table.with_columns(pl.col(floats).cast(pl.Float32))


def oof() -> None:
    table = matched_table(ROUND5 / "train.parquet")
    columns, categoricals = list(features.FEATURE_COLUMNS), list(features.CATEGORICAL_COLUMNS)
    month = table[schema.MVT_TIME].dt.month()
    parts = []
    for block in BLOCKS:
        held = month.is_in(list(block))
        fit_rows, pred_rows = table.filter(~held), table.filter(held)
        mixture = train._fit_mixture(
            fit_rows, columns, categoricals,
            {**train.PARAMS, "num_threads": THREADS}, train.JOINED_ROUNDS,
            {**train.JOINED_CLASSIFIER_PARAMS, "num_threads": THREADS}, train.JOINED_CLASSIFIER_ROUNDS,
            anchored=True,
        )
        p, r, gap = mixture.components(pred_rows)
        parts.append(pred_rows.select(
            schema.MVT_ID, schema.ADEP, schema.TARGET, schema.MVT_TIME, schema.SCHED_TIME, schema.LOBT,
            pl.lit(True).alias("matched"),
        ).with_columns(pl.Series("p", p), pl.Series("r", r), pl.Series("gap", gap)))
        print(f"block {block} done", flush=True)
    OUT.mkdir(exist_ok=True)
    pl.concat(parts).sort(schema.MVT_ID).write_parquet(OUT / "oof.parquet")


def with_base(components: pl.DataFrame, taxi: float, rates: dict) -> pl.DataFrame:
    return components.with_columns(pl.Series("base_pred", round4.predictions(components, taxi, rates)))


def stack_matrix(table: pl.DataFrame, stacked: pl.DataFrame):
    columns, categoricals = list(features.FEATURE_COLUMNS), list(features.CATEGORICAL_COLUMNS)
    joined = table.join(stacked.select(schema.MVT_ID, *STACK_INPUTS), on=schema.MVT_ID, how="inner")
    return joined, train._to_frame(joined, columns + STACK_INPUTS, categoricals)


def corrector() -> None:
    taxi, rates = round4.fitted_constants()
    stacked = with_base(pl.read_parquet(OUT / "oof.parquet"), taxi, rates)
    table, x = stack_matrix(matched_table(ROUND5 / "train.parquet"), stacked)
    residual = (table[schema.TARGET].cast(pl.Float64) - table["base_pred"]).to_numpy()
    booster = lgb.train(
        CORRECTOR_PARAMS,
        lgb.Dataset(x, label=residual, categorical_feature=list(features.CATEGORICAL_COLUMNS)),
        num_boost_round=CORRECTOR_ROUNDS,
    )
    booster.save_model(str(OUT / "corrector.txt"))
    print("corrector fitted; out-of-fold base RMSE on matched training rows:",
          round(float(np.sqrt(np.mean(residual ** 2))), 1), flush=True)


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def score() -> None:
    taxi, rates = round4.fitted_constants()
    booster = lgb.Booster(model_file=str(OUT / "corrector.txt"))
    valid = matched_table(ROUND5 / "valid.parquet")
    base = round4.prod_components()
    prod = round5.blend(round5.production(), pl.col("r_lgb"), pl.col("r_cb"))
    variants = {}
    for name, components in (("base", base), ("prod", prod)):
        stacked = with_base(components, taxi, rates)
        joined, x = stack_matrix(valid, stacked.filter(pl.col("matched")))
        correction = dict(zip(joined[schema.MVT_ID].to_list(), booster.predict(x)))
        pred = stacked["base_pred"].to_numpy()
        ids = stacked[schema.MVT_ID].to_list()
        low, high = train._lobt_window(stacked)
        for shrink in (0.0, 0.5, 1.0):
            corrected = np.array([v + shrink * correction.get(i, 0.0) for i, v in zip(ids, pred)])
            matched = stacked["matched"].to_numpy()
            variants[f"{name} + {shrink:.1f} corrector"] = np.where(matched, np.clip(corrected, low, high), corrected)
    truth = base[schema.TARGET].to_numpy().astype(float)
    month = base[schema.MVT_TIME].dt.month().to_numpy()
    rng = np.random.default_rng(7)
    draws = [rng.integers(0, len(truth), len(truth)) for _ in range(300)]
    for name, pred in variants.items():
        reference = variants[name.split(" + ")[0] + " + 0.0 corrector"]
        diffs = [rmse(pred[i], truth[i]) - rmse(reference[i], truth[i]) for i in draws]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        jan = rmse(pred[month == 1], truth[month == 1]) - rmse(reference[month == 1], truth[month == 1])
        jul = rmse(pred[month == 7], truth[month == 7]) - rmse(reference[month == 7], truth[month == 7])
        print(f"{name:22s} {rmse(pred, truth):6.1f}s  {rmse(pred, truth) - rmse(reference, truth):+5.1f}s "
              f"(95% CI {lo:+.1f} to {hi:+.1f})  Jan {jan:+.1f}  Jul {jul:+.1f}")


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"oof": oof, "corrector": corrector, "--score": score}[step]()
