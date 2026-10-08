"""Round 9: a winter fold for the ADS-B corrector.

The corrector (`taxiout.correct`) learns the model's error from adsb.lol's
ground observations, and needs honest held-out predictions to learn from.
Until now those came from one fold, January and July 2025, two mild
months; the test set is January and July 2026, and January 2026 was not
mild. This round adds a second fold: the production model fitted on the
other ten months and predicting February and December 2025, which carry
the winter disruption days the first fold lacks (LTFM's February storm,
December de-icing).

Variants, all scored by cross-validation over whole days with the
production rules (LOBT window, orphan cap and floor) applied after the
correction, and compared by a bootstrap over days:

- `janjul`: production, the corrector fitted on January and July only.
- `pooled`: fitted on all 124 days of both folds.
- `level`: pooled, plus the airport's live taxi level
  (`train.neighbour_taxi_level`) as an input, since adsb.lol dates
  push-backs late when the airport is congested.

Results (121 days with ADS-B, 653,828 departures; 31 December 2025 has no
adsb.lol release). February and December score 238.9s before any
correction, January and July 316.5s:

| Corrector | Jan+Jul | Feb+Dec |
|---|---|---|
| none | 316.5s | 234.9s |
| `janjul` | 308.6s | 226.7s |
| `pooled` | **308.0s** | **224.4s** |
| `level` | 308.1s | 224.5s |

`pooled` against `janjul`: -0.60s on January and July (95% CI over days
-1.26 to -0.04), -2.28s on February and December (-3.00 to -1.73),
better in every month (December -3.9s). The congestion level adds
nothing (+0.11s, -0.01 to +0.23). `build_final.py` now fits the corrector
on every fold.

`build` fits the chosen variant on every day and applies it to the final
model's 2026 predictions (`data/build_final/`), writing
`data/submission_v14.parquet`.

Usage: python pipeline/experiments_round9.py winter | --score | build [variant]
"""

import datetime as dt
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import adsb, correct, data, features, schema, submit, train  # noqa: E402

OUT = Path("data/round9")
STAGES = Path("data/build_final")
JANJUL = [dt.date(2025, m, d) for m in (1, 7) for d in range(1, 32)]
FEBDEC = [dt.date(2025, m, d) for m, n in ((2, 28), (12, 31)) for d in range(1, n + 1)]
VARIANTS = {
    "janjul": (correct.FEATURES, "janjul"),
    "pooled": (correct.FEATURES, "all"),
    "level": ([*correct.FEATURES, "level"], "all"),
}


def winter() -> None:
    """The production model fitted without February and December, and its inputs for those months."""
    model, held_out, _ = train.validate(data.load_training(), (2, 12))
    print(f"February + December RMSE before the correction: {model.validation_rmse:.1f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    correct.inputs(model, held_out).write_parquet(OUT / "winter_rows.parquet")


def table() -> pl.DataFrame:
    """Both folds' held-out rows, with their observations and the live taxi level."""
    summer = pickle.load(open(STAGES / "validation_rows.pkl", "rb"))
    rows = pl.concat([summer, pl.read_parquet(OUT / "winter_rows.parquet")], how="vertical_relaxed")
    rows = correct._join(rows, adsb.load_observations(JANJUL + FEBDEC))
    return with_level(rows).with_columns(pl.col(schema.MVT_TIME).dt.date().alias("day"))


def with_level(rows: pl.DataFrame) -> pl.DataFrame:
    level = train.neighbour_taxi_level(
        rows[schema.ADEP].to_numpy(),
        rows[schema.MVT_TIME].dt.epoch("s").to_numpy().astype(float),
        rows["nm_taxi"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
    )
    return rows.with_columns(pl.Series("level", level))


def matrix(frame: pl.DataFrame, columns: list[str]):
    x = frame.select(columns).with_columns(pl.col(c).cast(pl.Float64) for c in columns).to_pandas()
    x["ADEP"] = frame[schema.ADEP].to_pandas().astype("category")
    return x


def fit(frame: pl.DataFrame, columns: list[str]) -> lgb.Booster:
    seen = frame.filter(pl.col("adsb_matched"))
    residual = (seen[schema.TARGET].cast(pl.Float64) - seen["pred"]).to_numpy()
    return lgb.train(correct.PARAMS, lgb.Dataset(matrix(seen, columns), label=residual, categorical_feature=["ADEP"]),
                     num_boost_round=correct.ROUNDS)


def cross_fitted(frame: pl.DataFrame, columns: list[str], train_on: str) -> np.ndarray:
    """Each day's correction from a corrector that never saw that day."""
    day = frame["day"].to_numpy()
    days = np.array(sorted(set(day)), dtype=object)
    summer = np.isin(day, np.array(JANJUL, dtype=object))
    seen = frame["adsb_matched"].to_numpy()
    correction = np.zeros(frame.height)
    for k in range(10):
        held = days[k::10]
        test = np.isin(day, held)
        usable = ~test & (summer if train_on == "janjul" else True)
        booster = fit(frame.filter(pl.Series(usable)), columns)
        rows = test & seen
        correction[rows] = booster.predict(matrix(frame.filter(pl.Series(rows)), columns))
    return correction


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def score() -> None:
    frame = table()
    truth = frame[schema.TARGET].to_numpy().astype(float)
    pred = frame["pred"].to_numpy()
    print(f"{frame['day'].n_unique()} days, {frame.height} departures, ADS-B matched {frame['adsb_matched'].mean():.1%}, "
          f"push-back seen {frame['adsb_pushback_seen'].fill_null(False).mean():.1%}", flush=True)
    results = {"none": correct.rules(frame, pred)}
    for name, (columns, train_on) in VARIANTS.items():
        results[name] = correct.rules(frame, pred + cross_fitted(frame, columns, train_on))
        print(f"fitted {name}", flush=True)

    month = frame[schema.MVT_TIME].dt.month().to_numpy()
    day = frame["day"].to_numpy()
    days = np.array(sorted(set(day)), dtype=object)
    index = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(9)
    draws = [np.concatenate([index[d] for d in rng.choice(days, len(days))]) for _ in range(300)]
    slices = {"Jan+Jul": np.isin(month, (1, 7)), "Feb+Dec": np.isin(month, (2, 12)), "all": np.ones(len(day), bool)}
    for label, mask in slices.items():
        line = [f"{label:8s}"] + [f"{name} {rmse(p[mask], truth[mask]):.1f}" for name, p in results.items()]
        print("  ".join(line))
    for a, b in (("none", "janjul"), ("janjul", "pooled"), ("pooled", "level"), ("janjul", "level")):
        for label, mask in slices.items():
            pa, pb = results[a], results[b]
            diffs = []
            for i in draws:
                i = i[mask[i]]
                diffs.append(rmse(pb[i], truth[i]) - rmse(pa[i], truth[i]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"{b} vs {a}, {label}: {rmse(pb[mask], truth[mask]) - rmse(pa[mask], truth[mask]):+.2f}s "
                  f"(95% CI over days {lo:+.2f} to {hi:+.2f})")
        for m in (1, 7, 2, 12):
            mm = month == m
            print(f"    month {m:2d}: {rmse(results[b][mm], truth[mm]) - rmse(results[a][mm], truth[mm]):+.2f}s")


def build(variant: str = "pooled") -> None:
    columns, train_on = VARIANTS[variant]
    frame = table()
    if train_on == "janjul":
        frame = frame.filter(pl.col("day").is_in(JANJUL))
    booster = fit(frame, columns)
    print(f"{variant}: corrector fitted on {frame['day'].n_unique()} days, {int(frame['adsb_matched'].sum())} ADS-B rows")
    final = pickle.load(open(STAGES / "final_model.pkl", "rb"))
    ranking = data.load_ranking()
    rows = features.build(data.departures(ranking), final.unimpeded.lazy(), features.surroundings(ranking)).collect()
    days = [dt.date(2026, m, d) for m in (1, 7) for d in range(1, 32)]
    rows = with_level(correct._join(correct.inputs(final, rows), adsb.load_observations(days)))
    seen = rows["adsb_matched"].to_numpy()
    correction = np.zeros(rows.height)
    correction[seen] = booster.predict(matrix(rows.filter(pl.Series(seen)), columns))
    pred = correct.rules(rows, rows["pred"].to_numpy() + correction)
    predictions = dict(zip(rows[schema.MVT_ID].to_list(), pred.tolist()))
    path = Path("data/submission_v14.parquet")
    submit.write_submission(predictions, path)
    print(path, "problems:", submit.verify_submission(path, predictions))


if __name__ == "__main__":
    step = sys.argv[1]
    {"winter": winter, "--score": score, "build": lambda: build(*sys.argv[2:])}[step]()
