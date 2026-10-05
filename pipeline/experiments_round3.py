"""Round 3: day-shifted rows, linear leaves, and the LOBT window.

Fits the production pipeline (`train.validate`) once per arm and saves each
arm's held-out mixture components (p, r, gap) to `data/round3/<arm>.parquet`,
so post-prediction rules can be scored without refitting:

  base        the current model
  noshift     the orphan normal-taxi regressor fitted without rows above
              80,000s (a day-shifted off-block, see below)
  linear      noshift + linear leaves (`linear_tree`) on the matched regressor
  lobtfeat    noshift + take-off and schedule against LOBT as features
  anchor      noshift + the matched regressor boosted from take-off minus
              AOBT (`train._anchor`), learning only the deviation from it

Day-shifted rows. Fourteen 2025 departures have a taxi-out of a day plus a
normal taxi (84,240s to 88,392s), with no at-schedule off-block, and carry
17% of the target's variance on their own. Nearly all are orphans at LIRF,
and there every orphan 14h to 26h after schedule is either at schedule or
day-shifted, never a normal taxi. `train.DAY_SHIFT` makes the mixture's
second component a day plus a normal taxi in that band.

Measured 6 October 2026 (+lobt is the LOBT window, +dayshift the rule):

  base                 343.6s
  noshift              391.8s   orphan regressor without day-shifted rows:
                                it had learnt part of the day, so worse
  linear               951.2s   linear leaves extrapolate to +-300,000s on
                                a few rows; 292s matched even when bounded
  anchor               342.5s   -1.1s, 95% CI -1.7s to -0.6s
  base+lobt            339.8s   -3.8s, 95% CI -8.6s to -1.1s
  anchor+lobt+dayshift 329.8s

An earlier day-shift rule added (1 - p) * q * 86,400 with q a fitted share;
it reached 334.2s at LIRF only and cost 5.3s when applied everywhere.

Usage: python pipeline/experiments_round3.py [arm ...]   (fits)
       python pipeline/experiments_round3.py --score     (scores saved arms)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import data, features, schema, train  # noqa: E402

OUT = Path("data/round3")
DAY = 86_400
DAY_SHIFT_MIN_SEC = 80_000
# Gap bands (hours) in which day-shifted orphans occur often enough to price.
BANDS_H = [14, 20, 26]
LOBT = "LOBT_flt"
LOBT_WINDOW_SEC = 3606
LOBT_FEATURES = ["lobt_to_takeoff_sec", "sched_vs_lobt_sec"]


def configure(arm: str) -> None:
    train.ORPHAN_NORMAL_MAX_SEC = None if arm == "base" else DAY_SHIFT_MIN_SEC
    if arm == "linear":
        train.PARAMS = {**train.PARAMS, "linear_tree": True, "linear_lambda": 1.0}
    train.JOINED_ANCHORED = arm == "anchor"
    if arm == "lobtfeat":
        features.FEATURE_COLUMNS += [c for c in LOBT_FEATURES if c not in features.FEATURE_COLUMNS]


def fit(arm: str) -> None:
    configure(arm)
    model, frame, _ = train.validate(data.load_training())
    matched = frame.select(train.has_flight_record()).to_numpy().ravel()
    parts = []
    for is_matched, mixture in ((True, model.joined), (False, model.orphan)):
        rows = frame.filter(pl.Series(matched == is_matched))
        p, r, gap = mixture.components(rows)
        parts.append(rows.select(
            schema.MVT_ID, schema.ADEP, schema.TARGET, schema.MVT_TIME, schema.SCHED_TIME, LOBT,
            pl.lit(is_matched).alias("matched"),
        ).with_columns(pl.Series("p", p), pl.Series("r", r), pl.Series("gap", gap)))
    out = pl.concat(parts).sort(schema.MVT_ID)
    OUT.mkdir(exist_ok=True)
    out.write_parquet(OUT / f"{arm}.parquet")
    print(f"{arm}: validation {model.validation_rmse:.1f}s", flush=True)


def day_shift_taxi() -> float:
    departures = data.departures(data.load_training())
    fitted, _ = data.train_validation_split(departures)
    return train.normal_orphan_taxi(fitted.filter(~train.has_flight_record()).collect())


def predictions(frame: pl.DataFrame, day_shift_taxi: float | None, lobt: bool) -> np.ndarray:
    """`train.TrainedModel.predict`'s rules, applied to saved components."""
    p = frame["p"].to_numpy().copy()
    r = frame["r"].to_numpy().copy()
    gap = frame["gap"].to_numpy()
    matched = frame["matched"].to_numpy()
    if lobt:
        p[matched & ~train._sched_in_window(frame)] = 0.0
    if day_shift_taxi is not None:
        band = ~matched & train._day_shift_band(frame, gap)
        r[band] = train.DAY_SEC + day_shift_taxi
    pred = p * gap + (1 - p) * r
    if lobt:
        low, high = train._lobt_window(frame)
        pred = np.where(matched, np.clip(pred, low, high), pred)
    return pred


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def score() -> None:
    rng = np.random.default_rng(7)
    rates = day_shift_taxi()
    variants = {}
    base = pl.read_parquet(OUT / "base.parquet")
    for path in sorted(OUT.glob("*.parquet")):
        frame = pl.read_parquet(path)
        arm = path.stem
        if arm not in ("base", "noshift"):
            # Matched-group arms: keep the base orphan model, which `noshift`
            # showed to be the better one, so the comparison isolates them.
            frame = pl.concat([frame.filter(pl.col("matched")), base.filter(~pl.col("matched"))]).sort(schema.MVT_ID)
        variants[arm] = predictions(frame, None, False)
        variants[f"{arm}+lobt"] = predictions(frame, None, True)
        variants[f"{arm}+lobt+dayshift"] = predictions(frame, rates, True)
    truth = frame[schema.TARGET].to_numpy().astype(float)
    matched = frame["matched"].to_numpy()
    airport = frame[schema.ADEP].to_numpy()
    for name, pred in variants.items():
        lirf = airport == "LIRF"
        print(f"{name:28s} overall {rmse(pred, truth):6.1f}s  matched {rmse(pred[matched], truth[matched]):6.1f}s  "
              f"orphan {rmse(pred[~matched], truth[~matched]):7.1f}s  LIRF {rmse(pred[lirf], truth[lirf]):6.1f}s")
    reference = variants["base"] if "base" in variants else next(iter(variants.values()))
    n = len(truth)
    draws = [rng.integers(0, n, n) for _ in range(300)]
    for name, pred in variants.items():
        diffs = [rmse(pred[i], truth[i]) - rmse(reference[i], truth[i]) for i in draws]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  base -> {name}: {rmse(pred, truth) - rmse(reference, truth):+.1f}s, 95% CI {lo:+.1f}s to {hi:+.1f}s")


if __name__ == "__main__":
    if sys.argv[1:] == ["--score"]:
        score()
    else:
        for arm in sys.argv[1:] or ["base", "noshift", "linear"]:
            fit(arm)
