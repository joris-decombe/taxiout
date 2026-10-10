"""Round 11: the corrector cross-fitted over the whole year.

Rounds 9 and 10 gave the ADS-B corrector three folds of held-out 2025
predictions (January + July, February + December, June + August). This
round adds March + September, April + October and May + November, each
held out of a fit on the other ten months (`build_final.py` FOLDS), with
adsb.lol streamed for every other day of those months.

Variants, scored by cross-validation over whole days with the production
rules applied after the correction, compared by a bootstrap over days:

- `three`: the corrector fitted on the first three folds (round 10).
- `six`: fitted on all six.

Results (365 days; adsb.lol has no releases for 29 and 31 May 2025):

| Fold | none | `three` | `six` | `six` vs `three`, 95% CI over days |
|---|---|---|---|---|
| Jan + Jul | 316.5s | 308.1s | **307.8s** | -0.26s (-0.49 to -0.06) |
| Feb + Dec | 234.9s | 224.2s | **224.2s** | -0.04s (-0.16 to +0.08) |
| Jun + Aug | 279.5s | 276.1s | **276.1s** | -0.04s (-0.19 to +0.13) |
| Mar + Sep | 220.5s | 215.7s | **215.6s** | -0.19s (-0.30 to -0.10) |
| Apr + Oct | 208.3s | 203.6s | **203.5s** | -0.18s (-0.28 to -0.10) |
| May + Nov | 258.0s | 253.9s | **253.7s** | -0.12s (-0.24 to -0.04) |
| all | 255.9s | 250.0s | **249.9s** | -0.14s (-0.20 to -0.07) |

No fold is worse with six; `build_final.py` uses six. Against no
correction the six-fold corrector gains -6.0s over the year (95% CI -6.9
to -5.2), from -3.4s (June + August, half the days streamed) to -10.7s
(February + December).

`rules` repeats round 10's audit, each production rule switched off, on
every fold whose model is saved (all but January + July and February +
December, fitted before models were kept).

Usage: python pipeline/experiments_round11.py --score | rules
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
import experiments_round9 as round9  # noqa: E402
import experiments_round10 as round10  # noqa: E402
from build_final import FOLD_DAYS, FOLDS  # noqa: E402
from taxiout import adsb, correct, data, features, schema, train  # noqa: E402

STAGES = Path("data/build_final")


def table() -> pl.DataFrame:
    names = ["validation_rows" if f == (1, 7) else f"fold_{f[0]}_{f[1]}_rows" for f in FOLDS]
    rows = pl.concat([pickle.load(open(STAGES / f"{n}.pkl", "rb")) for n in names], how="vertical_relaxed")
    rows = correct._join(rows, adsb.load_observations(FOLD_DAYS))
    return rows.with_columns(pl.col(schema.MVT_TIME).dt.date().alias("day"))


def score() -> None:
    frame = table()
    truth = frame[schema.TARGET].to_numpy().astype(float)
    pred = frame["pred"].to_numpy()
    print(f"{frame['day'].n_unique()} days, {frame.height} departures, ADS-B matched {frame['adsb_matched'].mean():.1%}")
    results = {
        "none": correct.rules(frame, pred),
        "three": correct.rules(frame, pred + round10.cross_fitted(frame, (1, 7, 2, 12, 6, 8))),
        "six": correct.rules(frame, pred + round10.cross_fitted(frame, tuple(range(1, 13)))),
    }
    month = frame[schema.MVT_TIME].dt.month().to_numpy()
    day = frame["day"].to_numpy()
    days = np.array(sorted(set(day)), dtype=object)
    index = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(13)
    draws = [np.concatenate([index[d] for d in rng.choice(days, len(days))]) for _ in range(300)]
    slices = {f"{a}+{b}": (a, b) for a, b in FOLDS} | {"all": tuple(range(1, 13))}
    for label, months in slices.items():
        mask = np.isin(month, months)
        print(f"{label:6s}  " + "  ".join(f"{n} {round9.rmse(p[mask], truth[mask]):.1f}" for n, p in results.items()))
    for a, b in (("none", "six"), ("three", "six")):
        for label, months in slices.items():
            mask = np.isin(month, months)
            diffs = []
            for i in draws:
                i = i[mask[i]]
                diffs.append(round9.rmse(results[b][i], truth[i]) - round9.rmse(results[a][i], truth[i]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            d = round9.rmse(results[b][mask], truth[mask]) - round9.rmse(results[a][mask], truth[mask])
            print(f"{b} vs {a}, {label}: {d:+.2f}s (95% CI over days {lo:+.2f} to {hi:+.2f})")


def rules() -> None:
    training = data.load_training()
    around = features.surroundings(training)
    for fold in [(6, 8), (3, 9), (4, 10), (5, 11)]:
        model = pickle.load(open(STAGES / f"fold_{fold[0]}_{fold[1]}_model.pkl", "rb"))
        _, held_out = data.train_validation_split(data.departures(training), fold)
        frame = features.build(held_out, model.unimpeded.lazy(), around).collect()
        truth = frame[schema.TARGET].to_numpy().astype(float)

        def predictions(floor: bool = True) -> np.ndarray:
            rows = correct.inputs(model, frame)
            pred = rows["pred"].to_numpy()
            return train.orphan_congestion_floor(rows, pred) if floor else pred

        base = predictions()
        variants = {"orphan floor": predictions(floor=False)}
        for name, (constant, off) in round10.RULES.items():
            kept = getattr(train, constant)
            setattr(train, constant, off)
            variants[name] = predictions()
            setattr(train, constant, kept)
        day = frame[schema.MVT_TIME].dt.date().to_numpy()
        days = np.array(sorted(set(day)), dtype=object)
        index = {d: np.flatnonzero(day == d) for d in days}
        rng = np.random.default_rng(14)
        draws = [np.concatenate([index[d] for d in rng.choice(days, len(days))]) for _ in range(300)]
        print(f"fold {fold}, {len(truth)} departures: all rules {round9.rmse(base, truth):.2f}s", flush=True)
        for name, pred in variants.items():
            d = round9.rmse(pred, truth) - round9.rmse(base, truth)
            diffs = [round9.rmse(pred[i], truth[i]) - round9.rmse(base[i], truth[i]) for i in draws]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"  without {name:17s}: {d:+.2f}s (95% CI {lo:+.2f} to {hi:+.2f}), {int(np.sum(pred != base))} rows", flush=True)


if __name__ == "__main__":
    {"--score": score, "rules": rules}[sys.argv[1]]()
