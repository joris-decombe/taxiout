"""Round 10: a June fold for the ADS-B corrector.

The final ranking adds February and June 2026 to January and July. Round 9
gave the corrector a February + December 2025 fold; this round adds June +
August 2025, the summer months beside July, held out from the production
model the same way (`build_final.py` FOLDS). adsb.lol was streamed for
every other day of June and August: the corrector needs examples, not
every day.

Variants, scored by cross-validation over whole days with the production
rules applied after the correction, compared by a bootstrap over days:

- `two`: the corrector fitted on the January + July and February +
  December folds (round 9's `pooled`).
- `three`: fitted on all three folds.

`rules` switches off each production rule in turn on the June + August
fold, whose model never saw those months (`fold_6_8_model.pkl`): the rules
were all chosen while validating on January and July, so this is where a
rule fitted to those months would show.

Results. adsb.lol has no releases for 1 to 9 June 2025, so June has 10
streamed days, August 16. The fold scores 279.6s before any correction.

| Corrector | Jan+Jul | Feb+Dec | Jun+Aug | all |
|---|---|---|---|---|
| none | 316.5s | 234.9s | 279.5s | 280.3s |
| `two` | 308.1s | 224.6s | 276.7s | 273.6s |
| `three` | **308.0s** | **224.5s** | **276.1s** | **273.3s** |

`three` against `two`: -0.29s overall (95% CI over days -0.56 to -0.06),
-0.61s on June and August, no fold worse. `build_final.py` uses three.

Each rule switched off on June and August 2025 (RMSE change, positive
meaning the rule helps):

| Rule | Change | 95% CI over days | Rows |
|---|---|---|---|
| LOBT window | +1.65s | +0.25 to +3.85 | 12,893 |
| LIRF late-orphan rule | +4.34s | +1.44 to +7.38 | 355 |
| orphan cap at an hour | +1.03s | +0.24 to +2.19 | 10 |
| orphan shrinkage | +0.28s | +0.12 to +0.46 | 4,407 |
| orphan congestion floor | +0.01s | -0.00 to +0.02 | 44 |
| LIRF day shift | -1.54s | -7.10 to +1.62 | 3 |

Every rule chosen on January and July holds on months it never saw, except
the day shift, which three rows cannot judge either way. The floor has
nothing to do in summer; its test is the February + December fold (round 9).

Usage: python pipeline/experiments_round10.py --score | rules
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
import experiments_round9 as round9  # noqa: E402
from build_final import FOLD_DAYS  # noqa: E402
from taxiout import adsb, correct, data, features, schema, train  # noqa: E402

STAGES = Path("data/build_final")
FOLD_FILES = {(1, 7): "validation_rows", (2, 12): "fold_2_12_rows", (6, 8): "fold_6_8_rows"}


def table() -> pl.DataFrame:
    rows = pl.concat([pickle.load(open(STAGES / f"{name}.pkl", "rb")) for name in FOLD_FILES.values()],
                     how="vertical_relaxed")
    rows = correct._join(rows, adsb.load_observations(FOLD_DAYS))
    return rows.with_columns(pl.col(schema.MVT_TIME).dt.date().alias("day"))


def cross_fitted(frame: pl.DataFrame, train_months: tuple[int, ...]) -> np.ndarray:
    """Each day's correction from a corrector that never saw that day, fitted on `train_months` only."""
    day = frame["day"].to_numpy()
    days = np.array(sorted(set(day)), dtype=object)
    month = frame[schema.MVT_TIME].dt.month().to_numpy()
    allowed = np.isin(month, train_months)
    seen = frame["adsb_matched"].to_numpy()
    correction = np.zeros(frame.height)
    for k in range(10):
        test = np.isin(day, days[k::10])
        booster = round9.fit(frame.filter(pl.Series(~test & allowed)), correct.FEATURES)
        rows = test & seen
        correction[rows] = booster.predict(round9.matrix(frame.filter(pl.Series(rows)), correct.FEATURES))
    return correction


def score() -> None:
    frame = table()
    truth = frame[schema.TARGET].to_numpy().astype(float)
    pred = frame["pred"].to_numpy()
    print(f"{frame['day'].n_unique()} days, {frame.height} departures, ADS-B matched {frame['adsb_matched'].mean():.1%}")
    results = {
        "none": correct.rules(frame, pred),
        "two": correct.rules(frame, pred + cross_fitted(frame, (1, 7, 2, 12))),
        "three": correct.rules(frame, pred + cross_fitted(frame, (1, 7, 2, 12, 6, 8))),
    }
    month = frame[schema.MVT_TIME].dt.month().to_numpy()
    day = frame["day"].to_numpy()
    days = np.array(sorted(set(day)), dtype=object)
    index = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(10)
    draws = [np.concatenate([index[d] for d in rng.choice(days, len(days))]) for _ in range(300)]
    slices = {"Jan+Jul": (1, 7), "Feb+Dec": (2, 12), "Jun+Aug": (6, 8), "all": tuple(range(1, 13))}
    for label, months in slices.items():
        mask = np.isin(month, months)
        print(f"{label:8s}  " + "  ".join(f"{n} {round9.rmse(p[mask], truth[mask]):.1f}" for n, p in results.items()))
    for a, b in (("none", "three"), ("two", "three")):
        for label, months in slices.items():
            mask = np.isin(month, months)
            diffs = []
            for i in draws:
                i = i[mask[i]]
                diffs.append(round9.rmse(results[b][i], truth[i]) - round9.rmse(results[a][i], truth[i]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            d = round9.rmse(results[b][mask], truth[mask]) - round9.rmse(results[a][mask], truth[mask])
            print(f"{b} vs {a}, {label}: {d:+.2f}s (95% CI over days {lo:+.2f} to {hi:+.2f})")


RULES = {
    "LOBT window": ("LOBT_WINDOW", False),
    "LIRF orphan rule": ("LIRF_ORPHAN_RULE", False),
    "LIRF day shift": ("DAY_SHIFT", False),
    "orphan cap": ("ORPHAN_CAP_SEC", None),
    "orphan shrinkage": ("ORPHAN_SHRINK", 0.0),
}


def rules() -> None:
    """Each production rule switched off on the June + August fold, one at a time."""
    model = pickle.load(open(STAGES / "fold_6_8_model.pkl", "rb"))
    training = data.load_training()
    _, held_out = data.train_validation_split(data.departures(training), (6, 8))
    frame = features.build(held_out, model.unimpeded.lazy(), features.surroundings(training)).collect()
    truth = frame[schema.TARGET].to_numpy().astype(float)

    def predictions(floor: bool = True) -> np.ndarray:
        rows = correct.inputs(model, frame)
        pred = rows["pred"].to_numpy()
        return train.orphan_congestion_floor(rows, pred) if floor else pred

    base = predictions()
    variants = {"orphan floor": predictions(floor=False)}
    for name, (constant, off) in RULES.items():
        kept = getattr(train, constant)
        setattr(train, constant, off)
        variants[name] = predictions()
        setattr(train, constant, kept)

    day = frame[schema.MVT_TIME].dt.date().to_numpy()
    days = np.array(sorted(set(day)), dtype=object)
    index = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(11)
    draws = [np.concatenate([index[d] for d in rng.choice(days, len(days))]) for _ in range(300)]
    month = frame[schema.MVT_TIME].dt.month().to_numpy()
    print(f"June + August 2025, {len(truth)} departures: all rules {round9.rmse(base, truth):.2f}s")
    for name, pred in variants.items():
        changed = int(np.sum(pred != base))
        d = round9.rmse(pred, truth) - round9.rmse(base, truth)
        diffs = [round9.rmse(pred[i], truth[i]) - round9.rmse(base[i], truth[i]) for i in draws]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        by_month = "  ".join(f"m{m} {round9.rmse(pred[month == m], truth[month == m]) - round9.rmse(base[month == m], truth[month == m]):+.2f}"
                             for m in (6, 8))
        print(f"without {name:17s}: {d:+.2f}s (95% CI over days {lo:+.2f} to {hi:+.2f}), {changed} rows change; {by_month}")


if __name__ == "__main__":
    {"--score": score, "rules": rules}[sys.argv[1]]()
