"""The live excess signal, the refocused classifier, and the reference fallback.

Runs the production pipeline (`train.validate`) four times, switching one
change on at a time, and compares each step on a paired bootstrap:

  A. as v3 (none of the three)
  B. A + live excess: other departures' take-off minus NM off-block, over
     their reference, within 30/60 min (`features.add_live_excess`)
  C. B + the matched at-schedule classifier weighted by what a mistake
     costs, with an `EOBT == SCHED` flag and a stand-area categorical
  D. C + stand-group and runway fallback for the unimpeded reference

The orphan model is identical in A to C; D can change it only through the
reference on pairs unseen in training.

Measured 29 September 2026:

  A as v3          overall 346.6s  matched 240.7s  LIRF matched 445.1s
  B + live excess  overall 344.3s  matched 237.3s  LIRF matched 426.2s
  C + classifier   overall 344.4s  matched 237.5s  LIRF matched 430.6s
  D + fallback     overall 344.8s  matched 237.8s  LIRF matched 431.9s

  A -> B  -2.3s, 95% CI -6.5s to -0.3s   (kept)
  B -> C  +0.1s, 95% CI -0.9s to +0.9s   (removed from the code afterwards)
  C -> D  +0.4s, 95% CI -0.7s to +1.7s   (kept: it targets 2026-only stands)

The classifier variant no longer exists, so step C does not rerun as is.
"""

import sys

import numpy as np

sys.path.insert(0, "pipeline")
from taxiout import data, features, schema, train  # noqa: E402

rng = np.random.default_rng(7)
training = data.load_training()
ALL_NUM = list(features.FEATURE_COLUMNS)
ALL_CAT = list(features.CATEGORICAL_COLUMNS)
HINTS = set(features.CLASSIFIER_HINT_COLUMNS)
EXCESS = set(features.LIVE_EXCESS_COLUMNS)


def configure(excess: bool, classifier: bool, fallback: bool) -> None:
    drop = {"unimpeded_level"} if not fallback else set()
    if not excess:
        drop |= EXCESS
    if not classifier:
        drop |= HINTS
    features.FEATURE_COLUMNS[:] = [c for c in ALL_NUM if c not in drop]
    features.CATEGORICAL_COLUMNS[:] = [c for c in ALL_CAT if classifier or c != features.STAND_AREA]
    train.CLASSIFIER_GAP_WEIGHTING = classifier
    features.UNIMPEDED_FALLBACK = fallback


def run(**switches):
    configure(**switches)
    model, frame, pred = train.validate(training)
    order = np.argsort(frame[schema.MVT_ID].to_numpy())
    truth = frame[schema.TARGET].to_numpy().astype(float)[order]
    matched = frame.select(train.has_flight_record()).to_numpy().ravel()[order]
    airport = frame[schema.ADEP].to_numpy()[order]
    month = frame[schema.MVT_TIME].dt.month().to_numpy()[order]
    return pred[order], truth, matched, airport, month


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


runs = {
    "A as v3": run(excess=False, classifier=False, fallback=False),
    "B + live excess": run(excess=True, classifier=False, fallback=False),
    "C + classifier": run(excess=True, classifier=True, fallback=False),
    "D + fallback": run(excess=True, classifier=True, fallback=True),
}
configure(excess=True, classifier=True, fallback=True)

truth = next(iter(runs.values()))[1]
for name, (p, t, m, ap, mo) in runs.items():
    assert np.array_equal(t, truth)
    print(
        f"{name:18s} overall {rmse(p, t):6.1f}s  matched {rmse(p[m], t[m]):6.1f}s  "
        f"orphan {rmse(p[~m], t[~m]):7.1f}s  Jan matched {rmse(p[m & (mo == 1)], t[m & (mo == 1)]):6.1f}s  "
        f"LIRF matched {rmse(p[m & (ap == 'LIRF')], t[m & (ap == 'LIRF')]):6.1f}s",
        flush=True,
    )


def paired(a, b, reps=300):
    pa, pb = runs[a][0], runs[b][0]
    n = len(truth)
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        diffs.append(rmse(pb[idx], truth[idx]) - rmse(pa[idx], truth[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {a} -> {b}: {rmse(pb, truth) - rmse(pa, truth):+.1f}s, 95% CI {lo:+.1f}s to {hi:+.1f}s")


names = list(runs)
for a, b in zip(names, names[1:]):
    paired(a, b)
