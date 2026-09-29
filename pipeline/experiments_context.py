"""What the surroundings and the weather are each worth, on a paired bootstrap.

Runs the production pipeline (`train.validate`) three times, dropping feature
groups from `features.FEATURE_COLUMNS` for the reference runs:

  A. without the context and weather features
  B. with context, without weather
  C. everything

The matched group is a mixture in all three, and the orphan group too, so
the comparison isolates the features alone.

Measured 29 September 2026 (and the steps that led here, against the
mixture-free model of the time): context features 383.6s -> 378.0s, 95% CI
-6.8s to -4.7s; the flight-table fields a further -7.7s, CI -9.5s to -6.1s;
1,200 rounds instead of 400 -0.7s; the at-schedule mixture on the matched
group 369.6s -> 352.4s, CI -49.6s to -0.5s, LIRF's matched RMSE 642s -> 499s.
"""

import sys

import numpy as np

sys.path.insert(0, "pipeline")
from taxiout import context, data, features, schema, train, weather  # noqa: E402

rng = np.random.default_rng(7)
training = data.load_training()
ALL = list(features.FEATURE_COLUMNS)


def run(drop):
    features.FEATURE_COLUMNS[:] = [c for c in ALL if c not in drop]
    model, frame, pred = train.validate(training)
    ids = frame[schema.MVT_ID].to_numpy()
    order = np.argsort(ids)
    truth = frame[schema.TARGET].to_numpy().astype(float)[order]
    matched = frame.select(train.has_flight_record()).to_numpy().ravel()[order]
    airport = frame[schema.ADEP].to_numpy()[order]
    return pred[order], truth, matched, airport


def rmse(p, t, idx=None):
    if idx is not None:
        p, t = p[idx], t[idx]
    return float(np.sqrt(np.mean((p - t) ** 2)))


runs = {
    "A no context, no weather": run(set(context.FEATURE_COLUMNS) | set(weather.FEATURE_COLUMNS)),
    "B context, no weather": run(set(weather.FEATURE_COLUMNS)),
    "C everything": run(set()),
}
features.FEATURE_COLUMNS[:] = ALL

truth = next(iter(runs.values()))[1]
for name, (p, t, m, ap) in runs.items():
    assert np.array_equal(t, truth)
    per_airport = "  ".join(f"{a} {rmse(p[ap == a], t[ap == a]):.0f}" for a in np.unique(ap))
    print(f"{name:26s} overall {rmse(p, t):6.1f}s  matched {rmse(p[m], t[m]):6.1f}s", flush=True)
    print(f"    {per_airport}")


def paired(a, b, reps=300):
    pa, pb = runs[a][0], runs[b][0]
    n = len(truth)
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        diffs.append(rmse(pb, truth, idx) - rmse(pa, truth, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {a} -> {b}: {rmse(pb, truth) - rmse(pa, truth):+.1f}s, 95% CI {lo:+.1f}s to {hi:+.1f}s")


names = list(runs)
paired(names[0], names[1])
paired(names[1], names[2])
