"""Which of the claimed improvements survive a paired bootstrap?

An independent confidence interval on the orphan group is 1,276s wide, because
one row in 5,373 carries 22% of its squared error. But comparing two models is
a paired question: both see the same rows, so the right test resamples rows
once and scores both models on the same resample. That cancels most of the
"did the monster row land in this sample" variance and is far tighter.

Tested here:
  A. one model for everything      vs  two models        (the 471 -> 416 claim)
  B. two models                    vs  two + extra flags (the 416 -> 402 claim)
"""

import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import data, features, schema, train  # noqa: E402

rng = np.random.default_rng(7)

training = data.load_training()
tr, va = data.train_validation_split(data.departures(training))
unimpeded = features.unimpeded_taxi_reference(tr)
tr_f = features.build(tr, unimpeded).collect()
va_f = features.build(va, unimpeded).collect()
truth = va_f.select(schema.TARGET).to_numpy().ravel().astype(float)
matched = va_f.select(train.has_flight_record()).to_numpy().ravel()
cats = list(features.CATEGORICAL_COLUMNS)

FLAGS = [
    pl.col(schema.AIRCRAFT_TYPE).is_null().cast(pl.Int8).alias("f_no_actype"),
    pl.col(schema.STAND).is_null().cast(pl.Int8).alias("f_no_stand"),
    pl.col(schema.ADES).is_null().cast(pl.Int8).alias("f_no_ades"),
]
FLAG_NAMES = ["f_no_actype", "f_no_stand", "f_no_ades"]


def build(extra=()):
    cols = list(features.FEATURE_COLUMNS) + list(extra)
    tx = tr_f.with_columns(FLAGS) if extra else tr_f
    vx = va_f.with_columns(FLAGS) if extra else va_f

    def mat(f):
        x = f.select(cols + cats).to_pandas()
        for c in cats:
            x[c] = x[c].astype("category")
        return x

    return tx, vx, mat


def single_model():
    tx, vx, mat = build()
    m = lgb.train(train.PARAMS,
                  lgb.Dataset(mat(tx), label=tx.select(schema.TARGET).to_numpy().ravel(),
                              categorical_feature=cats), num_boost_round=400)
    return m.predict(mat(vx))


def split_model(extra=()):
    tx, vx, mat = build(extra)
    j = tx.filter(train.has_flight_record())
    o = tx.filter(~train.has_flight_record())
    mj = lgb.train(train.PARAMS,
                   lgb.Dataset(mat(j), label=j.select(schema.TARGET).to_numpy().ravel(),
                               categorical_feature=cats), num_boost_round=400)
    mo = lgb.train(train.SPARSE_PARAMS,
                   lgb.Dataset(mat(o), label=o.select(schema.TARGET).to_numpy().ravel(),
                               categorical_feature=cats), num_boost_round=400)
    p = np.empty(len(vx))
    p[matched] = mj.predict(mat(vx.filter(train.has_flight_record())))
    p[~matched] = mo.predict(mat(vx.filter(~train.has_flight_record())))
    return p


def rmse(p, t, idx=None):
    if idx is None:
        return float(np.sqrt(np.mean((p - t) ** 2)))
    return float(np.sqrt(np.mean((p[idx] - t[idx]) ** 2)))


def paired(name, p_a, p_b, reps=600):
    """Bootstrap the DIFFERENCE b - a on shared resamples. Negative = b better."""
    a, b = rmse(p_a, truth), rmse(p_b, truth)
    diffs = []
    n = len(truth)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        diffs.append(rmse(p_b, truth, idx) - rmse(p_a, truth, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "REAL" if hi < 0 else ("no effect" if lo < 0 < hi else "WORSE")
    print(f"  {name}")
    print(f"    {a:.1f}s -> {b:.1f}s   difference {b - a:+.1f}s")
    print(f"    95% CI on the difference: {lo:+.1f}s to {hi:+.1f}s   [{verdict}]")
    return b - a, lo, hi


print("fitting models...\n")
p_single = single_model()
p_split = split_model()
p_flags = split_model(FLAG_NAMES)

print("PAIRED BOOTSTRAP, 600 resamples of the full validation set\n")
paired("A. one model  ->  two models", p_single, p_split)
print()
paired("B. two models ->  two + missingness flags", p_split, p_flags)

print("\nFOR CONTRAST, the unpaired view of the same comparison B:")
for label, p in (("two models", p_split), ("two + flags", p_flags)):
    boot = []
    n = len(truth)
    for _ in range(300):
        idx = rng.integers(0, n, n)
        boot.append(rmse(p, truth, idx))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:14s} {rmse(p, truth):7.1f}s   95% CI {lo:.0f}s to {hi:.0f}s")
print("  The intervals overlap almost completely, which is why the paired test")
print("  is the one that can answer the question.")
