"""Does modelling the unmatched group's timestamp artifact beat least squares?

About half of LIRF's departures with no Network Manager record have a
BLOCK_TIME equal to SCHED_TIME to the second, so their taxi-out is exactly
MVT_TIME - SCHED_TIME: hours, not minutes. The rest taxi normally. Which kind
a row is depends strongly on the airline, which the model has never seen.

The joined model is fitted once and shared by every variant; only the
orphan model changes. Tested against the current two-model baseline:

  A. airline as a categorical feature in the orphan model
  B. an explicit mixture: a classifier for "off-block recorded at schedule"
     and a regressor for the normal case, combined as
         p * (MVT - SCHED) + (1 - p) * normal
     which is the least-squares prediction if p is calibrated
  C. B with the airline in both of its parts
"""

import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import data, features, schema, train  # noqa: E402

rng = np.random.default_rng(7)

AT_SCHED_TOLERANCE_SEC = 10

airline = (
    pl.coalesce(
        pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1),
        pl.col("FLIGHT_mvt").str.slice(0, 2),
    ).alias("airline")
)
at_sched = (
    (pl.col(schema.BLOCK_TIME) - pl.col(schema.SCHED_TIME)).dt.total_seconds().abs()
    <= AT_SCHED_TOLERANCE_SEC
).alias("at_sched")

training = data.load_training()
tr, va = data.train_validation_split(data.departures(training))
model = train.fit(tr)
unimpeded = model.unimpeded.lazy()
tr_f = features.build(tr, unimpeded).with_columns(airline, at_sched).collect()
va_f = features.build(va, unimpeded).with_columns(airline, at_sched).collect()
truth = va_f.select(schema.TARGET).to_numpy().ravel().astype(float)
matched = va_f.select(train.has_flight_record()).to_numpy().ravel()
s2t = va_f["sched_to_takeoff_sec"].to_numpy().astype(float)

tr_o = tr_f.filter(~train.has_flight_record())
va_o = va_f.filter(~train.has_flight_record())
print(f"orphans: {tr_o.height} train, {va_o.height} validation; "
      f"at_sched share {tr_o['at_sched'].mean():.3f} / {va_o['at_sched'].mean():.3f}")

BASE_CATS = list(features.CATEGORICAL_COLUMNS)


def matrix(frame, cats):
    x = frame.select(features.FEATURE_COLUMNS + cats).to_pandas()
    for c in cats:
        x[c] = x[c].astype("category")
    return x


def fit(frame, label, params, cats, rounds=400):
    return lgb.train(params, lgb.Dataset(matrix(frame, cats), label=label, categorical_feature=cats),
                     num_boost_round=rounds)


def assemble(orphan_pred):
    p = np.empty(len(truth))
    p[matched] = p_joined
    p[~matched] = orphan_pred
    return p


def rmse(p, t, idx=None):
    if idx is None:
        return float(np.sqrt(np.mean((p - t) ** 2)))
    return float(np.sqrt(np.mean((p[idx] - t[idx]) ** 2)))


def paired(name, p_a, p_b, reps=600):
    """Bootstrap the difference b - a on shared resamples. Negative = b better."""
    a, b = rmse(p_a, truth), rmse(p_b, truth)
    n = len(truth)
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        diffs.append(rmse(p_b, truth, idx) - rmse(p_a, truth, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "REAL" if hi < 0 else ("no effect" if lo < 0 < hi else "WORSE")
    o = ~matched
    print(f"  {name}")
    print(f"    overall {a:.1f}s -> {b:.1f}s ({b - a:+.1f}s), 95% CI {lo:+.1f}s to {hi:+.1f}s [{verdict}]")
    print(f"    unmatched group {rmse(p_a[o], truth[o]):.0f}s -> {rmse(p_b[o], truth[o]):.0f}s")


p_joined = model.joined.predict(matrix(va_f.filter(train.has_flight_record()), BASE_CATS))
y_o = tr_o.select(schema.TARGET).to_numpy().ravel()
s2t_o = va_o["sched_to_takeoff_sec"].to_numpy().astype(float)

p_base = assemble(model.orphan.predict(matrix(va_o, BASE_CATS)))

cats_a = BASE_CATS + ["airline"]
p_a = assemble(fit(tr_o, y_o, train.SPARSE_PARAMS, cats_a).predict(matrix(va_o, cats_a)))

CLASSIFIER = {**train.SPARSE_PARAMS, "objective": "binary", "metric": "binary_logloss"}


def mixture(cats):
    label = tr_o["at_sched"].cast(pl.Int8).to_numpy()
    clf = fit(tr_o, label, CLASSIFIER, cats, rounds=200)
    normal = tr_o.filter(~pl.col("at_sched"))
    reg = fit(normal, normal.select(schema.TARGET).to_numpy().ravel(), train.SPARSE_PARAMS, cats)
    p = clf.predict(matrix(va_o, cats))
    r = reg.predict(matrix(va_o, cats))
    at = va_o["at_sched"].to_numpy()
    brier = float(np.mean((p - at) ** 2))
    print(f"    classifier brier {brier:.4f} (base rate {np.mean((at.mean() - at) ** 2):.4f}); "
          f"normal-part RMSE on normal rows {rmse(r[~at], va_o.select(schema.TARGET).to_numpy().ravel()[~at]):.0f}s")
    return assemble(p * s2t_o + (1 - p) * r)


print("\nfitting variants...")
p_b = mixture(BASE_CATS)
p_c = mixture(cats_a)

print("\nPAIRED BOOTSTRAP, 600 resamples of the full validation set\n")
paired("A. baseline -> airline feature", p_base, p_a)
paired("B. baseline -> mixture", p_base, p_b)
paired("C. baseline -> mixture + airline", p_base, p_c)
paired("C vs A", p_a, p_c)

# Where the remaining error sits.
o = ~matched
err = (p_c - truth) ** 2
share = err[o].sum() / err.sum()
va_at = va_o["at_sched"].to_numpy()
print(f"\nafter C, the unmatched group carries {share:.0%} of the squared error")
print(f"  at_sched rows {va_at.sum()}: RMSE {rmse(p_c[o][va_at], truth[o][va_at]):.0f}s; "
      f"others: RMSE {rmse(p_c[o][~va_at], truth[o][~va_at]):.0f}s")
by_airport = pl.DataFrame({"adep": va_o[schema.ADEP], "e": err[o]}).group_by("adep").agg(
    pl.len(), (pl.col("e").sum() / err.sum()).alias("share_of_all")).sort("share_of_all", descending=True)
print(by_airport.head(5))

if "--residuals" in sys.argv:
    r = va_o.with_columns(pred=p_b[~matched], err=(p_b[~matched] - truth[~matched]))
    r = r.with_columns(sq_share=(pl.col("err") ** 2) / float(((p_b - truth) ** 2).sum()))
    pl.Config.set_tbl_rows(25); pl.Config.set_tbl_width_chars(220)
    print(r.sort("sq_share", descending=True).select(
        schema.ADEP, "FLIGHT_mvt", schema.AIRCRAFT_TYPE, schema.SCHED_TIME, schema.BLOCK_TIME, schema.MVT_TIME,
        schema.TARGET, "sched_to_takeoff_sec", "pred", "sq_share").head(20))
