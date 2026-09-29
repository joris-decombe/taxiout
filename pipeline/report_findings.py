"""Regenerates report/findings.json, the data behind *Taxi-Out, Measured*.

Run from the repo root:

    PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
      .venv/Scripts/python.exe pipeline/report_findings.py

Every figure is measured on the January + July 2025 validation split with
the current `train.py`, except the rows of the strategy ladder in
HISTORICAL: those score model configurations the code no longer contains,
so they are recorded as measured at the time.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import data, features, schema, train  # noqa: E402

OUT = Path("report/findings.json")
PAGE = Path("report/taxi-out-measured.html")
CLIP_CAPS = [1800, 2700, 3600, 5400, 7200, 10800, 14400, 21600, 43200, 86400]
HIST_BIN_SEC, HIST_BINS = 120, 40
S2T_BINS_H = [1, 2, 3, 4, 6, 8, 12, 18, 24]

# Measured on configurations since replaced; see the commit history.
HISTORICAL = {
    "mean": {"name": "Guess the group average for no-record flights", "rmse": 573.8, "verdict": "rejected",
             "note": "much worse: the model does find a pattern in these flights"},
    "flags": {"name": "Tell a single model which fields are blank", "rmse": 474.8, "verdict": "rejected",
              "note": "slightly worse than the 471.7s single model it was tested against"},
    "split": {"name": "A separate model for no-record flights", "rmse": 442.5, "verdict": "kept",
              "note": "splitting the problem in two"},
    "leaves": {"name": "Make that separate model simpler", "rmse": 415.8, "verdict": "kept",
               "note": "a small group needs a small model, or it memorises"},
    "airline": {"name": "Give the model the airline", "rmse": 419.3, "verdict": "rejected",
                "note": "worse than the 409.4s it was added to: the model memorised airlines"},
}


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def r1(x):
    return round(float(x), 1)


training = data.load_training()
departures = data.departures(training)
tr, va = data.train_validation_split(departures)

around = features.surroundings(training)
model = train.fit(tr, around)
va_f = features.build(va, model.unimpeded.lazy(), around).collect()
truth = va_f[schema.TARGET].to_numpy().astype(float)
pred = model.predict(va_f)
matched = va_f.select(train.has_flight_record()).to_numpy().ravel()

# Reference points, refitted here so the ladder is reproducible end to end.
tr_f = features.build(tr, model.unimpeded.lazy(), around).collect()
all_cols = list(features.FEATURE_COLUMNS)
orph_cols = train.orphan_columns()
single = lgb.train(train.PARAMS, train._to_dataset(tr_f, all_cols), num_boost_round=400)
p_single = single.predict(train._to_frame(va_f, all_cols))
orph_tr = tr_f.filter(~train.has_flight_record())
plain_orphan = lgb.train(train.SPARSE_PARAMS, train._to_dataset(orph_tr, orph_cols), num_boost_round=400)
p_plain = pred.copy()
p_plain[~matched] = plain_orphan.predict(train._to_frame(va_f.filter(~train.has_flight_record()), orph_cols))

sq = (pred - truth) ** 2
sq_plain = (p_plain - truth) ** 2
out = {
    "rmse_final": r1(rmse(pred, truth)),
    "rmse_single": r1(rmse(p_single, truth)),
    "rmse_before_mixture": r1(rmse(p_plain, truth)),
    "mae_final": r1(np.mean(np.abs(pred - truth))),
    "mae_single": r1(np.mean(np.abs(p_single - truth))),
    "rmse_predict_mean": r1(truth.std()),
    "rmse_if_orphans_perfect": r1(np.sqrt(sq[matched].sum() / len(truth))),
    "rmse_if_joined_perfect": r1(np.sqrt(sq[~matched].sum() / len(truth))),
    "target": {"mean": r1(truth.mean()), "sd": r1(truth.std()),
               "p50": r1(np.median(truth)), "p90": r1(np.quantile(truth, 0.9))},
}

# Error concentration, worst prediction first, denser at the head.
order = np.sort(sq)[::-1]
cum = np.concatenate([[0.0], np.cumsum(order) / order.sum()])
fracs = np.unique(np.concatenate([np.linspace(0, 0.02, 60), np.linspace(0.02, 0.2, 40), np.linspace(0.2, 1, 30)]))
out["concentration"] = [[round(float(f), 5), round(float(cum[int(round(f * len(order)))]), 4)] for f in fracs]

groups = {}
for name, mask in (("joined", matched), ("orphan", ~matched)):
    t, p = truth[mask], pred[mask]
    groups[name] = {
        "n": int(mask.sum()), "share": round(float(mask.mean()), 4),
        "rmse": r1(rmse(p, t)), "err_share": round(float(sq[mask].sum() / sq.sum()), 4),
        "rmse_before": r1(rmse(p_plain[mask], t)),
        "err_share_before": round(float(sq_plain[mask].sum() / sq_plain.sum()), 4),
        "mean": r1(t.mean()), "sd": r1(t.std()), "p50": r1(np.median(t)),
        "p99": r1(np.quantile(t, 0.99)), "max": r1(t.max()), "mae": r1(np.mean(np.abs(p - t))),
    }
out["groups"] = groups

out["hist"] = {"bin_sec": HIST_BIN_SEC, "n_bins": HIST_BINS, "counts": {
    name: np.bincount(np.clip(truth[mask] // HIST_BIN_SEC, 0, HIST_BINS).astype(int), minlength=HIST_BINS + 1).tolist()
    for name, mask in (("joined", matched), ("orphan", ~matched))}}

out["clip"] = [[c, r1(rmse(np.clip(pred, 0, c), truth))] for c in CLIP_CAPS]

out["monthly"] = [
    [int(m), r1(sd)] for m, sd in departures.group_by(pl.col(schema.MVT_TIME).dt.month().alias("m"))
    .agg(pl.col(schema.TARGET).std()).sort("m").collect().iter_rows()]

ident = departures.select(
    ((pl.col(schema.MVT_TIME) - pl.col(schema.BLOCK_TIME)).dt.total_seconds() - pl.col(schema.TARGET)).abs().max().alias("e"),
    pl.len().alias("n")).collect()
out["identity"] = {"max_abs_error": float(ident["e"][0]), "rows": int(ident["n"][0])}

out["strategies"] = [
    HISTORICAL["mean"],
    HISTORICAL["flags"],
    {"name": "One model for every flight", "rmse": out["rmse_single"], "verdict": "baseline",
     "note": "the starting point"},
    HISTORICAL["split"],
    HISTORICAL["airline"],
    HISTORICAL["leaves"],
    {"name": "Tell the separate model which fields are blank", "rmse": out["rmse_before_mixture"], "verdict": "kept",
     "note": "helps tell no-record flights apart from each other"},
    {"name": "Hedge between the schedule gap and a normal taxi", "rmse": out["rmse_final"], "verdict": "shipped",
     "note": "the blend described in section 08"},
]

# The recording artifact, over all of 2025's unmatched departures.
orph = departures.filter(~train.has_flight_record()).with_columns(
    at=train.off_block_at_schedule().fill_null(False),
    gap_h=(pl.col(schema.MVT_TIME) - pl.col(schema.SCHED_TIME)).dt.total_seconds() / 3600,
    y=pl.col(schema.TARGET).cast(pl.Float64),
).collect()
dev = (orph["y"] - orph["y"].mean()) ** 2
by_ap = orph.with_columns(d=dev).group_by(schema.ADEP).agg(
    pl.len().alias("n"), pl.col("at").mean().alias("at_share"),
    (pl.col("d").sum() / dev.sum()).alias("dev_share"),
).sort("dev_share", descending=True)
lirf = orph.filter(pl.col(schema.ADEP) == "LIRF")
edges = [-np.inf] + S2T_BINS_H + [np.inf]
bins = []
for lo, hi in zip(edges[:-1], edges[1:]):
    b = lirf.filter((pl.col("gap_h") > lo) & (pl.col("gap_h") <= hi))
    if b.height:
        bins.append({"lo_h": None if lo == -np.inf else lo, "hi_h": None if hi == np.inf else hi,
                     "n": b.height, "at_share": round(float(b["at"].mean()), 3),
                     "day_shift": int((b["y"] > 80_000).sum()),
                     "y_over_gap": round(float((b["y"] / (b["gap_h"] * 3600)).mean()), 2)})
long6 = departures.filter(pl.col(schema.TARGET) > 6 * 3600).with_columns(
    at=train.off_block_at_schedule().fill_null(False), orphan=~train.has_flight_record(),
    day=(pl.col(schema.TARGET) >= 86_400) & (pl.col(schema.TARGET) < 86_400 + 7_200),
).collect()
negative = departures.filter(pl.col(schema.TARGET) < 0).select(
    pl.len().alias("n"), (pl.col(schema.ADEP) == "LSZH").sum().alias("lszh"),
    pl.col(schema.TARGET).min().alias("min")).collect()
out["negative"] = {k: int(negative[k][0]) for k in ("n", "lszh", "min")}
out["artifact"] = {
    "orphans": orph.height,
    "lirf_orphans": lirf.height,
    "lirf_at_share": round(float(lirf["at"].mean()), 3),
    "other_at_share": round(float(orph.filter(pl.col(schema.ADEP) != "LIRF")["at"].mean()), 3),
    "lirf_dev_share": round(float(by_ap.filter(pl.col(schema.ADEP) == "LIRF")["dev_share"][0]), 3),
    "by_airport": [{"adep": a, "n": n, "at_share": round(s, 3), "dev_share": round(d, 3)} for a, n, s, d in by_ap.iter_rows()],
    "lirf_by_gap": bins,
    "over_6h": {"n": long6.height, "orphan": int(long6["orphan"].sum()),
                "at_schedule": int(long6["at"].sum()), "day_shift": int(long6["day"].sum()),
                "unexplained": int((~long6["at"] & ~long6["day"]).sum())},
}

OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

# The page carries its data inline; refresh it so the page publishes as is.
page = PAGE.read_text(encoding="utf-8")
opening = '<script type="application/json" id="d">'
start = page.index(opening) + len(opening)
end = page.index("</script>", start)
PAGE.write_text(page[:start] + OUT.read_text(encoding="utf-8") + page[end:], encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("concentration", "hist", "clip", "monthly", "artifact")}, indent=1))
