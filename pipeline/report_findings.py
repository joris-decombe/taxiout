"""Regenerates report/findings.json, the data behind *Taxi-Out, Measured*.

Run from the repo root:

    PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
      .venv/Scripts/python.exe pipeline/report_findings.py

Every model figure is measured on the January + July 2025 validation split
with the current `train.py`, except LADDER and REJECTED: those score model
configurations the code no longer contains, so they are recorded as
measured at the time. Airport, runway and weather figures come from the
2025 training files and the 2026 ranking file; nothing in the output is a
per-flight row.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import data, features, schema, train, weather  # noqa: E402

OUT = Path("report/findings.json")
PAGE = Path("report/taxi-out-measured.html")
CLIP_CAPS = [1800, 2700, 3600, 5400, 7200, 10800, 14400, 21600, 43200, 86400]
HIST_BIN_SEC, HIST_BINS = 120, 40
# Aligned with train.py's LIRF bands: late-orphan shares, then the day-shift band.
S2T_BINS_H = [1, 2, 3, 4, 6, 8, 14, 20, 26]

LADDER = [
    {"name": "One model for every flight", "rmse": 475.5},
    {"name": "A separate model for flights with no flight-plan record", "rmse": 442.5},
    {"name": "Make that model small, so it cannot memorise", "rmse": 415.8},
    {"name": "Tell it which fields are blank", "rmse": 409.4},
    {"name": "Hedge between the schedule gap and a normal taxi", "rmse": 383.6},
    {"name": "Read the airport around each flight: arrivals, stands, queues", "rmse": 378.0},
    {"name": "Use the flight plan: airline, destination, planned off-block", "rmse": 370.3},
    {"name": "Train longer (1,200 rounds)", "rmse": 369.6},
    {"name": "Hedge for flights with a record too", "rmse": 352.4},
    {"name": "Weather, and keep the small model's inputs small", "rmse": 346.1},
    {"name": "Read how far the airport's other departures run over their reference, live", "rmse": 344.3},
    {"name": "Borrow a reference from neighbouring stands for stands new in 2026", "rmse": 343.6},
    {"name": "Learn only the deviation from the Network Manager's own taxi time", "rmse": 342.5},
    {"name": "Keep the off-block inside the flight plan's two-hour window", "rmse": 339.1},
    {"name": "Rome, 14 to 26 hours late: at schedule, or a day plus a normal taxi", "rmse": 329.8},
    {"name": "Rome, past six hours late: trust the at-schedule share", "rmse": 324.6},
    {"name": "Blend in a CatBoost model of the normal taxi", "rmse": 321.8},
    {"name": "A CatBoost model for the no-record flights too", "rmse": 319.6},
    {"name": "Average three seeds of the LightGBM model", "rmse": 319.4},
    {"name": "A LightGBM per airport beside the global one, and a deeper CatBoost", "rmse": 318.5},
    {"name": "A CatBoost classifier beside LightGBM's, for flights with a record", "rmse": 317.9},
    {"name": "No-record flights outside Rome: never more than an hour", "rmse": 317.2},
    {"name": "Pull those towards their airport's usual taxi for that lateness", "rmse": 316.7},
    {"name": "Correct with what ADS-B receivers saw on the ground (cross-validated by day)", "rmse": 308.8},
    {"name": "No-record flights taxi at least as long as the departures around them", "rmse": 308.6},
    {"name": "Teach the ADS-B corrector from six held-out folds, the whole of 2025", "rmse": 307.8},
]
REJECTED = [
    {"name": "Weight the schedule-copy classifier by what a mistake costs", "rmse": 344.4, "against": 344.3},
    {"name": "Guess the group average for no-record flights", "rmse": 573.8, "against": 442.5},
    {"name": "Tell one model which fields are blank", "rmse": 474.8, "against": 471.7},
    {"name": "Give the no-record model the airline", "rmse": 419.3, "against": 409.4},
    {"name": "Linear leaves in the trees", "rmse": 951.2, "against": 343.6},
    {"name": "Drop day-late flights from the no-record model's training", "rmse": 391.8, "against": 343.6},
    {"name": "Count the queue as queueing theory does (adjusted traffic, busy periods)", "rmse": 324.5, "against": 324.6},
    {"name": "A CatBoost classifier for the no-record flights too", "rmse": 320.9, "against": 319.4},
    {"name": "Stronger regularisation, for a shifted 2026", "rmse": 319.8, "against": 319.4},
    {"name": "Each flight number's record in the other months (all of it Rome, and lost to 2026's new callsigns)", "rmse": 303.4, "against": 307.9},
]
UPLOADS = [
    {"v": "v1", "validation": 383.6, "test": 370.9},
    {"v": "v2", "validation": 346.1, "test": 363.1},
    {"v": "v3", "validation": 346.6, "test": 361.5},
    {"v": "v4", "validation": 344.3, "test": 360.6},
    {"v": "v5", "validation": 343.6, "test": 359.0},
    {"v": "v6", "validation": 321.8, "test": 311.3},
    {"v": "v7", "validation": 317.2, "test": 287.2},
    {"v": "v8", "validation": 316.7, "test": 287.0},
    {"v": "v9", "validation": 308.8, "test": 275.0},
    {"v": "v10", "validation": 308.6, "test": 274.1},
    {"v": "v11", "validation": 308.0, "test": 271.9},
    {"v": "v12", "validation": 307.8, "test": 272.1},
]
LEADERBOARD = {"date": "9 October 2026", "teams": 239, "leader": 213.0, "tenth": 222.9,
               "quartile": 268.7, "median": 294.6, "ours": 271.9, "rank": 68}
# Iowa Environmental Mesonet METAR archive.
METAR = {"station": "EHAM", "time": "2026-01-05 08:25 UTC",
         "raw": "EHAM 050825Z 20009KT 0700 R18C/1200N R27/1200U R18R/0700N R06/1400U SHSN VV005 00/M00 Q1008 TEMPO 2000"}


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

tr_f = features.build(tr, model.unimpeded.lazy(), around).collect()
orph_cols, orph_cats = train.orphan_columns(), train.ORPHAN_CATEGORICALS
orph_tr = tr_f.filter(~train.has_flight_record())
plain_orphan = lgb.train(
    train.SPARSE_PARAMS, train._to_dataset(orph_tr, orph_cols, orph_cats), num_boost_round=400
)
p_plain = pred.copy()
p_plain[~matched] = plain_orphan.predict(
    train._to_frame(va_f.filter(~train.has_flight_record()), orph_cols, orph_cats)
)

sq = (pred - truth) ** 2
sq_plain = (p_plain - truth) ** 2
out = {
    "rmse_final": r1(rmse(pred, truth)),
    "rmse_before_mixture": r1(rmse(p_plain, truth)),
    "mae_final": r1(np.mean(np.abs(pred - truth))),
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
        day = (b["y"] > 80_000) & ~b["at"]
        bins.append({"lo_h": None if lo == -np.inf else lo, "hi_h": None if hi == np.inf else hi,
                     "n": b.height, "at_share": round(float(b["at"].mean()), 3),
                     "at_n": int(b["at"].sum()), "day_shift": int(day.sum()),
                     "normal_n": int((~b["at"] & ~day).sum()),
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

# The flight plan's window: BLOCK_TIME against LOBT on every matched departure.
lobt = departures.filter(train.has_flight_record()).select(
    pl.len().alias("n"),
    (pl.col(schema.BLOCK_TIME) - pl.col(schema.LOBT)).dt.total_seconds().abs().max().alias("max_abs"),
    ((pl.col(schema.BLOCK_TIME) - pl.col(schema.LOBT)).dt.total_seconds().abs() <= train.LOBT_WINDOW_SEC).mean().alias("within"),
    ((pl.col(schema.SCHED_TIME) - pl.col(schema.LOBT)).dt.total_seconds().abs() > train.LOBT_WINDOW_SEC).sum().alias("sched_outside"),
).collect()
out["lobt"] = {"n": int(lobt["n"][0]), "max_abs": int(lobt["max_abs"][0]),
               "within": round(float(lobt["within"][0]), 6), "sched_outside": int(lobt["sched_outside"][0])}

# --- The airports, and what changed between 2025 and 2026 -------------------

AIRPORTS = {
    "EDDF": "Frankfurt", "EDDM": "Munich", "EGLL": "London Heathrow", "EHAM": "Amsterdam Schiphol",
    "LEBL": "Barcelona", "LEMD": "Madrid", "LFPG": "Paris Charles de Gaulle", "LIRF": "Rome Fiumicino",
    "LSZH": "Zurich", "LTFM": "Istanbul",
}
full = training.collect()
# The page describes the leaderboard's January and July 2026; the final ranking
# file adds February and June, which its counts must not include.
ranking = data.load_ranking().filter(pl.col(schema.MVT_TIME).dt.month().is_in([1, 7])).collect()
dep_all = full.filter(pl.col(schema.PHASE) == schema.DEPARTURE)
arr_all = full.filter(pl.col(schema.PHASE) == schema.ARRIVAL)
rk_dep = ranking.filter(pl.col(schema.PHASE) == schema.DEPARTURE)
month = pl.col(schema.MVT_TIME).dt.month()
taxi = pl.col(schema.TARGET)


def shares(frame, months):
    f = frame.filter(month.is_in(months))
    return {r: round(n / f.height, 4) for r, n in f.group_by(schema.RUNWAY).len().iter_rows()} if f.height else {}


airport_rows = []
for icao, name in AIRPORTS.items():
    dep = dep_all.filter(pl.col(schema.ADEP) == icao)
    arr = arr_all.filter(pl.col(schema.ADES) == icao)
    rk = rk_dep.filter(pl.col(schema.ADEP) == icao)
    s = dep.select(
        taxi.quantile(0.1).alias("p10"), taxi.median().alias("p50"), taxi.quantile(0.9).alias("p90"),
        train.off_block_at_schedule().fill_null(False).mean().alias("at"),
        pl.col(schema.AOBT).is_null().mean().alias("no_nm"),
    ).row(0, named=True)
    j25, l25, j26, l26 = shares(dep, [1]), shares(dep, [7]), shares(rk, [1]), shares(rk, [7])
    runways = [
        {"rwy": r, "share": round(n / dep.height, 4), "p50": r1(p50),
         "jan25": j25.get(r, 0), "jan26": j26.get(r, 0), "jul25": l25.get(r, 0), "jul26": l26.get(r, 0)}
        for r, n, p50 in dep.group_by(schema.RUNWAY).agg(pl.len(), taxi.median()).sort("len", descending=True).iter_rows()
        if n / dep.height >= 0.005
    ]
    arrivals = [
        {"rwy": r, "share": round(n / arr.height, 4)}
        for r, n in arr.group_by(schema.RUNWAY).len().sort("len", descending=True).iter_rows()
        if n / arr.height >= 0.01
    ]
    airport_rows.append({
        "icao": icao, "name": name, "deps": dep.height, "deps_2026": rk.height,
        "p10": r1(s["p10"]), "p50": r1(s["p50"]), "p90": r1(s["p90"]),
        "at_share": round(s["at"], 4), "no_nm": round(s["no_nm"], 4),
        "runways": runways, "arrivals": arrivals,
    })
out["airports"] = airport_rows

# Weather at take-off: snow or freezing precipitation, January 2025 vs 2026.
wx = weather.load()


def weathered(frame):
    return weather.join(frame.select(schema.MVT_ID, schema.ADEP, schema.MVT_TIME), wx).join(
        frame.select(schema.MVT_ID, schema.TARGET, schema.BLOCK_TIME, schema.SCHED_TIME), on=schema.MVT_ID)


snowy = (pl.col("wx_snow") == 1) | (pl.col("wx_freezing") == 1)
jan25 = weathered(dep_all.filter(month == 1))
jan26 = weathered(rk_dep.filter(month == 1))
snow_rows = []
for icao in AIRPORTS:
    a = jan25.filter(pl.col(schema.ADEP) == icao).select(snowy.mean()).item()
    b = jan26.filter(pl.col(schema.ADEP) == icao).select(snowy.mean()).item()
    snow_rows.append({"icao": icao, "jan25": round(a or 0.0, 4), "jan26": round(b or 0.0, 4)})
out["snow"] = snow_rows

# Where de-icing lands: winter 2025 departures in snow or freezing precipitation
# against mild dry ones. Remote de-icing lengthens taxi-out; on-stand de-icing
# lengthens the wait before off-block instead.
winter = weathered(dep_all.filter(month.is_in([1, 2, 11, 12]))).with_columns(
    ((pl.col(schema.BLOCK_TIME) - pl.col(schema.SCHED_TIME)).dt.total_seconds()).alias("delay"),
    snowy.alias("snowy"),
    ((pl.col("wx_temp_c") > 3) & (pl.col("wx_precip") == 0)).alias("mild"),
)
deice_rows = []
for icao in AIRPORTS:
    w = winter.filter(pl.col(schema.ADEP) == icao)
    sn, mi = w.filter(pl.col("snowy")), w.filter(pl.col("mild"))
    if sn.height < 50:
        deice_rows.append({"icao": icao, "n_snow": sn.height})
        continue
    deice_rows.append({
        "icao": icao, "n_snow": sn.height,
        "taxi_extra": r1(sn[schema.TARGET].median() - mi[schema.TARGET].median()),
        "delay_extra": r1(sn["delay"].median() - mi["delay"].median()),
    })
out["deicing"] = deice_rows

# LIRF's at-schedule records among flights WITH an NM record, by how late the
# NM saw the aircraft leave.
lirf_m = dep_all.filter((pl.col(schema.ADEP) == "LIRF") & train.has_flight_record()).with_columns(
    at=train.off_block_at_schedule().fill_null(False),
    late=(pl.col(schema.AOBT) - pl.col(schema.SCHED_TIME)).dt.total_seconds() / 60,
)
late_bins = [(-1e9, 0, "early"), (0, 5, "0–5 min"), (5, 15, "5–15 min"), (15, 60, "15–60 min"),
             (60, 120, "1–2 h"), (120, 1e9, "over 2 h")]
out["lirf_matched"] = {
    "n": lirf_m.height,
    "at_share": round(float(lirf_m["at"].mean()), 4),
    "long_at_share": round(float(lirf_m.filter(taxi > 3600)["at"].mean()), 4),
    "long_n": lirf_m.filter(taxi > 3600).height,
    "by_late": [
        {"label": lab, "n": b.height, "at_share": round(float(b["at"].mean()), 4)}
        for lo, hi, lab in late_bins
        for b in [lirf_m.filter((pl.col("late") >= lo) & (pl.col("late") < hi))]
        if b.height
    ],
}

# Measured at the time on the January + July 2025 validation split; each step
# adds to the one above. Test scores from the competition's result files.
out["ladder"] = LADDER
out["rejected"] = REJECTED
out["uploads"] = UPLOADS
out["leaderboard"] = LEADERBOARD
out["metar"] = METAR
out["arrivals_2026"] = ranking.filter(pl.col(schema.PHASE) == schema.ARRIVAL).height

OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

# The page carries its data inline; refresh it so the page publishes as is.
page = PAGE.read_text(encoding="utf-8")
opening = '<script type="application/json" id="d">'
start = page.index(opening) + len(opening)
end = page.index("</script>", start)
PAGE.write_text(page[:start] + OUT.read_text(encoding="utf-8") + page[end:], encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))}, indent=1))

# And the static copy GitHub Pages serves.
import build_site  # noqa: E402

print(build_site.build())
