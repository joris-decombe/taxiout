"""ADS-B observations of each departure on the ground, from adsb.lol.

The adsb.lol global history archive (https://github.com/adsblol, Open
Database License 1.0; feeder data CC0) holds every aircraft trace of a day
as one GitHub release per day (`v<YYYY.MM.DD>-planes-readsb-prod-0`, a tar
split into 2 GB parts). `fetch_day` streams a day without keeping it,
keeps the points within 12 km and below 8,000 ft of the ten airports, and
writes them to `data/external/adsblol/<date>_airports.parquet` (12 to 30 MB
a day). `observe_day` matches that day's departures to the traces and
writes one row per departure to `<date>_departures.parquet`.

What ADS-B shows, measured on 15 January 2025 (578 departures with the
whole sequence visible): an aircraft parked for ten minutes or more and
then moving, which dates the push-back to a median 35 s from BLOCK_TIME
(67% within a minute; RMSE 360 s, 138 s without the 8% of misses beyond
ten minutes), against 180 s for the Network Manager's AOBT on the same
flights. That sequence is visible for 12% to 20% of departures,
mostly at EDDM, EHAM, EGLL, LSZH and LEBL; LFPG, LIRF, LEMD and LTFM have
little or no receiver coverage on the stands. About 40% more are first
seen already taxiing, which bounds taxi-out from below.

The derived tables stay under `data/`, which is not committed: anything
derived from adsb.lol that is published must carry the ODbL, and these are
keyed to challenge rows.

Usage:
    python -m taxiout.adsb fetch 2026-01-15 [2026-01-16 ...]
    python -m taxiout.adsb observe 2026-01-15 [...]
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

from . import data, schema

OUT_DIR = data.DATA_DIR / "external" / "adsblol"

AIRPORTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4775, -0.4614),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
KEEP_KM = 12.0
KEEP_ALT_FT = 8000

# Movement detection.
CLUSTER_RADIUS_M = 40.0      # positions within this of a cluster's first one are "the same place"
STAND_DWELL_S = 600          # a stand is a place the aircraft sat for at least this long
APPEAR_GAP_S = 600           # a gap this long before a point means it "appeared" there
LOOKBACK_S = 4 * 3600
MIN_STAND_TO_TAKEOFF_S = 180  # a stand left less than this before take-off is a holding point
SHARED_RADIUS_M = 30.0       # dwell spots shared by three or more stand labels are pads, not stands
SHARED_MIN_LABELS = 3

OBSERVATION_COLUMNS = [
    "adsb_matched", "adsb_pushback_seen", "adsb_move_to_takeoff_sec", "adsb_first_ground_to_takeoff_sec",
    "adsb_appear_to_takeoff_sec", "adsb_stand_dwell_sec", "adsb_move_resolution_sec", "adsb_takeoff_offset_sec",
]


# --- Streaming and filtering ------------------------------------------------

def _prefixes(lo: float, hi: float) -> list[str]:
    out, x = [], math.floor(lo * 10) / 10
    while x <= hi + 1e-9:
        out.append(("-" if x < 0 else "") + re.escape(f"{abs(x):.1f}"))
        x = round(x + 0.1, 1)
    return out


def _prefilter() -> re.Pattern:
    """A byte regex that finds any trace point near any airport without parsing JSON."""
    branches = []
    for lat, lon in AIRPORTS.values():
        dlat = KEEP_KM / 111.0
        dlon = KEEP_KM / (111.0 * math.cos(math.radians(lat)))
        lats, lons = _prefixes(lat - dlat, lat + dlat), _prefixes(lon - dlon, lon + dlon)
        branches.append(f"(?:{'|'.join(lats)})\\d*,(?:{'|'.join(lons)})\\d")
    return re.compile(("\n\\[[0-9.]+,(?:" + "|".join(branches) + ")").encode())


PREFILTER = _prefilter()
COS_LAT = {k: math.cos(math.radians(v[0])) for k, v in AIRPORTS.items()}


def _nearest_airport(lat: float, lon: float) -> tuple[str, float]:
    best, best_d = None, 1e18
    for code, (alat, alon) in AIRPORTS.items():
        dy, dx = (lat - alat) * 111.195, (lon - alon) * 111.195 * COS_LAT[code]
        d = dx * dx + dy * dy
        if d < best_d:
            best, best_d = code, d
    return best, math.sqrt(best_d)


def _trace_rows(blob: bytes) -> list[tuple]:
    try:
        raw = zlib.decompress(blob, 47)
    except zlib.error:
        raw = blob
    if not PREFILTER.search(raw):
        return []
    try:
        doc = json.loads(raw)
    except ValueError:
        return []
    icao, t0, trace = doc.get("icao"), doc.get("timestamp", 0.0), doc.get("trace") or []
    # Callsigns appear intermittently: carry the last one forward, and the next
    # one backward within a leg (bit 2 of the flags marks a new leg).
    forward, cs = [], None
    for p in trace:
        d = p[8] if len(p) > 8 else None
        if isinstance(d, dict) and d.get("flight"):
            cs = d["flight"].strip() or cs
        forward.append(cs)
    backward, nxt = [None] * len(trace), None
    for i in range(len(trace) - 1, -1, -1):
        p = trace[i]
        d = p[8] if len(p) > 8 else None
        if isinstance(d, dict) and d.get("flight"):
            nxt = d["flight"].strip() or nxt
        backward[i] = nxt
        if len(p) > 6 and isinstance(p[6], int) and (p[6] & 2):
            nxt = None
    rows = []
    for i, p in enumerate(trace):
        lat, lon, alt = p[1], p[2], p[3]
        if lat is None or lon is None:
            continue
        ground = alt == "ground"
        if not ground and (alt is None or (isinstance(alt, (int, float)) and alt > KEEP_ALT_FT)):
            continue
        airport, dist_km = _nearest_airport(lat, lon)
        if dist_km > KEEP_KM:
            continue
        rows.append((icao, t0 + p[0], lat, lon, ground, None if ground or alt is None else int(alt),
                     p[4], airport, forward[i], backward[i]))
    return rows


def _batch_rows(batch: list[bytes]) -> list[tuple]:
    out = []
    for blob in batch:
        out.extend(_trace_rows(blob))
    return out


class _PartStream:
    """The release's tar parts read back to back as one stream, resuming on errors."""

    def __init__(self, urls: list[str]):
        self.urls, self.idx, self.off, self.resp, self.total = list(urls), 0, 0, None, 0

    def read(self, n: int = -1) -> bytes:
        out, want = bytearray(), n if n >= 0 else 1 << 62
        while len(out) < want and self.idx < len(self.urls):
            if self.resp is None:
                for attempt in range(8):
                    try:
                        req = urllib.request.Request(self.urls[self.idx])
                        if self.off:
                            req.add_header("Range", f"bytes={self.off}-")
                        self.resp = urllib.request.urlopen(req, timeout=60)
                        break
                    except OSError:
                        time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError(f"cannot open {self.urls[self.idx]}")
            try:
                chunk = self.resp.read(min(want - len(out), 1 << 20))
            except OSError:
                self.resp = None
                continue
            if not chunk:
                self.resp, self.idx, self.off = None, self.idx + 1, 0
                continue
            self.off += len(chunk)
            self.total += len(chunk)
            out += chunk
        return bytes(out)


def _release_parts(day: dt.date) -> list[str]:
    tag = f"v{day:%Y.%m.%d}-planes-readsb-prod-0"
    listing = subprocess.check_output(
        ["gh", "release", "view", tag, "-R", f"adsblol/globe_history_{day.year}", "--json", "assets"]
    )
    assets = sorted(json.loads(listing)["assets"], key=lambda a: a["name"])
    return [a["url"] for a in assets if ".tar" in a["name"]]


def fetch_day(day: dt.date, out_dir: Path = OUT_DIR, workers: int = 12) -> Path:
    """Streams one day's archive and keeps the points near the ten airports."""
    out = out_dir / f"{day.isoformat()}_airports.parquet"
    stream = _PartStream(_release_parts(day))
    rows = []

    def batches():
        archive = tarfile.open(fileobj=stream, mode="r|")
        batch = []
        try:
            for member in archive:
                if member.isfile() and "trace_full_" in member.name:
                    batch.append(archive.extractfile(member).read())
                    if len(batch) >= 64:
                        yield batch
                        batch = []
        except (tarfile.ReadError, EOFError):
            pass
        if batch:
            yield batch

    with Pool(workers) as pool:
        for part in pool.imap(_batch_rows, batches()):
            rows.extend(part)
    names = ["icao", "t", "lat", "lon", "ground", "alt", "gs", "apt", "callsign", "callsign_next"]
    types = [pl.Utf8, pl.Float64, pl.Float64, pl.Float64, pl.Boolean, pl.Int32, pl.Float32, pl.Utf8, pl.Utf8, pl.Utf8]
    frame = pl.DataFrame(rows, schema=list(zip(names, types)), orient="row")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out, compression="zstd")
    return out


# --- Matching and movement --------------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def _track(track: dict, takeoff: float, excluded=None) -> dict:
    """What one aircraft's trace says about its departure taking off near `takeoff`."""
    t, ground, gs = track["t"], track["ground"], track["gs"]
    lat, lon, alt = track["lat"], track["lon"], track["alt"]
    res: dict = {}
    near = np.where((t >= takeoff - 1800) & (t <= takeoff + 1800))[0]
    lift = None
    for k in near:
        if k > 0 and not ground[k] and ground[k - 1] and t[k] - t[k - 1] < 120:
            if lift is None or abs(t[k] - takeoff) < abs(lift - takeoff):
                lift = t[k]
    res["lift"] = lift
    end = lift if lift is not None else takeoff + 60
    g = np.where(ground & (t <= end) & (t >= takeoff - LOOKBACK_S))[0]
    if len(g) == 0:
        return res
    before = np.where(~ground & (t < t[g[-1]]) & (t >= takeoff - LOOKBACK_S))[0]
    if len(before):
        g = g[t[g] > t[before[-1]]]
    if len(g) == 0:
        return res
    tg, la, lo = t[g], lat[g], lon[g]
    sp = np.nan_to_num(gs[g], nan=0.0)
    n = len(g)
    res["first_ground"] = tg[0]
    gaps = np.where(np.diff(tg) >= APPEAR_GAP_S)[0]
    res["appear"] = tg[gaps[-1] + 1] if len(gaps) else tg[0]
    clusters, start, cla, clo = [], 0, la[0], lo[0]
    for k in range(1, n):
        if _haversine_m(cla, clo, la[k], lo[k]) > CLUSTER_RADIUS_M:
            if sp[k] >= 1.0 or k == n - 1 or _haversine_m(cla, clo, la[k + 1], lo[k + 1]) > CLUSTER_RADIUS_M:
                clusters.append((start, k - 1))
                start, cla, clo = k, la[k], lo[k]
    clusters.append((start, n - 1))
    dwell = [(a, b) for a, b in clusters if tg[b] - tg[a] >= STAND_DWELL_S]
    res["dwell_spots"] = [(float(np.median(la[a:b + 1])), float(np.median(lo[a:b + 1]))) for a, b in dwell]
    if excluded is not None:
        dwell = [(a, b) for a, b in dwell if not excluded(float(np.median(la[a:b + 1])), float(np.median(lo[a:b + 1])))]
    dwell = [(a, b) for a, b in dwell if end - tg[b] >= MIN_STAND_TO_TAKEOFF_S]
    if not dwell:
        return res
    a, b = max(dwell, key=lambda ab: tg[ab[1]] - tg[ab[0]])
    res["stand_dwell"] = tg[b] - tg[a]
    k = b
    while k > a and sp[k] >= 0.5 and tg[k] - tg[k - 1] <= 60:
        k -= 1
    move = k if sp[k] >= 0.5 else k + 1
    if move == a:
        move = a + 1 if a + 1 <= b else a
    if move > b and b == n - 1:
        return res
    res["move"] = tg[move]
    res["resolution"] = tg[move] - tg[move - 1]
    return res


def _departures(day: dt.date) -> pl.DataFrame:
    source = data.load_ranking() if day.year == 2026 else data.load_training()
    return data.departures(source).filter(pl.col(schema.MVT_TIME).dt.date() == day).select(
        schema.MVT_ID, schema.ADEP, schema.STAND,
        (pl.col(schema.MVT_TIME).dt.epoch("ms") / 1000.0).alias("_takeoff"),
        pl.col(schema.CALLSIGN).str.strip_chars().str.to_uppercase().alias("_cs"),
    ).collect()


def observe_day(day: dt.date, out_dir: Path = OUT_DIR) -> Path:
    """One row per departure of `day`: what ADS-B saw of it on the ground."""
    points = pl.read_parquet(out_dir / f"{day.isoformat()}_airports.parquet").with_columns(
        pl.col("callsign").str.strip_chars().str.to_uppercase(),
        pl.col("callsign_next").str.strip_chars().str.to_uppercase(),
    )
    deps = _departures(day)
    by_callsign: dict = {}
    for col in ("callsign", "callsign_next"):
        for airport, cs, icao in points.select("apt", col, "icao").unique().iter_rows():
            if cs:
                by_callsign.setdefault((airport, cs), set()).add(icao)
    tracks = {}
    for (icao, airport), sub in points.sort("t").group_by(["icao", "apt"], maintain_order=True):
        tracks[(icao, airport)] = {
            "t": sub["t"].to_numpy(), "ground": sub["ground"].to_numpy(),
            "gs": sub["gs"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
            "lat": sub["lat"].to_numpy(), "lon": sub["lon"].to_numpy(),
            "alt": sub["alt"].cast(pl.Float64).fill_null(0.0).to_numpy(), "cs": sub["callsign"].to_list(),
        }
    lifts: dict = {}
    for (icao, airport), tr in tracks.items():
        k = np.where(~tr["ground"][1:] & tr["ground"][:-1] & (np.diff(tr["t"]) < 120))[0] + 1
        for kk in k:
            lifts.setdefault(airport, []).append((tr["t"][kk], icao))

    def match(excluded=None) -> list[dict]:
        rows = []
        for mvt_id, airport, stand, takeoff, cs in deps.iter_rows():
            best = None
            if cs is not None:
                for icao in by_callsign.get((airport, cs), ()):
                    tr = tracks[(icao, airport)]
                    window = np.where((tr["t"] >= takeoff - LOOKBACK_S) & (tr["t"] <= takeoff + 900))[0]
                    seen = sum(1 for k in window if tr["cs"][k] == cs)
                    if not seen:
                        continue
                    res = _track(tr, takeoff, excluded)
                    score = (res.get("lift") is not None and abs(res["lift"] - takeoff) < 600,
                             res.get("first_ground") is not None, seen)
                    if best is None or score > best[0]:
                        best = (score, res)
            if best is None:
                near = [icao for t, icao in lifts.get(airport, []) if abs(t - takeoff) <= 60]
                if len(near) == 1:
                    best = (None, _track(tracks[(near[0], airport)], takeoff, excluded))
            rows.append({"MVT_ID": mvt_id, "stand": stand, "takeoff": takeoff, **(best[1] if best else {}),
                         "matched": best is not None})
        return rows

    # First pass finds every long dwell; spots shared by several stand labels
    # (de-icing pads, remote holding) are excluded in the second pass.
    spots, labels = [], []
    for row in match():
        for la, lo in row.get("dwell_spots", []):
            spots.append((la, lo))
            labels.append(row["stand"])
    shared = []
    if spots:
        arr = np.array(spots)
        for la, lo in arr:
            near = np.where(_haversine_m(la, lo, arr[:, 0], arr[:, 1]) <= SHARED_RADIUS_M)[0]
            if len({labels[j] for j in near}) >= SHARED_MIN_LABELS:
                shared.append((la, lo))
    shared = np.array(shared) if shared else np.zeros((0, 2))

    def excluded(la, lo):
        return len(shared) > 0 and bool((_haversine_m(la, lo, shared[:, 0], shared[:, 1]) <= SHARED_RADIUS_M).any())

    def since(row, key):
        return row["takeoff"] - row[key] if row.get(key) is not None else None

    out = pl.DataFrame([{
        schema.MVT_ID: row["MVT_ID"],
        "adsb_matched": bool(row["matched"]),
        # A move implying a taxi of over an hour is a tow between stands, not the push-back.
        "adsb_pushback_seen": bool(row.get("move") is not None and 60 <= row["takeoff"] - row["move"] <= 3600),
        "adsb_move_to_takeoff_sec": since(row, "move"),
        "adsb_first_ground_to_takeoff_sec": since(row, "first_ground"),
        "adsb_appear_to_takeoff_sec": since(row, "appear"),
        "adsb_stand_dwell_sec": row.get("stand_dwell"),
        "adsb_move_resolution_sec": row.get("resolution"),
        "adsb_takeoff_offset_sec": (row["lift"] - row["takeoff"]) if row.get("lift") is not None else None,
    } for row in match(excluded)], schema_overrides={c: pl.Float64 for c in OBSERVATION_COLUMNS[2:]})
    path = out_dir / f"{day.isoformat()}_departures.parquet"
    out.write_parquet(path)
    return path


def load_observations(days: list[dt.date], out_dir: Path = OUT_DIR) -> pl.DataFrame:
    """The observed departures of the given days that have been processed."""
    files = [out_dir / f"{d.isoformat()}_departures.parquet" for d in days]
    return pl.concat([pl.read_parquet(f) for f in files if f.exists()])


if __name__ == "__main__":
    command, *dates = sys.argv[1:]
    for d in (dt.date.fromisoformat(x) for x in dates):
        print(fetch_day(d) if command == "fetch" else observe_day(d), flush=True)
