"""Features from the movements around each departure, arrivals included.

The ranking set is not departures alone: it carries every arrival in the same
months too, with its in-block time intact, and every departure's takeoff time.
Only departure off-block times are blanked. So the state of the surface around
each departure is observable from the other rows, and none of it needs a
departure's BLOCK_TIME:

  * Stand. An arrival that blocks in at our stand before we take off either
    is our inbound aircraft (we left after it) or the next occupant (we left
    before it). Either way its in-block time brackets our off-block time.
  * Queue. Other departures' takeoff times and Network Manager off-block
    times say how many aircraft were on the surface between our pushback and
    our takeoff, and how busy the runway was around it, before and after.

Everything here is computed per dataset, over all of its movements, so it must
be built from the full movement table (both phases) of one dataset, and joined
back to departures by MVT_ID.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import schema

AIRPORT = "airport"
RUNWAY_KEY = "runway_key"
STAND_KEY = "stand_key"

# Windows around takeoff, in minutes, for runway and airport movement counts.
AROUND_TAKEOFF_MIN = (5, 10, 20, 40)

FEATURE_COLUMNS = [
    # Stand.
    "stand_prev_in_gap",
    "stand_prev2_in_gap",
    "stand_next_in_gap",
    "stand_prev_land_gap",
    "stand_prev_dep_gap",
    "stand_next_dep_gap",
    "stand_in_since_aobt",
    # Queue between the Network Manager's off-block and takeoff.
    "rwy_deps_since_aobt",
    "apt_deps_since_aobt",
    "rwy_arrs_since_aobt",
    "apt_surface_at_takeoff",
    "apt_surface_at_aobt",
    # Around takeoff.
    *[f"rwy_deps_before_{m}m" for m in AROUND_TAKEOFF_MIN],
    *[f"rwy_deps_after_{m}m" for m in AROUND_TAKEOFF_MIN],
    *[f"rwy_arrs_around_{m}m" for m in AROUND_TAKEOFF_MIN],
    *[f"apt_deps_around_{m}m" for m in AROUND_TAKEOFF_MIN],
    "rwy_prev_dep_gap",
    "rwy_next_dep_gap",
    # Runway configuration in the hour of takeoff.
    "cfg_dep_runways",
    "cfg_arr_runways",
    "cfg_rwy_dep_share",
    "cfg_rwy_mixed",
]

# Queueing-theory measures, kept apart until they earn a place in
# FEATURE_COLUMNS (`experiments_round4.py`). Simaiakis and Balakrishnan's
# adjusted traffic is the aircraft taxiing out when a flight pushes back plus
# those pushing back while it taxis; taxi-out grows with it, convex and
# non-decreasing. A runway busy period is a run of departures less than
# BUSY_GAP_SEC apart: a flight deep into one most likely waited in a queue.
QUEUEING_COLUMNS = [
    "apt_pushbacks_during_taxi",
    "apt_adjusted_traffic",
    "rwy_busy_rank",
    "rwy_busy_elapsed",
]
BUSY_GAP_SEC = 150.0


def _seconds(expr: pl.Expr) -> pl.Expr:
    return expr.dt.total_seconds()


def _keyed(movements: pl.LazyFrame) -> pl.DataFrame:
    is_dep = pl.col(schema.PHASE) == schema.DEPARTURE
    airport = pl.when(is_dep).then(pl.col(schema.ADEP)).otherwise(pl.col(schema.ADES))
    return (
        movements.with_columns(airport.alias(AIRPORT))
        .with_columns(
            pl.concat_str([AIRPORT, schema.RUNWAY], separator="|").alias(RUNWAY_KEY),
            pl.concat_str([AIRPORT, schema.STAND], separator="|").alias(STAND_KEY),
        )
        .select(
            schema.MVT_ID, schema.PHASE, AIRPORT, RUNWAY_KEY, STAND_KEY,
            schema.MVT_TIME, schema.BLOCK_TIME, schema.AOBT,
        )
        .collect()
    )


def _stand_features(deps: pl.DataFrame, arrs: pl.DataFrame) -> pl.DataFrame:
    """In-block, landing and departure times at the same stand, relative to takeoff."""
    inblocks = (
        arrs.filter(pl.col(schema.BLOCK_TIME).is_not_null() & pl.col(STAND_KEY).is_not_null())
        .select(STAND_KEY, pl.col(schema.BLOCK_TIME).alias("in_t"))
        .sort(STAND_KEY, "in_t")
        .with_columns(pl.col("in_t").shift(1).over(STAND_KEY).alias("in_prev_t"))
        .sort("in_t")
    )
    landings = (
        arrs.filter(pl.col(STAND_KEY).is_not_null())
        .select(STAND_KEY, pl.col(schema.MVT_TIME).alias("land_t"))
        .sort("land_t")
    )
    departs = (
        deps.filter(pl.col(STAND_KEY).is_not_null())
        .select(STAND_KEY, schema.MVT_ID, pl.col(schema.MVT_TIME).alias("dep_t"))
        .sort("dep_t")
    )
    takeoff = pl.col(schema.MVT_TIME)

    out = deps.select(schema.MVT_ID, STAND_KEY, schema.MVT_TIME, schema.AOBT).sort(schema.MVT_TIME)
    out = out.join_asof(inblocks, left_on=schema.MVT_TIME, right_on="in_t", by=STAND_KEY, strategy="backward")
    out = out.join_asof(
        inblocks.select(STAND_KEY, pl.col("in_t").alias("next_in_t")),
        left_on=schema.MVT_TIME, right_on="next_in_t", by=STAND_KEY, strategy="forward",
    )
    out = out.join_asof(landings, left_on=schema.MVT_TIME, right_on="land_t", by=STAND_KEY, strategy="backward")
    # The departure's own row would match itself, so step one microsecond off.
    shifted = out.with_columns(
        (takeoff - pl.duration(microseconds=1)).alias("_before"),
        (takeoff + pl.duration(microseconds=1)).alias("_after"),
    )
    shifted = shifted.sort("_before").join_asof(
        departs.select(STAND_KEY, pl.col("dep_t").alias("prev_dep_t")),
        left_on="_before", right_on="prev_dep_t", by=STAND_KEY, strategy="backward",
    )
    shifted = shifted.sort("_after").join_asof(
        departs.select(STAND_KEY, pl.col("dep_t").alias("next_dep_t")),
        left_on="_after", right_on="next_dep_t", by=STAND_KEY, strategy="forward",
    )
    # First in-block at the stand at or after the Network Manager's off-block.
    with_aobt = shifted.filter(pl.col(schema.AOBT).is_not_null()).sort(schema.AOBT).join_asof(
        inblocks.select(STAND_KEY, pl.col("in_t").alias("in_after_aobt_t")),
        left_on=schema.AOBT, right_on="in_after_aobt_t", by=STAND_KEY, strategy="forward",
    ).select(schema.MVT_ID, "in_after_aobt_t")
    shifted = shifted.join(with_aobt, on=schema.MVT_ID, how="left")

    return shifted.select(
        schema.MVT_ID,
        _seconds(takeoff - pl.col("in_t")).alias("stand_prev_in_gap"),
        _seconds(takeoff - pl.col("in_prev_t")).alias("stand_prev2_in_gap"),
        _seconds(pl.col("next_in_t") - takeoff).alias("stand_next_in_gap"),
        _seconds(takeoff - pl.col("land_t")).alias("stand_prev_land_gap"),
        _seconds(takeoff - pl.col("prev_dep_t")).alias("stand_prev_dep_gap"),
        _seconds(pl.col("next_dep_t") - takeoff).alias("stand_next_dep_gap"),
        # Negative when the next occupant was already in before takeoff.
        _seconds(pl.col("in_after_aobt_t") - takeoff).alias("stand_in_since_aobt"),
    )


def _count_between(sorted_times: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """How many of `sorted_times` fall in [lo, hi). NaN bounds give NaN."""
    out = (np.searchsorted(sorted_times, hi, side="left") - np.searchsorted(sorted_times, lo, side="left")).astype(float)
    out[np.isnan(lo) | np.isnan(hi)] = np.nan
    return out


def _nearest_gaps(sorted_times: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seconds to the previous and next time strictly before and after `t`."""
    i_lo = np.searchsorted(sorted_times, t, side="left") - 1
    i_hi = np.searchsorted(sorted_times, t, side="right")
    prev = np.full(len(t), np.nan)
    nxt = np.full(len(t), np.nan)
    ok = i_lo >= 0
    prev[ok] = t[ok] - sorted_times[i_lo[ok]]
    ok = i_hi < len(sorted_times)
    nxt[ok] = sorted_times[i_hi[ok]] - t[ok]
    return prev, nxt


def _busy_period(sorted_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per take-off: departures before it in its busy period, and seconds since it began."""
    n = len(sorted_times)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    starts = np.ones(n, dtype=bool)
    starts[1:] = np.diff(sorted_times) > BUSY_GAP_SEC
    start_idx = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    return (np.arange(n) - start_idx).astype(float), sorted_times - sorted_times[start_idx]


def _epoch(series: pl.Series) -> np.ndarray:
    return series.dt.epoch("ms").cast(pl.Float64).to_numpy() / 1000.0


def _queue_features(deps: pl.DataFrame, arrs: pl.DataFrame) -> pl.DataFrame:
    """Counts of movements on the same runway and airport around each departure."""
    parts = []
    for (airport,), group in deps.group_by(AIRPORT):
        apt_arrs = arrs.filter(pl.col(AIRPORT) == airport)
        takeoff = _epoch(group[schema.MVT_TIME])
        aobt = _epoch(group[schema.AOBT])
        apt_dep_t = np.sort(takeoff)
        apt_arr_t = np.sort(_epoch(apt_arrs[schema.MVT_TIME]))
        apt_aobt_t = np.sort(aobt[~np.isnan(aobt)])

        cols: dict[str, np.ndarray] = {
            "apt_deps_since_aobt": _count_between(apt_dep_t, aobt, takeoff),
            # Aircraft pushed back (by the NM clock) but not yet airborne.
            "apt_surface_at_takeoff": np.searchsorted(apt_aobt_t, takeoff, "right").astype(float)
            - np.searchsorted(apt_dep_t, takeoff, "left"),
            "apt_surface_at_aobt": np.searchsorted(apt_aobt_t, aobt, "left").astype(float)
            - np.searchsorted(apt_dep_t, aobt, "right"),
        }
        cols["apt_surface_at_aobt"][np.isnan(aobt)] = np.nan
        # Strictly after our own push-back, so the flight does not count itself.
        cols["apt_pushbacks_during_taxi"] = _count_between(apt_aobt_t, aobt + 1e-3, takeoff)
        cols["apt_adjusted_traffic"] = cols["apt_surface_at_aobt"] + cols["apt_pushbacks_during_taxi"]
        for m in AROUND_TAKEOFF_MIN:
            w = 60.0 * m
            cols[f"apt_deps_around_{m}m"] = _count_between(apt_dep_t, takeoff - w, takeoff + w)

        rwy_cols = {name: np.full(len(group), np.nan) for name in (
            "rwy_deps_since_aobt", "rwy_arrs_since_aobt", "rwy_prev_dep_gap", "rwy_next_dep_gap",
            "rwy_busy_rank", "rwy_busy_elapsed",
            *[f"rwy_deps_before_{m}m" for m in AROUND_TAKEOFF_MIN],
            *[f"rwy_deps_after_{m}m" for m in AROUND_TAKEOFF_MIN],
            *[f"rwy_arrs_around_{m}m" for m in AROUND_TAKEOFF_MIN],
        )}
        runway = group[RUNWAY_KEY].to_numpy()
        arr_runway = apt_arrs[RUNWAY_KEY].to_numpy()
        arr_t_all = _epoch(apt_arrs[schema.MVT_TIME])
        for key in np.unique(runway[runway != None]):  # noqa: E711
            idx = np.flatnonzero(runway == key)
            t = takeoff[idx]
            a = aobt[idx]
            rwy_dep_t = np.sort(t)
            rwy_arr_t = np.sort(arr_t_all[arr_runway == key])
            rwy_cols["rwy_deps_since_aobt"][idx] = _count_between(rwy_dep_t, a, t)
            rwy_cols["rwy_arrs_since_aobt"][idx] = _count_between(rwy_arr_t, a, t)
            rank, elapsed = _busy_period(rwy_dep_t)
            pos = np.searchsorted(rwy_dep_t, t, side="left")
            rwy_cols["rwy_busy_rank"][idx] = rank[pos]
            rwy_cols["rwy_busy_elapsed"][idx] = elapsed[pos]
            prev, nxt = _nearest_gaps(rwy_dep_t, t)
            rwy_cols["rwy_prev_dep_gap"][idx] = prev
            rwy_cols["rwy_next_dep_gap"][idx] = nxt
            for m in AROUND_TAKEOFF_MIN:
                w = 60.0 * m
                rwy_cols[f"rwy_deps_before_{m}m"][idx] = _count_between(rwy_dep_t, t - w, t)
                # Strictly after takeoff: shift the lower bound past our own row.
                rwy_cols[f"rwy_deps_after_{m}m"][idx] = _count_between(rwy_dep_t, t + 1e-3, t + w)
                rwy_cols[f"rwy_arrs_around_{m}m"][idx] = _count_between(rwy_arr_t, t - w, t + w)

        parts.append(pl.DataFrame({schema.MVT_ID: group[schema.MVT_ID], **cols, **rwy_cols}))
    return pl.concat(parts)


def _config_features(deps: pl.DataFrame, arrs: pl.DataFrame) -> pl.DataFrame:
    """Which runways the airport ran in the hour of each takeoff.

    Closures for works or maintenance, and configuration changes for wind,
    show up here as fewer active runways, traffic moved to another runway,
    or departures and arrivals sharing one. Derived from the movements
    themselves, so it needs no NOTAM archive and holds on the ranking set.
    """
    hour = pl.col(schema.MVT_TIME).dt.truncate("1h").alias("_hour")
    dep_hour = deps.select(schema.MVT_ID, AIRPORT, RUNWAY_KEY, hour)
    arr_hour = arrs.select(AIRPORT, RUNWAY_KEY, hour)
    dep_cfg = dep_hour.group_by(AIRPORT, "_hour").agg(
        pl.col(RUNWAY_KEY).n_unique().alias("cfg_dep_runways"), pl.len().alias("_deps")
    )
    arr_cfg = arr_hour.group_by(AIRPORT, "_hour").agg(pl.col(RUNWAY_KEY).n_unique().alias("cfg_arr_runways"))
    rwy_deps = dep_hour.group_by(RUNWAY_KEY, "_hour").agg(pl.len().alias("_rwy_deps"))
    rwy_arrs = arr_hour.group_by(RUNWAY_KEY, "_hour").agg(pl.len().alias("_rwy_arrs"))
    return (
        dep_hour.join(dep_cfg, on=[AIRPORT, "_hour"], how="left")
        .join(arr_cfg, on=[AIRPORT, "_hour"], how="left")
        .join(rwy_deps, on=[RUNWAY_KEY, "_hour"], how="left")
        .join(rwy_arrs, on=[RUNWAY_KEY, "_hour"], how="left")
        .select(
            schema.MVT_ID,
            pl.col("cfg_dep_runways").cast(pl.Float64),
            pl.col("cfg_arr_runways").cast(pl.Float64).fill_null(0.0),
            (pl.col("_rwy_deps") / pl.col("_deps")).alias("cfg_rwy_dep_share"),
            (pl.col("_rwy_arrs").fill_null(0) > 0).cast(pl.Float64).alias("cfg_rwy_mixed"),
        )
    )


def build(movements: pl.LazyFrame) -> pl.DataFrame:
    """Context features for every departure in `movements`, keyed by MVT_ID.

    `movements` must be one dataset's full movement table, both phases.
    Departure BLOCK_TIME is never read, so training and ranking compute alike.
    """
    keyed = _keyed(movements)
    deps = keyed.filter(pl.col(schema.PHASE) == schema.DEPARTURE).drop(schema.BLOCK_TIME)
    arrs = keyed.filter(pl.col(schema.PHASE) == schema.ARRIVAL)
    stand = _stand_features(deps, arrs)
    queue = _queue_features(deps, arrs)
    config = _config_features(deps, arrs)
    return (
        stand.join(queue, on=schema.MVT_ID, how="full", coalesce=True)
        .join(config, on=schema.MVT_ID, how="left")
    ).select(
        schema.MVT_ID, *FEATURE_COLUMNS, *QUEUEING_COLUMNS
    )
