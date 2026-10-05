"""Feature construction for the taxi-out model.

Three families:

  * Per row -- stand/runway geometry, aircraft type, calendar, and the
    Network Manager's timings against takeoff.
  * Surroundings -- stand occupancy, queue counts and runway configuration
    from the other movements in the same dataset (`context.py`), and the
    airport's weather at takeoff (`weather.py`). These need the full
    movement table, arrivals included, so they are built once per dataset by
    `surroundings()` and joined in by MVT_ID.
  * The unimpeded stand/runway reference, fitted on training targets only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import context, schema, weather

CONGESTION_WINDOWS_MIN = (15, 30, 60)

# Windows, in minutes either side of take-off, for the live excess features.
EXCESS_WINDOWS_MIN = (30, 60)

# Whether pairs never seen in training fall back to a stand-group and then a
# runway reference. Exists so experiments can switch it off. Validation cannot
# judge it (+0.4s, 95% CI -0.7s to +1.7s): unseen stands are under 0.1% of
# validation rows but 4.6% of EDDM's 2026 departures.
UNIMPEDED_FALLBACK = True

STAND_GROUP = "stand_group"


def stand_group() -> pl.Expr:
    """Neighbouring stands, by name: all but the last character.

    Stand names are mostly apron letters and sequential numbers, so "103" to
    "108" share "10" and "V333" to "V337" share "V33". Short names keep only
    their first character. It is a proxy for adjacency, not a map.
    """
    stand = pl.col(schema.STAND)
    return (
        pl.when(stand.str.len_chars() >= 3).then(stand.str.head(-1)).otherwise(stand.str.head(1))
    )


def unimpeded_taxi_reference(training: pl.LazyFrame) -> pl.LazyFrame:
    """The taxi time when the airport is quiet, at three levels of detail.

    The low quantile is the point: the median of a congested pair still has
    queueing baked into it, whereas the 10th percentile approximates the
    geometry alone (EUROCONTROL's reference uses the same P10 per stand and
    runway). Rows carry a `level`: "pair" per stand and runway, "group" per
    stand group and runway, "runway" per runway. `build` uses the most
    specific one available, so a stand that opened after the training period
    (EDDM 103-108 and EDDF's Terminal 3 in 2026) still gets a reference from
    its neighbours instead of a null.
    """
    departures = training.filter(pl.col(schema.PHASE) == schema.DEPARTURE).with_columns(
        stand_group().alias(STAND_GROUP)
    )
    p10 = pl.col(schema.TAXITIME).quantile(0.10).alias("unimpeded_taxi_sec")
    pair = departures.group_by([schema.ADEP, schema.STAND, schema.RUNWAY]).agg(
        p10, pl.len().alias("pair_movements")
    )
    group = departures.group_by([schema.ADEP, STAND_GROUP, schema.RUNWAY]).agg(p10)
    runway = departures.group_by([schema.ADEP, schema.RUNWAY]).agg(p10)
    return pl.concat(
        [
            pair.with_columns(pl.lit("pair").alias("level"), pl.lit(None, pl.String).alias(STAND_GROUP)),
            group.with_columns(
                pl.lit("group").alias("level"), pl.lit(None, pl.String).alias(schema.STAND),
                pl.lit(None, pl.UInt32).alias("pair_movements"),
            ),
            runway.with_columns(
                pl.lit("runway").alias("level"), pl.lit(None, pl.String).alias(schema.STAND),
                pl.lit(None, pl.String).alias(STAND_GROUP), pl.lit(None, pl.UInt32).alias("pair_movements"),
            ),
        ],
        how="diagonal",
    )


def join_unimpeded(frame: pl.LazyFrame, unimpeded: pl.LazyFrame) -> pl.LazyFrame:
    """Joins the most specific reference available: pair, then group, then runway."""
    level = lambda name: unimpeded.filter(pl.col("level") == name)  # noqa: E731
    frame = frame.with_columns(stand_group().alias(STAND_GROUP)).join(
        level("pair").select(schema.ADEP, schema.STAND, schema.RUNWAY, "unimpeded_taxi_sec", "pair_movements"),
        on=[schema.ADEP, schema.STAND, schema.RUNWAY], how="left",
    )
    if not UNIMPEDED_FALLBACK:
        return frame.with_columns(
            pl.when(pl.col("unimpeded_taxi_sec").is_not_null()).then(0).alias("unimpeded_level")
        )
    frame = frame.join(
        level("group").select(schema.ADEP, STAND_GROUP, schema.RUNWAY, pl.col("unimpeded_taxi_sec").alias("_ref_group")),
        on=[schema.ADEP, STAND_GROUP, schema.RUNWAY], how="left",
    ).join(
        level("runway").select(schema.ADEP, schema.RUNWAY, pl.col("unimpeded_taxi_sec").alias("_ref_runway")),
        on=[schema.ADEP, schema.RUNWAY], how="left",
    )
    ref = pl.col("unimpeded_taxi_sec")
    return frame.with_columns(
        pl.when(ref.is_not_null()).then(0)
        .when(pl.col("_ref_group").is_not_null()).then(1)
        .when(pl.col("_ref_runway").is_not_null()).then(2)
        .alias("unimpeded_level"),
        pl.coalesce(ref, "_ref_group", "_ref_runway").alias("unimpeded_taxi_sec"),
    ).drop("_ref_group", "_ref_runway")


def add_calendar_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    time = pl.col(schema.MVT_TIME)
    return frame.with_columns(
        time.dt.hour().alias("hour"),
        time.dt.weekday().alias("weekday"),
        time.dt.month().alias("month"),
        time.dt.ordinal_day().alias("day_of_year"),
    )


def add_schedule_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Timings available at predict time, anchored on takeoff.

    The obvious feature here is how late the aircraft left the stand --
    BLOCK_TIME - SCHED_TIME. It is not available: BLOCK_TIME is blank on the
    ranking set, because the target is MVT_TIME minus it.

    AOBT is the Network Manager's off-block time and does survive, so
    `aobt_to_takeoff_sec` is the closest available thing to the target
    itself. Measured over the full 2025 training set, on the rows that have
    an AOBT, it predicts taxi-out at 385s RMSE against a 417s standard
    deviation for those rows: a real edge, but a modest one.

    It is null for the ~1.5% of departures with no Network Manager record,
    which is why train.py fits those rows as a separate model -- this
    feature carries zero gain there, by construction.
    """
    takeoff = pl.col(schema.MVT_TIME)
    return frame.with_columns(
        (takeoff - pl.col(schema.AOBT)).dt.total_seconds().alias("aobt_to_takeoff_sec"),
        (takeoff - pl.col(schema.SCHED_TIME)).dt.total_seconds().alias("sched_to_takeoff_sec"),
        (pl.col(schema.AOBT) - pl.col(schema.EOBT)).dt.total_seconds().alias("off_block_delay_sec"),
        (takeoff - pl.col(schema.EOBT)).dt.total_seconds().alias("eobt_to_takeoff_sec"),
        (takeoff - pl.col(schema.IOBT)).dt.total_seconds().alias("iobt_to_takeoff_sec"),
        # How late the NM saw the aircraft leave against the timetable. At
        # LIRF a large value is the strongest sign that BLOCK_TIME was
        # recorded at the schedule instead (see train.py).
        (pl.col(schema.AOBT) - pl.col(schema.SCHED_TIME)).dt.total_seconds().alias("aobt_vs_sched_sec"),
        pl.col(schema.AOBT).dt.second().alias("aobt_second"),
        (takeoff - pl.col(schema.LOBT)).dt.total_seconds().alias("lobt_to_takeoff_sec"),
        (pl.col(schema.SCHED_TIME) - pl.col(schema.LOBT)).dt.total_seconds().alias("sched_vs_lobt_sec"),
    )


def add_live_excess(frame: pl.DataFrame) -> pl.DataFrame:
    """How far the airport's other departures ran over their reference, just now.

    For each departure j with an NM record, excess_j = (MVT_j - AOBT_j) -
    unimpeded_j: take-off minus the Network Manager off-block, less the
    stand-runway reference. For each row, the mean of that over the same
    airport's (or runway's) other departures taking off within the window
    either side of this one, leaving the row itself out.

    It measures what the METARs only guess at: remote de-icing, snow
    clearance, a closed runway or a queue, as it is actually happening. On
    icing rows its hourly version correlated 0.36-0.56 with a flight's
    taxi-out excess, against 0.07-0.46 for the weather features. Nothing in
    it reads BLOCK_TIME, so it computes the same on the ranking set, and
    rows with no NM record still get their neighbours' value.

    Excess is clipped to [-900s, 3600s] so one recording artifact cannot move
    a window mean by much.
    """
    frame = frame.with_row_index("_row")
    excess = (
        (pl.col(schema.MVT_TIME) - pl.col(schema.AOBT)).dt.total_seconds() - pl.col("unimpeded_taxi_sec")
    ).clip(-900, 3600)
    base = frame.select(
        "_row", schema.ADEP, schema.RUNWAY,
        (pl.col(schema.MVT_TIME).dt.epoch("ms").cast(pl.Float64) / 1000.0).alias("_t"),
        excess.cast(pl.Float64).alias("_e"),
    )
    out = {name: np.full(frame.height, np.nan) for name in LIVE_EXCESS_COLUMNS}

    def window_means(group: pl.DataFrame, minutes: int) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(group["_t"].to_numpy(), kind="stable")
        rows = group["_row"].to_numpy()[order]
        t = group["_t"].to_numpy()[order]
        e = group["_e"].to_numpy()[order]
        valid = ~np.isnan(e)
        own = np.where(valid, e, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(own)])
        cn = np.concatenate([[0], np.cumsum(valid)])
        lo = np.searchsorted(t, t - 60.0 * minutes, "left")
        hi = np.searchsorted(t, t + 60.0 * minutes, "right")
        s = cs[hi] - cs[lo] - own
        n = cn[hi] - cn[lo] - valid
        mean = np.full(len(t), np.nan)
        mean[n > 0] = s[n > 0] / n[n > 0]
        return rows, mean

    for _, group in base.group_by(schema.ADEP):
        for m in EXCESS_WINDOWS_MIN:
            rows, mean = window_means(group, m)
            out[f"apt_excess_{m}m"][rows] = mean
        for _, runway_group in group.group_by(schema.RUNWAY):
            rows, mean = window_means(runway_group, 60)
            out["rwy_excess_60m"][rows] = mean
    # A window with no neighbours is missing, not a number.
    return frame.with_columns([pl.Series(k, v).fill_nan(None) for k, v in out.items()]).drop("_row")


def add_record_completeness(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Which fields the movement record is missing.

    LightGBM routes nulls on its own, so these look redundant, and inside a
    single model over all departures they are: adding them there measured
    slightly worse. They earn their place in the model fitted to the ~1.5% of
    departures with no Network Manager record, where they separate within the
    group instead of restating what 2M nulls already say.

    Worth 7.9s of RMSE (415.8s to 407.9s), 95% CI -13.6s to -2.9s on a paired
    bootstrap. The unpaired intervals overlap almost entirely, so only the
    paired comparison can see an effect this size.
    """
    return frame.with_columns(
        pl.col(schema.AIRCRAFT_TYPE).is_null().cast(pl.Int8).alias("no_aircraft_type"),
        pl.col(schema.STAND).is_null().cast(pl.Int8).alias("no_stand"),
        pl.col(schema.ADES).is_null().cast(pl.Int8).alias("no_destination"),
    )

def add_pushback_congestion(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Movements per airport in the windows before each pushback.

    Windowed on MVT_TIME, which is takeoff for a departure and landing for
    an arrival. Both survive on the ranking set, so these compute
    identically there; it is BLOCK_TIME that is blanked, not MVT_TIME.
    """
    frame = frame.sort(schema.MVT_TIME)
    expressions = []
    for minutes in CONGESTION_WINDOWS_MIN:
        window = f"{minutes}m"
        expressions.append(
            pl.len()
            .rolling(index_column=schema.MVT_TIME, period=window)
            .over(schema.ADEP)
            .alias(f"movements_prev_{minutes}m")
        )
    return frame.with_columns(expressions)


def surroundings(movements: pl.LazyFrame) -> pl.DataFrame:
    """Per-departure features that need the whole dataset, keyed by MVT_ID.

    `movements` is one dataset's full table, arrivals included: the stand,
    queue and runway-configuration features in `context` read other rows,
    and the weather join reads the METAR archive.
    """
    around = context.build(movements)
    deps = (
        movements.filter(pl.col(schema.PHASE) == schema.DEPARTURE)
        .select(schema.MVT_ID, schema.ADEP, schema.MVT_TIME)
        .collect()
    )
    wx = weather.join(deps, weather.load()).drop(schema.ADEP, schema.MVT_TIME)
    return around.join(wx, on=schema.MVT_ID, how="left")


def build(frame: pl.LazyFrame, unimpeded: pl.LazyFrame, around: pl.DataFrame) -> pl.LazyFrame:
    """`around` is `surroundings()` of the dataset `frame` was drawn from.

    The live excess features read the other rows of `frame` itself, so
    `frame` should hold all of a dataset's departures for the period it
    covers, as the train/validation split and the ranking set do.
    """
    frame = join_unimpeded(frame, unimpeded)
    frame = add_calendar_features(frame)
    frame = add_schedule_features(frame)
    frame = add_record_completeness(frame)
    frame = add_pushback_congestion(frame)
    frame = add_live_excess(frame.collect()).lazy()
    return frame.join(around.lazy(), on=schema.MVT_ID, how="left")


FEATURE_COLUMNS = [
    "unimpeded_taxi_sec",
    "pair_movements",
    "hour",
    "weekday",
    "month",
    "day_of_year",
    "aobt_to_takeoff_sec",
    "sched_to_takeoff_sec",
    "off_block_delay_sec",
    "no_aircraft_type",
    "no_stand",
    "no_destination",
    *[f"movements_prev_{m}m" for m in CONGESTION_WINDOWS_MIN],
    "eobt_to_takeoff_sec",
    "iobt_to_takeoff_sec",
    "aobt_vs_sched_sec",
    "aobt_second",
    *context.FEATURE_COLUMNS,
    *weather.FEATURE_COLUMNS,
]

# Worth 2.3s on validation (346.6s to 344.3s, 95% CI -6.5s to -0.3s), most
# of it at LIRF. January 2025 was mild, so its de-icing side is unmeasured
# there. The orphan model does not see these (see `train.orphan_columns`).
LIVE_EXCESS_COLUMNS = [*[f"apt_excess_{m}m" for m in EXCESS_WINDOWS_MIN], "rwy_excess_60m"]
FEATURE_COLUMNS += [*LIVE_EXCESS_COLUMNS, "unimpeded_level"]

CATEGORICAL_COLUMNS = [
    schema.ADEP, schema.RUNWAY, schema.STAND, schema.AIRCRAFT_TYPE,
    schema.OPERATOR, schema.MARKET_SEGMENT, schema.FLIGHT_TYPE, schema.WAKE_CATEGORY, schema.ADES,
]
