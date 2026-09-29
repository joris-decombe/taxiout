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

import polars as pl

from . import context, schema, weather

CONGESTION_WINDOWS_MIN = (15, 30, 60)


def unimpeded_taxi_reference(training: pl.LazyFrame) -> pl.LazyFrame:
    """Per stand/runway pair, the taxi time when the airport is quiet.

    The low quantile is the point: the median of a congested pair still has
    queueing baked into it, whereas the 10th percentile approximates the
    geometry alone. Falls back to the airport-level value for rare pairs.
    """
    return (
        training.filter(pl.col(schema.PHASE) == schema.DEPARTURE)
        .group_by([schema.ADEP, schema.STAND, schema.RUNWAY])
        .agg(
            pl.col(schema.TAXITIME).quantile(0.10).alias("unimpeded_taxi_sec"),
            pl.len().alias("pair_movements"),
        )
    )


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
    )


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
    """`around` is `surroundings()` of the dataset `frame` was drawn from."""
    frame = frame.join(unimpeded, on=[schema.ADEP, schema.STAND, schema.RUNWAY], how="left")
    frame = add_calendar_features(frame)
    frame = add_schedule_features(frame)
    frame = add_record_completeness(frame)
    frame = add_pushback_congestion(frame)
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

CATEGORICAL_COLUMNS = [
    schema.ADEP, schema.RUNWAY, schema.STAND, schema.AIRCRAFT_TYPE,
    schema.OPERATOR, schema.MARKET_SEGMENT, schema.FLIGHT_TYPE, schema.WAKE_CATEGORY, schema.ADES,
]
