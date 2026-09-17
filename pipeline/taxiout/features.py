"""Feature construction for the taxi-out model.

Two families:

  * Static -- stand/runway geometry, aircraft type, calendar. Available for
    every movement in both the training and ranking sets.
  * Congestion -- how busy the surface was at pushback. These carry most of
    the signal AND most of the difficulty: the honest versions need takeoff
    times, which are blanked on the ranking set. Only the features computable
    from off-block times alone live here; the rest come from the simulator.
"""

from __future__ import annotations

import polars as pl

from . import schema

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


def build(frame: pl.LazyFrame, unimpeded: pl.LazyFrame) -> pl.LazyFrame:
    frame = frame.join(unimpeded, on=[schema.ADEP, schema.STAND, schema.RUNWAY], how="left")
    frame = add_calendar_features(frame)
    frame = add_schedule_features(frame)
    frame = add_pushback_congestion(frame)
    return frame


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
    *[f"movements_prev_{m}m" for m in CONGESTION_WINDOWS_MIN],
]

CATEGORICAL_COLUMNS = [schema.ADEP, schema.RUNWAY, schema.STAND, schema.AIRCRAFT_TYPE]
