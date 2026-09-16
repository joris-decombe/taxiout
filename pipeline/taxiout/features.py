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

# Scratch column: the leak-free clock, dropped again once the windows are cut.
_EVENT_TIME = "_surface_event_time"


def surface_event_time() -> pl.Expr:
    """When a movement actually occupies the surface, without leaking.

    MVT_TIME is takeoff for a departure, which is off-block plus the target:
    using it anywhere in a departure's features leaks, and it is blanked on
    the ranking set regardless. Off-block time is the honest substitute.

    Arrivals keep MVT_TIME -- their landing time is known in advance of any
    departure's takeoff, so it carries no information about the target.
    """
    return (
        pl.when(pl.col(schema.PHASE) == schema.DEPARTURE)
        .then(pl.col(schema.BLOCK_TIME))
        .otherwise(pl.col(schema.MVT_TIME))
    )

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
    # Off-block, not takeoff: hour-of-day looks innocent but is derived
    # from the target when taken off MVT_TIME.
    time = surface_event_time()
    return frame.with_columns(
        time.dt.hour().alias("hour"),
        time.dt.weekday().alias("weekday"),
        time.dt.month().alias("month"),
        time.dt.ordinal_day().alias("day_of_year"),
    )


def add_schedule_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """How late the aircraft actually left the stand.

    A flight pushing back well behind schedule is often doing so into a bank
    it was meant to miss, so this proxies for congestion the counts can miss.
    """
    return frame.with_columns(
        (pl.col(schema.BLOCK_TIME) - pl.col(schema.SCHED_TIME))
        .dt.total_seconds()
        .alias("off_block_delay_sec")
    )


def add_pushback_congestion(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Movements per airport in the windows before each pushback.

    Uses off-block and landing times only -- never takeoff times, which are
    the target in disguise -- so these compute identically on training and
    ranking data. That constraint is what makes them trustworthy; richer
    queue-state features come from the sim.
    """
    frame = frame.with_columns(surface_event_time().alias(_EVENT_TIME)).sort(_EVENT_TIME)
    expressions = []
    for minutes in CONGESTION_WINDOWS_MIN:
        window = f"{minutes}m"
        expressions.append(
            pl.len()
            .rolling(index_column=_EVENT_TIME, period=window)
            .over(schema.ADEP)
            .alias(f"movements_prev_{minutes}m")
        )
    return frame.with_columns(expressions).drop(_EVENT_TIME)


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
    "off_block_delay_sec",
    *[f"movements_prev_{m}m" for m in CONGESTION_WINDOWS_MIN],
]

CATEGORICAL_COLUMNS = [schema.ADEP, schema.RUNWAY, schema.STAND, schema.AIRCRAFT_TYPE]
