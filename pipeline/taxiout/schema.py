"""Column names from the published dataset description.

These are transcribed from the challenge's data page, NOT from the real
parquet files -- nobody has opened those yet. Run `verify_schema` against the
first training file the moment the access keys arrive; every column here is a
guess until it passes.
"""

from __future__ import annotations

import polars as pl

# Movement table.
MVT_ID = "MVT_ID_mvt"
FLIGHT_ID = "FLIGHT_ID_mvt"
PHASE = "PHASE_mvt"
ADEP = "ADEP_mvt"
ADES = "ADES_mvt"
MVT_TIME = "MVT_TIME_UTC_mvt"
BLOCK_TIME = "BLOCK_TIME_UTC_mvt"
SCHED_TIME = "SCHED_TIME_UTC_mvt"
AIRCRAFT_TYPE = "AIRCRAFT_TYPE_mvt"
RUNWAY = "RUNWAY_mvt"
STAND = "STAND_mvt"
TAXITIME = "TAXITIME_SEC_mvt"

# Flight table (Network Manager).
CALLSIGN = "CALLSIGN_flt"
WAKE_CATEGORY = "WK_TBL_CAT_flt"
EOBT = "EOBT_1_flt"
AOBT = "AOBT_3_flt"

DEPARTURE = "DEP"
ARRIVAL = "ARR"

TARGET = TAXITIME

REQUIRED_COLUMNS = [
    MVT_ID, PHASE, ADEP, ADES, MVT_TIME, BLOCK_TIME, SCHED_TIME,
    AIRCRAFT_TYPE, RUNWAY, STAND, TAXITIME,
]


def verify_schema(frame: pl.LazyFrame) -> list[str]:
    """Returns the required columns that are missing. Empty means we guessed right."""
    present = set(frame.collect_schema().names())
    return [column for column in REQUIRED_COLUMNS if column not in present]
