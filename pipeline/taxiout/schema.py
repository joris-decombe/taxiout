"""Column names from the published dataset description.

Verified against the real training files on 17 September 2026: every name
below is present in training_2025-01-01_2025-02-01.parquet, which carries 30
columns in total. `verify_schema` passes.
"""

from __future__ import annotations

import polars as pl

# Movement table.
MVT_ID = "MVT_ID_mvt"
FLIGHT_ID = "FLIGHT_ID_mvt"
PHASE = "PHASE_mvt"
ADEP = "ADEP_mvt"
ADES = "ADES_mvt"
# The movement time: takeoff for a departure, landing for an arrival.
#
# For departures this is the TARGET IN DISGUISE. Verified against the
# January 2025 file: MVT_TIME - BLOCK_TIME == TAXITIME_SEC for 100% of
# 153,706 departure rows. Any feature derived from it -- including
# something as innocent-looking as hour-of-day -- leaks, and it is blank
# on the ranking set anyway. Use BLOCK_TIME for departures instead.
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
