"""Column names from the published dataset description.

Verified against the real files on 17 September 2026: every name below is
present, `verify_schema` passes, and the training parquets carry 30 columns.

WHAT IS BLANKED ON THE RANKING SET (departure rows, all 344,841 of them):

    BLOCK_TIME_UTC_mvt   100% null
    TAXITIME_SEC_mvt     100% null   <- the target

and nothing else. MVT_TIME, SCHED_TIME, RUNWAY, STAND and the flight-table
columns all survive. Since TAXITIME == MVT_TIME - BLOCK_TIME exactly, those
two had to go together: the task is to reconstruct the off-block time given
the takeoff time, not the other way round.

A RECORDING ARTIFACT in the departures with no flight record (null AOBT):
about half of LIRF's have BLOCK_TIME == SCHED_TIME to the second, so their
target is exactly MVT_TIME - SCHED_TIME, often several hours. A smaller set,
spread across airports, has an off-block a day early (target ~86,400s plus a
normal taxi). train.py models the first; nothing observable flags the second.
"""

from __future__ import annotations

import polars as pl

# Movement table.
MVT_ID = "MVT_ID_mvt"
FLIGHT_ID = "FLIGHT_ID_mvt"
PHASE = "PHASE_mvt"
ADEP = "ADEP_mvt"
ADES = "ADES_mvt"
# Movement time: takeoff for a departure, landing for an arrival. Present
# on the ranking set, so it is a legitimate feature.
MVT_TIME = "MVT_TIME_UTC_mvt"
# Off-block time. BLANK ON THE RANKING SET -- it is the target subtracted
# from MVT_TIME, so anything derived from it leaks in training and cannot
# be computed at predict time. Use AOBT as the off-block proxy instead.
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
# The Network Manager's actual off-block time. Present on the ranking set
# (98.5% of departures), and NOT the same quantity as BLOCK_TIME: the two
# agree within a minute only 21% of the time, sd of the difference 374s.
# That gap is why this is a feature rather than the answer.
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
