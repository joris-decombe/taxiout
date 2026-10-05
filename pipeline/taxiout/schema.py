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

THE RANKING SET ALSO HOLDS ARRIVALS: 344,693 of them in its 689,534 rows,
with BLOCK_TIME (in-block) intact. Row order follows MVT_TIME and MVT_ID
follows the schedule, so neither encodes a departure's off-block. The
arrivals are what `context.py` reads for stand occupancy and runway use.

A RECORDING ARTIFACT: some departures have BLOCK_TIME == SCHED_TIME to the
second, so their target is exactly MVT_TIME - SCHED_TIME, often several
hours. About half of LIRF's departures with no flight record (null AOBT) and
18% of its matched ones do; elsewhere it is 1-6% and mostly harmless, since
those flights leave near schedule. A smaller set, spread across airports, has
an off-block a day early (target ~86,400s plus a normal taxi). train.py
models the first in both groups; nothing observable flags the second.
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
IOBT = "IOBT_flt"
# The last off-block time in the flight plan. On every 2025 departure that
# has one, BLOCK_TIME lies within +-3606s of it: the movement and flight
# tables look to have been joined on that window.
LOBT = "LOBT_flt"
# An opaque hash, but a stable one: the same operator gets the same value.
OPERATOR = "AIRCRAFT_OPERATOR_flt"
MARKET_SEGMENT = "MARKET_SEGMENT_flt"
FLIGHT_TYPE = "FLIGHT_TYPE_flt"
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
