"""Fake parquet files shaped like the challenge data.

Nobody has opened the real files yet, so every column name in `schema.py` is a
transcription from the data page. These fixtures do not make those guesses any
more correct -- what they buy is that the *pipeline* is exercised end to end
before the real data lands, so the first genuine parquet meets working code
instead of a stack of API errors.

Delete this module once the real files are in `data/`; it is scaffolding, not
a test double worth maintaining.
"""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import polars as pl

from . import schema

AIRPORTS = ["EHAM", "LFPG", "EDDF"]
RUNWAYS = {"EHAM": ["18L", "24"], "LFPG": ["08L", "26R"], "EDDF": ["18", "25C"]}
STANDS = [f"{bay}{number}" for bay in "ABCD" for number in range(1, 6)]
AIRCRAFT = ["B738", "A320", "A21N", "B77W", "A359", "E190"]
WAKES = ["M", "M", "M", "H", "H", "J"]


def _movements(rng: random.Random, start: dt.datetime, days: int, per_day: int) -> dict:
    rows: dict[str, list] = {name: [] for name in schema.REQUIRED_COLUMNS}
    rows[schema.FLIGHT_ID] = []
    rows[schema.WAKE_CATEGORY] = []

    identifier = 1
    for day in range(days):
        midnight = start + dt.timedelta(days=day)
        for _ in range(per_day):
            airport = rng.choice(AIRPORTS)
            runway = rng.choice(RUNWAYS[airport])
            stand = rng.choice(STANDS)
            index = AIRCRAFT.index(rng.choice(AIRCRAFT))

            # Departure banks in the morning and late afternoon, so the
            # congestion features have something to bite on.
            hour = rng.choice([6, 7, 8, 9, 12, 16, 17, 18, 19, 21])
            off_block = midnight + dt.timedelta(
                hours=hour, minutes=rng.randrange(60), seconds=rng.randrange(60)
            )
            scheduled = off_block - dt.timedelta(minutes=rng.gauss(5, 12))

            unimpeded = 240 + 40 * (ord(stand[0]) - ord("A"))
            taxi = max(60.0, rng.gauss(unimpeded + 180, 240))
            takeoff = off_block + dt.timedelta(seconds=taxi)

            rows[schema.MVT_ID].append(identifier)
            rows[schema.FLIGHT_ID].append(identifier * 10)
            rows[schema.PHASE].append(schema.DEPARTURE)
            rows[schema.ADEP].append(airport)
            rows[schema.ADES].append(rng.choice(["EGLL", "LEMD", "LIRF"]))
            rows[schema.MVT_TIME].append(takeoff)
            rows[schema.BLOCK_TIME].append(off_block)
            rows[schema.SCHED_TIME].append(scheduled)
            rows[schema.AIRCRAFT_TYPE].append(AIRCRAFT[index])
            rows[schema.RUNWAY].append(runway)
            rows[schema.STAND].append(stand)
            rows[schema.TAXITIME].append(round(taxi, 1))
            rows[schema.WAKE_CATEGORY].append(WAKES[index])
            identifier += 1

    return rows


def write(data_dir: Path, seed: int = 0, days: int = 60, per_day: int = 120) -> list[Path]:
    """Writes training/ranking/template parquets and returns what it wrote."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    written: list[Path] = []

    # Two training files, so the glob and the multi-file scan are exercised.
    # Spanning a full year matters: the validation split holds out January and
    # July, and an empty validation set would make `train` fail confusingly.
    for part, start in enumerate(
        [dt.datetime(2025, 1, 1), dt.datetime(2025, 7, 1)], start=1
    ):
        frame = pl.DataFrame(_movements(rng, start, days, per_day))
        path = data_dir / f"training_{part}.parquet"
        frame.write_parquet(path)
        written.append(path)

    # The ranking set is January and July 2026 with the target blanked out.
    ranking = pl.DataFrame(_movements(rng, dt.datetime(2026, 1, 1), days, per_day))
    ranking = ranking.with_columns(pl.lit(None, dtype=pl.Float64).alias(schema.TAXITIME))
    ranking_path = data_dir / "ranking.parquet"
    ranking.write_parquet(ranking_path)
    written.append(ranking_path)

    template = ranking.select(schema.MVT_ID, schema.TAXITIME)
    template_path = data_dir / "submitting.parquet"
    template.write_parquet(template_path)
    written.append(template_path)

    return written


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    for written_path in write(target):
        print(f"wrote {written_path}")
