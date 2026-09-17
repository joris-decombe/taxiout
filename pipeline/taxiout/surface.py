"""Running the C++ surface simulator over the challenge data.

The simulator models one aerodrome's departure surface: aircraft push back,
cross the apron, join a FIFO runway queue subject to wake separation, and
arrivals preempt them. It answers a question the movement counts cannot, which
is how much of a given departure's ground time was queueing rather than transit.

Its input is the awkward part. It wants an off-block time per departure, and
`BLOCK_TIME_UTC_mvt` is blank on the scored set precisely because the target is
derived from it. `AOBT_3_flt` is the substitute: the Network Manager's own
off-block time, present for about 98.5% of departures on both the training and
the scored data. It is not the same quantity (the two differ with a standard
deviation of 384s) so the reconstructed queue delay inherits that noise, but it
is the only off-block clock that survives.

Times are passed to the simulator as seconds within the day being simulated,
not as a Unix epoch: one airport-day per invocation keeps the numbers small and
the queue state meaningful, since a queue does not carry across a night.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import polars as pl

from . import data, schema

# Built by `cmake --build sim/build`. MSVC puts binaries under a config
# subdirectory; single-config generators do not.
BINARY_CANDIDATES = (
    Path("sim/build/Debug/taxiout_sim.exe"),
    Path("sim/build/Release/taxiout_sim.exe"),
    Path("sim/build/taxiout_sim.exe"),
    Path("sim/build/taxiout_sim"),
)

CSV_HEADER = "id,phase,wake,runway,off_block_time,unimpeded_taxi_sec,landing_time"

# Fallback when a movement has no wake category. Medium is by far the most
# common at these airports, so it is the least distorting guess.
DEFAULT_WAKE = "M"

# Fallback transit when a stand/runway pair has no unimpeded reference, in
# seconds. The airport-wide median is a better guess than zero, which would
# make the aircraft appear at the runway threshold instantly.
DEFAULT_UNIMPEDED_SEC = 600.0


def find_binary() -> Path:
    for candidate in BINARY_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "taxiout_sim not built. Run:\n"
        "  cmake -S sim -B sim/build\n"
        "  cmake --build sim/build --config Debug"
    )


def to_movements(frame: pl.DataFrame, unimpeded: pl.DataFrame) -> pl.DataFrame:
    """One airport-day of movements in the simulator's CSV shape.

    Departures contribute an off-block time and an unimpeded transit estimate;
    arrivals contribute only a landing time, which is what lets them take the
    runway away from the departure queue.
    """
    day_start = pl.col(schema.MVT_TIME).dt.truncate("1d")
    # The reference is keyed on ADEP, which only means anything for the
    # departures; arrivals take the default and never use it.
    frame = frame.join(
        unimpeded, on=[schema.ADEP, schema.STAND, schema.RUNWAY], how="left"
    ).with_columns(
        (pl.col(schema.AOBT) - day_start).dt.total_seconds().cast(pl.Float64).alias("_off_block"),
        (pl.col(schema.MVT_TIME) - day_start).dt.total_seconds().cast(pl.Float64).alias("_mvt"),
    )

    is_departure = pl.col(schema.PHASE) == schema.DEPARTURE
    return frame.select(
        pl.col(schema.MVT_ID).cast(pl.Int64).alias("id"),
        pl.when(is_departure).then(pl.lit("DEP")).otherwise(pl.lit("ARR")).alias("phase"),
        pl.col(schema.WAKE_CATEGORY).fill_null(DEFAULT_WAKE).alias("wake"),
        # The simulator keys runway state by this string, so a null would put
        # every unknown-runway movement into one shared phantom runway.
        pl.col(schema.RUNWAY).fill_null("UNK").alias("runway"),
        pl.when(is_departure).then(pl.col("_off_block")).otherwise(0.0).alias("off_block_time"),
        pl.when(is_departure)
        .then(pl.col("unimpeded_taxi_sec").fill_null(DEFAULT_UNIMPEDED_SEC))
        .otherwise(0.0)
        .alias("unimpeded_taxi_sec"),
        pl.when(is_departure).then(0.0).otherwise(pl.col("_mvt")).alias("landing_time"),
    ).drop_nulls("off_block_time")


def run_once(movements: pl.DataFrame, binary: Path | None = None) -> pl.DataFrame:
    """Runs the simulator over one airport-day and returns its predictions."""
    binary = binary or find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "movements.csv"
        out_path = Path(tmp) / "predictions.csv"
        movements.write_csv(in_path)
        result = subprocess.run(
            [str(binary), str(in_path), str(out_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"taxiout_sim failed: {result.stderr.strip()}")
        return pl.read_csv(out_path)


def aerodrome() -> pl.Expr:
    """The airport whose surface a movement occupies.

    ADEP is the origin and ADES the destination, so only the departure rows
    are keyed by ADEP. An arrival at Schiphol carries ADEP of wherever it flew
    from; grouping arrivals by ADEP scatters them across every origin airport
    in Europe and leaves the simulated hub with no arrivals to preempt its
    departure queue.
    """
    return (
        pl.when(pl.col(schema.PHASE) == schema.DEPARTURE)
        .then(pl.col(schema.ADEP))
        .otherwise(pl.col(schema.ADES))
    )


def queue_delay(
    frame: pl.DataFrame, unimpeded: pl.DataFrame, binary: Path | None = None
) -> pl.DataFrame:
    """Simulated queue delay per departure, as (MVT_ID, sim_queue_delay_sec).

    Runs one simulation per aerodrome-day. A runway queue does not persist
    across a night, and the simulator keys its state by runway name, so mixing
    airports or days into one run would have aircraft queueing behind each
    other across a continent.
    """
    binary = binary or find_binary()
    frame = frame.with_columns(
        pl.col(schema.MVT_TIME).dt.date().alias("_day"),
        aerodrome().alias("_aerodrome"),
    ).filter(pl.col("_aerodrome").is_not_null())

    pieces: list[pl.DataFrame] = []
    for (_airport, _day), group in frame.group_by(["_aerodrome", "_day"], maintain_order=True):
        movements = to_movements(group, unimpeded)
        if not movements.filter(pl.col("phase") == "DEP").height:
            continue
        pieces.append(run_once(movements, binary).select("id", "queue_delay_sec"))

    if not pieces:
        return pl.DataFrame(
            schema={schema.MVT_ID: pl.Float64, "sim_queue_delay_sec": pl.Float64}
        )

    return (
        pl.concat(pieces)
        .rename({"id": schema.MVT_ID, "queue_delay_sec": "sim_queue_delay_sec"})
        .with_columns(
            pl.col(schema.MVT_ID).cast(pl.Float64),
            # Floating-point noise puts exact zeros a few 1e-13 either side.
            pl.col("sim_queue_delay_sec").clip(0.0, None).round(1),
        )
    )


def add_queue_delay(
    frame: pl.DataFrame, unimpeded: pl.DataFrame, binary: Path | None = None
) -> pl.DataFrame:
    """Left-joins the simulated queue delay onto a feature frame."""
    return frame.join(queue_delay(frame, unimpeded, binary), on=schema.MVT_ID, how="left")


if __name__ == "__main__":
    import sys

    from . import features

    airport = sys.argv[1] if len(sys.argv) > 1 else "EHAM"
    training = data.load_training()
    reference = features.unimpeded_taxi_reference(training).collect()
    one_day = (
        training.filter(pl.col(schema.ADEP) == airport)
        .filter(pl.col(schema.MVT_TIME).dt.date() == pl.date(2025, 1, 15))
        .collect()
    )
    print(f"{airport} on 2025-01-15: {len(one_day)} movements")
    delays = queue_delay(one_day, reference)
    print(delays.describe())
