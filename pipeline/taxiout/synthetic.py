"""Synthetic movements, so the simulator can be exercised before the real data.

Generates a plausible day at a two-runway hub: departure banks separated by
quiet periods, a steady arrival stream, and a realistic wake-category mix.
The point is not realism for its own sake -- it is having something whose
congestion behaviour we know, to check the simulator reproduces it.
"""

from __future__ import annotations

import random
from pathlib import Path

WAKE_MIX = [("L", 0.03), ("M", 0.78), ("H", 0.17), ("J", 0.02)]
RUNWAYS = ["09L", "09R"]

# Departure banks, as (start hour, number of departures).
BANKS = [(6, 60), (9, 45), (12, 55), (15, 40), (18, 65), (20, 30)]


def _sample_wake(rng: random.Random) -> str:
    roll = rng.random()
    cumulative = 0.0
    for code, share in WAKE_MIX:
        cumulative += share
        if roll < cumulative:
            return code
    return "M"


def generate(path: Path, seed: int = 0) -> Path:
    rng = random.Random(seed)
    rows: list[tuple] = []
    next_id = 1

    for start_hour, count in BANKS:
        bank_start = start_hour * 3600
        for _ in range(count):
            # Pushbacks cluster over ~40 minutes, which is what builds a queue.
            off_block = bank_start + rng.gauss(1200, 700)
            rows.append(
                (
                    next_id,
                    "DEP",
                    _sample_wake(rng),
                    rng.choice(RUNWAYS),
                    round(off_block, 1),
                    round(rng.gauss(300, 90), 1),  # unimpeded apron transit
                    0.0,
                )
            )
            next_id += 1

    # Arrivals run all day at roughly two per five minutes.
    for landing in range(5 * 3600, 23 * 3600, 150):
        rows.append(
            (next_id, "ARR", "M", rng.choice(RUNWAYS), 0.0, 0.0, float(landing + rng.gauss(0, 40)))
        )
        next_id += 1

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,phase,wake,runway,off_block_time,unimpeded_taxi_sec,landing_time\n")
        for row in rows:
            handle.write(",".join(str(field) for field in row) + "\n")

    return path


if __name__ == "__main__":
    import sys

    output = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_movements.csv")
    generate(output)
    print(f"wrote {output}")
