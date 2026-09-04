# taxiout

PRC Data Challenge 2026 — predicting taxi-out time at 11 major European airports.

Two tracks that meet in the middle:

- **`pipeline/`** — Python. Loads the challenge parquet files, builds features,
  trains a LightGBM baseline, writes `submitting.parquet`. This is the yardstick.
- **`sim/`** — C++20. An event-driven model of the departure surface: pushback,
  apron transit, runway queue with wake separation, arrival preemption. This is
  the part that can beat the yardstick.

## Why simulate at all

Taxi-out decomposes into unimpeded transit plus queue delay, and the queue delay
is where all the variance lives. The strongest congestion features need takeoff
times — which are blanked on the ranking set. They depend on the very quantity
being predicted, so they cannot be measured, only *reconstructed* by running the
whole airport forward in time. That is the simulator's job, and its output
(`queue_delay_sec`) becomes a feature for the model.

## Status

Scaffold only. No real data yet — bucket keys arrive with team approval.

Everything in `pipeline/taxiout/schema.py` is transcribed from the challenge's
published data page, **not** from the actual files. Run `verify_schema` against
the first training parquet before trusting any of it.

## Build

```
cmake -S sim -B sim/build
cmake --build sim/build
ctest --test-dir sim/build
```

Needs a C++20 compiler and CMake 3.20+. Tests pull doctest via FetchContent, so
the first configure needs network access.

## Run the simulator on synthetic movements

```
python pipeline/taxiout/synthetic.py data/synthetic.csv
./sim/build/taxiout_sim data/synthetic.csv data/predictions.csv
```

One airport per invocation — the surface model is only meaningful within a
single aerodrome, and running them separately parallelises for free.

## Validation split

Train on 2025 minus January and July; validate on January and July 2025. The
test set is January and July 2026, and taxi-out has a strong seasonal signal, so
any other split flatters the model.

## Key dates

Submissions close **11 October 2026, 23:59:59 CET**.
