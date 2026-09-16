# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The one thing to get right

The target is `TAXITIME_SEC_mvt`, and it equals `MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt`
**exactly**, on 100% of departure rows. So the ranking set blanks both of the
right-hand columns and nothing else:

| Column | Training | Ranking (departures) |
|---|---|---|
| `MVT_TIME_UTC_mvt` (takeoff) | present | **present** |
| `BLOCK_TIME_UTC_mvt` (off-block) | present | **100% null** |
| `TAXITIME_SEC_mvt` (target) | present | 100% null |
| `SCHED_TIME`, `RUNWAY`, `STAND`, flight-table columns | present | present |

Consequences worth holding onto, because each one has already been got wrong
once in this repo's history:

- **`BLOCK_TIME` is the leak.** Any feature derived from it is unbuildable at
  predict time. `MVT_TIME` is fine and is the natural clock for departures.
- The task is reconstructing the *off-block* time from the takeoff time, not
  the reverse. Anything phrased the other way round is confused.
- **`AOBT_3_flt` is not `BLOCK_TIME`.** The Network Manager's off-block time
  survives on the ranking set and is a genuinely different quantity — they
  agree within a minute only 21% of the time, sd of the difference 374s.
  `MVT_TIME - AOBT` alone scores 377s RMSE against a target sd of 605s. That is
  the number any change has to beat, not zero.
- Fit anything derived from the target — `unimpeded_taxi_reference` is a low
  quantile of it — on the training half only, never on all of `training`.

`pipeline/taxiout/schema.py` carries this contract in its docstring. Update it
there when something new is learned about the data.

## Commands

The venv is at `.venv/` and is not on PATH; call its interpreter directly.
`data/` is gitignored and holds the real challenge parquets.

```
.venv/Scripts/python.exe -m pip install -r pipeline/requirements.txt
```

Train the baseline (run from the repo root; `pipeline` must be on the path):

```
PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
  .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'pipeline'); \
  from taxiout import data, train; print(train.train(data.load_training()).validation_rmse)"
```

Both environment variables matter in practice: the parquets contain an
`arrow.r.vctrs` extension type that warns on every scan, and polars' table
output is not cp1252-encodable, so printing a DataFrame on Windows raises
`UnicodeEncodeError` without the first.

Pull data or push a submission (credentials are read from
`~/.opensky/object_store_creds.json`, never from the repo):

```
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'pipeline'); \
  from taxiout import bucket; print(bucket.pull_dataset())"
```

Build and test the simulator:

```
cmake -S sim -B sim/build
cmake --build sim/build
ctest --test-dir sim/build
```

`ctest` runs doctest cases individually, so `-R <name>` selects one. The first
configure fetches doctest v2.4.11 over the network.

CMake is not on this shell's PATH; prepend `C:\Program Files\CMakein`.
The MSVC generator is multi-config, so `--config Debug` is needed on the
build and `-C Debug` on `ctest`.
There is no Python test suite.

## Architecture

Two tracks that are meant to meet, but currently do not.

**`pipeline/`** is the working path: parquet → features → LightGBM → submission.
`data.py` scans `data/training_*.parquet` and splits January and July out for
validation, matching the test set's months because taxi-out is strongly
seasonal. `features.py` builds a per-stand/runway unimpeded reference (a low
quantile, to approximate geometry without queueing) plus calendar, schedule and
rolling congestion features. `train.py` is the yardstick. `submit.py` refuses to
write a partial file rather than shipping zeros for missing rows.

`bucket.py` talks to the object store: the challenge data is in the shared
`prc-2026-datasets` bucket, submissions go to `prc-2026-gentle-octopus` under a
filename pattern the scorer silently requires. The S3 API is at
`https://s3.opensky-network.org` — the console URL in the provisioning email is
a web UI, not an endpoint — and it is MinIO, so path-style addressing is
required.

**`sim/`** is an event-driven C++20 model of the departure surface: aircraft
push back, cross the apron, join a FIFO runway queue subject to wake
separation, and arrivals preempt them. `Simulator::run` is a single sorted pass
over the merged movement stream, with the sort standing in for an event queue.

**The simulator's premise is stale.** It was written believing takeoff times
were blanked on the ranking set, which would have made congestion features
measurable only by reconstruction. The opposite is true, so congestion around
takeoff is directly computable and the sim is not needed for it. It also
consumes `off_block_time` as an input, which the ranking set does not provide.
Its remaining case is estimating how much of the pushback-to-takeoff gap was
queueing rather than transit — but that case has not been made or acted on, and
its output is not yet wired into `features.py` as a feature.

`pipeline/taxiout/fixtures.py` and `synthetic.py` generate stand-in data for the
two tracks. `fixtures.py` exists because the real files were unavailable for the
first part of this project; it is scaffolding to delete, not a maintained test
double.

## Competition rules

These come from the provisioning email and are reproduced in `README.md`:
team `gentle-octopus`, submissions named `gentle-octopus_v<N>.parquet` into
`prc-2026-gentle-octopus`, deadline 11 October 2026 23:59:59 CET. A submission
whose filename does not match the pattern is dropped with no result file and no
error, which is why `bucket.submission_name` centralises it. Derive N from
`bucket.next_version()` rather than locally — overwriting a submission loses its
result file.
