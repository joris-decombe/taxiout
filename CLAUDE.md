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
| Arrival rows, with their `BLOCK_TIME` (in-block) | present | **present** |

Consequences worth holding onto, because each one has already been got wrong
once in this repo's history:

- **`BLOCK_TIME` is the leak.** Any feature derived from it is unbuildable at
  predict time. `MVT_TIME` is fine and is the natural clock for departures.
- The task is reconstructing the *off-block* time from the takeoff time, not
  the reverse. Anything phrased the other way round is confused.
- **`AOBT_3_flt` is not `BLOCK_TIME`.** The Network Manager's off-block time
  survives on the ranking set and is a genuinely different quantity. The two
  agree within a minute only 21% of the time, sd of the difference 384s.
  `MVT_TIME - AOBT` alone scores 385s RMSE against a 417s sd on the rows that
  have it: a real edge, not a dramatic one.
- **~1.5% of departures have no Network Manager record at all** (no AOBT, no
  callsign, no market segment) and those rows carry **about half of the
  squared error**: RMSE ~2,006s against ~241s for the rest (validation,
  29 September 2026). `train.py` fits them as a separate small model.
- **Half of LIRF's unmatched departures, and 18% of its matched ones, have
  `BLOCK_TIME == SCHED_TIME`** to the second, so their target is exactly
  `MVT_TIME - SCHED_TIME`, often hours. Both models are mixtures built on
  that (`p * gap + (1 - p) * normal`), worth 26s on the orphans and 17s on
  the matched group. Predictions above the schedule gap are not a bug: past
  12h of gap, about a third of those rows also carry a day-early off-block.
- **The ranking set is not departures only.** It carries every arrival too,
  in-block time intact, so stand occupancy, runway use and queue lengths are
  all computable around each departure (`context.py`). Row order and MVT_ID
  were checked and do not encode off-block times.
- **Do not clip predictions tightly.** Real taxi-out exceeds two hours often
  enough that clipping there cost 92s of RMSE in a measured run. `submit.py`
  caps at 172,800s purely as a runaway guard (a day plus a
  normal taxi is a legitimate LIRF prediction).
- Fit anything derived from the target (`unimpeded_taxi_reference` is a low
  quantile of it) on the training half only, never on all of `training`.

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

Build a verified submission file locally (fits on all twelve months; it does
not upload, `bucket.upload_submission(path, bucket.next_version())` does):

```
PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage   .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'pipeline');   from pathlib import Path; from taxiout import submit;   print(submit.build_submission(Path('data/submission_local.parquet')))"
```

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
configure fetches doctest v2.4.12 over the network.

CMake installs to `C:\Program Files\CMake\bin`, which is on the
machine PATH but missing from shells started before it was installed.
The MSVC generator is multi-config, so `--config Debug` is needed on the
build and `-C Debug` on `ctest`. All 11 tests pass.

There is no Python test suite.

## Domain knowledge

`.claude/skills/air-traffic-controller/` holds the operational background:
what each timestamp means (APDF and A-CDM definitions), EUROCONTROL's
reference taxi-time method, separation, de-icing, and a section per airport
with measured runway use and the 2025-to-2026 differences. Its
`scripts/airport_profile.py` regenerates the measured profiles.

## Open work

[TODO.md](TODO.md) is the working list: what is blocked, what is worth
attacking next, and what is merely housekeeping. Update it there rather than
scattering status across docstrings.

## Architecture

Two tracks that are meant to meet, but currently do not.

**`pipeline/`** is the working path: parquet → features → LightGBM → submission.
`data.py` scans `data/training_*.parquet` and splits January and July out for
validation, matching the leaderboard's months because taxi-out is strongly
seasonal; `train.validate(training, months)` holds out any other pair, which
the folds in `build_final.py` use. `features.py` builds a per-stand/runway unimpeded reference (a low
quantile, to approximate geometry without queueing) plus calendar, schedule and
rolling congestion features, and `surroundings()` adds the dataset-wide ones:
`context.py` (stand occupancy, queue counts, runway configuration, from all
movements including arrivals) and `weather.py` (METARs at takeoff).
`surroundings()` must see one dataset's full movement table, so training and
ranking each build their own. `train.py` fits two mixtures (matched and
orphan groups, the orphan one on a smaller feature set) and `validate()`
returns held-out predictions for experiments. `submit.py` refuses to write a
partial file rather than shipping zeros for missing rows, and
`verify_submission` re-reads the written file and checks it by keyed join.

Weather comes from the Iowa Environmental Mesonet METAR archive into
`data/weather/`, once: `weather.fetch()`. The archive rate-limits with HTTP
429, which `fetch` retries.

ADS-B comes from the adsb.lol archive (ODbL): `adsb.fetch_day` streams a
day's GitHub release (3 to 4 GB) and keeps the points near the airports,
`adsb.observe_day` dates each departure's push-back where the aircraft is
seen parked and then moving. `correct.py` fits a corrector of the model's
error on those observations, from held-out predictions for every month of
2025 (six folds, `build_final.py` FOLDS), and `pipeline/build_final.py`
runs the whole chain. Never commit anything under
`data/external/`: it is derived from ODbL data and keyed to challenge rows.

`bucket.py` talks to the object store: the challenge data is in the shared
`prc-2026-datasets` bucket, submissions go to `prc-2026-gentle-octopus` under a
filename pattern the scorer silently requires. The S3 API is at
`https://s3.opensky-network.org`. The console URL in the provisioning email is
a web UI, not an endpoint, and it is MinIO, so path-style addressing is
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
queueing rather than transit, but that case has not been made or acted on, and
its output is not yet wired into `features.py` as a feature.

`pipeline/taxiout/synthetic.py` generates stand-in movements for the
simulator. `sim/` stays in the repo as documented exploration: the submitted
model does not use it.

## Competition rules

Scored on **RMSE in seconds**, which is why `train.py` optimises RMSE,
confirmed on the challenge site rather than assumed. The leaderboard scores
January and July 2026. **The final ranking** (announced 8 October 2026) is
one submission, `gentle-octopus_final.parquet` over
`final_submitting.parquet`, covering January, February, June and July 2026,
ranked per pair of months and combined, with a review of code and docs.
It exists to reward generalisation, so nothing should be tuned to January
and July alone. `data.RANKING_FILE` and `data.SUBMISSION_TEMPLATE` point at
the final files; `data.LEADERBOARD_TEMPLATE` keeps the January and July one.
The final file is uploaded once, and only when the user says so.
RMSE is dominated by the worst predictions, so the target's -12s..87,177s
tail matters more than its bulk.

The rules require the final solution on a **public** GitHub repo under
**GPLv3** (`LICENSE` holds the text), with reproducible documentation and any
external data openly licensed. The repo has been public since 6 October 2026
(history audited first: no parquet, no credential), so anything committed is
published: keep data, credentials and scores-by-row out of it.

Solutions must be **original**: reusing another implementation needs its
authors' permission and significant changes. Other teams publish their
repos during the challenge. Read their write-ups for ideas if useful, never
their code, verify any idea on our own data, and credit it in the README's
prior-work section.

The ranking page also states that submissions are monitored for "attempts
to learn from or exploit the ranking process", which the organisers
consider unfair. Upload only genuine candidates for best model, never
variants built to measure a slice or a single change on the test set.

**Submission rules** are on the challenge site's ranking page
(<https://prc-data-challenge-2026.netlify.app/ranking.html>), not the overview
or eligibility pages: teams are ranked on their **best** RMSE across all
uploads, the limit is **5 uploads per day** and 1 GB per bucket, and the
scorer rejects a file with a mismatched, missing or extra `MVT_ID_mvt`. An
upload therefore costs a daily slot and nothing else, but it is still an
outward-facing action: upload only when the user asks. The scorer writes
`<name>_result.json` (with `"score"`) beside the upload within a minute or so.
The public leaderboard is JSON at
`https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`,
paginated by `cursor=<nextCursor>`. v1 scored 370.9s on the test set against
384s on validation, so validation runs slightly pessimistic. The leader was
at 220.7s on 29 September 2026.

These come from the provisioning email and are reproduced in `README.md`:
team `gentle-octopus`, submissions named `gentle-octopus_v<N>.parquet` into
`prc-2026-gentle-octopus`, deadline 14 October 2026 10:00 UTC (extended from
11 October with the final phase). A submission
whose filename does not match the pattern is dropped with no result file and no
error, which is why `bucket.submission_name` centralises it. Derive N from
`bucket.next_version()` rather than locally: overwriting a submission loses its
result file.
