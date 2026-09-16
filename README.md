# taxiout

PRC Data Challenge 2026 — predicting taxi-out time at 10 major European
airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.

Two tracks that meet in the middle:

- **`pipeline/`** — Python. Loads the challenge parquet files, builds features,
  trains a LightGBM baseline, writes `submitting.parquet`. This is the yardstick.
- **`sim/`** — C++20. An event-driven model of the departure surface: pushback,
  apron transit, runway queue with wake separation, arrival preemption. This is
  the part that was meant to beat the yardstick.

## Why simulate at all

Taxi-out decomposes into unimpeded transit plus queue delay, and the queue
delay is where all the variance lives.

The scaffold assumed takeoff times were blanked on the ranking set, which would
have made congestion features unmeasurable and the simulator the only way to
get them. **That is backwards.** The real ranking set blanks
`BLOCK_TIME_UTC_mvt` and the target, and keeps `MVT_TIME_UTC_mvt`. Since the
target is exactly `MVT_TIME - BLOCK_TIME`, the two had to be blanked together —
the task is reconstructing the *off-block* time from the takeoff time, not the
reverse.

So congestion features around takeoff are directly computable and need no
simulator. What the simulator can still offer is the counterfactual: how much
of the gap between pushback and takeoff was queueing rather than transit. Its
input assumptions need revisiting first, since it was written to consume
off-block times that the ranking set does not provide.

The strongest single feature is `MVT_TIME - AOBT_3_flt`: the Network Manager's
off-block time is *not* blanked, and differs from the movement table's by a
standard deviation of 374s. On its own it predicts the target at 377s RMSE
against a target sd of 605s.

## Status

Full 2025 dataset local: 12 monthly training files, `ranking.parquet`,
`submitting.parquet`.

| | |
|---|---|
| Python baseline | **471s validation RMSE**, honest features only |
| Bar to beat | 377s — `MVT_TIME - AOBT` with no model at all |
| Target spread | 605s standard deviation |
| C++ simulator | builds, 11/11 tests pass, premise needs the rethink above |

The simulator's `queue_delay_sec` is not yet wired into `features.py`, so the
two tracks do not actually meet yet.

Known data-quality issues not yet handled: the target runs from -12s to
87,177s, so both tails need a decision before they distort the loss.

## Getting the data

The console login is interactive SSO, so the first credential is a manual step:
sign in, generate a service-account key pair, and save the JSON it gives you
outside this repo (`~/.opensky/object_store_creds.json` is where `bucket.py`
looks). Everything after that is scripted.

```
python -c "from taxiout import bucket; print(bucket.pull_dataset())"
```

The challenge data lives in the shared `prc-2026-datasets` bucket;
`prc-2026-gentle-octopus` is ours and holds only what we upload.
`bucket.upload_submission` pushes under the mandated
`gentle-octopus_v<N>.parquet` name, and `bucket.next_version()` reads the
bucket to pick N so a previous submission's result file is never overwritten.

The S3 API is at `https://s3.opensky-network.org` — the console URL below is a
web UI, not an endpoint. The store is MinIO, so path-style addressing is
required.

## Build

```
cmake -S sim -B sim/build
cmake --build sim/build --config Debug
ctest --test-dir sim/build -C Debug
```

Needs a C++20 compiler and CMake 3.20+. Tests pull doctest via FetchContent, so
the first configure needs network access; the pin is v2.4.12, because v2.4.11
declares a `cmake_minimum_required` that CMake 4.x refuses. `--config`/`-C` are
only needed on multi-config generators such as MSVC.

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

Note that the split keys on off-block month, which is blank on the ranking set —
that is fine, because the split only ever runs over training data.

## Team and submission rules

From the provisioning email (OpenSky Network, 4 September 2026). These are the
competition's rules, not our conventions — getting any of them wrong means the
submission is silently ignored.

| | |
|---|---|
| Team name | `gentle-octopus` |
| Submission bucket | `prc-2026-gentle-octopus` |
| Submission filename | `gentle-octopus_v<N>.parquet`, N being the version number |
| Console | <https://s3-console.opensky-network.org> |
| Deadline | 11 October 2026, 23:59:59 CET |

**Logging in.** The console's default form will not accept OpenSky credentials.
Click *Other Authentication Methods*, choose *Login with SSO*, and authenticate
against the Keycloak IAM that it redirects to.

**Getting a score.** Upload to the team bucket; a result file appears in the
same bucket shortly afterwards if the submission parsed. No result file means
the submission was rejected — check the filename against the pattern above
first, since that is the easiest thing to get wrong.

**Contact.** Discord <https://discord.gg/RPh89jpVVz>, or
<challenge@opensky-network.org>.

## Key dates

Submissions close **11 October 2026, 23:59:59 CET**.
