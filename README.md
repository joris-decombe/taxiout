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

## Getting the data

The console login is interactive SSO, so the first download is a manual
step: sign in, then either save the parquets into `data/` by hand or
generate an access key pair and use `taxiout.bucket`.

```
export TAXIOUT_S3_ENDPOINT=...   # S3 API endpoint, not the console URL
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

`bucket.pull_dataset` then fetches the parquets, and `bucket.upload_submission`
pushes one under the mandated `gentle-octopus_v<N>.parquet` name --
`bucket.next_version()` reads the bucket to pick N so a previous
submission's result file is never overwritten.

## Working before the data arrives

`taxiout.fixtures` writes parquets shaped like the real ones, so the
pipeline can be run end to end today:

```
python -m taxiout.fixtures data
python -c "from taxiout import data, train; print(train.train(data.load_training()).validation_rmse)"
```

The RMSE this produces is meaningless -- the fixtures are random. What it
checks is that the code path from parquet to trained model has no errors
in it, which is worth knowing before the real files land rather than after.
Delete the module once they do.

## Validation split

Train on 2025 minus January and July; validate on January and July 2025. The
test set is January and July 2026, and taxi-out has a strong seasonal signal, so
any other split flatters the model.

## Team and submission rules

From the provisioning email (OpenSky Network, 4 September 2026). These are
the competition's rules, not our conventions -- getting any of them wrong
means the submission is silently ignored.

| | |
|---|---|
| Team name | `gentle-octopus` |
| Submission bucket | `prc-2026-gentle-octopus` |
| Submission filename | `gentle-octopus_v<N>.parquet`, N being the version number |
| Console | <https://s3-console.opensky-network.org> |
| Deadline | 11 October 2026, 23:59:59 CET |

**Logging in.** The console's default form will not accept OpenSky
credentials. Click *Other Authentication Methods*, choose *Login with SSO*,
and authenticate against the Keycloak IAM that it redirects to.

**Getting a score.** Upload to the team bucket; a result file appears in the
same bucket shortly afterwards if the submission parsed. No result file
means the submission was rejected -- check the filename against the pattern
above first, since that is the easiest thing to get wrong.

**Contact.** Discord <https://discord.gg/RPh89jpVVz>, or
<challenge@opensky-network.org>.

## Key dates

Submissions close **11 October 2026, 23:59:59 CET**.
