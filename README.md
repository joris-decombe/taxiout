# taxiout

PRC Data Challenge 2026: predicting taxi-out time at 10 major European
airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.

Two tracks that meet in the middle:

- **`pipeline/`**. Python. Loads the challenge parquet files, builds features,
  trains a LightGBM baseline, writes `submitting.parquet`. This is the yardstick.
- **`sim/`**. C++20. An event-driven model of the departure surface: pushback,
  apron transit, runway queue with wake separation, arrival preemption. This is
  the part that was meant to beat the yardstick.

## What the competition is

The PRC Data Challenge is run annually by EUROCONTROL's Performance Review
Commission with the OpenSky Network. The 2026 edition asks for the most
accurate prediction of **taxi-out time**, the seconds between a departing
flight leaving its stand and getting airborne, at 10 major European hubs.

The stated motivation is that taxi-out is hard to predict and worth
predicting: it identifies periods of constrained airport operations, and
the excess is directly convertible into fuel burn and CO2.

| | |
|---|---|
| Metric | **RMSE**, in seconds |
| Ranked on | January and July 2026 movements |
| Training data | all movements at the 10 airports, full year 2025 |
| Prize | EUR 5,000 shared between the top three teams |
| Open | 1 September to 11 October 2026, 23:59:59 CET |

RMSE is worth taking literally: it is dominated by the worst predictions,
so the long tail in the target (see Status) matters more than its bulk.

**Entries must be open-sourced.** The final solution goes on a *public* GitHub
repo under **GNU GPLv3** (`LICENSE` holds the text), with enough documentation
to reproduce the results and any external dataset openly licensed. The
organisers fork the repo for administration. Failing any of those makes an
entry ineligible for a prize, though not for scoring. An open-access paper in
the Journal of Open Aviation Science is encouraged.

This repo is currently **private**, which suits work in progress and does not
suit the deadline. The history has been checked and has never contained a
parquet or a credential, so the flip is safe whenever it happens.

**Submissions are on hold.** Neither the challenge site nor the eligibility
page documents how many submissions a team may make, whether there is a daily
cap, or whether the final ranking takes the best submission or the most recent
one. Nothing is uploaded until that is confirmed with the organisers, because
the cost of a bad submission cannot be judged without it. Ask via the Discord
or challenge@opensky-network.org.

Teams from sanctioned countries are excluded; the full terms are on the
eligibility page linked from the challenge site.

Source: <https://ansperformance.eu/study/data-challenge/dc2026/> (redirects
to the 2026 site). Note the announcement email says 11 airports and the
site says both 10 and 11 in different places; the data itself has 10, which
is what this repo goes by. The site does not document the submission file
format; that comes from `submitting.parquet` and the provisioning email.

## Why simulate at all

Taxi-out decomposes into unimpeded transit plus queue delay, and the queue
delay is where all the variance lives.

The scored set blanks `BLOCK_TIME_UTC_mvt` and the target, and keeps
`MVT_TIME_UTC_mvt`. Since the target is exactly `MVT_TIME - BLOCK_TIME`, the
two had to be blanked together. The prediction therefore runs from the
takeoff time back to the off-block time, not forwards.

That makes congestion features around takeoff directly computable, so the
simulator is not needed for them. What it can still offer is the
counterfactual: how much of the gap between pushback and takeoff was
queueing rather than transit. Its input assumptions need revisiting first,
since it consumes off-block times that the scored set does not provide.

`MVT_TIME - AOBT_3_flt` is the strongest single feature. The Network
Manager's off-block time is not blanked, and differs from the movement
table's by a standard deviation of 384s. On the rows that have it, that
difference alone predicts the target at 385s RMSE against a 417s standard
deviation: a real edge, but a modest one.

The structure that matters more is in Status.
## Status

Full 2025 dataset local: 12 monthly training files, `ranking.parquet`,
`submitting.parquet`. Nothing submitted yet; the bucket is empty.

| | |
|---|---|
| Python baseline | **416s validation RMSE** (Jan + Jul 2025) |
| Target sd, those months | 686s, the two hardest months of the year |
| Target sd, full year | 546s |
| C++ simulator | builds, 11/11 tests pass, premise needs the rethink above |

### The thing that dominates everything

About 1.5% of departures fail to join to a Network Manager flight record:
no AOBT, and with it no callsign, no market segment, no flight rule. It is
one join failing, not four independent gaps.

Those rows carry **62% of the squared error**. Their RMSE is ~2,970s against
~293s for everything else. Their target distribution is a different
animal: sd ~3,960s against ~476s, and 98.6% of all departures over six
hours live there. Fitting them as a separate, much smaller model is worth
55s of RMSE on its own (471s → 416s).

Two plausible alternatives were measured and rejected: substituting the
group's mean instead of modelling it (574s), and explicit missingness flags
in a single model (475s).

Within the group there is a recording artifact. About half of LIRF's
unmatched departures have an off-block time equal to the scheduled time to
the second, so their taxi-out is exactly takeoff minus schedule, routinely
several hours. The orphan model is a mixture: a classifier for that
artifact, a regressor for a normal taxi, combined as
`p × (takeoff − schedule) + (1 − p) × normal`. That takes validation RMSE
from 409s to 384s (paired-bootstrap 95% CI −46s to −11s).

**Taxi-Out, Measured** (<https://claude.ai/artifact/Ez8LT8SdgrUbdp8oAeMq1i>)
is the long-form account: what taxi-out is and why it is worth predicting,
why a squared metric changes the question, which timestamps the scored data
keeps, the error decomposition, the at-schedule recording artifact, and the
eight strategies tried.
Written to be read with no prior knowledge of the dataset or of aviation,
and intended as the basis for the open-access write-up the rules encourage.

Open work is tracked in [TODO.md](TODO.md). The short version: the unmatched
group is 62% of the score and everything else is noise next to it, submissions
are blocked on a rules question, and the simulator's `queue_delay_sec` is not
yet wired into `features.py`, so the two tracks do not actually meet.
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

The S3 API is at `https://s3.opensky-network.org`. The console URL below is a
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

One airport per invocation: the surface model is only meaningful within a
single aerodrome, and running them separately parallelises for free.

### Looking at the result

<https://claude.ai/artifact/V6pWhi5jVwR7V4ZeotiFSt> reads both CSVs and draws
the run: each departure as a bar from pushback to wheels-up, split into apron
transit and queue delay, with arrivals marked on the runway lane. Load the two
files with the pickers at the top; nothing is uploaded, it parses in the page.
It ships with a sample run so it is not an empty shell on first open.

## Validation split

Train on 2025 minus January and July; validate on January and July 2025. The
test set is January and July 2026, and taxi-out has a strong seasonal signal, so
any other split flatters the model.

Note that the split keys on off-block month, which is blank on the ranking set.
That is fine, because the split only ever runs over training data.

## Team and submission rules

From the provisioning email (OpenSky Network, 4 September 2026). These are the
competition's rules, not our conventions. Getting any of them wrong means the
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
the submission was rejected. Check the filename against the pattern above
first, since that is the easiest thing to get wrong.

**Contact.** Discord <https://discord.gg/RPh89jpVVz>, or
<challenge@opensky-network.org>.

## Key dates

Submissions close **11 October 2026, 23:59:59 CET**.
