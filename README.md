# taxiout

PRC Data Challenge 2026: predicting taxi-out time at 10 major European
airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.

Two tracks:

- **`pipeline/`**. Python. The submitted model: loads the challenge parquet
  files and the open data (METARs, adsb.lol ADS-B), builds features, fits
  LightGBM and CatBoost mixtures, corrects them from ADS-B ground
  observations, and writes a verified submission (`pipeline/build_final.py`).
- **`sim/`**. C++20. An event-driven model of the departure surface: pushback,
  apron transit, runway queue with wake separation, arrival preemption. It
  was meant to supply queueing estimates; the queue turned out to be
  readable directly from the data, and the submitted model does not use it.

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

This repo has been **public** since 6 October 2026. Its history was checked
first and has never contained a parquet or a credential.

**Ranking takes each team's best submission**, and the limit is 5 uploads a
day (1 GB per bucket). Both are on the site's
[ranking page](https://prc-data-challenge-2026.netlify.app/ranking.html),
along with a public leaderboard.

Teams from sanctioned countries are excluded; the full terms are on the
eligibility page linked from the challenge site.

Source: <https://ansperformance.eu/study/data-challenge/dc2026/> (redirects
to the 2026 site). Note the announcement email says 11 airports and the
site says both 10 and 11 in different places; the data itself has 10, which
is what this repo goes by. The submission format is `submitting.parquet`
with `TAXITIME_SEC_mvt` filled in for every `MVT_ID_mvt`, as the ranking page
describes.

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

| | |
|---|---|
| Best upload | **v10: 274.1s** on the test set, rank 68 of 237 (8 October 2026; leader 213.5s, median 295.9s) |
| Validation (Jan + Jul 2025) | **308.6s**, 316.7s before the ADS-B correction |
| Target sd, those months | 686s, the two hardest months of the year |
| Uploads | v1 370.9s, v5 359.0s, v6 311.3s, v7 287.2s, v8 287.0s, v9 275.0s, v10 274.1s; see [TODO.md](TODO.md) |

### How the model works

Take-off time survives in the scored data and off-block time does not, so
predicting taxi-out means reconstructing when the aircraft left its stand.
Three other clocks bear on that, and the model is built around them:

- **The Network Manager's off-block (AOBT).** The matched regressor starts
  from take-off minus AOBT and learns only the deviation from it.
- **The flight plan's last off-block (LOBT).** The airport's off-block lies
  within ±3,606s of it on every 2025 departure that has one, so
  predictions are projected into that window.
- **The schedule.** Some airports, Rome above all, record the scheduled
  time as the off-block. Each group is a mixture: `p × (take-off −
  schedule) + (1 − p) × normal taxi`. At Rome, departures with no flight
  record follow two clean rules from the 2025 records: past six hours late
  the at-schedule share is used directly, and 14 to 26 hours late the
  alternative to the schedule is a day plus a normal taxi.

About 1.5% of departures have no Network Manager record at all and carry
about half of the squared error. They get their own smaller model; outside
Rome their predictions are capped at an hour and pulled towards their
airport and lateness band's mean, because in 2025 they taxied normally
however late they left. "Normally" means like the flights around them:
when the matched departures at their airport are taking over 25 minutes,
as on Amsterdam's de-icing days in January 2026, they are predicted at
least 0.95 times that level.

The regressors are LightGBM and CatBoost blends (three LightGBM seeds, a
LightGBM per airport, a CatBoost twin for each group, and a CatBoost
classifier beside LightGBM's for the matched group). Finally, a small
corrector learns the model's error from what adsb.lol's ADS-B receivers saw
on the ground, the push-back itself for about one 2026 departure in five,
fitted on the validation days only.

[RESEARCH.md](RESEARCH.md) explains why each of these works and records what
was tried and rejected. **[Taxi-Out, Measured](https://joris-decombe.github.io/taxiout/)**
is the long-form account for readers with no prior knowledge of the dataset
or of aviation, and the basis for the open-access write-up the rules
encourage. Open work is in [TODO.md](TODO.md).

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

### External data

- **METARs** from the Iowa Environmental Mesonet ASOS archive
  (<https://mesonet.agron.iastate.edu/request/download.phtml>), fetched once
  by `weather.fetch()` into `data/weather/`.
- **ADS-B traces** from the adsb.lol global history archive
  (<https://github.com/adsblol/globe_history_2025>,
  <https://github.com/adsblol/globe_history_2026>), available under the
  [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/),
  with feeder data under CC0. `taxiout.adsb` streams each day of January and
  July 2025 and 2026 (about 390 GB read in all, nothing kept whole) and keeps
  only the points near the ten airports (about 3 GB) in
  `data/external/adsblol/`. Those derived tables are not committed: anything
  derived from adsb.lol that is published must carry the ODbL, and they are
  keyed to challenge rows. The code in this repo is the method that rebuilds
  them. Contains information from adsb.lol, which is made available under
  the ODbL.

### The final submission, end to end

```
PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
  .venv/Scripts/python.exe pipeline/build_final.py
```

streams and observes the ADS-B days, fits the model on ten months for
honest January and July 2025 predictions, fits the ADS-B corrector
(`taxiout.correct`) on them, refits on all twelve months, and writes and
verifies `data/submission_final.parquet`. About four hours of compute plus
two of streaming the first time.

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

## Prior work

Everything in `pipeline/` is written here. Ideas that came from elsewhere
are listed below, each reimplemented from its description and checked on
the 2025 data before use; no code was taken from any of them.

**Literature.** [RESEARCH.md](RESEARCH.md) surveys it: why the mixture,
the LOBT projection and the anchor work, and which models to try next.

- I. Simaiakis and H. Balakrishnan, "A queuing model of the airport
  departure process", *Transportation Science*, 2016
  ([PDF](https://www.mit.edu/~hamsa/pubs/SimaiakisBalakrishnan_TS2014.pdf)):
  taxi-out as unimpeded time plus runway queue plus surface congestion, and
  the number of take-offs between push-back and take-off as the queue
  measure, which `context.py`'s `*_deps_since_aobt` counts also measure.
- EUROCONTROL's additional taxi-out time indicator: the unimpeded reference
  per stand and runway (`features.unimpeded_taxi_reference`).

**Other teams' public write-ups.** Read for ideas (READMEs only, never
their source), each idea verified on our data before use:

- **The LOBT window**, from
  [EnioAguiar/prc-taxiout-2026](https://github.com/EnioAguiar/prc-taxiout-2026):
  `BLOCK_TIME` lies within ±3,606s of `LOBT_flt` on every 2025 departure
  that has one. Confirmed here on the same 2,062,577 rows; `train.py`
  projects matched predictions into that window (`LOBT_WINDOW`).
- **Linear leaves** (LightGBM's `linear_tree`), reported as a large gain by
  [radekacar/joyous-rainbow](https://github.com/radekacar/joyous-rainbow).
  Tried and rejected: on this model a few rows extrapolate to ±300,000s
  (`experiments_round3.py`).
- **ADS-B ground tracks from adsb.lol** as a source of off-block times, from
  the same EnioAguiar README, which reports a large gain from them. The
  extraction and the corrector here (`taxiout.adsb`, `taxiout.correct`)
  were written from scratch and measured on our own validation days.
- **LIRF's day-shifted orphans** were found here in the 2025 data
  (`train.DAY_SHIFT`). The README of
  [skylinkapi/prc-data-challenge-2026-kind-mango](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango)
  describes a similar 24-hour fallback for LIRF.
