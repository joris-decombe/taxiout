# TODO

Submissions close **11 October 2026, 23:59:59 CET**. Ranking takes each
team's **best** score, at up to 5 uploads a day, so an upload never costs
anything but a slot.

Current model: **346.1s validation RMSE** on January + July 2025 (240.7s on
the rows with a Network Manager record, 2,006s on the rest), against a 686s
standard deviation for those months. `data/submission_v2.parquet` is built
from it and verified, not yet uploaded.

On the test set, v1 (the 384s-validation model) scored **370.9s**, rank 139
of 200 on 29 September 2026. The leader was at 220.7s, the 10th team at
237.0s, the median team at 299.2s.

## Model, in priority order

**Measure improvements with a paired bootstrap, not a single RMSE.** The
unmatched group has ~5,400 validation rows and a target sd near 4,000s, and
one row carries 22% of its squared error, five rows carry 52%. An
independent 95% CI on its RMSE is 1,276s wide, so a lone score from that
group means almost nothing. Resampling the validation set once and scoring
both models on the same resample cancels most of that and can resolve
differences under 10s. `pipeline/experiments_paired_bootstrap.py` is the
harness.

- [x] **LIRF's at-schedule artifact.** Half of LIRF's unmatched departures
      record off-block at the scheduled time to the second, so the target
      is `MVT - SCHED`. A mixture model for the orphan group (classifier for
      the artifact, regressor for a normal taxi) took validation from 409.4s
      to 383.6s, 95% CI -45.9s to -10.5s. `experiments_unmatched.py`.
- [ ] **The unmatched group, what is left.** Still 45% of squared error.
      Two LFPG easyJet rows with a day-early off-block (84,240s and 58,206s
      against a 30-minute schedule gap) carry 20% of *all* validation squared
      error on their own; nothing observable flags them. LIRF day-shift rows
      (target ~86,400s + a normal taxi) are the rest of the tail.
      Measured and rejected against the mixture: the airline as a raw
      categorical (+10s worse), a smoothed per-airline at-schedule rate in
      the classifier (-1.0s, CI -4.3s to +2.4s: nothing), and dropping the
      February 2025 days from training (+1.9s, CI +0.3s to +3.9s: worse).
      What remains looks like irreducible recording noise; the next gains
      are more likely in the matched 55% of the error.
- [x] **The surroundings, arrivals included.** The ranking set carries every
      arrival with its in-block time, which the pipeline used to drop.
      `context.py` builds stand, queue and runway-configuration features
      from all movements; `weather.py` adds METARs. With the flight-table
      fields the model had never used (operator, market segment, flight
      type, wake category, destination, EOBT and IOBT against takeoff), the
      matched group went from 292.3s to 273.4s before the mixture below.
      After it, context is worth 4.1s and weather 1.0s on the matched group.
      Both made the orphan group worse (2,005s to ~2,080s), so it keeps the
      smaller feature set. `experiments_context.py`.
- [x] **The at-schedule artifact in the matched group.** LIRF was 7.6% of
      matched departures and 42.5% of their squared error: 18% of them
      record off-block at the schedule. The mixture now covers both groups:
      matched 273.4s to 249.2s, LIRF matched 642s to 499s.
- [ ] **LIRF is still the worst airport** by a factor of two, and the
      classifier is at 0.87 AUC. The artifact rows are delayed flights; a
      LIRF-specific classifier, or features on how late the NM saw the
      aircraft leave, may separate them better.
- [ ] **Two LFPG rows** (the day-early easyJet off-blocks) keep LFPG's
      validation RMSE near 580s. Nothing observable flags them yet.
- [ ] **Tuning.** Parameters are unchanged since the first baseline apart from
      the round counts. `num_leaves`, `min_data_in_leaf` and the learning
      rate have not been searched, nor an ensemble of seeds.
## Submitting

- [x] **Pre-upload verification.** `submit.verify_submission` re-reads the
      written file and checks schema and dtypes against the template, a 1:1
      keyed join on every template ID, no null, non-finite or negative value,
      each written value equal to the prediction for that ID, and the median
      against the training target. A positionally scrambled copy of a good
      file fails it; the good file passes.
- [x] **End-to-end run on `ranking.parquet`.** `submit.build_submission`
      fits on all twelve months (`train.train_final`), predicts the 344,841
      departures and writes `data/submission_local.parquet`, which passes
      verification. About a minute to fit. None of the feared breakages
      happened. The target column is written as Int32 to match the template.
      Predictions: median 960s, 94 over 2h (97 expected at the training
      rate), max 72,502s. The largest are LIRF departures with no Network
      Manager record that took off 15-17h after schedule; the model reads that
      gap as taxi-out. Whether it is right to is the unmatched-group question.
- [x] **Submit v1.** `gentle-octopus_v1.parquet`, uploaded 29 September
      2026 from the 384s-validation model. The result file,
      `<name>_result.json` in the team bucket, reads `"status": "Succeeded"`,
      `"used_pairs": 344841`, `"score": 370.8992`. A `_persist.json` appears
      beside it. The scorer took the file in the template's row order.

## Data quality, unexplained

- [ ] **289 training departures have a negative taxi-out**, 241 of them at LSZH,
      minimum -12s. Not a model defect: the model reproduces them faithfully and
      the truth on those rows has a median of 8s. But the two timestamps
      disagree there and nothing explains why.
- [ ] **69 departures exceed six hours**, up to 131,167s. Day-boundary clock
      errors were the obvious candidate and were tested: subtracting 24h lands
      only 20% of them anywhere plausible, so that is not what they are.

## Simulator

`surface.py` bridges it to the parquet data: AOBT as the off-block clock,
one run per aerodrome-day, `queue_delay_sec` joined back by `MVT_ID`.
Measured over March 2025, 163,367 departures:

- Correlation with the real excess over geometry: **pearson +0.21**.
- It reconstructs **17% of the real excess** (41s simulated against 241s
  real). Departures it flags as queued averaged 292s of real excess against
  205s for the rest, so the signal is real but weak.
- EGLL is the exception: 195s simulated against 384s real, pearson +0.23.
  The model bites hardest where the airfield is genuinely at capacity.

- [ ] **Decide whether +0.21 earns a feature slot.** Add
      `sim_queue_delay_sec` to `FEATURE_COLUMNS` and measure the RMSE
      change. It is behind the unmatched group in priority.
- [ ] **If it stays, calibrate the separation matrix per airport.** The
      current values are ICAO defaults converted to time, and generating a
      sixth of the real delay suggests they are too permissive.
## Compliance and housekeeping

- [x] **GPLv3 licence.** `LICENSE` holds the canonical text.
- [ ] **Make the repo public** before the deadline. Prize eligibility requires
      it. The history has been audited: 28 paths ever added, no parquet, no
      credential, so the flip is safe.
- [ ] **Reproducible documentation.** An eligibility requirement in its own
      right, not just good practice. *Taxi-Out, Measured* is now covered:
      `pipeline/report_findings.py` rebuilds its data and page source in
      `report/`. The surface replay page and `synthetic_run.json` are not.
- [ ] **`experiments_paired_bootstrap.py` and `experiments_unmatched.py`
      call `features.build` and `train.fit` with their pre-surroundings
      signatures**, so they no longer run. They record measurements of an
      older model; port them or mark them historical.
- [ ] **Delete `pipeline/taxiout/fixtures.py`.** Scaffolding from before the
      real data arrived.
- [ ] **Confirm the "Run the simulator on synthetic movements" README section.**
      The two commands were verified once end to end; the section predates that.
- [ ] Optional: the open-access paper in the Journal of Open Aviation Science
      that the rules encourage. `Taxi-Out, Measured` is the draft material.
