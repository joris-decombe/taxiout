# TODO

Submissions close **11 October 2026, 23:59:59 CET**. Nothing has been submitted;
the team bucket is empty.

Current model: **409s validation RMSE** on January + July 2025, against a 686s
standard deviation for those months.

## Blocked

- [ ] **Confirm the submission rules with the organisers.** Neither the
      challenge site nor the eligibility page states how many submissions a team
      may make, whether there is a daily cap, or whether the final ranking takes
      the best upload or the most recent one. "Best" and "latest" imply
      completely different endgames: under "latest", a careless final upload
      undoes everything. Ask on the Discord or at
      challenge@opensky-network.org. **Everything under "Submitting" waits on
      this.**

## Model, in priority order

**Measure improvements with a paired bootstrap, not a single RMSE.** The
unmatched group has ~5,400 validation rows and a target sd near 4,000s, and
one row carries 22% of its squared error, five rows carry 52%. An
independent 95% CI on its RMSE is 1,276s wide, so a lone score from that
group means almost nothing. Resampling the validation set once and scoring
both models on the same resample cancels most of that and can resolve
differences under 10s. `pipeline/experiments_paired_bootstrap.py` is the
harness.

- [ ] **The unmatched group, continued.** 1.5% of departures, still the
      majority of the error. What is known: their long flights concentrate
      hard at LIRF (50% of its unmatched departures exceed an hour, against
      a 4.84% base rate) and on runways 25 and 34L; 69% of all long ones
      have a null aircraft type; two days in February 2025 hold 15% of them.
      Untested: whether LIRF deserves its own treatment, whether the
      February days are an outage worth excluding from training, and whether
      a quantile or two-stage model beats least squares on a distribution
      this skewed.
- [ ] **Congestion features carry almost nothing** (1.2% of gain combined),
      which is odd for a problem whose physics is queueing. Either the
      windows are wrong, the counts too coarse, or `unimpeded_taxi_sec`
      already absorbs the signal.
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
- [ ] **Submit v1** and record what the result file contains.

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
      right, not just good practice. Concretely: the scripts that generate
      `report/findings.json` and the published figures live outside the repo,
      so neither page can currently be regenerated from a clean checkout.
- [ ] **Delete `pipeline/taxiout/fixtures.py`.** Scaffolding from before the
      real data arrived.
- [ ] **Confirm the "Run the simulator on synthetic movements" README section.**
      The two commands were verified once end to end; the section predates that.
- [ ] Optional: the open-access paper in the Journal of Open Aviation Science
      that the rules encourage. `Taxi-Out, Measured` is the draft material.
