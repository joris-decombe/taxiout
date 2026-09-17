# TODO

Submissions close **11 October 2026, 23:59:59 CET**. Nothing has been submitted;
the team bucket is empty.

Current model: **416s validation RMSE** on January + July 2025, against a 686s
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

## Submitting

- [ ] **Write the pre-upload verification.** The dangerous failure is silent:
      `submitting.parquet` and `ranking.parquet` hold the same 344,841 IDs in a
      different order, so a positional assignment instead of a keyed join
      produces a perfectly valid file that scores like noise, indistinguishable
      from a bad model. Check: every template ID matched through a keyed join,
      no NaN or infinity, prediction distribution comparable to the training
      target distribution, dtypes surviving the parquet round-trip.
- [ ] **Run the pipeline end to end on `ranking.parquet` once.** Never done.
      Candidates for breaking: predicting on a frame where `BLOCK_TIME` is 100%
      null, the `MVT_ID_mvt` Float64 dtype in the template join, and the
      partial-file guard in `submit.py` firing on a dropped row.
- [ ] **Submit v1** and record what the result file contains.

## Model

- [ ] **The unmatched group.** 1.5% of departures, 62% of the squared error,
      still around 2,700s RMSE. Their *median* taxi-out is normal (976s against
      923s), so the question is specifically what lengthens their tail. Untested
      ideas: whether missingness correlates with time of day or particular
      stands; whether the long ones are cargo or non-scheduled traffic; whether
      a quantile or two-stage model beats least squares on a distribution that
      skewed.
- [ ] **Congestion features carry almost nothing** (1.2% of gain combined),
      which is odd for a problem whose physics is queueing. Either the windows
      are wrong, the counts are too coarse, or `unimpeded_taxi_sec` already
      absorbs the signal. Worth one experiment before adding more features.

## Data quality, unexplained

- [ ] **289 training departures have a negative taxi-out**, 241 of them at LSZH,
      minimum -12s. Not a model defect: the model reproduces them faithfully and
      the truth on those rows has a median of 8s. But the two timestamps
      disagree there and nothing explains why.
- [ ] **69 departures exceed six hours**, up to 131,167s. Day-boundary clock
      errors were the obvious candidate and were tested: subtracting 24h lands
      only 20% of them anywhere plausible, so that is not what they are.

## Simulator

- [ ] **Decide its fate.** It builds and passes 11 tests, but it was written
      believing takeoff times were blanked on the scored set, and they are not.
      Its remaining case is estimating how much of the pushback-to-takeoff gap
      was queueing rather than transit. Either feed it AOBT as the off-block
      input, wire `queue_delay_sec` into `features.py` and measure whether it
      beats the plain congestion counts, or retire it explicitly. Limbo is the
      worst option with a public repo required.

## Compliance and housekeeping

- [x] **GPLv3 licence.** `LICENSE` holds the canonical text.
- [ ] **Make the repo public** before the deadline. Prize eligibility requires
      it. The history has been audited: 28 paths ever added, no parquet, no
      credential, so the flip is safe.
- [ ] **Reproducible documentation.** An eligibility requirement in its own
      right, not just good practice.
- [ ] **Delete `pipeline/taxiout/fixtures.py`.** Scaffolding from before the
      real data arrived.
- [ ] **Confirm the "Run the simulator on synthetic movements" README section.**
      The two commands were verified once end to end; the section predates that.
- [ ] Optional: the open-access paper in the Journal of Open Aviation Science
      that the rules encourage. `Taxi-Out, Measured` is the draft material.
