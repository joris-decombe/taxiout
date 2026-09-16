# taxiout

PRC Data Challenge 2026 — predicting taxi-out time at 11 major European airports.

Two tracks that meet in the middle:

- **`pipeline/`** — Python. Loads the challenge parquet files, builds features,
  trains a LightGBM baseline, writes `submitting.parquet`. This is the yardstick.
- **`sim/`** — C++20. An event-driven model of the departure surface: pushback,
  apron transit, runway queue with wake separation, arrival preemption. This is
  the part that can beat the yardstick.

## Why simulate at all

Taxi-out decomposes into unimpeded transit plus queue delay, and the queue
delay is where all the variance lives.

The scaffold assumed takeoff times were blanked on the ranking set, which
would have made congestion features unmeasurable and the simulator the only
way to get them. **That is backwards.** The real ranking set blanks
`BLOCK_TIME_UTC_mvt` and the target, and keeps `MVT_TIME_UTC_mvt`. Since the
target is exactly `MVT_TIME - BLOCK_TIME`, the two had to be blanked
together -- the task is reconstructing the *off-block* time from the takeoff
time, not the reverse.

So congestion features around takeoff are directly computable and need no
simulator. What the simulator can still offer is the counterfactual: how
much of the gap between pushback and takeoff was queueing rather than
transit. Its input assumptions need revisiting first, since it was written
to consume off-block times that the ranking set does not provide.

The strongest single feature is `MVT_TIME - AOBT_3_flt`: the Network
Manager's off-block time is *not* blanked, and differs from the movement
table's by a standard deviation of 374s. On its own it predicts the target
at 377s RMSE against a target sd of 605s.
## Status

Full 2025 dataset local (12 monthly training files, `ranking.parquet`,
`submitting.parquet`). Python baseline runs end to end: **471s validation
RMSE**, honest features only. The C++ simulator compiles nowhere yet -- the
MSVC toolchain is not installed -- and its premise needs the rethink above.
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
