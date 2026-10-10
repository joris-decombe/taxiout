# TODO

Submissions close **14 October 2026, 10:00 UTC**. The ranking is the
**final phase**: one upload of `gentle-octopus_final.parquet`, predicting
January, February, June and July 2026, ranked on January + July,
February + June and all four, with a review of the code and documentation.
The leaderboard (January and July only, best of up to 5 uploads a day)
stays open as a check.

**Best upload: v11, 271.9s on the leaderboard, rank 68 of 239** (9 October
2026; leader 213.0s, 10th 222.9s, median 294.6s). v11 is the January and
July part of the final build, uploaded to check it and nothing else. Its
final model scores **307.8s** on the January and July 2025 validation days
(316.7s before the ADS-B correction) and 249.9s over all of 2025, each
month held out of its own fit (six folds).

## Uploads

Bucket names (`gentle-octopus_v<N>`) and local file names differ from v6 on;
the last column maps them.

| Upload | Validation | Test | What changed | Local file |
|---|---|---|---|---|
| v1 | 384s | 370.9s | first submission | `submission_v1` |
| v2 | 346.1s | 363.1s | surroundings, weather, flight fields, matched mixture | `submission_v2` |
| v3 | 346.6s | 361.5s | v2's matched predictions, v1's orphan predictions | `submission_v3` |
| v4 | 344.3s | 360.6s | live excess features | `submission_v4_nofallback` |
| v5 | 343.6s | 359.0s | stand-group/runway reference fallback | `submission_v4` |
| v6 | 321.8s | 311.3s | anchored regressor, LOBT window, LIRF day-shift and late-orphan rules, CatBoost blend | `submission_v8` |
| v7 | 317.2s | 287.2s | CatBoost orphan twin, three seeds, per-airport models, deeper CatBoost, CatBoost classifier, orphan cap outside LIRF, two-day guard | `submission_v10` |
| v8 | 316.7s | 287.0s | orphans outside LIRF shrunk towards their lateness band's mean | `submission_v11` |
| v9 | 308.8s | 275.0s | ADS-B corrector from adsb.lol ground observations | `submission_v12` |
| v10 | 308.6s | 274.1s | orphans outside LIRF floored at 0.95x their airport's live NM taxi level (431 rows, nearly all on EHAM's de-icing days of 3 to 9 January 2026) | `submission_v13` |
| v11 | 308.0s | **271.9s** | the final build's January and July: corrector pooled over three folds, fresh twelve-month fit | `submission_leaderboard` |

From v6 on, every upload scored better on the test set than on
validation, and by more than validation predicted. Most of the gap came
from orphans (no Network Manager record) in 2026 that the free-form orphan
model put at many hours, which the structural rules removed.

**Before each upload:**

- Diff the new predictions against the last scored file by group and
  airport, and read the largest changes (validation alone missed the Nice
  predictions in v2).
- Upload only genuine candidates for best model. The ranking page monitors
  submissions for "attempts to learn from or exploit the ranking process";
  variants built to measure one change or one slice on the test set are
  out.

## Open work, in priority order

- [x] **Reproducible final submission.** `pipeline/build_final.py` rebuilds
      v10's chain from the challenge data and the adsb.lol archive (about
      four hours, each fitted stage saved under `data/build_final/`). Its
      file matches `submission_v13` to fit noise: RMS difference 12.6s on
      matched rows and 37s on orphans, none above 359s; validation 316.7s
      before the correction, as before. Not uploaded on its own: it is the
      same model, and the next upload will be this build plus round 9.
- [ ] **Upload the final file**, `data/submission_final.parquet` from
      `build_final.py` with six folds (670,790 rows, verified), once and
      with the user's go-ahead. Against the three-fold build scored as v11
      (271.9s on January and July) it moves predictions 14s RMS on matched
      rows and 46s on orphans, none above 513s.
- [x] **A June fold** (round 10): June + August 2025, held out like the
      others. Pooling three folds beats two by -0.29s (95% CI -0.56 to
      -0.06), no fold worse. adsb.lol has no releases for 1 to 9 June 2025.
- [x] **Six folds** (round 11): March + September, April + October,
      May + November added. The corrector is cross-fitted over the whole
      year, -0.14s against three folds (95% CI -0.20 to -0.07), no fold
      worse; every rule re-checked on four held-out folds.
- [x] **A second fold for the corrector** (round 9): February and
      December 2025. Pooled with January and July it beats January and
      July alone, -0.60s there (95% CI -1.26 to -0.04) and -2.28s on
      February and December. More folds remain worth having: the best
      public write-up (EnioAguiar, 243.95s) cross-fits its corrector over
      all twelve months.
- [x] **ADS-B on congested days.** adsb.lol dates push-backs late when
      the airport is congested (`experiments_round8.py adsb`), but giving
      the corrector the live congestion level added nothing with winter
      days to learn from (+0.11s, round 9). The NM anchor protects matched
      rows and the floor protects orphans.
- [x] **Winter check** on a February + December 2025 holdout: 238.9s
      before the correction, 224.4s after (round 9).
- [ ] **Two LFPG easyJet orphans** with an off-block logged a day early
      (84,240s and 58,206s against a 30-minute schedule gap) carry about a
      quarter of all validation squared error. Nothing recorded flags them.

What was tried, and why the current model is built the way it is, is in
[RESEARCH.md](RESEARCH.md) and the `pipeline/experiments_round*.py`
docstrings (rounds 3 to 9 cover 6 to 9 October 2026).

## Data quality, unexplained

- [ ] **289 training departures have a negative taxi-out**, 241 of them at
      LSZH, minimum -12s. The model reproduces them; the two timestamps
      disagree there and nothing explains why.
- [ ] **69 departures exceed six hours**, up to 131,167s. 52 are at-schedule
      copies and 14 day-shifted off-blocks; 4 remain unexplained.

## Simulator

Not used by the submitted model. `surface.py` bridges it to the parquet
data; over March 2025 its queue delay correlates +0.21 with the real excess
over geometry and reconstructs 17% of it. Queueing features computed
directly from take-off times (round 4) added nothing either, because take-off
minus the Network Manager's off-block already contains the queue.

- [ ] Decide whether to keep `sim/` in the final repo as documented
      exploration or remove it.

## Compliance and housekeeping

- [x] **GPLv3 licence.** `LICENSE` holds the canonical text.
- [x] **Public repo** since 6 October 2026, history audited first.
- [x] **External data openly licensed:** METARs (Iowa Environmental
      Mesonet) and adsb.lol (ODbL, attribution in the README; derived tables
      stay uncommitted under `data/external/`). OPDI and OpenSky's
      historical database were not used: no open licence.
- [x] **Ideas from other teams credited** in the README's prior-work section.
- [ ] **Reproducible documentation.** `build_final.py` covers the
      submission, `report_findings.py` the report. The surface replay page
      and `synthetic_run.json` are not reproducible from the repo.
- [ ] **`experiments_paired_bootstrap.py` and `experiments_unmatched.py`**
      call `features.build` and `train.fit` with old signatures and no
      longer run. Port them or mark them historical.
- [ ] **Delete `pipeline/taxiout/fixtures.py`**, scaffolding from before the
      real data arrived.
- [ ] Optional: the open-access paper in the Journal of Open Aviation Science
      that the rules encourage. *Taxi-Out, Measured* is the draft material.
