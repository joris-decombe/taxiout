# TODO

Submissions close **11 October 2026, 23:59:59 CET**. Ranking takes each
team's **best** score, at up to 5 uploads a day.

**Best upload: v10, 274.1s on the test set, rank 68 of 237** (8 October
2026; leader 213.5s, 10th 223.7s, median 295.9s). Its model scores
**308.6s** on the January and July 2025 validation days (308.8s without
the orphan congestion floor, 316.7s before the ADS-B correction), against a
686s standard deviation for those months.

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
| v10 | 308.6s | **274.1s** | orphans outside LIRF floored at 0.95x their airport's live NM taxi level (431 rows, nearly all on EHAM's de-icing days of 3 to 9 January 2026) | `submission_v13` |

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
- [ ] **A full-year ADS-B corrector**, in progress as round 9
      (`experiments_round9.py`): a February + December 2025 fold first.
      The corrector is fitted on 62 validation days. The best public write-up (EnioAguiar, 243.95s)
      reports its largest single step, 314.8s to 275.9s with the LOBT
      window, from a corrector fitted on base predictions cross-fitted
      over all twelve months. That needs the other ~300 days of 2025
      streamed (about 1 TB; the last 124 days took about two hours) and
      five production-size fits for the out-of-fold base. The most
      credible large gain left, and the most expensive.
- [ ] **ADS-B on congested days.** adsb.lol dates push-backs late when
      the airport is congested: BLOCK_TIME minus the observed push-back
      has a median of 13s to 23s where the NM taxi level is under 1,200s
      and 111s to 219s above it (de-icing pads and queues look like stands
      to a dwell detector; `experiments_round8.py adsb`). The NM anchor
      protects matched rows and the floor protects orphans; giving the
      corrector the congestion level would address it directly.
- [ ] **Winter check of the live excess signal** on a February + December
      2025 holdout: January 2025 was mild, January 2026 was not.
- [ ] **Two LFPG easyJet orphans** with an off-block logged a day early
      (84,240s and 58,206s against a 30-minute schedule gap) carry about a
      quarter of all validation squared error. Nothing recorded flags them.

What was tried, and why the current model is built the way it is, is in
[RESEARCH.md](RESEARCH.md) and the `pipeline/experiments_round*.py`
docstrings (rounds 3 to 8 cover 6 to 8 October 2026).

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
