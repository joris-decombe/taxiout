# Models for taxi-out: what the literature says, and why ours works

A survey of the modelling literature behind this repo's approach, and the
candidates it suggests. Measurements are on the January and July 2025
validation split unless stated; `experiments_round*.py` hold the details.

## The problem, stated mathematically

Take-off time is known exactly on the ranking set, so predicting taxi-out
`y = MVT − BLOCK` is the same as estimating the off-block time. The data
carries three noisy clocks for it:

- **AOBT** (Network Manager off-block). `MVT − AOBT` alone scores 385s RMSE.
- **LOBT** (flight plan). `|BLOCK − LOBT| ≤ 3,606s` on all 2,062,577 matched
  2025 departures, so it bounds the answer to a two-hour window.
- **SCHED**, which the recording system copies into BLOCK on a sizeable
  share of LIRF's departures (the at-schedule artifact).

The score is RMSE, whose minimiser is the conditional mean `E[y | x]`. On a
target with a tail to 88,000s, 1.5% of rows hold half the squared error, so
the rare regimes matter more than the bulk.

## Why the current model works

1. **Mixtures compute the conditional mean exactly.** By the law of total
   expectation, `E[y|x] = P(k=art|x)·gap + P(k=normal|x)·E[y|x,normal]`
   (+ a day-shift component at LIRF). A single regressor would have to
   learn a jump of hours with piecewise-constant trees, which is high
   variance. Split, each component is smooth, and the artifact component is
   known exactly (it equals the gap). This is a mixture of experts (Jacobs,
   Jordan, Nowlan and Hinton, 1991) with a known expert.
2. **The mixture is only as good as its weights.** The expected cost of a
   row is about `p(1−p)(gap − r)²`, so calibration of `p` matters most where
   the gap is large. Boosted trees are known to give poorly calibrated
   probabilities (Niculescu-Mizil and Caruana, ICML 2005). Replacing `p`
   with LIRF's empirical band share past 6h late gained 5.2s on its own.
3. **Projection onto a set that contains the truth cannot hurt.** For a
   convex set C with `y ∈ C`, `|proj_C(ŷ) − y| ≤ |ŷ − y|` for every row. The
   LOBT window is such a set, which is why it gains on every row where the
   bound holds (−3.8s).
4. **Anchoring.** Boosting from `init_score = clip(MVT − AOBT)` leaves the
   trees only the deviation from the NM's own taxi time, instead of
   approximating the identity in steps (−1.1s).
5. **Structure transfers, free fits do not.** Rules derived from the data's
   structure (the day-shift and late-orphan rules) carried to validation;
   free-form fits produced wild values: linear leaves extrapolated to
   ±300,000s on a few rows, and the orphan regressor ranged from −3,500s to
   20,000s on rows whose truth is ~1,000s. The validation-to-test gap (v1:
   384s to 371s, v5: 344s to 359s) says 2026 is shifted, which favours
   fewer degrees of freedom.

## Why gradient-boosted trees

- On tabular data, trees beat neural networks because targets are irregular
  and many features uninformative (Grinsztajn, Oyallon and Varoquaux, 2022,
  [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)).
- EUROCONTROL's taxi-time model for six A-CDM airports (Swatowska and
  Gabagnou, SESAR Innovation Days 2025,
  [paper](https://www.sesarju.eu/sites/default/files/documents/sid/2025/papers/SIDs_2025_paper_119-final%20v2.pdf))
  benchmarked linear models, random forests and CatBoost on 4.1 million
  flights; CatBoost won. Its SHAP shares for departures: runway 19%, stand
  14%, airport 9%, de-icing 7%, airline 5%, weather 3% in all.
- TabArena (2025, [arXiv:2506.16791](https://arxiv.org/abs/2506.16791)) finds
  that validation protocol and post-hoc ensembling change rankings more than
  architecture. Tabular foundation models (TabPFN) target small datasets,
  not two million rows.

## The surface as a queue

- Idris et al. (MIT, early 2000s) found the number of take-offs between
  push-back and take-off the strongest predictor of taxi-out at Boston.
  `context.py`'s `*_deps_since_aobt` counts measure it.
- Simaiakis and Balakrishnan, "A queuing model of the airport departure
  process" (Transportation Science, 2016,
  [PDF](https://www.mit.edu/~hamsa/pubs/SimaiakisBalakrishnan_TS2014.pdf)):
  - taxi-out = unimpeded time + ramp/taxiway interaction + runway queue;
  - interaction is linear in the aircraft taxiing when a flight pushes back;
  - mean taxi-out is a convex, non-decreasing function of **adjusted
    traffic**: aircraft taxiing out at push-back plus those pushing back
    while it taxis;
  - zero-queue flights are a biased sample (the fastest ones), so a low
    quantile underestimates the unimpeded time; they fit the flat region of
    the convex curve instead;
  - unimpeded times are right-skewed (lognormal or Erlang);
  - the runway is a D(t)/E_k(t)/1 queue whose service rate depends on the
    configuration.
- Ravizza, Atkin and Burke (2013 to 2014, Stockholm-Arlanda and Zurich)
  combined a ground-movement model (taxi distance, turns) with multiple
  linear regression and fuzzy rule-based systems; the TSK fuzzy system was
  the most accurate of those compared.

## Candidates, by expected value

| # | Candidate | Basis | Status |
|---|---|---|---|
| 1 | Queueing features: adjusted traffic, runway busy-period position | Simaiakis and Balakrishnan | no effect: -0.1s (95% CI -0.3s to +0.1s); take-off minus AOBT already holds the queue's outcome |
| 2 | Calibrated mixture weights: cross-fitted isotonic calibration of `p` | Niculescu-Mizil and Caruana; the LIRF result | ruled out: an in-sample oracle gains at most 0.5s once the LIRF rules apply |
| 3 | CatBoost blended with LightGBM | Ordered target statistics avoid prediction shift (Prokhorenkova et al., 2018, [arXiv:1706.09516](https://arxiv.org/abs/1706.09516)); EUROCONTROL's benchmark | **in: -2.7s** (95% CI -3.4s to -2.2s) at 50/50, both months better; `train.JOINED_CATBOOST_WEIGHT` |
| 4 | Stacking on out-of-fold predictions | Wolpert (1992); cross-fitting (Chernozhukov et al., 2018, [arXiv:1608.00060](https://arxiv.org/abs/1608.00060)) | rejected: -0.4s on production, all in July; January worse at full strength (`experiments_round6.py`) |
| 5 | Seed averaging | Variance reduction | **in: -0.2s** (95% CI -0.3s to -0.2s), three seeds |
| 6 | Monotone constraints on queue counts | The convex, non-decreasing relation above | dropped with 1 |
| 7 | CatBoost for the no-record group | Ordered boosting suits small, noisy groups | **in: -2.2s** (95% CI -3.7s to -0.9s) at 50/50 |
| 8 | Per-airport LightGBM beside the global one; deeper CatBoost | Local structure; capacity | **in: -0.9s** together (95% CI -1.2s to -0.6s) |
| 9 | CatBoost classifier beside LightGBM's | Ensemble of classifiers ranks records better | **in for matched: -0.6s**; orphans +1.5s, not used |
| 10 | LightGBM tuning: learning rate 0.03, 511 leaves, stronger L2 | Standard tuning | no effect (+0.5s to -0.1s) |
| 11 | Cap orphan predictions outside LIRF at 3,600s | 2025 structure: no lateness band there averaged over 1,360s | **in: -0.8s** (95% CI -1.5s to -0.2s) |
| 12 | Shrink orphan predictions outside LIRF a quarter of the way to their airport and lateness band's mean | Empirical Bayes; a plausibility audit of 2026 predictions against 2025 outcomes | **in: -0.5s** (95% CI -0.7s to -0.3s) |
| – | Clip matched predictions to 2025's range around the NM taxi time | The same audit | rejected: +3s to +13s; the extremes are real (at-schedule copies) |
| 13 | ADS-B ground observations (adsb.lol) through a corrector fitted on validation days | Direct observation of the push-back; idea credited in the README | **in: -7.3s** on 57 validation days (95% CI -9.7s to -5.6s; January -5.5s, July -9.2s), cross-validated by day (`experiments_round7.py`) |
| – | Neural nets, TabPFN, distributional boosting (NGBoost) | Wrong scale, or they model a full distribution when RMSE needs the mean | not pursued |

## Observing the push-back: ADS-B

The strongest model input is a different clock, and ADS-B can be one.
Surveyed sources (round 7 notes in `experiments_round7.py`):

- **adsb.lol** (ODbL, open download, every day of 2025 and 2026): the
  aircraft is seen parked and then moving for 12% of 2025 departures and
  18% to 20% of 2026's (half of Munich's, up to 45% of Amsterdam's and
  Heathrow's, none at Istanbul, Paris, Rome). Where seen, the push-back
  lands a median 35 s from BLOCK_TIME, 67% within a minute, against 180 s
  for the Network Manager's AOBT. Most other departures are first seen
  already taxiing, which still bounds the taxi-out.
- **OPDI** (PRC and OpenSky flight events): ground coverage only at Zurich
  and Frankfurt, nine days of January 2026 missing, and no stated open
  licence. Not used.
- **OpenSky's historical database**: an account and a research licence,
  not openly reusable. Not used.

A corrector (LightGBM on the model's error, from the observations, the
prediction, the at-schedule probability, the NM taxi time and the
schedule gap) fitted on held-out validation predictions turns this into
-7.3s; on the departures whose push-back was seen, 200 s to 164 s.

## Tried and rejected

- **Queueing features** (adjusted traffic, busy-period position): -0.1s.
  Counting the queue adds nothing once the model sees `MVT − AOBT`, which
  is the waiting itself.
- **The flight table's arrival times** (`ARVT_1`, `ARVT_3`: airborne time,
  planned and actual block-to-arrival, arrival delay): correlation with
  the current model's residual about 0.01 in every form.

- **Linear leaves** (`linear_tree`): +607s, extrapolation on a few rows;
  still 292s matched against 229s when bounded.
- **Orphan regressor without day-shifted rows**: +48s; it had learnt part
  of the day-shift mass, and removing it left nothing in its place.
- **A fitted day-shift share** `(1 − p)·q·86,400` at every airport: +5.3s.
  The structural LIRF rule replaced it.
