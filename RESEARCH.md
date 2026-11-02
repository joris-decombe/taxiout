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
  2025 departures, so it bounds the answer to a two-hour window. It holds
  because it is the join condition: a departure recorded more than an hour
  from its LOBT loses its NM record and becomes an orphan.
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
| 14 | Floor orphans outside LIRF at 0.95x the median take-off minus AOBT of their airport's matched departures within ±30 min, where it exceeds 1,500s | An orphan taxis like its neighbours (0.95x to 1.0x at every congestion level in 2025, outside LTFM); the orphan model has no live inputs | **in: -0.16s** on validation (95% CI -0.40s to 0.00s: two mild months), **-0.96s on the test set** (v10, 431 rows, nearly all on EHAM's de-icing days of 3 to 9 January 2026); **-3.98s on the February + December 2025 fold** (251 rows, 75% improved, mostly LTFM's storm), a check of the structure since 0.95 and 1,500s were read from all ten training months, February included; `experiments_round8.py`, `experiments_round9.py` |
| – | Each flight number's history in the other ten months (copy rate, long-taxi rate, residual against the NM taxi), as corrector inputs | Target encoding of a stable identity | rejected: -4.5s on validation, but all of it LIRF orphans and 59% from 20 rows, matched rows +0.6s; LIRF's July 2026 flight numbers match 2025 for only half the departures |
| – | Neighbours' ADS-B-observed residuals; a route and scheduled-time key; flight-number string patterns | Residuals are shared within an airport-runway-hour | rejected: +0.4s, +0.2s, worse than the history alone |
| – | Neural nets, TabPFN, distributional boosting (NGBoost) | Wrong scale, or they model a full distribution when RMSE needs the mean | not pursued; the winners' ensembles used them ([after the challenge](#after-the-challenge-what-the-winners-did-differently)) |

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

The dwell detector reads late on congested days: BLOCK minus the observed
push-back has a median of 13s to 23s where the airport's NM taxi level is
under 1,200s and 111s to 219s above it, presumably de-icing pads and
queues taken for stands. Matched predictions keep the NM anchor; orphans
are protected by the congestion floor (candidate 14).

## Orphans on congested days

An orphan's taxi-out tracks the live taxi level of the matched departures
around it: the median `MVT − AOBT` at its airport within ±30 min. Outside
LTFM, 2025 orphans took 0.95x to 1.0x that level in every band from under
900s to over 3,000s (LTFM's February 2025 snowstorm ran at about 2x). The
orphan model, built on the smaller feature set, sees none of it, and
cross-fitted over the ten training months it under-predicts most where it
matters: neighbours at 2,000s to 3,000s, orphans 3,991s, predicted 1,762s.
Matched predictions show no such bias (the NM anchor carries the level).
January 2026 had what the validation months lacked: EHAM's de-icing days
of 3 to 9 January put 223 orphans at levels of 2,800s to 3,700s, predicted
at about 1,240s. Floored at 0.95x the level above 1,500s they scored 274.1s
against 275.0s. On the February and December 2025 fold the floor moves 251
orphans and the fold's RMSE from 238.9s to 235.0s, three quarters of the
rows improving, most of it LTFM's February storm. The ratio and the
threshold were read from all ten training months, February included, so
this checks the structure rather than untouched parameters.

## Generalisation beyond the leaderboard months

The final ranking adds February and June 2026 to the leaderboard's January
and July, because the organisers saw teams tune to those two months. Every
component here was chosen while validating on January and July 2025, so
each is checked on months the model never saw: six folds pair every month
of 2025, each held out of a fit on the other ten (`build_final.py` FOLDS,
rounds 9 to 11).

Per month, before the ADS-B correction, against a baseline of take-off
minus the Network Manager's off-block (the month's median where there is
none):

| Month, held out | Target sd | Baseline | Model | Model / baseline |
|---|---|---|---|---|
| January | 605s | 560s | 344s | 0.61 |
| February | 628s | 561s | 260s | 0.46 |
| March | 462s | 434s | 201s | 0.46 |
| April | 436s | 416s | 197s | 0.47 |
| May | 469s | 456s | 291s | 0.64 |
| June | 555s | 544s | 306s | 0.56 |
| July | 745s | 722s | 292s | 0.40 |
| August | 534s | 524s | 251s | 0.48 |
| September | 560s | 558s | 237s | 0.42 |
| October | 437s | 421s | 219s | 0.52 |
| November | 532s | 497s | 214s | 0.43 |
| December | 515s | 489s | 211s | 0.43 |

January and July sit at both ends of the range rather than ahead of it.
January is the hardest month: two LFPG orphans with an off-block logged a
day early carry about a quarter of its squared error.

Each rule switched off on the folds whose models are saved (RMSE change,
positive: the rule helps; rounds 10 and 11):

| Rule | Jun + Aug | Mar + Sep | Apr + Oct | May + Nov |
|---|---|---|---|---|
| LOBT window | +1.65s | +1.92s | +0.77s | +0.30s |
| LIRF late-orphan rule | +4.34s | +2.56s | +0.05s | +1.81s |
| LIRF day shift (1 to 5 rows) | -1.54s | +5.50s | -0.06s | +2.69s |
| orphan cap at an hour | +1.03s | +0.62s | +0.31s | +0.71s |
| orphan shrinkage | +0.28s | +0.48s | +0.19s | +0.11s |
| orphan congestion floor | +0.01s | +0.68s | -0.13s | +0.28s |

The window, the cap and the shrinkage help on every fold with 95%
intervals over days above zero; no rule is significantly worse on any. The
floor matters on congested days: +3.98s on February + December.

Where the rules' numbers come from: the LOBT window's ±3,606s from every
2025 departure that has a LOBT; the LIRF at-schedule rates, the day-shift
taxi and the shrinkage's band means are computed inside each fit from its
own training months; the hour cap from 2025's orphan lateness bands; the
floor's 0.95 and 1,500s from the ten training months. None was read from a
test score.

The ADS-B corrector learns from all six folds' held-out predictions, the
whole of 2025. Each fold added helped (rounds 9 to 11):

| Corrector fitted on | Jan + Jul | Feb + Dec | Jun + Aug | whole year |
|---|---|---|---|---|
| no correction | 316.5s | 234.9s | 279.5s | 255.9s |
| January + July | 308.6s | 226.7s | n/a | n/a |
| two folds | 308.1s | 224.6s | 276.7s | n/a |
| three folds | 308.1s | 224.2s | 276.1s | 250.0s |
| six folds | **307.8s** | **224.2s** | **276.1s** | **249.9s** |

Six against three: -0.14s over the year (95% CI -0.20 to -0.07), no fold
worse.

**The leaderboard.** Uploads v3 to v5 (29 September) were close to
single-change comparisons, made before this repo adopted a rule against
them. v5's stand-group fallback was kept on its test result (validation
+0.4s, test -1.6s); its reason, stands in 2026 that 2025 never used, holds
for every 2026 month. From v6 every upload was a candidate for best model,
and since the final phase no leaderboard score informs a choice: v11 was
uploaded only to check the final build's January and July part.

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

## After the challenge: what the winners did differently

The results came out on 30 October 2026
([winners page](https://ansperformance.eu/study/data-challenge/dc2026/winners.html)).
Only the top ten were published, 176.16s to 188.22s; this entry is not
among them and its final score was not released. 510 teams registered,
61 shared a repo, 96 sent a final file and 56 were eligible (final file
and public repo).

The final ranking reshuffled the leaderboard, as it was built to:

| Final rank | Team | Final RMSE | Best leaderboard score | Leaderboard rank |
|---|---|---|---|---|
| 1 | charming-eel | 176.16s | 213.25s | 5 |
| 2 | zesty-puzzle | 180.18s | 221.16s | 15 |
| 3 | youthful-giraffe | 183.66s | 223.31s | 17 |
| 4 | vigorous-whistle | 183.84s | 214.34s | 7 |
| 5 | gentle-igloo | 184.03s | 230.45s | 19 |
| 6 | jolly-lobster | 185.06s | 213.29s | 6 |
| 7 | strong-moose | 186.18s | 217.48s | 10 |
| 8 | strong-crane | 186.56s | 211.73s | 2 |
| 9 | versatile-watermelon | 186.87s | 208.11s | 1 |
| 10 | spirited-tiger | 188.22s | 227.67s | 18 |

Final scores sit 21s to 46s below the same teams' best leaderboard
scores, so the two are not on one scale. The winners page does not say
how the published figure combines the months (zesty-puzzle's README says
the ranking is on February and June), and a best-of-many leaderboard
score also carries selection on the test months. This entry's 271.9s was
77th of 249 when uploads closed.

The top three's READMEs describe their methods
([charming-eel](https://github.com/TheAryamanC/prc-data-challenge-2026),
[zesty-puzzle](https://github.com/Lys-fy/prc-data-challenge-2026-zesty-puzzle),
[youthful-giraffe](https://github.com/mattmotoki/youthful-giraffe-prc2026)).
Every one of the top ten uses adsb.lol. The numbers below are theirs, as
reported, and none is verified on our data yet.

### Where the gap is

**Matched rows, not orphans.** zesty-puzzle's laptop baseline (one
LightGBM on the residual to `MVT − AOBT_3`, 21 features from the
challenge files) scores 209s to 244s on matched rows over six 2025
month-pair folds; its full pipeline scores 166.5s. This model scores
222.8s on matched rows of January and July 2025 before the ADS-B
correction: at their baseline's level. The orphan structure (schedule
copies, the LOBT window, LIRF's day shift) was found by the top teams
too, and their mixtures have the same form as ours.

**1. ADS-B as a model input, on most departures.** Here adsb.lol gives a
push-back for 12% of 2025 departures (18% to 20% of 2026's), seen parked
and then moving, and enters only through a corrector fitted on the
model's error (-7.3s). zesty-puzzle has an ADS-B ground clock for 55% of
matched rows, which score 124s against 208s without one. charming-eel
reports that "left the stand" alone lies within 60s of `BLOCK_TIME` for
77% to 83% of departures at well-covered airports, against about 20% for
the Network Manager's AOBT. What makes the coverage:

- Matching every movement, arrivals too, to an airframe in passes: by
  callsign confirmed by the runway speed transition (slow to fast at
  take-off) near `MVT_TIME`, then the nearest unused take-off transition
  within 120s, then the same restricted to the aircraft type.
- A stand map from where arrivals park at their in-block time, which the
  ranking set keeps, so "parked" means parked at a known stand.
- About 190 track features per departure (first seen, dwells, leaving
  the parked position, reception gaps, path still to go, the state on a
  time grid before take-off) fed to the model, not to a corrector.
- A link for orphans too: zesty-puzzle's, by callsign with the take-off
  within 60s, is worth -75 to -197 s² of MSE per fold at half strength.

**2. Neural and sequence members in the ensemble.** RESEARCH.md ruled
neural networks out on the tabular literature. charming-eel's 160 network
jobs (ResNet, TabM, RealMLP, attention over the 16 or 32 neighbouring
take-offs), averaged and corrected by an XGBoost second stage, were worth
about 7s; single networks differed by 60s to 70s rms between seeds, so
the gain is in the averaging. youthful-giraffe adds a transformer over
the 64 movements either side of each departure; zesty-puzzle an
airport-day set transformer and NVIDIA's Kumo Tabular foundation model
used in context, its largest single gain (-405 to -552 s² of MSE per
fold).

**3. The orphan's own flight plan, from its arrival record.** When an
orphan's destination is one of the ten airports, its arrival row there
can carry the flight-plan link the departure lost. That says why the
departure's join failed: an AOBT more than an hour from LOBT means the
block time is genuine; a schedule more than an hour from LOBT means a
real off-block would have joined, so the block holds the schedule. On
2025 LIRF zesty-puzzle finds the block equal to the schedule in 26 of 28
rows of the first kind and 0 of 100 of the second. charming-eel uses the
same link for its no-link raises and caps.

**4. Smaller signals for the schedule-copy classifier.** Whether the
previous arrival at the same stand had its block copied from the
schedule raises LIRF's departure copy rate from 16% to 39%
(charming-eel). The airframe's turnaround (time since it last landed,
from OPDI or ADS-B) feeds its Rome models. A copy-cost weight
`clip(|msched - vbase|, 300, 14400) / 300` on the classifier, which
failed here in an earlier form (344.4s against 344.3s), is in theirs.

**5. Published aggregates of the scored period.** charming-eel shifts
each airport-month by EUROCONTROL's monthly taxi-out figures;
zesty-puzzle calibrates each day to EUROCONTROL's daily pre-departure
delay (-207 to -629 s² of MSE per fold). Both read figures published
after the scored months, which zesty-puzzle reports the organisers
allowed. Check the rules of the edition before relying on it.

**6. Stricter validation.** charming-eel holds out every fourth ISO
week; zesty-puzzle seals 20% of airport-days, drawn once, and admits a
change only if it beats a placebo with permuted inputs on at least three
of four month-pair folds. Our six month-pair folds cover the year but
were reused for every choice.

### Where they agree with this model

- Queue and arrival-interference features add nothing (charming-eel;
  candidate 1 here).
- METAR in the value model, a log target, Huber loss and capping during
  training do not help (charming-eel).
- The mixture `p·gap + (1 − p)·normal`, the LOBT window and the NM anchor
  `MVT − AOBT_3` are in all three.

### Why this entry did not do these

What the repo recorded at the time, and where it recorded nothing:

- **ADS-B as an input.** Recorded: `experiments_round7.py` chose a
  corrector over retraining because a year of adsb.lol meant streaming
  1.2 TB, on 7 October 2026, four days before the deadline as it then
  stood. By 10 October 357 days were extracted for the six-fold corrector,
  which removed that reason, but the design stayed a corrector. The low
  coverage follows from two choices in `taxiout.adsb`, neither weighed
  against an alternative. A clock needs ten minutes parked and then
  moving (12% to 20% of departures; the 40% first seen already taxiing
  give only a lower bound). And a departure without a callsign match,
  every orphan among them, is matched only when a single lift-off lies
  within ±60s of its take-off at that airport: no nearest-candidate
  pass, no aircraft-type pass. A stand map from arrivals was not
  considered.
- **Neural and sequence models.** Recorded: "Why gradient-boosted trees"
  above, on the tabular benchmarks and EUROCONTROL's CatBoost result, and
  candidates marked "not pursued" for being the wrong scale. It rests on
  the literature, with no experiment. The TabArena result cited there
  (ensembling matters more than architecture) is the one the winners
  acted on, and the foundation-model objection to two million rows does
  not hold for in-context use on a subsample of each airport's rows.
- **The orphan's arrival record.** Not recorded anywhere. The orphan
  analysis reached the LOBT join window ("Orphans on congested days" and
  the LOBT credit in the README) and stopped short of reading why each
  orphan's join failed.
- **Published aggregates.** Not recorded; never surveyed as a source.
- **Sealed days.** Not recorded. The six folds were the only check, and
  they were also used to choose.
- **The last four days.** The deadline moved from 11 to 14 October on 8
  October. The final file was uploaded on 10 October, and the final phase
  allows one upload, so no later improvement could reach the ranking.
  The repo records the upload but no reason for making it early, and no
  work after it.

### For a next edition, in order

1. Match every movement to its airframe in adsb.lol in passes, build the
   stand map from arrivals, and feed the ground clock and track features
   to the main models, orphans included.
2. Link orphans to their arrival records and apply the join-failure test
   above.
3. Add neural and sequence members (attention over neighbouring
   movements) and a second stage fitted on held-out folds.
4. Seal a share of airport-days for final checks only, apart from the
   folds used to choose.
