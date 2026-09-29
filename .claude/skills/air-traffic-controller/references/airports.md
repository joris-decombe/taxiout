# The ten airports

Contents: summary table · one section per airport (published operating
concept, reconciled with the data) · measured profiles (appendix, generated).

Tags as in SKILL.md. "[data]" figures come from the appendix, which
`scripts/airport_profile.py` regenerates from `data/`; taxi-out figures are
2025 training departures, runway shares compare January and July of 2025
(training) with the same months of 2026 (ranking set).

## Summary

| ICAO | Median taxi-out | Main departure runways (2025 share) | Notable |
|---|---|---|---|
| EDDF | 837s | 18 (61%), 25C/07C (39%) | Runway 18 is departures-only, southbound |
| EDDM | 774s | 26L, 26R, 08R, 08L | Mixed mode on both runways |
| EGLL | 1,319s | 27R/27L alternating, 09R easterly | Longest taxi of the ten; easterly ops far more common in 2026 |
| EHAM | 742s | 24, 36L, 18L, 36C | Polderbaan (36L) median 1,049s vs 627s on 24 |
| LEBL | 906s | 24L (69%), 06R (29%) | Segregated parallels; 02 for night arrivals |
| LEMD | 985s | 36R/36L north, 14L/14R south | Two-configuration airport |
| LFPG | 954s | 08L/26R, 09R/27L | North doublet departs from the inner runway |
| LIRF | 1,025s | 25 (90%) | 18.1% of off-blocks recorded at schedule |
| LSZH | 711s | 28 (63%), 32, 16 | Time-of-day concepts; most negative taxi-outs |
| LTFM | 963s | 36, 35L north; 18, 17R south | South config common in January only; weakest AOBT agreement |

## EDDF Frankfurt

- Runways: 07L/25R (northwest, landings only), 07C/25C, 07R/25L, and 18
  (West, 4,000 m, take-offs southbound only) [doc: Fraport; secondary for
  role details].
- Data: departures on 18 (60.8%), 25C (20.0%), 07C (18.8%); arrivals on 25L,
  25R, 07R, 07L and some on 25C [data]. 25C is the only runway with both.
- Runway 18 has the longest median taxi (888s) [data].
- Several remote de-icing pads [secondary].
- 2026: 07C departures up in both months (Jan 11% → 20%, Jul 7% → 15%),
  i.e. more easterly operation [data].
- AOBT_3 vs BLOCK_TIME: median +87s, 21.0% within 60s [data].

## EDDM Munich

- Two parallel 4,000 m runways, 08L/26R (north) and 08R/26L (south),
  2,300 m apart, approved for independent parallel operations [secondary].
  North-bound traffic tends to use the northern runway [secondary].
- Data: all four runway directions carry both departures and arrivals, i.e.
  mixed mode [data]. West operation dominates in July (26L+26R 79% of
  departures in July 2025, 77% in July 2026) [data].
- Remote de-icing areas, about 9,000 de-icings in an average winter
  [secondary].
- Off-block at schedule 5.0% [data].

## EGLL London Heathrow

- Two runways, 09L/27R and 09R/27L. Westerly preference; in westerly
  operation landing and departure runways swap at 15:00 daily (runway
  alternation), and the pattern alternates weekly. In easterly operation the
  Cranford Agreement legacy puts departures on 09R and landings on 09L
  [doc: Heathrow runway alternation pages].
- Data: departures 27R 32.2%, 27L 31.2%, 09R 36.6% of 2025; arrivals 09L,
  27L, 27R [data]. Consistent with the above.
- **2026 shift**: 09R departures 20% → 46% in January and 11% → 38% in July
  [data]. More easterly wind days or an operational change; not
  established [judgement]. Taxi times by runway are similar (medians
  1,310–1,368s), so the effect on taxi-out goes through stand–runway pairs,
  not the runway alone [data].
- Longest taxi-out of the ten: median 1,319s, P10 909s [data].
- Two sets of remote de-icing pads [doc: HADIP].

## EHAM Amsterdam Schiphol

- Six runways. Preferential use: take-offs on 36L (Polderbaan) and 24
  (Kaagbaan), landings on 06 and 18R, because their routes overfly less
  populated areas; LVNL decides the combination from wind, visibility and
  environmental rules [doc: Schiphol].
- The Polderbaan is about 5 km from the terminal [doc: Schiphol news].
  Data: its median taxi-out is 1,049s against 627s for runway 24 [data].
- Data: departures 24 (34.4%), 36L (23.6%), 18L (22.4%), 36C (12.5%) [data].
- **2026 shift**: July 36L share 29.7% → 40.8%, 36C 18% → 22%, 18L 16% →
  8.6% [data], i.e. more of the long-taxi runway.
- AOBT_3 vs BLOCK_TIME: median +107s [data].

## LEBL Barcelona

- Runways 06L/24R, 06R/24L (parallel) and 02/20 (crossing). The crossing
  runway takes night arrivals to reduce noise over Gavà and Castelldefels,
  wind permitting [secondary].
- Data: segregated parallels. West: departures 24L (69.3%), arrivals 24R
  (65.2%). East: departures 06R, arrivals 06L. Runway 02 takes 12.2% of
  arrivals [data].
- **2026 shift**: July east operation more common (06R departures 23.2% →
  38.7%) [data].
- AOBT_3 vs BLOCK_TIME: median −54s [data].

## LEMD Madrid

- Four runways in two pairs. North configuration (preferential): landings
  32L/32R, take-offs 36L/36R. South configuration: landings 18L/18R,
  take-offs 14L/14R. At night (23:00–07:00 local) a single runway each way:
  32R/36L north, 18L/14L south [secondary: AIP summary].
- Data confirms: departures 36R+36L 79.6%, 14L+14R 20.4%; arrivals 32R+32L
  80.0%, 18L+18R 20.1% [data].
- 36R and 14L have the longer taxis (medians 1,050–1,051s) [data].
- 2026: south configuration more common in July (14L+14R 20.5% → 34.9%)
  [data].

## LFPG Paris Charles de Gaulle

- Four parallel runways in two doublets: south 08L/26R (outer, ~4,200 m)
  and 08R/26L (inner, ~2,700 m); north 09L/27R (outer, ~4,200 m) and 09R/27L
  (inner, ~2,700 m) [secondary].
- Data: the doublets differ. South: departures from the outer 08L/26R
  (61.4%), arrivals on the inner 26L/08R (60.6%). North: departures from
  the inner 09R/27L (24.6%), arrivals on the outer 09L/27R (32.9%) [data].
  A "long runway for departures" rule only holds in the south.
- **Anomaly**: in July 2025, 16.1% of departures used 27R, which normally
  carries arrivals, and 27L fell to 14.2%. In July 2026 27R is back to
  0.1% [data]. Consistent with works on the north doublet in summer 2025
  [judgement: not verified]. Training rows from that period have
  stand–runway pairs that will not recur in 2026.
- 2026: July east operation far more common (08L departures 12.3% → 31.3%,
  26R 49.5% → 25.3%) [data].
- Two LFPG departures with a day-early off-block dominate the validation
  error [data, see TODO.md].

## LIRF Rome Fiumicino

- Runways 07/25, 16L/34R, 16R/34L; 16C/34C is used mostly as a taxiway.
  07/25 is used for take-offs westbound [secondary].
- Data: departures 25 (90.1%), 16R (7.0%), 34L (2.6%); arrivals 16L
  (69.8%), 16R, 34R [data].
- **Recording artifact**: 18.1% of departures have BLOCK_TIME within ±10s of
  SCHED_TIME, against 1.1–5.6% elsewhere [data]. Read it the right way
  round:
  - **Most flagged flights left close to schedule.** Median AOBT_3 − SCHED on
    flagged rows is about +4 min. The flag rate is 36% when AOBT_3 is 0–5 min
    late, 5% at 1–2 h late, and 0.6% beyond that. On these rows the aircraft
    taxied normally: MVT − AOBT_3 has a median of 898s, the same as other
    rows [data].
  - **The few delayed ones carry the error.** 82% of LIRF departures with a
    target over one hour are flagged rows [data].
  - **Minute resolution.** LIRF off-blocks sit within ±6s of a whole minute
    and schedules are on the minute, so "equal to the second" means the same
    minute. The count at exactly 0 min off schedule is about 29,000, against
    2,000–5,900 for each neighbouring minute from −5 to +5 [data].
  - **What the flag goes with**: the airline (one operator at 54%, several
    at 4–6%), night schedules (41–70% between 22:00 and 01:00 UTC), stand
    area (1xx and 2xx stands high), EOBT_1 still equal to the schedule (30%
    against 9%), and no NM record (48%) [data]. That points to a handler or
    feed that never captures an actual and keeps or keys the scheduled time
    [judgement].
  - The rate is steady by month (15% in July, about 20% in winter). Arrival
    in-blocks at LIRF also sit at the schedule: 8.9% in July 2025, 10.6% in
    July 2026 [data]. Arrivals keep their BLOCK_TIME on the ranking set, so
    this is a 2026 check on whether the habit persists.
- Median BLOCK_TIME − AOBT_3 is −118s, the most negative of the ten [data].
- Taxi-out sd 1,332s, 1.2% over one hour [data].
- 2026: 34L departures nearly vanish in January (9.6% → 0.4%) [data].

## LSZH Zurich

- Three runways: 10/28, 14/32, 16/34. Operating concepts (local time)
  [doc: Flughafen Zürich]:
  - South approach (Mon–Fri 06:00–07:00; weekends and Baden-Württemberg
    holidays 06:00–09:00): land 34, depart 32/34 north and 28 west.
  - North approach (Mon–Fri 07:00–21:00; weekends 09:00–20:00): land 14/16,
    depart 28 west and 16 south.
  - East approach (Mon–Fri 21:00–23:30; weekends 20:00–23:30; also daytime
    in strong westerly wind): land 28, depart 32/34 north.
  - Bise (north-easterly wind): land 14/16, depart 10 east.
  - German airspace rules restrict northern approaches in early mornings,
    late evenings and on Baden-Württemberg holidays [doc].
- Data: departures 28 (62.9%), 32 (24.4%), 16 (9.6%), 10 (1.9%), 34 (1.0%);
  arrivals 14 (69.8%), 28 (20.1%), 34 (9.7%) [data]. Runway 16 has the longest
  taxi (median 973s) and only 14 departures an hour at the 99th percentile
  [data].
- Because the concept is set by clock and day type, the runway, and so the
  taxi time, is largely predictable from local hour, weekday and German
  holidays [judgement].
- 241 of the 289 negative taxi-outs in training are at LSZH [data].

## LTFM Istanbul

- Five runways: 16L/34R, 16R/34L, 17L/35R, 17R/35L, 18/36. North
  configuration (34/35/36) is preferential; parallel departure operations
  are used to speed departures [secondary: IVAO briefing].
- Data: departures 36 (46.2%), 35L (29.3%), 18 (11.6%), 17R (7.7%);
  arrivals 35R (40.5%), 34L (30.0%) [data].
- Strongly seasonal: the south configuration (18, 17R) carries 44–50% of
  January departures but 3–5% of July's [data].
- Not listed among NM-connected A-CDM airports; BLOCK_TIME agrees with
  AOBT_3 within 60s for only 8.7% of departures, median difference +296s
  [data].

## What is different in 2026

Measured on the ranking set against the same months of 2025. The target is
blank in 2026, so taxi-out comparisons use `MVT − AOBT_3` as a proxy. It
exists in both years and scores about 385s RMSE against the truth, so compare
differences between groups or models, not levels [data].

- **January 2026 was a winter-operations month; January 2025 was not.**
  Share of departures with snow or freezing precipitation at take-off,
  January 2025 → January 2026: EDDF 0.7% → 7.5%, EDDM 5.2% → 10.7%, EHAM 3.5%
  → 5.2%, LFPG 2.5% → 1.4%, LSZH 4.6% → 13.4%, LTFM 0.0% → 8.5% [data]. LTFM's
  only snow in training is February 2025.
- **EHAM disruption, 2–7 January 2026**: 194–469 departures a day against
  about 600, 20–40% of them with no NM record, and a P90 of the proxy
  taxi-out of 4,300–5,400s [data]. LFPG was elevated on 5–7 January [data].
- **Less pre-departure delay in July 2026.** Median AOBT − schedule: LFPG
  960s → 600s, LIRF 1,080s → 900s, EGLL 480s → 240s, EHAM 660s → 420s [data].
  Summer 2025 had French ATC strikes [secondary]. Fewer delayed flights
  means fewer of the costly at-schedule rows at LIRF.
- **New stands**: 7.9% of EDDM's July 2026 departures use stands 103–108,
  which never appear in 2025. At EDDF, 1.45% use J5, J10 and V333–V337 [data],
  consistent with Terminal 3 opening [secondary]. Stand-based references
  fall back to defaults on these rows.
- **Runway mix**: see the per-airport sections and the table below.

## Measured profiles (generated)

Columns: departure share over 2025; taxi-out P10/median/P90; gap between
successive take-offs on that runway at the 5th and 25th percentiles; 99th
percentile of departures per clock hour; runway share in January and July,
2025 vs 2026. Runways under 0.5% of departures are omitted.

#### EDDF

2025 departures 230,141, arrivals 230,122; 2026 ranking departures 36,317. Stands used by departures: 246.
Taxi-out 2025: P10 478s, median 837s, P90 1251s, sd 330s, over 1h 0.0%. No NM record 0.8%. Off-block at schedule (±10s) 1.2%. BLOCK_TIME within 60s of NM AOBT 21.0%, median BLOCK_TIME - AOBT +87s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 18 | 60.8% | 491 / 888 / 1279 | 56 / 80 | 40 | 63.6% | 62.4% | 64.7% | 63.4% |
| 25C | 20.0% | 479 / 735 / 1161 | 78 / 120 | 25 | 25.1% | 17.4% | 28.0% | 20.8% |
| 07C | 18.8% | 423 / 777 / 1214 | 58 / 87 | 44 | 11.0% | 20.0% | 7.0% | 15.0% |

Arrival runways 2025: 25L 28.9%, 25R 21.3%, 07R 21.1%, 07L 17.5%, 25C 11.0%.
Runways carrying both departures and arrivals: 25C.

#### EDDM

2025 departures 167,334, arrivals 167,323; 2026 ranking departures 25,966. Stands used by departures: 203.
Taxi-out 2025: P10 533s, median 774s, P90 1135s, sd 302s, over 1h 0.0%. No NM record 0.6%. Off-block at schedule (±10s) 5.0%. BLOCK_TIME within 60s of NM AOBT 25.7%, median BLOCK_TIME - AOBT -2s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 26L | 34.5% | 472 / 720 / 1078 | 55 / 109 | 35 | 34.8% | 32.8% | 47.5% | 47.4% |
| 26R | 23.4% | 545 / 774 / 1091 | 58 / 119 | 23 | 28.1% | 24.9% | 31.9% | 29.1% |
| 08R | 22.2% | 649 / 847 / 1194 | 56 / 114 | 33 | 17.8% | 17.1% | 12.4% | 13.8% |
| 08L | 19.9% | 534 / 721 / 1146 | 57 / 116 | 32 | 19.3% | 25.2% | 8.2% | 9.7% |

Arrival runways 2025: 26R 31.6%, 26L 26.2%, 08R 23.0%, 08L 19.1%.
Runways carrying both departures and arrivals: 08L, 08R, 26L, 26R.

#### EGLL

2025 departures 239,546, arrivals 239,511; 2026 ranking departures 39,840. Stands used by departures: 232.
Taxi-out 2025: P10 909s, median 1319s, P90 1806s, sd 422s, over 1h 0.3%. No NM record 0.6%. Off-block at schedule (±10s) 4.9%. BLOCK_TIME within 60s of NM AOBT 19.4%, median BLOCK_TIME - AOBT +54s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 09R | 36.6% | 905 / 1310 / 1791 | 52 / 59 | 49 | 20.2% | 46.4% | 11.3% | 38.3% |
| 27R | 32.2% | 960 / 1368 / 1854 | 52 / 59 | 49 | 40.9% | 26.3% | 44.5% | 29.9% |
| 27L | 31.2% | 898 / 1317 / 1811 | 52 / 59 | 49 | 38.8% | 27.3% | 44.3% | 31.2% |

Arrival runways 2025: 09L 35.1%, 27L 32.6%, 27R 30.9%, 09R 1.4%.
Runways carrying both departures and arrivals: 09R, 27L, 27R.

#### EHAM

2025 departures 247,951, arrivals 247,706; 2026 ranking departures 38,182. Stands used by departures: 261.
Taxi-out 2025: P10 458s, median 742s, P90 1157s, sd 318s, over 1h 0.0%. No NM record 1.6%. Off-block at schedule (±10s) 1.3%. BLOCK_TIME within 60s of NM AOBT 23.7%, median BLOCK_TIME - AOBT +107s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 24 | 34.4% | 426 / 627 / 944 | 61 / 81 | 38 | 43.3% | 32.0% | 32.6% | 22.8% |
| 36L | 23.6% | 798 / 1049 / 1398 | 57 / 77 | 38 | 12.6% | 11.7% | 29.7% | 40.8% |
| 18L | 22.4% | 513 / 701 / 969 | 57 / 78 | 40 | 37.1% | 37.4% | 16.0% | 8.6% |
| 36C | 12.5% | 495 / 711 / 979 | 55 / 72 | 43 | 2.9% | 4.9% | 18.0% | 22.4% |
| 09 | 3.2% | 580 / 800 / 1102 | 52 / 78 | 41 | 1.3% | 5.4% | 0.0% | 1.8% |
| 22 | 1.9% | 125 / 238 / 472 | 134 / 624 | 6 | 2.1% | 1.6% | 1.7% | 1.3% |
| 04 | 1.1% | 176 / 341 / 528 | 137 / 604 | 6 | 0.2% | 0.7% | 1.6% | 2.2% |
| 18C | 0.8% | 637 / 848 / 1140 | 57 / 82 | 39 | 0.4% | 6.2% | 0.3% | 0.1% |

Arrival runways 2025: 18R 37.3%, 06 24.3%, 18C 15.2%, 36R 11.5%, 27 4.2%, 22 3.9%, 36C 2.9%.
Runways carrying both departures and arrivals: 18C, 22, 36C.

#### LEBL

2025 departures 179,705, arrivals 179,081; 2026 ranking departures 30,080. Stands used by departures: 232.
Taxi-out 2025: P10 595s, median 906s, P90 1357s, sd 331s, over 1h 0.0%. No NM record 1.0%. Off-block at schedule (±10s) 5.6%. BLOCK_TIME within 60s of NM AOBT 26.4%, median BLOCK_TIME - AOBT -54s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 24L | 69.3% | 614 / 914 / 1341 | 65 / 78 | 41 | 80.2% | 74.9% | 74.5% | 59.3% |
| 06R | 29.1% | 557 / 876 / 1378 | 67 / 83 | 39 | 18.6% | 23.0% | 23.2% | 38.7% |
| 24R | 1.0% | 718 / 1184 / 1645 | 305 / 1647 | 4 | 0.7% | 1.5% | 1.6% | 1.2% |
| 06L | 0.6% | 324 / 882 / 1587 | 240 / 1584 | 4 | 0.5% | 0.7% | 0.7% | 0.9% |

Arrival runways 2025: 24R 65.2%, 06L 19.4%, 02 12.2%, 24L 3.1%.
Runways carrying both departures and arrivals: 06L, 24L, 24R.

#### LEMD

2025 departures 212,242, arrivals 211,091; 2026 ranking departures 36,954. Stands used by departures: 374.
Taxi-out 2025: P10 656s, median 985s, P90 1402s, sd 313s, over 1h 0.0%. No NM record 0.4%. Off-block at schedule (±10s) 4.5%. BLOCK_TIME within 60s of NM AOBT 26.2%, median BLOCK_TIME - AOBT +2s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 36R | 39.9% | 752 / 1050 / 1445 | 69 / 87 | 33 | 33.8% | 42.5% | 38.3% | 30.0% |
| 36L | 39.7% | 593 / 903 / 1333 | 78 / 103 | 29 | 35.3% | 39.8% | 41.2% | 35.1% |
| 14L | 12.1% | 743 / 1051 / 1476 | 69 / 86 | 35 | 18.7% | 10.6% | 11.9% | 18.5% |
| 14R | 8.3% | 630 / 910 / 1340 | 77 / 106 | 29 | 12.3% | 7.1% | 8.6% | 16.4% |

Arrival runways 2025: 32R 43.0%, 32L 37.0%, 18L 11.6%, 18R 8.5%.

#### LFPG

2025 departures 239,552, arrivals 238,835; 2026 ranking departures 39,872. Stands used by departures: 410.
Taxi-out 2025: P10 655s, median 954s, P90 1444s, sd 453s, over 1h 0.2%. No NM record 1.6%. Off-block at schedule (±10s) 4.1%. BLOCK_TIME within 60s of NM AOBT 21.9%, median BLOCK_TIME - AOBT -4s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 08L | 31.4% | 656 / 903 / 1380 | 55 / 66 | 36 | 27.2% | 21.8% | 12.3% | 31.3% |
| 26R | 30.0% | 608 / 906 / 1450 | 55 / 66 | 36 | 27.9% | 36.3% | 49.5% | 25.3% |
| 09R | 13.4% | 653 / 1012 / 1557 | 56 / 116 | 31 | 19.1% | 12.2% | 6.7% | 22.3% |
| 27L | 11.2% | 659 / 908 / 1375 | 56 / 114 | 32 | 25.6% | 29.5% | 14.2% | 20.4% |
| 27R | 9.0% | 772 / 1030 / 1496 | 57 / 118 | 28 | 0.0% | 0.0% | 16.1% | 0.1% |
| 09L | 4.8% | 727 / 1085 / 1510 | 57 / 118 | 26 | 0.1% | 0.0% | 1.1% | 0.1% |

Arrival runways 2025: 26L 31.6%, 08R 29.0%, 09L 17.0%, 27R 15.9%, 09R 3.5%, 27L 2.1%.
Runways carrying both departures and arrivals: 09L, 09R, 27L, 27R.

#### LIRF

2025 departures 160,704, arrivals 160,504; 2026 ranking departures 26,899. Stands used by departures: 142.
Taxi-out 2025: P10 670s, median 1025s, P90 1689s, sd 1332s, over 1h 1.2%. No NM record 0.9%. Off-block at schedule (±10s) 18.1%. BLOCK_TIME within 60s of NM AOBT 18.6%, median BLOCK_TIME - AOBT -118s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 25 | 90.1% | 665 / 1013 / 1626 | 53 / 62 | 38 | 79.1% | 91.9% | 90.8% | 88.6% |
| 16R | 7.0% | 956 / 1332 / 2148 | 59 / 121 | 30 | 10.9% | 7.4% | 8.0% | 9.7% |
| 34L | 2.6% | 848 / 1211 / 1974 | 56 / 114 | 33 | 9.6% | 0.4% | 0.9% | 1.0% |

Arrival runways 2025: 16L 69.8%, 16R 14.5%, 34R 12.8%, 34L 2.8%.
Runways carrying both departures and arrivals: 16R, 34L.

#### LSZH

2025 departures 134,907, arrivals 134,691; 2026 ranking departures 23,150. Stands used by departures: 208.
Taxi-out 2025: P10 400s, median 711s, P90 1080s, sd 385s, over 1h 0.0%. No NM record 1.7%. Off-block at schedule (±10s) 1.1%. BLOCK_TIME within 60s of NM AOBT 24.2%, median BLOCK_TIME - AOBT -33s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 28 | 62.9% | 350 / 673 / 998 | 76 / 94 | 33 | 62.0% | 71.2% | 57.3% | 58.8% |
| 32 | 24.4% | 474 / 719 / 1044 | 87 / 107 | 33 | 26.4% | 17.2% | 31.3% | 28.8% |
| 16 | 9.6% | 697 / 973 / 1394 | 86 / 116 | 14 | 8.6% | 10.4% | 9.9% | 9.9% |
| 10 | 1.9% | 551 / 822 / 1193 | 88 / 118 | 26 | 1.2% | 0.0% | 0.0% | 0.9% |
| 34 | 1.0% | 513 / 876 / 1313 | 105 / 334 | 6 | 1.2% | 0.9% | 1.3% | 1.2% |

Arrival runways 2025: 14 69.8%, 28 20.1%, 34 9.7%.
Runways carrying both departures and arrivals: 28, 34.

#### LTFM

2025 departures 272,965, arrivals 273,886; 2026 ranking departures 47,581. Stands used by departures: 417.
Taxi-out 2025: P10 662s, median 963s, P90 1502s, sd 428s, over 1h 0.3%. No NM record 1.3%. Off-block at schedule (±10s) 5.1%. BLOCK_TIME within 60s of NM AOBT 8.7%, median BLOCK_TIME - AOBT +296s.

| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |
|---|---|---|---|---|---|---|---|---|
| 36 | 46.2% | 610 / 891 / 1266 | 54 / 62 | 39 | 34.0% | 25.6% | 53.3% | 51.7% |
| 35L | 29.3% | 671 / 1011 / 1567 | 58 / 118 | 34 | 21.3% | 17.3% | 37.8% | 39.2% |
| 18 | 11.6% | 779 / 1018 / 1389 | 55 / 64 | 37 | 26.5% | 30.3% | 1.7% | 2.6% |
| 17R | 7.7% | 890 / 1267 / 1868 | 58 / 119 | 34 | 17.1% | 19.5% | 1.5% | 1.9% |
| 34L | 3.2% | 907 / 1314 / 2689 | 55 / 112 | 33 | 0.0% | 4.3% | 4.6% | 4.3% |
| 35R | 1.2% | 669 / 1028 / 1729 | 57 / 115 | 34 | 0.2% | 0.4% | 0.9% | 0.1% |
| 16R | 0.6% | 1260 / 1672 / 2111 | 55 / 67 | 27 | 0.0% | 2.4% | 0.1% | 0.1% |

Arrival runways 2025: 35R 40.5%, 34L 30.0%, 17L 10.3%, 36 9.5%, 16R 7.9%, 18 1.8%.
Runways carrying both departures and arrivals: 16R, 18, 34L, 35R, 36.


