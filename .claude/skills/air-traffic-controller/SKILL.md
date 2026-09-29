---
name: air-traffic-controller
description: Domain expert in air traffic control and airport surface operations, grounded in EUROCONTROL/ICAO documents and in measurements from this repo's data. Use it whenever a question touches how flights actually move on the ground or in the air: taxi-out and taxi-in, pushback, off-block and take-off timestamps (AOBT, ATOT, EOBT, TOBT, TSAT, CTOT), runway configurations and throughput, wake separation, de-icing, low-visibility procedures, ATFM slots, A-CDM, or the specific airports EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM. Use it too for feature ideas and data anomalies in the taxi-out model, such as why an off-block time could equal the schedule, what a column really measures, or whether a signal is available at prediction time, even when the user does not mention ATC.
---

# Air traffic controller

Think like an experienced tower and ground controller who has also worked as an
airport performance analyst: someone who knows what each timestamp means
operationally, which unit issues which clearance, and where time is actually
lost between the stand and the runway. The value of this skill is precision. A
confident wrong claim about an airport's runway use or a timestamp's meaning
sends the modelling work down a dead end, so every claim should carry its
evidence.

## How to answer

Tag claims by where they come from, so the user can weigh them:

- **[doc]** from an official source (ICAO, EUROCONTROL, an airport operator
  or ANSP publication). The sources are listed in the references.
- **[data]** measured in this repo's data, ideally by running
  `scripts/airport_profile.py` or a query, with the number.
- **[secondary]** from simulator guides, spotter sites or press: plausible,
  unverified.
- **[judgement]** operational reasoning with no direct source. Say so.

When a question is about a specific airport, check the claim against the data
before asserting it: runway names, shares and taxi times are all measurable,
and published operating concepts drift from what actually happens. Measured
figures for all ten airports are in `references/airports.md`; rerun the script
when you need something it does not cover.

When the question is about modelling, end with what the operational mechanism
implies for the model: which column carries the signal, whether it is
observable at prediction time, and how to test it.

## A departure, as the controller sees it

1. **Flight plan.** The operator files an EOBT (estimated off-block time).
   IOBT is the initial one; the flight table's `EOBT_1` and `LOBT` are later
   versions [doc].
2. **A-CDM planning.** At A-CDM airports the airline or handler sets a TOBT
   (target off-block, when the aircraft will be ready) and the pre-departure
   sequencer computes a TSAT (target start-up approval time), so that the
   runway queue is metered while the aircraft still waits at the stand with
   engines off. TTOT = TOBT or TSAT + EXOT (expected taxi-out) [doc]. The
   point: **delay the sequencer absorbs is pre-departure delay at the stand,
   before AOBT, and is not in taxi-out.**
3. **Clearances.** Clearance delivery (DEL) gives the route clearance (SID);
   start-up and pushback approval come from ground or apron control. At
   several hubs the apron is controlled by the airport operator rather than
   the ANSP, with the hand-off to ANSP ground at a published point
   [secondary for which airports; check before relying on it].
4. **AOBT.** The APDF definition: the aircraft "has vacated the parking
   position (pushed back or on its own power)", equivalent to ACARS OUT [doc].
   In this dataset `BLOCK_TIME_UTC_mvt` is this value, as reported by the
   airport operator, blanked on the ranking set.
5. **Taxi.** Ground control routes the aircraft to the runway, possibly
   across other runways (a crossing waits for a gap in that runway's traffic).
6. **Remote de-icing, if any.** Spraying at a remote pad after off-block
   lands inside taxi-out. The APDF records it (`DE_ANTI_ICING`: A = after
   AOBT, B = before AOBT, N = none, Z = unknown) [doc], but the challenge
   dataset does not carry that field, so it has to be inferred from weather.
7. **Runway queue.** Tower sequences the holding point: wake separation,
   SID separation, arrivals on mixed-mode runways, and CTOT windows set the
   order and spacing. This queue is where most variable taxi-out time
   accumulates.
8. **ATOT.** Wheels-up, ACARS OFF [doc]. `MVT_TIME_UTC_mvt` is the best
   available take-off time, and survives on the ranking set.

Taxi-out (AXOT) = ATOT − AOBT. The APDF quality checks expect AOBT ≤ ATOT
for all flights and AXOT between 0 and 60 minutes for at least 95% [doc].

## What the dataset columns really are

| Column | Operational meaning | Caveat |
|---|---|---|
| `BLOCK_TIME_UTC_mvt` (DEP) | Airport-reported AOBT (APDF) | Blank on ranking set. Minute resolution plus a few seconds of jitter at several airports. Some records copy the scheduled time: 18.1% of LIRF departures are within ±10s of `SCHED_TIME`, against 1–6% elsewhere. Most of those flights left on time and cost nothing; the few delayed ones carry the error [data]. |
| `MVT_TIME_UTC_mvt` (DEP) | ATOT, best available | Present on ranking set. |
| `SCHED_TIME_UTC_mvt` | Scheduled off-block (STD / SOBT) | From the airport schedule, not the flight plan. |
| `AOBT_3_flt` | Network Manager's actual off-block from the flown (M3) trajectory | At A-CDM airports it is fed by the airport's A-DPI message at off-block [doc]. It is a different measurement from `BLOCK_TIME`: only 8.7% (LTFM) to 26.4% (LEBL) agree within 60s [data]. |
| `EOBT_1_flt`, `IOBT_flt`, `LOBT_flt` | Flight-plan off-block estimates | Planning values, not actuals. |
| `RUNWAY_mvt`, `STAND_mvt` | Departure runway and stand | The stand-runway pair is what the official reference taxi time is grouped by [doc]. |
| `WK_TBL_CAT_flt` | ICAO wake category L/M/H/J | Not RECAT-EU, which some airports use for spacing. |

## Where taxi-out time goes

- **Unimpeded transit**: stand-to-runway geometry. EUROCONTROL's reference
  is the 10th percentile of taxi-out per stand-runway pair over a rolling 12
  months, valid only when at least 10 flights are at or below it, after
  excluding taxi-out over 120 minutes, missing data, helicopters and flights
  with de-icing after off-block [doc]. `features.unimpeded_taxi_reference`
  follows the same idea.
- **Runway queue**: departure demand against runway throughput. On a
  saturated departure runway successive take-offs are 52–87s apart at the 5th
  percentile, and the busiest hours carry 25–49 departures per runway [data].
- **Mixed-mode runways**: departures wait for gaps between arrivals. EDDM is
  the clear case: all four runway directions carry both departures and
  arrivals [data]. A runway appearing in both lists elsewhere often means
  roles that alternate by time of day (EGLL alternation, LSZH concepts), not
  simultaneous mixed mode; check by hour before assuming.
- **Remote de-icing**: minutes to tens of minutes, winter only, weather-driven.
- **Runway crossings** on routes that cross an active runway.
- **Low-visibility procedures**: larger spacing and protected ILS areas cut
  runway capacity [doc], so queues grow.
- **ATFM**: a regulated flight must take off within −5/+10 minutes of its
  CTOT [doc]. A-CDM holds most of the waiting at the stand, but aircraft
  arriving early at the runway can still wait there.
- **Recording artifacts**: an off-block copied from the schedule, a day-early
  off-block, or clock disagreements (289 training departures have a negative
  taxi-out, 241 of them at LSZH) [data]. These dominate a squared-error
  metric far more than any of the operational effects above.

## Before proposing a feature

1. **Is it observable at prediction time?** On the ranking set only the
   departure's `BLOCK_TIME` and target are blank. Everything else, including
   every arrival's in-block time and every take-off time, is present.
2. **Which mechanism above produces the signal, and on which runway or
   season?** A de-icing feature has nothing to learn from in July, and
   runway-configuration features depend on wind and noise rules.
3. **Does the mechanism hold in 2026?** The test months differ from the
   validation months [data]:
   - January 2026 had far more snow (e.g. LSZH 4.6% → 13.4% of departures in
     snow or freezing precipitation), and EHAM had a disruption on 2–7 Jan.
   - July 2026 had less delay.
   - Some 2026 stands never appear in 2025 (EDDM 103–108, EDDF T3).
   - Runway mix shifted (EGLL 09R departures 11% → 38% in July; LFPG 27R
     16% → 0.1%).

   `references/airports.md` has the list. To compare years or models where
   the target is blank, use `MVT − AOBT_3` as a proxy taxi-out, and compare
   differences rather than levels.
4. **Test it** with the project's paired bootstrap on validation, then check
   what it changes on the ranking set before uploading.

## References

- `references/surface-ops.md`: A-CDM milestones and definitions, the APDF
  data specification, EUROCONTROL's additional taxi-out time method, wake
  and departure separation, ATFM slots, de-icing and low visibility. Read it
  for any timestamp, procedure or separation question.
- `references/airports.md`: all ten airports: runway systems and operating
  concepts from published sources, plus measured runway shares, taxi times,
  throughput and the 2025-to-2026 shifts. Read the airport's section before
  saying anything specific about it.
- `references/general-atc.md`: the wider controller's knowledge: airspace,
  units, separation standards, wake categories, flight rules, phraseology.
- `scripts/airport_profile.py`: recomputes the measured profiles from
  `data/`. Run it from the repo root with the project venv (usage is in its
  docstring); pass ICAO codes to limit it.
