# Surface and departure operations: definitions and methods

Contents: 1 Timestamps and A-CDM · 2 The APDF data specification ·
3 EUROCONTROL additional taxi-out time · 4 Wake and departure separation ·
5 ATFM slots · 6 De-icing · 7 Low visibility · 8 Sources

Tags as in SKILL.md: [doc], [data], [secondary], [judgement].

## 1 Timestamps and A-CDM

| Term | Meaning | Set by |
|---|---|---|
| SOBT / STD | Scheduled off-block time | Airline schedule, airport slot |
| EOBT | Estimated off-block time in the flight plan | Aircraft operator |
| IOBT | Initial off-block time (first known EOBT) | NM |
| TOBT | Target off-block time: when the aircraft will be ready | Airline or handler (A-CDM) |
| TSAT | Target start-up approval time | Pre-departure sequencer (A-CDM) |
| AOBT | Actual off-block time | Airport (APDF), also sent to NM by A-DPI |
| EXOT | Estimated taxi-out time, used to compute TTOT | A-CDM platform |
| TTOT | Target take-off time: TOBT or TSAT plus EXOT | A-CDM platform |
| CTOT | Calculated take-off time, from an ATFM regulation | NM |
| ATOT | Actual take-off time (wheels-up) | Airport (APDF), ATC, NM |
| AXOT | Actual taxi-out time = ATOT − AOBT | Derived |

[doc: EUROCONTROL A-CDM specification 2025; APDF specification]

A-CDM airports send Departure Planning Information (DPI) messages to the
Network Manager so that its flight model uses the airport's taxi time and
TTOT; the A-DPI is sent at actual off-block [doc: NM DPI implementation
guide]. The dataset's `AOBT_3_flt` is the NM's actual off-block on the flown
(M3) trajectory [doc: challenge data page], so at A-CDM airports it should
largely reflect A-DPI. It still differs from the airport's APDF AOBT: the
median of BLOCK_TIME − AOBT_3 ranges from −118s (LIRF) to +296s (LTFM), and
agreement within 60s from 8.7% (LTFM) to 26.4% (LEBL) [data]. Why they differ
is not documented: different capture points (pushback start vs. parking
position vacated), manual entry, and message timing are all plausible
[judgement].

A-CDM status. Fully implemented A-CDM airports include Amsterdam,
Barcelona, Frankfurt, London Heathrow, Madrid, Munich, Paris CDG, Rome
Fiumicino and Zurich [secondary: Wikipedia list; doc: NM Network Operations
Report 2025 Annex II lists 33 NM-connected A-CDM airports at end-2025].
Istanbul (LTFM) does not appear in those lists, which fits its much weaker
AOBT agreement [judgement, data].

Design intent of A-CDM: meter the runway queue by holding aircraft at the
stand until TSAT, so that "departure and arrival delay needed for other
reasons than sequencing and metering" is absorbed at the stand [doc: ATXOT
indicator document, 2.2]. Consequence for taxi-out prediction: the queue that
remains after off-block is the short-term, un-metered part, plus whatever the
sequencer got wrong [judgement].

## 2 The APDF data specification (EUROCONTROL-SPEC-175)

The movement table comes from the Airport Operator Data Flow: airports report
every movement monthly to EUROCONTROL [doc]. Definitions that matter here:

- **AOBT (ODI 11)**: "the actual date and time the aircraft has vacated the
  parking position (pushed back or on its own power)"; alternative
  definition "time the aircraft pushes back / vacates the parking position
  (equivalent to Airline / Handlers ATD, ACARS=OUT)". Format
  `DD-MM-YYYY hh:nn:ss`. Quality checks: 100% of flights with AOBT ≤ ATOT;
  at least 95% with AXOT > 0 min; at least 95% with AXOT < 60 min; at least
  95% of airport-vs-airline values within ±3 min [doc].
- **ATOT (ODI 12)**: "the date and time that an aircraft has taken off from
  the runway (wheels-up)", equivalent to ACARS OFF [doc].
- **Scheduled time of departure (ODI 09)**: when the flight is scheduled to
  depart from the stand [doc].
- **De-icing (ODI 17)**: `A` after AOBT (remote), `B` before AOBT (on
  stand), `N` none, `Z` unknown [doc]. Not in the challenge data.
- Delay codes (IATA) include 75 "de-icing of aircraft" and ATC codes for
  start-up or pushback delay and for de-icing [doc].

The dataset's negative taxi-outs (289 training departures, 241 at LSZH)
violate the AOBT ≤ ATOT check [data], so the published quality checks are not
enforced on the challenge extract.

## 3 EUROCONTROL additional taxi-out time

The indicator measures "the accumulated (i.e. additional) time spent in the
departure queue during taxi operations on the apron and taxiway system,
including queuing at the runway threshold" [doc: ATXOT indicator, Ed. 01.00,
March 2023].

- Actual taxi-out time = ATOT − AOBT.
- Reference (unimpeded) time per **stand–departure runway** pair = **10th
  percentile** of taxi-out over the **rolling 12 months** up to the month
  analysed, selected by local take-off time.
- A reference is valid only with **at least 10 flights** at or below it;
  flights of other pairs get no reference.
- Excluded before computing: taxi-out **> 120 min**, missing runway, stand
  or off-block, **helicopters**, **de-icing after AOBT**.
- Aircraft type is deliberately not a grouping factor: it fragments samples
  without changing results much.
- Factors acknowledged as not captured: taxi route, taxi speed, special
  events such as apron works.

Additional time = actual − reference. Airports' monthly averages are published
on ansperformance.eu.

## 4 Wake and departure separation

ICAO wake turbulence categories by maximum take-off mass: **Light** ≤ 7,000
kg, **Medium** 7,000–136,000 kg, **Heavy** ≥ 136,000 kg, and **Super (J)**
for the A380 [doc: ICAO Doc 4444].

ICAO time-based departure wake separation, same runway (Doc 4444 §5.8.3;
check the edition in force):

- 2 minutes: Light or Medium taking off behind a Heavy; Light behind a Medium.
- 3 minutes: the same pairs when the follower departs from an intermediate
  part of the runway.
- Behind a Super: 2 minutes for a Heavy, 3 minutes for a Medium or Light.

RECAT-EU splits Heavy into Upper and Lower Heavy and uses six categories
(A–F), with departure separations from 80 to 240 seconds depending on the
pair; the pair-wise variant saves up to 30 seconds more on some frequent
departure pairs [doc: EUROCONTROL RECAT-EU]. Which of the ten airports apply
RECAT-EU on departure is not established here; the dataset only carries ICAO
categories (`WK_TBL_CAT_flt`).

Beyond wake, departure spacing depends on SID geometry: departures on
diverging routes can go closer together than two on the same route [doc:
ICAO Doc 4444 departure separation provisions]. Measured on saturated
runways in this data: 5th-percentile gap between take-offs 52–87s, 99th
percentile of hourly departures 25–49 per runway [data].

## 5 ATFM slots

A regulated flight receives a CTOT and must take off in the **slot tolerance
window, 5 minutes before to 10 minutes after the CTOT** [doc: ansperformance
slot tolerance definition]. The window exists for ATC to sequence
departures [secondary]. ATFM delay itself is taken at the stand, before
AOBT [doc: ATXOT 2.2]. A CTOT is not in the challenge data. NM regulation
data (which airports and sectors were regulated, when) is published on the
Aviation Intelligence Portal and could show when the network was
constrained [judgement: availability and licence to verify].

## 6 De-icing

- Types: on-stand (before AOBT, not in taxi-out) or remote/pad (after
  AOBT, inside taxi-out) [doc: APDF ODI 17].
- Remote facilities: Frankfurt operates several de-icing pads; Munich
  operates remote de-icing areas and de-ices about 9,000 aircraft in an
  average winter; Heathrow has two sets of remote de-icing pads on the way to
  the runway [secondary: airport and trade-press summaries; Frankfurt's CDM
  site publishes a de-icing operations document].
- Holdover time: anti-icing fluid protects for a limited time, which is why
  remote pads sit near runway ends [secondary].
- EUROCONTROL excludes flights de-iced after AOBT from the reference taxi
  time, because de-icing is not queueing [doc].
- Signal in this data: no de-icing field. Which airports de-ice after
  off-block has to be read from the data. Remote de-icing raises taxi-out in
  cold weather; on-stand de-icing raises off-block delay (BLOCK − SCHED)
  instead. Taxi-in barely moves on snow days (+11 to +68s), so a departure
  excess in snow is de-icing, not slower taxiing [data].

  | Airport | Winter 2025 reading | Evidence |
  |---|---|---|
  | LFPG | Remote, largest effect: taxi-out excess about +1,100s in snow or freezing precipitation | [data] |
  | EDDM | Remote, about +670s in snow | [data] |
  | LTFM | Remote, about +570–650s mean in snow; February 2025 only | [data] |
  | LSZH | Remote, moderate (+200–430s) | [data] |
  | EHAM | Remote, moderate (+130–390s), and NM AOBT absorbs part of it | [data] |
  | EDDF | Weak: the median is flat, only the tail grows (>10 min excess 2% → 10%) | [data] |
  | EGLL | Mostly on stand: taxi-out barely rises, off-block delay goes from 116s to 900s in snow | [data], despite published remote pads [doc: HADIP] |
  | LEMD | Frost mornings only, small | [data] |
  | LEBL, LIRF | Nothing measurable | [data] |

  The magnitudes come from crude baselines (two independent measurements
  agree on the ranking, not the exact seconds). Treat them as an ordering.
- **NM AOBT under snow**: at EDDM, LFPG, LSZH and EDDF, AOBT_3 runs 100–300s
  later than BLOCK_TIME in snow hours, so `MVT − AOBT_3` understates taxi-out
  exactly when de-icing happens. At EHAM and EGLL the gap does not move with
  snow [data].
- **A live de-icing signal** from other flights: the median over the
  airport's other departures within ±30–60 min of `(MVT − AOBT_3)` minus
  their stand-runway reference. On icing rows it correlates 0.36–0.56 with a
  flight's taxi-out excess, against 0.07–0.46 for the METAR features [data].
  It needs no BLOCK_TIME, so it is legal on the ranking set. Leave the row
  itself out.

## 7 Low visibility

Low-visibility procedures (LVP) apply below published RVR or ceiling
thresholds and protect ILS sensitive areas: aircraft hold at more distant
CAT II/III holding points and spacing increases, so runway capacity drops
[doc: SKYbrary]. Airport-specific LVP departure rates are in each AIP
(AD 2.22) [judgement: not retrieved]. In this data, visibility and ceiling
come from METARs (`wx_visibility_mi`, `wx_ceiling_ft`, `wx_fog`).

## 8 Sources

- PRC Data Challenge 2026 data description:
  https://prc-data-challenge-2026.netlify.app/data.html
- EUROCONTROL, Additional taxi-out time performance indicator document,
  Ed. 01.00, 16 March 2023:
  https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf
- EUROCONTROL, Airport Operator Data Flow data specification, Ed. 00-11:
  https://www.eurocontrol.int/sites/default/files/2019-10/ao-data-flow-specs.pdf
- EUROCONTROL Specification for A-CDM, Ed. 1.0, January 2025:
  https://www.eurocontrol.int/sites/default/files/2025-01/eurocontrol-specification-for-acdm.pdf
- EUROCONTROL NM DPI Implementation Guide, Ed. 2.700:
  https://www.eurocontrol.int/sites/default/files/2025-06/eurocontrol-dpi-impl-guide-2-700.pdf.pdf
- EUROCONTROL Network Operations Report 2025, Annex II Airports:
  https://www.eurocontrol.int/sites/default/files/2026-05/eurocontrol-annual-network-operations-report-2025-annex-ii.pdf
- Slot tolerance window: https://ansperformance.eu/definition/slot-tolerance-window/
- RECAT-EU: https://www.eurocontrol.int/publication/european-wake-turbulence-categorisation-and-separation-minima-approach-and-departure
- EASA assignment of ICAO types to RECAT-EU categories:
  https://www.easa.europa.eu/en/assignment-icao-aircraft-types-recat-eu-wake-turbulence-categories
- SKYbrary, Low Visibility Procedures: https://www.skybrary.aero/index.php/Low_Visibility_Procedures
- Heathrow Aircraft De-icing Plan (HADIP):
  https://www.heathrow.com/content/dam/heathrow/web/common/documents/company/team-heathrow/airside/winter-operations/HADIP-Winter-2019-20.pdf
