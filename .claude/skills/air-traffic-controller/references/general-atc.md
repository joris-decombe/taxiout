# General air traffic control

The wider knowledge behind the surface-operations focus: who controls what,
how aircraft are separated, and the vocabulary. Standards are ICAO unless
stated; states publish differences in their AIP (GEN 1.7), so check the
state's AIP before relying on a number for a specific airport.

Contents: 1 Units and positions · 2 Airspace · 3 Flight rules and types ·
4 Separation · 5 Runway operations · 6 Flow management · 7 Phraseology ·
8 Sources

## 1 Units and positions

From the stand to the cruise, control passes through:

| Position | Responsibility |
|---|---|
| Clearance delivery (DEL) | Route clearance: SID, initial level, squawk; in A-CDM, start-up approval against TSAT |
| Apron / ramp | Pushback and movements on the apron; at some airports run by the airport operator, not the ANSP |
| Ground (GND) | Taxiways and runway crossings outside the runway strip in use |
| Tower (TWR) | The runway: line-up, take-off and landing clearances, departure sequence at the holding point |
| Approach / terminal (APP) | Arrivals and departures in the terminal area, arrival sequencing |
| Area control centre (ACC) | En-route |

The Network Manager (EUROCONTROL NMOC) is not a control unit: it balances
demand and capacity across Europe and allocates ATFM slots [doc].

## 2 Airspace

ICAO classes A–G. A: IFR only, all flights separated. B: IFR and VFR, all
separated. C: IFR separated from IFR and VFR; VFR gets traffic information on
other VFR. D: IFR separated from IFR; traffic information otherwise. E: IFR
separated from IFR; VFR not controlled. F and G: uncontrolled [doc: ICAO
Annex 11]. Hub control zones (CTR) and terminal areas are typically class C
or D in Europe; check the AIP [judgement].

## 3 Flight rules and types (as in the flight table)

- Flight rules (`FLIGHT_RULE_flt`): I = IFR, V = VFR, Y = IFR first then
  VFR, Z = VFR first then IFR [doc: ICAO flight plan item 8].
- Flight type (`FLIGHT_TYPE_flt`): S = scheduled air service, N =
  non-scheduled, G = general aviation, M = military, X = other [doc].
  The challenge data dictionary lists S, N, G and X; military flights were
  removed [doc: challenge data page].

## 4 Separation

- **Vertical**: 1,000 ft between IFR flights up to FL410 in RVSM airspace
  (FL290–FL410), 2,000 ft above [doc: ICAO Doc 4444].
- **Radar/surveillance horizontal**: 5 NM minimum, reducible to 3 NM where
  approved, typically in terminal areas; 2.5 NM between successive
  arrivals on the same final under specific conditions [doc: Doc 4444].
- **Wake turbulence, distance-based (arrivals and surveillance)**: behind a
  Heavy, 4 NM for a Heavy, 5 NM for a Medium, 6 NM for a Light; Light behind
  Medium 5 NM; behind a Super (A380), 6 NM Heavy, 7 NM Medium, 8 NM Light
  [doc: Doc 4444 §8.7].
- **Wake turbulence, time-based (departures)**: see surface-ops.md §4.
- **Wake categories**: Light ≤ 7,000 kg MTOW, Medium 7,000–136,000 kg,
  Heavy ≥ 136,000 kg, Super for the A380 [doc]. RECAT-EU refines this to six
  categories (A–F) [doc: EUROCONTROL].
- **Time-based separation on final** (TBS): arrivals spaced by time rather
  than distance, which recovers landing rate in strong headwinds; in use at
  Heathrow among others [secondary].

## 5 Runway operations

- **Modes**: segregated (one runway for arrivals, one for departures),
  mixed (both on one runway), independent parallel (simultaneous operations
  on runways far enough apart) [doc].
- **Runway in use** is chosen by ATC from wind (tailwind and crosswind
  limits), noise preferential runway schemes, and capacity. Noise rules
  often set the configuration by time of day (Zurich, Heathrow alternation)
  [doc: airport publications; see airports.md].
- **Runway incursion protection**: stop bars at holding points, and during
  low visibility aircraft hold further back at CAT II/III holding points
  to protect the ILS signal [doc: SKYbrary].
- **Landing minima** (approximate ICAO categories): CAT I decision height
  ≥ 200 ft and RVR ≥ 550 m; CAT II DH 100–200 ft, RVR ≥ 300 m; CAT III below
  that [doc: ICAO Annex 6]. LVP are in force for CAT II/III operations and
  low-visibility take-offs [doc].

## 6 Flow management

- **ATFM regulations**: when forecast demand exceeds the capacity of an
  airport or sector, NM issues a regulation and allocates departure slots
  (CTOT) to the flights concerned; they wait on the ground [doc].
- **Slot tolerance**: take-off must fall within CTOT −5 / +10 minutes
  [doc: ansperformance.eu].
- **A-CDM**: airport, airlines, handlers and ATC share milestone times
  (TOBT, TSAT, TTOT) so pre-departure sequencing can hold aircraft at the
  stand; A-CDM airports exchange DPI messages with NM [doc]. Details in
  surface-ops.md §1.
- **Airport slots** (IATA WSG) are a different thing: the planned schedule
  time a coordinated airport allocates to an airline, used for
  `SCHED_TIME` [doc].

## 7 Phraseology

Standard ICAO phraseology (Doc 9432) for a departure, in order [doc]:

- "*Callsign*, cleared to *destination* via *SID*, climb *level*, squawk
  *code*." (DEL)
- "*Callsign*, start-up approved." / "pushback approved, facing *direction*."
  (DEL/apron/GND)
- "*Callsign*, taxi to holding point *runway* via *taxiways*." (GND)
- "*Callsign*, line up and wait runway *runway*." (TWR)
- "*Callsign*, wind *direction/speed*, runway *runway*, cleared for
  take-off." (TWR)

European hubs also use "line up behind *traffic*" (conditional clearance)
and multiple line-up at intermediate take-off positions, which is where the
3-minute intermediate-departure wake rule applies [judgement].

## 8 Sources

- ICAO Doc 4444, Procedures for Air Navigation Services – Air Traffic
  Management (PANS-ATM). Paywalled; figures here are the widely published
  values and should be checked against the edition in force.
- ICAO Annex 11 (Air Traffic Services), Annex 6 (Operation of Aircraft),
  Doc 9432 (Manual of Radiotelephony).
- EUROCONTROL RECAT-EU:
  https://www.eurocontrol.int/publication/european-wake-turbulence-categorisation-and-separation-minima-approach-and-departure
- SKYbrary: https://skybrary.aero
- Aviation Intelligence Portal definitions: https://ansperformance.eu/definition/
