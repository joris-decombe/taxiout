"""Per-airport operating profile, measured from the challenge data.

Prints a markdown profile of each airport's departures: runway use and
taxi-out by runway (training), how runway use shifts between the 2025
validation months and the 2026 ranking months, stand counts, record
completeness, and the at-schedule off-block artifact rate.

Run from the repository root with the project venv:

    PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
      .venv/Scripts/python.exe .claude/skills/air-traffic-controller/scripts/airport_profile.py [ICAO ...]

With no arguments it profiles all ten airports. Taxi-out statistics come
from training only: the ranking set has no BLOCK_TIME for departures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "pipeline"))
from taxiout import data, schema as s  # noqa: E402


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def profile(airport: str, training: pl.DataFrame, ranking: pl.DataFrame) -> str:
    dep = training.filter((pl.col(s.PHASE) == s.DEPARTURE) & (pl.col(s.ADEP) == airport))
    arr = training.filter((pl.col(s.PHASE) == s.ARRIVAL) & (pl.col(s.ADES) == airport))
    rk_dep = ranking.filter((pl.col(s.PHASE) == s.DEPARTURE) & (pl.col(s.ADEP) == airport))
    taxi = pl.col(s.TAXITIME)
    gap = (pl.col(s.BLOCK_TIME) - pl.col(s.SCHED_TIME)).dt.total_seconds().abs()
    nm_gap = (pl.col(s.BLOCK_TIME) - pl.col(s.AOBT)).dt.total_seconds()

    lines = [f"### {airport}", ""]
    lines.append(
        f"2025 departures {dep.height:,}, arrivals {arr.height:,}; "
        f"2026 ranking departures {rk_dep.height:,}. "
        f"Stands used by departures: {dep[s.STAND].n_unique():,}."
    )
    q = dep.select(
        taxi.quantile(0.10).alias("p10"), taxi.median().alias("p50"),
        taxi.quantile(0.90).alias("p90"), taxi.std().alias("sd"),
        (taxi > 3600).mean().alias("over1h"),
        pl.col(s.AOBT).is_null().mean().alias("no_nm"),
        (gap <= 10).mean().alias("at_sched"),
        (nm_gap.abs() <= 60).mean().alias("nm_within_60s"),
        nm_gap.median().alias("nm_gap_median"),
    ).row(0, named=True)
    lines.append(
        f"Taxi-out 2025: P10 {q['p10']:.0f}s, median {q['p50']:.0f}s, P90 {q['p90']:.0f}s, "
        f"sd {q['sd']:.0f}s, over 1h {pct(q['over1h'])}. "
        f"No NM record {pct(q['no_nm'])}. Off-block at schedule (±10s) {pct(q['at_sched'])}. "
        f"BLOCK_TIME within 60s of NM AOBT {pct(q['nm_within_60s'])}, "
        f"median BLOCK_TIME - AOBT {q['nm_gap_median']:+.0f}s."
    )
    lines.append("")

    # Runway throughput: the gap between successive takeoffs on the same
    # runway, and the busiest clock hours. The low gap percentile is close to
    # the separation actually applied when the runway is saturated.
    spacing = (
        dep.sort(s.MVT_TIME)
        .with_columns(
            pl.col(s.MVT_TIME).diff().over(s.RUNWAY).dt.total_seconds().alias("gap_s"),
            pl.col(s.MVT_TIME).dt.truncate("1h").alias("hour"),
        )
        .group_by(s.RUNWAY)
        .agg(pl.col("gap_s").quantile(0.05).alias("gap_p05"), pl.col("gap_s").quantile(0.25).alias("gap_p25"))
    )
    hourly = (
        dep.with_columns(pl.col(s.MVT_TIME).dt.truncate("1h").alias("hour"))
        .group_by(s.RUNWAY, "hour").len()
        .group_by(s.RUNWAY).agg(pl.col("len").quantile(0.99).alias("hour_p99"))
    )
    by_rwy = (
        dep.group_by(s.RUNWAY)
        .agg(pl.len().alias("n"), taxi.quantile(0.10).alias("p10"), taxi.median().alias("p50"),
             taxi.quantile(0.90).alias("p90"))
        .with_columns((pl.col("n") / dep.height).alias("share"))
        .filter(pl.col("share") >= 0.005)
        .join(spacing, on=s.RUNWAY, how="left")
        .join(hourly, on=s.RUNWAY, how="left")
        .sort("n", descending=True)
    )

    def shares(frame: pl.DataFrame, months: tuple[int, ...]) -> dict[str, float]:
        f = frame.filter(pl.col(s.MVT_TIME).dt.month().is_in(months))
        if f.height == 0:
            return {}
        c = f.group_by(s.RUNWAY).len()
        return {r: n / f.height for r, n in c.iter_rows()}

    jan25, jul25 = shares(dep, (1,)), shares(dep, (7,))
    jan26, jul26 = shares(rk_dep, (1,)), shares(rk_dep, (7,))
    lines.append(
        "| Dep. runway | Share 2025 | Taxi P10 / median / P90 (s) | Takeoff gap P5 / P25 (s) | "
        "Deps/h P99 | Jan 25 | Jan 26 | Jul 25 | Jul 26 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in by_rwy.iter_rows(named=True):
        rw = r[s.RUNWAY]
        lines.append(
            f"| {rw} | {pct(r['share'])} | {r['p10']:.0f} / {r['p50']:.0f} / {r['p90']:.0f} | "
            f"{r['gap_p05']:.0f} / {r['gap_p25']:.0f} | {r['hour_p99']:.0f} | "
            f"{pct(jan25.get(rw, 0))} | {pct(jan26.get(rw, 0))} | {pct(jul25.get(rw, 0))} | {pct(jul26.get(rw, 0))} |"
        )
    arr_rwy = (
        arr.group_by(s.RUNWAY).len().with_columns((pl.col("len") / arr.height).alias("share"))
        .filter(pl.col("share") >= 0.01).sort("len", descending=True)
    )
    lines.append("")
    lines.append(
        "Arrival runways 2025: "
        + ", ".join(f"{r} {pct(sh)}" for r, _, sh in arr_rwy.iter_rows())
        + "."
    )
    both = set(by_rwy[s.RUNWAY].to_list()) & set(arr_rwy[s.RUNWAY].to_list())
    if both:
        lines.append(f"Runways carrying both departures and arrivals: {', '.join(sorted(both))}.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    airports = sys.argv[1:] or ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
    training = data.load_training().collect()
    ranking = data.load_ranking().collect()
    for airport in airports:
        print(profile(airport, training, ranking))


if __name__ == "__main__":
    main()
