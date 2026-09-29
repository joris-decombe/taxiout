"""Airport weather from METAR reports, as features for each departure.

Weather reaches taxi-out in two ways. De-icing: in freezing or wet-cold
conditions aircraft are sprayed after leaving the stand, often at a remote
pad, which lands inside taxi-out and can add tens of minutes. Throughput: low
visibility procedures and strong winds cut how many aircraft a runway takes
per hour, so queues grow. January is half of the scored months, so the first
effect is where most of this is expected to pay.

Source: the Iowa Environmental Mesonet ASOS/METAR archive
(https://mesonet.agron.iastate.edu), which redistributes the global METAR
feed freely. Fetched once per airport into data/weather/, which is gitignored
like the rest of data/.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

from . import data, schema

WEATHER_DIR = data.DATA_DIR / "weather"
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")

# Training is 2025; the ranking set is January and July 2026.
START = (2025, 1, 1)
END = (2026, 8, 1)

URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={station}"
    "&data=tmpc&data=dwpc&data=vsby&data=wxcodes&data=sknt&data=gust&data=skyl1&data=skyc1"
    "&year1={y1}&month1={m1}&day1={d1}&year2={y2}&month2={m2}&day2={d2}"
    "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=null&trace=T&direct=no"
    "&report_type=3&report_type=4"
)

FEATURE_COLUMNS = [
    "wx_temp_c",
    "wx_dewpoint_spread_c",
    "wx_visibility_mi",
    "wx_wind_kt",
    "wx_gust_kt",
    "wx_ceiling_ft",
    "wx_precip",
    "wx_freezing",
    "wx_snow",
    "wx_fog",
    "wx_icing_risk",
    "wx_icing_hours_6h",
]


def fetch(weather_dir: Path = WEATHER_DIR) -> list[Path]:
    """Downloads each airport's METARs once; existing files are kept."""
    weather_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for station in AIRPORTS:
        path = weather_dir / f"{station}.csv"
        if not path.exists():
            url = URL.format(station=station, y1=START[0], m1=START[1], d1=START[2],
                             y2=END[0], m2=END[1], d2=END[2])
            path.write_bytes(_get(url))
        written.append(path)
    return written


def _get(url: str, attempts: int = 6) -> bytes:
    """The archive rate-limits with HTTP 429, so back off and retry."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(30 * (attempt + 1))
    raise AssertionError("unreachable")


def load(weather_dir: Path = WEATHER_DIR) -> pl.DataFrame:
    """One row per METAR, with the derived conditions, sorted by time."""
    frames = [
        pl.read_csv(path, null_values=["null", "M"], infer_schema_length=0)
        for path in sorted(weather_dir.glob("*.csv"))
    ]
    raw = pl.concat(frames)
    num = lambda c: pl.col(c).cast(pl.Float64, strict=False)  # noqa: E731
    codes = pl.col("wxcodes").fill_null("")
    temp = num("tmpc")
    wet = codes.str.contains("RA|DZ|SN|SG|PL|GR|GS|FZ|BR|FG")
    out = raw.select(
        pl.col("station").alias(schema.ADEP),
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("wx_time"),
        temp.alias("wx_temp_c"),
        (temp - num("dwpc")).alias("wx_dewpoint_spread_c"),
        num("vsby").alias("wx_visibility_mi"),
        num("sknt").alias("wx_wind_kt"),
        num("gust").alias("wx_gust_kt"),
        # Ceiling only when the lowest layer is broken or overcast.
        pl.when(pl.col("skyc1").is_in(["BKN", "OVC", "VV"])).then(num("skyl1")).otherwise(99_999.0)
        .alias("wx_ceiling_ft"),
        codes.str.contains("RA|DZ|SN|SG|PL|GR|GS").cast(pl.Int8).alias("wx_precip"),
        codes.str.contains("FZ").cast(pl.Int8).alias("wx_freezing"),
        codes.str.contains("SN|SG|PL").cast(pl.Int8).alias("wx_snow"),
        codes.str.contains("FG").cast(pl.Int8).alias("wx_fog"),
        # De-icing conditions, roughly: at or below 3C with visible moisture
        # or a small dewpoint spread (frost), or any freezing or frozen
        # precipitation.
        (
            ((temp <= 3) & (wet | ((temp - num("dwpc")) <= 2)))
            | codes.str.contains("FZ|SN|SG|PL")
        ).cast(pl.Int8).alias("wx_icing_risk"),
    ).drop_nulls("wx_time").sort(schema.ADEP, "wx_time")
    # How much of the previous six hours carried icing risk: frost and snow
    # left on a parked aircraft outlast the report that caused them.
    return out.with_columns(
        pl.col("wx_icing_risk").cast(pl.Float64)
        .rolling_mean_by("wx_time", window_size="6h").over(schema.ADEP)
        .alias("wx_icing_hours_6h")
    ).sort("wx_time")


def join(departures: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """The latest METAR at or before each takeoff, within two hours.

    Keeps the input's row order: the as-of join needs a sort, and callers
    line predictions up against their own frame.
    """
    order = departures.columns
    joined = (
        departures.with_row_index("_row")
        .sort(schema.MVT_TIME)
        .join_asof(
            weather, left_on=schema.MVT_TIME, right_on="wx_time", by=schema.ADEP,
            strategy="backward", tolerance="2h",
        )
        .sort("_row")
    )
    return joined.select(*order, *FEATURE_COLUMNS)
