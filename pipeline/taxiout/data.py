"""Loading the challenge parquet files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import schema

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TRAINING_GLOB = "training_*.parquet"
# The final phase (announced 8 October 2026) scores one submission on four
# months of 2026: January, February, June and July. Its ranking set holds the
# original January and July departures unchanged, plus February and June.
RANKING_FILE = "final_ranking.parquet"
SUBMISSION_TEMPLATE = "final_submitting.parquet"
# The public leaderboard still scores January and July only, on this template.
LEADERBOARD_TEMPLATE = "submitting.parquet"

# The leaderboard months were January and July 2026, so validation holds out
# the same two months of 2025: taxi-out has a strong seasonal signal and these
# two months bracket it. Other folds (`experiments_round9.py`) check the rest.
VALIDATION_MONTHS = (1, 7)


def load_training(data_dir: Path = DATA_DIR) -> pl.LazyFrame:
    files = sorted(data_dir.glob(TRAINING_GLOB))
    if not files:
        raise FileNotFoundError(
            f"no {TRAINING_GLOB} under {data_dir} -- the bucket keys arrive with team approval"
        )
    return pl.scan_parquet(files)


def load_ranking(data_dir: Path = DATA_DIR) -> pl.LazyFrame:
    return pl.scan_parquet(data_dir / RANKING_FILE)


def load_submission_template(data_dir: Path = DATA_DIR, name: str = SUBMISSION_TEMPLATE) -> pl.DataFrame:
    return pl.read_parquet(data_dir / name)


def departures(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.filter(pl.col(schema.PHASE) == schema.DEPARTURE)


def train_validation_split(
    frame: pl.LazyFrame, months: tuple[int, ...] = VALIDATION_MONTHS
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    month = pl.col(schema.MVT_TIME).dt.month()
    is_validation = month.is_in(months)
    return frame.filter(~is_validation), frame.filter(is_validation)
