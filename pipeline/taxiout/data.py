"""Loading the challenge parquet files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import schema

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TRAINING_GLOB = "training_*.parquet"
RANKING_FILE = "ranking.parquet"
SUBMISSION_TEMPLATE = "submitting.parquet"

# The test set is January and July 2026, so validation holds out the same two
# months of 2025. Any other split flatters the model: taxi-out has a strong
# seasonal signal and these two months bracket it deliberately.
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


def load_submission_template(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return pl.read_parquet(data_dir / SUBMISSION_TEMPLATE)


def departures(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.filter(pl.col(schema.PHASE) == schema.DEPARTURE)


def train_validation_split(frame: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    # Split on off-block, not MVT_TIME: for a departure MVT_TIME is the
    # takeoff time, so a flight that pushed back at 23:50 on 31 January
    # would land in the training half purely because it taxied past
    # midnight -- exactly the flights whose taxi-out was worst.
    month = pl.col(schema.BLOCK_TIME).dt.month()
    is_validation = month.is_in(VALIDATION_MONTHS)
    return frame.filter(~is_validation), frame.filter(is_validation)
