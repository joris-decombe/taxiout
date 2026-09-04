"""Writing submitting.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import data, schema


def write_submission(
    predictions: dict[int, float],
    output_path: Path,
    data_dir: Path = data.DATA_DIR,
) -> Path:
    """Fills the template with predicted taxi-out seconds.

    Refuses to write a partial file: a missing MVT_ID means the ranking set
    has departures the pipeline dropped somewhere, and shipping zeros for
    those would quietly wreck the RMSE.
    """
    template = data.load_submission_template(data_dir)
    ids = template.select(schema.MVT_ID).to_numpy().ravel()

    missing = [int(i) for i in ids if int(i) not in predictions]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(ids)} template rows have no prediction "
            f"(first few: {missing[:5]})"
        )

    values = np.array([predictions[int(i)] for i in ids], dtype=float)
    # Taxi-out cannot be negative, and no European hub taxis for over two hours.
    values = np.clip(values, 0.0, 7200.0)

    submission = template.with_columns(pl.Series(schema.TAXITIME, values))
    submission.write_parquet(output_path)
    return output_path
