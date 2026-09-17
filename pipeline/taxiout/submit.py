"""Writing submitting.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import data, schema

# A full day. The longest taxi-out in the 2025 training data is 131,167s,
# which is a data artifact rather than an aircraft, but the honest values run
# well past two hours and must not be flattened.
MAX_PLAUSIBLE_TAXI_SEC = 86_400.0


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
    # Taxi-out cannot be negative, so the floor stays.
    #
    # The ceiling used to be 7200s, on the reasoning that no European hub
    # taxis for over two hours. Hubs do: 4,126 training departures exceed an
    # hour and 584 exceed two, and under RMSE those rows are expensive --
    # clipping a 2025 validation run at 7200s cost 92s of RMSE (534s against
    # 442s unclipped). The cap is now only a guard against a runaway
    # prediction, set beyond anything the training data reaches.
    values = np.clip(values, 0.0, MAX_PLAUSIBLE_TAXI_SEC)

    submission = template.with_columns(pl.Series(schema.TAXITIME, values))
    submission.write_parquet(output_path)
    return output_path
