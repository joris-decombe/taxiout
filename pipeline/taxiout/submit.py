"""Predicting the ranking set, writing the submission, and checking it.

The dangerous failure here is silent. `submitting.parquet` and
`ranking.parquet` hold the same 344,841 IDs in a different order, so a
positional assignment instead of a keyed join produces a perfectly valid file
that scores like noise, indistinguishable from a bad model. Everything below
joins on MVT_ID, and `verify_submission` re-reads the file from disk to prove
it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import data, features, schema, train

# Two days. The longest taxi-out in the 2025 training data is 131,167s,
# which is a data artifact rather than an aircraft, but the honest values run
# well past two hours and must not be flattened, and LIRF's day-shifted
# orphans (train.DAY_SHIFT) are predicted at a day plus a normal taxi, which
# a one-day guard used to cut by about 1,000s.
MAX_PLAUSIBLE_TAXI_SEC = 172_800.0

# How far the submission's median may drift from the training target's
# before verification refuses it. Loose on purpose: January and July differ
# from the annual median, and the check is for a scrambled or unit-confused
# file (a median in minutes, or centred on zero), not for a model shift.
MEDIAN_TOLERANCE = 0.5


def predict_ranking(model: train.TrainedModel, data_dir: Path = data.DATA_DIR) -> dict[int, float]:
    """Predicted taxi-out per departure MVT_ID on the ranking set.

    The surroundings are built from the whole ranking set, arrivals
    included, exactly as training builds them from the whole training set.
    The row features are built from departures only: the rolling congestion
    windows count departures in training, because `train` filters to them
    first, so counting arrivals here would shift them away from what the
    boosters learnt.
    """
    ranking = data.load_ranking(data_dir)
    around = features.surroundings(ranking)
    frame = features.build(data.departures(ranking), model.unimpeded.lazy(), around).collect()
    predictions = model.predict(frame)
    ids = frame.select(schema.MVT_ID).to_numpy().ravel()
    return {int(i): float(p) for i, p in zip(ids, predictions)}


def write_submission(
    predictions: dict[int, float],
    output_path: Path,
    data_dir: Path = data.DATA_DIR,
    template_name: str = data.SUBMISSION_TEMPLATE,
) -> Path:
    """Fills the template with predicted taxi-out seconds.

    Refuses to write a partial file: a missing MVT_ID means the ranking set
    has departures the pipeline dropped somewhere, and shipping zeros for
    those would quietly wreck the RMSE.
    """
    template = data.load_submission_template(data_dir, template_name)
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

    # The template's target column is Int32. Keep its dtype rather than
    # guess whether the scorer casts: rounding costs under half a second on
    # any row, which is nothing against a ~400s RMSE.
    target_dtype = template.schema[schema.TAXITIME]
    submission = template.with_columns(
        pl.Series(schema.TAXITIME, np.rint(values)).cast(target_dtype)
    )
    submission.write_parquet(output_path)
    return output_path


def verify_submission(
    path: Path,
    predictions: dict[int, float],
    data_dir: Path = data.DATA_DIR,
    template_name: str = data.SUBMISSION_TEMPLATE,
) -> list[str]:
    """Problems with the file on disk. Empty means it is safe to upload.

    Reads the written file back rather than trusting the frame that produced
    it, so the parquet round-trip is part of what is checked.
    """
    problems: list[str] = []
    written = pl.read_parquet(path)
    template = data.load_submission_template(data_dir, template_name)

    if written.schema != template.schema:
        problems.append(f"schema {dict(written.schema)} != template {dict(template.schema)}")
    if written.height != template.height:
        problems.append(f"{written.height} rows, template has {template.height}")
    if written[schema.MVT_ID].is_duplicated().any():
        problems.append("duplicate MVT_IDs")

    # Keyed, not positional: every template ID must be present exactly once.
    joined = template.select(schema.MVT_ID).join(
        written, on=schema.MVT_ID, how="left", validate="1:1"
    )
    target = joined[schema.TAXITIME]
    if target.null_count():
        problems.append(f"{target.null_count()} template IDs have no value after a keyed join")
    as_float = target.cast(pl.Float64)
    if not as_float.drop_nulls().is_finite().all():
        problems.append("non-finite values")
    if (as_float < 0).any():
        problems.append("negative values")

    # The written value for each ID must be the prediction made for that ID.
    # This is the test a positional mix-up fails and every other check passes.
    expected = np.clip(
        np.array([predictions[int(i)] for i in joined[schema.MVT_ID]], dtype=float),
        0.0,
        MAX_PLAUSIBLE_TAXI_SEC,
    )
    worst = float(np.max(np.abs(as_float.fill_null(np.nan).to_numpy() - expected)))
    if not worst <= 0.5:
        problems.append(f"written values differ from predictions by up to {worst:.1f}s")

    # Against the training target, to catch unit or centring mistakes.
    reference = (
        data.departures(data.load_training(data_dir))
        .select(pl.col(schema.TARGET).median())
        .collect()
        .item()
    )
    median = float(as_float.median())
    if abs(median - reference) > MEDIAN_TOLERANCE * reference:
        problems.append(f"median {median:.0f}s is far from the training median {reference:.0f}s")

    return problems


def build_submission(output_path: Path, data_dir: Path = data.DATA_DIR) -> Path:
    """Fits on all of training, predicts the ranking set, writes and verifies.

    Writes a local file only. Uploading is a separate, deliberate step:
    `bucket.upload_submission` with N from `bucket.next_version()`.
    """
    model = train.train_final(data.load_training(data_dir))
    predictions = predict_ranking(model, data_dir)
    write_submission(predictions, output_path, data_dir)
    problems = verify_submission(output_path, predictions, data_dir)
    if problems:
        raise ValueError("submission failed verification:\n  " + "\n  ".join(problems))
    return output_path
