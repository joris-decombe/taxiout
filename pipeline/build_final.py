"""Builds the final submission end to end, from the challenge data and adsb.lol.

1. ADS-B: streams every day the build needs from the adsb.lol archive and
   observes each departure (`taxiout.adsb`): the 2025 months of the folds
   below, and the four final months of 2026. Days already done are
   skipped; `--skip-fetch` skips streaming altogether.
2. Folds (`train.validate`): the model fitted without each pair of months
   in FOLDS, with honest held-out predictions for that pair.
3. The ADS-B corrector (`taxiout.correct`) fitted on every fold's held-out
   predictions at once (`experiments_round9.py`: pooling January, July,
   February and December beat January and July alone on both).
4. The final fit on all twelve months, its predictions for the final
   ranking set (January, February, June and July 2026), corrected, orphans
   floored at their airport's live taxi level (`correct.rules`), written to
   `data/submission_final.parquet` and verified. The same predictions on
   the leaderboard's January and July template go to
   `data/submission_leaderboard.parquet`.

About two hours per fold and two for the final fit on a 20-thread machine
with 32 GB, plus three to four minutes of streaming per day the first time
(about 4 GB read per day, under 1% kept). Each fitted stage is saved under
`data/build_final/` and reused on a rerun, so an interrupted build resumes,
and adding a fold refits only that fold and the corrector.

Usage: python pipeline/build_final.py [--skip-fetch]
"""

import calendar
import datetime as dt
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import adsb, correct, data, features, schema, submit, train  # noqa: E402

FOLDS = [(1, 7), (2, 12), (6, 8)]
FINAL_MONTHS = [(2026, 1), (2026, 2), (2026, 6), (2026, 7)]
OUTPUT = Path("data/submission_final.parquet")
LEADERBOARD_OUTPUT = Path("data/submission_leaderboard.parquet")
STAGES = Path("data/build_final")


def month_days(year: int, month: int) -> list[dt.date]:
    return [dt.date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]


FOLD_DAYS = [day for fold in FOLDS for m in fold for day in month_days(2025, m)]
FINAL_DAYS = [day for y, m in FINAL_MONTHS for day in month_days(y, m)]


def stage(name: str, make):
    """`make()`'s result, saved the first time and read back on later runs."""
    path = STAGES / f"{name}.pkl"
    if path.exists():
        log(f"reusing {path}")
        return pickle.load(open(path, "rb"))
    value = make()
    STAGES.mkdir(parents=True, exist_ok=True)
    pickle.dump(value, open(path, "wb"))
    return value


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def fold_rows(training: pl.LazyFrame, months: tuple[int, int]) -> pl.DataFrame:
    """The corrector's inputs for one fold's held-out months."""

    def make():
        model, held_out, _ = train.validate(training, months)
        log(f"fold {months}: RMSE before the correction {model.validation_rmse:.1f}s")
        # Kept so each rule can be switched off on held-out months (experiments_round10.py).
        STAGES.mkdir(parents=True, exist_ok=True)
        pickle.dump(model, open(STAGES / f"fold_{months[0]}_{months[1]}_model.pkl", "wb"))
        return correct.inputs(model, held_out)

    # (1, 7) keeps the name it had when it was the only fold.
    return stage("validation_rows" if months == (1, 7) else f"fold_{months[0]}_{months[1]}_rows", make)


def main() -> None:
    days = FOLD_DAYS + FINAL_DAYS
    if "--skip-fetch" not in sys.argv:
        for day in days:
            if not (adsb.OUT_DIR / f"{day.isoformat()}_airports.parquet").exists():
                try:
                    adsb.fetch_day(day)
                except Exception as e:  # 2025-12-31 has no release in the archive
                    log(f"no ADS-B for {day}: {type(e).__name__}")
                    continue
                log(f"fetched {day}")
    for day in days:
        points = adsb.OUT_DIR / f"{day.isoformat()}_airports.parquet"
        if points.exists() and not (adsb.OUT_DIR / f"{day.isoformat()}_departures.parquet").exists():
            adsb.observe_day(day)
    observations = adsb.load_observations(days)
    log(f"ADS-B observations for {observations.height} departures")

    training = data.load_training()
    rows = pl.concat([fold_rows(training, months) for months in FOLDS], how="vertical_relaxed")
    name = "corrector_" + "-".join(f"{a}_{b}" for a, b in FOLDS)
    corrector = stage(name, lambda: correct.fit(rows, observations))
    corrected = correct.apply(corrector, rows, observations)
    truth = corrected[schema.TARGET].to_numpy().astype(float)
    log(f"folds' RMSE after the correction, in sample: "
        f"{np.sqrt(np.mean((corrected['pred'].to_numpy() - truth) ** 2)):.1f}s "
        "(cross-validated by day in experiments_round9.py)")

    final = stage("final_model", lambda: train.train_final(training))
    ranking = data.load_ranking()
    frame = features.build(data.departures(ranking), final.unimpeded.lazy(), features.surroundings(ranking)).collect()
    out = correct.apply(corrector, correct.inputs(final, frame), observations)
    predictions = dict(zip(out[schema.MVT_ID].to_list(), out["pred"].to_list()))
    for path, template in ((OUTPUT, data.SUBMISSION_TEMPLATE), (LEADERBOARD_OUTPUT, data.LEADERBOARD_TEMPLATE)):
        submit.write_submission(predictions, path, template_name=template)
        problems = submit.verify_submission(path, predictions, template_name=template)
        if problems:
            raise ValueError(f"{path} failed verification:\n  " + "\n  ".join(problems))
        log(f"wrote {path}")


if __name__ == "__main__":
    main()
