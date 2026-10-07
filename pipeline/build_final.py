"""Builds the final submission end to end, from the challenge data and adsb.lol.

1. ADS-B: streams every day of January and July 2025 and 2026 from the
   adsb.lol archive and observes each departure (`taxiout.adsb`). Days
   already done are skipped; `--skip-fetch` skips streaming altogether.
2. Validation fit (`train.validate`): the model fitted on ten months, with
   honest held-out predictions for January and July 2025.
3. The ADS-B corrector (`taxiout.correct`) fitted on those predictions.
4. The final fit on all twelve months, its predictions for the ranking set,
   corrected, written to `data/submission_final.parquet` and verified.

About four hours on a 20-thread machine with 32 GB, plus about two hours
of streaming (390 GB read, 3 GB kept) the first time.

Usage: python pipeline/build_final.py [--skip-fetch]
"""

import datetime as dt
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline")
from taxiout import adsb, correct, data, features, schema, submit, train  # noqa: E402

DAYS = [dt.date(y, m, d) for y in (2025, 2026) for m in (1, 7) for d in range(1, 32)]
OUTPUT = Path("data/submission_final.parquet")


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def main() -> None:
    if "--skip-fetch" not in sys.argv:
        for day in DAYS:
            if not (adsb.OUT_DIR / f"{day.isoformat()}_airports.parquet").exists():
                adsb.fetch_day(day)
                log(f"fetched {day}")
    for day in DAYS:
        if not (adsb.OUT_DIR / f"{day.isoformat()}_departures.parquet").exists():
            adsb.observe_day(day)
    observations = adsb.load_observations(DAYS)
    log(f"ADS-B observations for {observations.height} departures")

    training = data.load_training()
    model, held_out, _ = train.validate(training)
    log(f"validation RMSE before the correction: {model.validation_rmse:.1f}s")
    rows = correct.inputs(model, held_out)
    corrector = correct.fit(rows, observations)
    corrected = correct.apply(corrector, rows, observations)
    truth = corrected[schema.TARGET].to_numpy().astype(float)
    log(f"validation RMSE after it, in sample: {np.sqrt(np.mean((corrected['pred'].to_numpy() - truth) ** 2)):.1f}s "
        "(cross-validated by day in experiments_round7.py)")

    final = train.train_final(training)
    pickle.dump({"model": final, "corrector": corrector}, open("data/model_final.pkl", "wb"))
    ranking = data.load_ranking()
    frame = features.build(data.departures(ranking), final.unimpeded.lazy(), features.surroundings(ranking)).collect()
    out = correct.apply(corrector, correct.inputs(final, frame), observations)
    predictions = dict(zip(out[schema.MVT_ID].to_list(), out["pred"].to_list()))
    submit.write_submission(predictions, OUTPUT)
    problems = submit.verify_submission(OUTPUT, predictions)
    if problems:
        raise ValueError("submission failed verification:\n  " + "\n  ".join(problems))
    log(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
