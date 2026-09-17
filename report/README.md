# report/

Data behind the two published pages. Both files are derived, small, and
committed deliberately, unlike anything under `data/`.

- **`findings.json`** backs *Taxi-Out, Measured*
  (<https://claude.ai/artifact/Ez8LT8SdgrUbdp8oAeMq1i>). Aggregates only:
  per-group RMSE and percentiles, an error-concentration curve, a target
  histogram in 120-second bins, the clip curve, per-month spread. No per-flight
  rows, so it carries nothing the challenge data licence would object to.
- **`synthetic_run.json`** backs *Departure Surface Replay*
  (<https://claude.ai/artifact/V6pWhi5jVwR7V4ZeotiFSt>). One simulator run over
  `synthetic.py` output, so it contains no challenge data at all.

The scripts that produce these still live outside the repo. Moving them in is
tracked in [../TODO.md](../TODO.md); the rules require documentation sufficient
to reproduce the results, and a figure whose generator is missing does not meet
that.
