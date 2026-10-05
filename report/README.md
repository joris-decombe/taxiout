# report/

Data behind the two published pages. Both files are derived, small, and
committed deliberately, unlike anything under `data/`.

- **`findings.json`** backs *Taxi-Out, Measured*, served as a static page
  at <https://joris-decombe.github.io/taxiout/> from `docs/index.html`.
  Aggregates only:
  per-group RMSE and percentiles, an error-concentration curve, a target
  histogram in 120-second bins, the clip curve, per-month spread, and the
  at-schedule artifact broken down by airport and by schedule gap. No
  per-flight rows, so it carries nothing the challenge data licence would
  object to.
- **`taxi-out-measured.html`** is that page's source, with `findings.json`
  inlined. `pipeline/report_findings.py` regenerates both from the current
  model, then `pipeline/build_site.py` wraps the page into `docs/index.html`
  for GitHub Pages.
- **`synthetic_run.json`** backs *Departure Surface Replay*
  (<https://claude.ai/artifact/V6pWhi5jVwR7V4ZeotiFSt>). One simulator run over
  `synthetic.py` output, so it contains no challenge data at all.

`synthetic_run.json`'s generator and the replay page's source still live
outside the repo, tracked in [../TODO.md](../TODO.md): the rules require
documentation sufficient to reproduce the results.
