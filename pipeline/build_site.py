"""Builds docs/index.html, the static copy of *Taxi-Out, Measured*.

GitHub Pages serves `docs/` on the main branch at
https://joris-decombe.github.io/taxiout/. The page source in
`report/taxi-out-measured.html` is a fragment (title, styles, body content)
written for a host that supplies the document skeleton; this wraps it in
one, with the same reset that host applies, so it renders identically as a
plain file. `report_findings.py` calls it after regenerating the page.

Usage: python pipeline/build_site.py
"""

from pathlib import Path

PAGE = Path("report/taxi-out-measured.html")
SITE = Path("docs/index.html")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Predicting taxi-out time at ten European airports: what the data keeps, where the error lives, and what worked. Team gentle-octopus, PRC Data Challenge 2026.">
<style>
:root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
body { margin: 0; font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; background: #fafaf8; }
img { max-width: 100%; }
[hidden] { display: none !important; }
</style>
"""


def build(page: Path = PAGE, site: Path = SITE) -> Path:
    source = page.read_text(encoding="utf-8")
    # The fragment opens with its title, font links and stylesheet; those
    # belong in the head, everything after the first stylesheet in the body.
    split = source.index("</style>") + len("</style>")
    site.parent.mkdir(exist_ok=True)
    site.write_text(
        HEAD + source[:split] + "\n</head>\n<body>\n" + source[split:] + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    # Plain HTML: skip GitHub's Jekyll pass.
    (site.parent / ".nojekyll").touch()
    return site


if __name__ == "__main__":
    print(build())
