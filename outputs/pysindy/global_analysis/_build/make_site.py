"""Assemble a minimal, self-contained ``dist/`` folder for static hosting.

The full ``global_analysis`` directory is ~1.7 GB and contains files the site
never loads (the 58 MB ``configs.csv``, per-part sources, redundant figures).
This script copies only what ``index.html`` actually requests -- the dashboard
payload and the exact figures referenced in it -- into ``_build/dist/``, ready to
drag-and-drop onto Cloudflare Pages or Netlify.

Figures are hardlinked (free on the same filesystem), so ``dist/`` adds no
meaningful disk usage. Static hosts do not run Jekyll, so the ``_build/``
underscore path is served fine and no ``.nojekyll`` file is needed.

Run after ``build_index.py``:

    .venv/bin/python outputs/pysindy/global_analysis/_build/make_site.py

Then deploy the printed ``dist/`` path.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
GLOBAL_ANALYSIS_DIR = BUILD_DIR.parent
DIST_DIR = BUILD_DIR / "dist"

# Small companion files worth shipping (referenced by name in the UI text).
EXTRA_FILES = ["annotations.csv", "associations.md"]


def referenced_figures(dashboard_js: Path) -> list[str]:
    """Return the figure paths embedded in the dashboard payload.

    Args:
        dashboard_js: Path to the generated ``dashboard_data.js``.

    Returns:
        Sorted unique figure paths, relative to ``global_analysis``.
    """
    text = dashboard_js.read_text()
    match = re.search(r"window\.DASHBOARD_DATA = (\[.*\]);", text, re.S)
    if not match:
        raise SystemExit("Could not parse DASHBOARD_DATA; run build_index.py first.")
    data = json.loads(match.group(1))
    figs: set[str] = set()
    for record in data:
        figs.update(record.get("figures", []))
    return sorted(figs)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst`` when possible, else copy.

    Args:
        src: Existing source file.
        dst: Destination path (parents are created).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


def main() -> int:
    """Assemble ``dist/`` and report its size and file count.

    Returns:
        Process exit code (0 on success).
    """
    dashboard_js = BUILD_DIR / "dashboard_data.js"
    index_html = GLOBAL_ANALYSIS_DIR / "index.html"
    if not dashboard_js.exists() or not index_html.exists():
        raise SystemExit("Missing index.html or dashboard_data.js; run build_index.py.")

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)

    # Entry point and payload (payload keeps the _build/ path the HTML expects).
    shutil.copy2(index_html, DIST_DIR / "index.html")
    link_or_copy(dashboard_js, DIST_DIR / "_build" / "dashboard_data.js")

    for name in EXTRA_FILES:
        src = (GLOBAL_ANALYSIS_DIR / name) if name == "annotations.csv" else (BUILD_DIR / name)
        if src.exists():
            shutil.copy2(src, DIST_DIR / name)

    figs = referenced_figures(dashboard_js)
    missing = 0
    for rel in figs:
        src = GLOBAL_ANALYSIS_DIR / rel
        if not src.exists():
            missing += 1
            continue
        link_or_copy(src, DIST_DIR / rel)

    n_files = sum(1 for _ in DIST_DIR.rglob("*") if _.is_file())
    total_bytes = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file())
    print(f"dist assembled at: {DIST_DIR}")
    print(f"  figures linked: {len(figs) - missing} (missing on disk: {missing})")
    print(f"  total files: {n_files}")
    print(f"  apparent size: {total_bytes / 1e6:.0f} MB "
          f"(hardlinked -- little extra disk used)")
    print(f"  largest file under 25 MB Cloudflare limit: "
          f"{max((f.stat().st_size for f in DIST_DIR.rglob('*') if f.is_file()), default=0) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
