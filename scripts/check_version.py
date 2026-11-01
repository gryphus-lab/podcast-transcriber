#!/usr/bin/env python3
"""Verify the semantic version is identical across every version surface.

The package keeps one semantic version (see VERSIONING.md), written in several
files. This script fails (exit 1) if any disagree, so drift is caught in CI
instead of shipping mismatched artifacts (the FastAPI apps and the CLI all read
the version back from ``podcast_transcriber.__version__``).

Surfaces checked:
  - pyproject.toml                               (podcast-transcriber)
  - pyproject.converter.toml                     (podcast-converter)
  - src/podcast_transcriber/__init__.py          (__version__)

Run:  uv run python scripts/check_version.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _toml_version(path: Path) -> str:
    with path.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _init_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise ValueError(f"__version__ not found in {path}")
    return match.group(1)


def main() -> int:
    surfaces = {
        "pyproject.toml": _toml_version(ROOT / "pyproject.toml"),
        "pyproject.converter.toml": _toml_version(ROOT / "pyproject.converter.toml"),
        "src/podcast_transcriber/__init__.py": _init_version(
            ROOT / "src/podcast_transcriber/__init__.py"
        ),
    }

    semver_re = re.compile(r"^\d+\.\d+\.\d+$")
    versions = set(surfaces.values())
    errors: list[str] = []

    if len(versions) != 1:
        errors.append("Version mismatch across surfaces:")
        for name, ver in surfaces.items():
            errors.append(f"    {ver:<12} {name}")
    else:
        (semver,) = versions
        if not semver_re.match(semver):
            errors.append(f"Version '{semver}' is not valid MAJOR.MINOR.PATCH.")

    if errors:
        print("\n".join(["version check failed", *errors]), file=sys.stderr)
        return 1

    (semver,) = versions
    print(f"version consistent: {semver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
