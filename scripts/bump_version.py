#!/usr/bin/env python3
"""Bump every version surface for the podcast-transcriber project."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _next_version(version: str, level: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Version '{version}' is not valid MAJOR.MINOR.PATCH.")
    major, minor, patch = map(int, match.groups())
    if level == "major":
        major, minor, patch = major + 1, 0, 0
    elif level == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def _updated_text(path: Path, pattern: str, version: str) -> str:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"Expected one version entry in {path}")
    return re.sub(
        pattern,
        rf"\g<1>{version}\g<2>",
        text,
        count=1,
        flags=re.MULTILINE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=("patch", "minor", "major"))
    level = parser.parse_args().level

    with (ROOT / "pyproject.toml").open("rb") as fh:
        current_version = tomllib.load(fh)["project"]["version"]
    new_version = _next_version(current_version, level)

    targets = {
        ROOT / "pyproject.toml": r'(?m)^(version = ")[^"]+("$)',
        ROOT / "pyproject.converter.toml": r'(?m)^(version = ")[^"]+("$)',
        ROOT / "src/podcast_transcriber/__init__.py": (
            r'(?m)^(__version__ = ")[^"]+("$)'
        ),
        ROOT / "uv.lock": (
            r'(?m)^(\[\[package\]\]\nname = "podcast-transcriber"\n'
            r'version = ")[^"]+("$)'
        ),
    }

    updates = {
        path: _updated_text(path, pattern, new_version)
        for path, pattern in targets.items()
    }
    for path, text in updates.items():
        path.write_text(text, encoding="utf-8")

    print(f"Bumped version: {current_version} -> {new_version} ({level})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"version bump failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
