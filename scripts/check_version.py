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
  - uv.lock                                       (editable project package)

Run:  uv run python scripts/check_version.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _lock_version(path: Path) -> str:
    with path.open("rb") as fh:
        packages = tomllib.load(fh)["package"]
    versions = [
        package["version"]
        for package in packages
        if package["name"] == "podcast-transcriber"
        and package.get("source") == {"editable": "."}
    ]
    if len(versions) != 1:
        raise ValueError(f"Expected one editable podcast-transcriber package in {path}")
    return versions[0]


def _toml_version(path: Path) -> str:
    with path.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _init_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise ValueError(f"__version__ not found in {path}")
    return match.group(1)


def _versions() -> dict[str, str]:
    return {
        "pyproject.toml": _toml_version(ROOT / "pyproject.toml"),
        "pyproject.converter.toml": _toml_version(ROOT / "pyproject.converter.toml"),
        "src/podcast_transcriber/__init__.py": _init_version(
            ROOT / "src/podcast_transcriber/__init__.py"
        ),
        "uv.lock": _lock_version(ROOT / "uv.lock"),
    }


def _version_tuple(version: str) -> tuple[int, int, int]:
    semver_re = re.compile(r"^\d+\.\d+\.\d+$")
    if not semver_re.fullmatch(version):
        raise ValueError(f"Version '{version}' is not valid MAJOR.MINOR.PATCH.")
    return tuple(map(int, version.split(".")))


def _check_base_version(version: str, base_ref: str) -> bool:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:pyproject.toml"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    base_version = tomllib.loads(result.stdout)["project"]["version"]
    try:
        current_tuple = _version_tuple(version)
        base_tuple = _version_tuple(base_version)
    except ValueError as exc:
        print(f"version bump check failed: {exc}", file=sys.stderr)
        return False

    if current_tuple <= base_tuple:
        print(
            f"version bump check failed: {version} must be greater than "
            f"{base_version} from {base_ref}",
            file=sys.stderr,
        )
        return False

    print(f"version bump check passed: {version} > {base_version} ({base_ref})")
    return True


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        help="also require the current version to be greater than this Git ref",
    )
    args = parser.parse_args()

    surfaces = _versions()
    versions = set(surfaces.values())
    errors: list[str] = []

    if len(versions) != 1:
        errors.append("Version mismatch across surfaces:")
        for name, ver in surfaces.items():
            errors.append(f"    {ver:<12} {name}")
    else:
        (semver,) = versions
        try:
            _version_tuple(semver)
        except ValueError:
            errors.append(f"Version '{semver}' is not valid MAJOR.MINOR.PATCH.")

    if errors:
        print("\n".join(["version check failed", *errors]), file=sys.stderr)
        return 1

    (semver,) = versions
    if args.base_ref and not _check_base_version(semver, args.base_ref):
        return 1
    print(f"version consistent: {semver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
