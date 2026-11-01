import pytest

from scripts.bump_version import _next_version


@pytest.mark.parametrize(
    ("version", "level", "expected"),
    [
        ("0.3.24", "patch", "0.3.25"),
        ("0.3.24", "minor", "0.4.0"),
        ("0.3.24", "major", "1.0.0"),
    ],
)
def test_next_version(version, level, expected):
    assert _next_version(version, level) == expected


def test_next_version_rejects_invalid_semver():
    with pytest.raises(ValueError, match="not valid MAJOR.MINOR.PATCH"):
        _next_version("0.3", "patch")
