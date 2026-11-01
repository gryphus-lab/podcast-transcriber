# Versioning

This package uses [Semantic Versioning](https://semver.org):
**MAJOR.MINOR.PATCH**, baselined at **0.1.0**. It is still pre-1.0, so the API
is allowed to evolve under MINOR bumps until it stabilises at 1.0.0.

One version covers the whole package. It lives in a few files that must stay in
lockstep; a CI check (`scripts/check_version.py`, wired into `mise run test`)
fails the build if they drift.

## Version surfaces

| File | Field | Purpose |
|------|-------|---------|
| `src/podcast_transcriber/__init__.py` | `__version__` | single source of truth — read back by both FastAPI apps and the CLI |
| `pyproject.toml` | `project.version` | the `podcast-transcriber` distribution |
| `pyproject.converter.toml` | `project.version` | the `podcast-converter` distribution |

Because the apps read `__version__`, a running service self-identifies: both
FastAPI apps report the version in their OpenAPI schema / `/docs`, and the CLI
answers `transcribe --version`.

## When to bump

Bump the version on every change merged to `main`, by the highest-impact change
in the PR:

| Change | Bump | Example |
|--------|------|---------|
| Breaking — removed/renamed API route or CLI flag, response-shape break | **MAJOR** once 1.0; while pre-1.0, a **MINOR** | `feat!:` / `BREAKING CHANGE:` |
| New backward-compatible capability | **MINOR** | REST API, async jobs + web console |
| Bug fix, build/CI/Docker/config, refactor, chore, docs | **PATCH** | `fix:`, Sonar fixes, CI resilience |

### Dependabot updates

Classify by impact on the product, not by the dependency's own version jump. A
routine library bump does not change what the tool does, so:

- **Any Dependabot bump → PATCH** (runtime libs like transformers, torch,
  lightning, aiohttp, pillow, nltk; build/dev tooling like ruff, setuptools; CI
  actions; base images).
- Promote to **MINOR** only if a bump is adopted specifically to ship a new
  capability; **MAJOR** only if it forces a break in our public CLI/API surface.

## How the current version was derived

Computed from all **44** PRs merged to `main` from the `0.1.0` baseline. Two
introduced genuine product capabilities (**MINOR**); the remaining 42
(Dependabot, Sonar/Docker fixes, the OOM fix, CI, warning suppression, metadata)
are **PATCH**.

The two MINOR features, in merge order:

| PR | Feature | Version |
|----|---------|---------|
| #4 | REST API + Docker support | 0.2.0 |
| #22 | Async transcription jobs + web console (RQ/Redis) | 0.3.0 |

After #22, 24 further patch-level PRs were merged, landing the patch component
at **24**.

**Current version: `0.3.24`.**

> Note: this also reconciled pre-existing drift — the transcriber API had been
> hardcoded to `0.2.0` and the converter service to `0.1.0`. Both now read
> `__version__`, so they can't fall out of sync again.
