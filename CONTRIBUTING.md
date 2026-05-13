# Contributing to Nalu

Thanks for your interest. Nalu is Apache 2.0 and welcomes contributions.

## Ground rules

1. **Fully local.** No code path may add a cloud API call, telemetry, or "phone home." If a feature genuinely needs the network (e.g. downloading a model on first run), it must be opt-in and route through the existing model-download path.
2. **Real data only.** Don't ship UIs backed by mock arrays. Build the real infra first; render real outputs.
3. **No emojis in UIs.** Use [lucide](https://lucide.dev/) icons in the dashboard.
4. **Tests are not optional.** New behavior gets unit tests. Side-effect-heavy code (audio I/O, Cocoa, subprocess) gets a pure-Python core that's tested directly, with the I/O wrapper kept thin.

## Setup

```bash
git clone <this repo> nalu
cd nalu
uv sync
uv run nalu onboard
```

The wizard verifies macOS permissions, downloads voice/STT models, and warms the vision model. Re-run it any time.

## Running tests

```bash
uv run pytest                 # full suite, offline
uv run pytest tests/test_xxx  # single file
```

The full suite must pass before you open a PR.

## Project layout

See [`BUILD_PLAN.md`](./BUILD_PLAN.md) and the Architecture section of the [README](./README.md). In short: stand-alone agents that talk over a Unix domain socket bus.

## Where work lives

[`BUILD_PLAN.md`](./BUILD_PLAN.md) is the single source of truth for what's built and what's next. Update it whenever you ship a phase item.

## Pull request checklist

- [ ] Tests added/updated and `uv run pytest` is green.
- [ ] No new network dependencies.
- [ ] No mock data in shipped UIs.
- [ ] [`BUILD_PLAN.md`](./BUILD_PLAN.md) updated if you shipped a roadmap item.
- [ ] [`CHANGELOG.md`](./CHANGELOG.md) gets an entry under `## Unreleased`.

## License

By contributing, you agree your work is licensed under Apache 2.0.
