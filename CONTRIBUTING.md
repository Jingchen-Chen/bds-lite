# Contributing

This repository is a research codebase. Code quality matters because experiment results must be reproducible months later.

## Development Rules

- Work on a branch, not directly on `main`.
- Keep commits small and focused.
- Do not commit datasets, checkpoints, generated run directories, or private notes.
- Update configs and docs in the same pull request as code that changes behavior.
- Add or update tests for reusable utilities, losses, metrics, and dataset transforms.
- Update `docs/devlog.md` after each meaningful experiment or implementation session.

## Branch Names

Use one of these prefixes:

```text
feat/<short-topic>
fix/<short-topic>
test/<short-topic>
docs/<short-topic>
refactor/<short-topic>
exp/<dataset-or-topic>
paper/<section-or-figure>
chore/<short-topic>
```

Examples:

```text
feat/boundary-targets
exp/synapse-unet-baseline
paper/method-section
```

## Commit Messages

Use conventional commit style:

```text
<type>(optional-scope): short imperative summary
```

Allowed types:

```text
feat fix test docs refactor perf chore exp paper
```

Rules:

- use lowercase type
- keep the first line under 72 characters
- use imperative mood: `add`, `fix`, `update`, not `added` or `fixed`
- mention experiment IDs or config paths in the body when relevant

Good examples:

```text
feat(data): add boundary target generation
fix(loss): match boundary target channel count
test(metrics): cover empty-mask hd95 behavior
exp(synapse): add bdslite unet ablation config
paper(intro): refine contribution statements
```

## Required Checks Before Commit

Run:

```bash
ruff format .
ruff check --fix .
python3 -m pytest
```

The same checks run through `pre-commit` and GitHub Actions.

## Pull Request Checklist

Before opening a PR:

- branch is up to date with `main`
- tests pass locally
- new behavior has a config, test, or short doc note
- devlog entry exists for experiment-facing work
- generated artifacts are not staged

## Research Result Policy

Do not overwrite old result files in place if the numbers are used for paper decisions. Create a new file with a version suffix, for example:

```text
results/tables/ablation_synapse_v1.csv
results/tables/ablation_synapse_v2.csv
```

If a result is superseded, document why in `docs/devlog.md`.

