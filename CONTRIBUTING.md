# Contributing

Bug reports and focused improvements to the code or documentation are welcome.

## Reporting a problem

Include the command, configuration path, Python/PyTorch versions and the relevant traceback or log.
For a numerical discrepancy, identify the study, run or result file and describe the expected value.
Avoid attaching dataset images, checkpoints or private review material.

## Making a change

Work on a branch and keep the change focused. Update documentation and configurations when behavior
changes, and add or update tests for changes to models, losses, metrics or data processing.
Document experiment changes with their configuration, seed, split and evaluation procedure.

Install the development dependencies and run the relevant checks:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest -q
```

The [reproduction guide](docs/reproducing.md#tests) lists the inputs and dependencies needed by the
full test suite. Include the checks you ran and any missing dependencies in the pull request.

## Study records and results

Protocols, registered configurations, run queues, evidence archives and their checksum manifests
preserve the paper's experiment records. Keep these files unchanged when proposing new experiments;
use separate configurations and output paths. Record amendments or deviations alongside the relevant
study, and retain earlier results when reporting a replacement analysis.

Do not commit datasets, checkpoints, generated training folders or private notes. Third-party
contributions must retain the attribution and license information described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
