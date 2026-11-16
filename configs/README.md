# Configurations

- `base.yaml`, `datasets/`, `experiments/`: defaults and basic training examples for
  `scripts/train.py`. These examples use directory-based splits.
- `revision/`: locked configurations for the paper studies. Use the corresponding study trainer
  and registered manifests, as described in the [reproduction guide](../docs/reproducing.md#paper-experiments).

Paper configurations retain their recorded bytes and hashes. Restore their protocols and queues
with `python evidence/restore.py --group protocols`. Development-only tuning examples have been
removed from `experiments/`; they are not the entry points for reproducing the paper.
