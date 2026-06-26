# Configuration of record (resolved snapshots)

These 18 YAML files are the fully expanded `config_resolved.yaml` snapshots written by the trainer at run time — the authoritative record of what produced each reported number (`{isic2018,acdc,synapse}` × `{unet,bds_lite_full}` × seeds `{1,2,3}`).

## ⚠️ `split_file` is VESTIGIAL — read before drawing any leakage conclusion

Every snapshot contains a `split_file:` field pointing at a top-level `splits/<dataset>_split.json`. **The data loader (`NpzSegmentationDataset`) does not read this field.** It selects samples by globbing `data/processed/<dataset>/<split>/*.npz`. Consequently:

- The actual partition used for **both training and evaluation** is the **seed-2026** split, whose manifests are in `splits/<dataset>/` (ISIC 2075/519, ACDC 1350/186/366, Synapse 1738/473/1568). The processed-directory counts match these exactly.
- We verified **0 train/val overlap** and matching case IDs for the seed-2026 split.
- The path named in `split_file` is a **stale seed-1 split** (e.g. ISIC 1815/389/390) that is **not shipped in this repository and was never consumed** by the loader.

A reviewer therefore **cannot** infer train/eval leakage from the `split_file` value. Each snapshot also carries an inline `ADDED NOTE` to this effect directly above the field. Full detail: [`../../docs/config_of_record.md`](../../docs/config_of_record.md) and [`../../docs/naming.md`](../../docs/naming.md).

## `model.name: bds_lite`

The snapshots use `model.name: bds_lite`; the cleaned public dispatch key is `bdslite_unet`. The public `build_model` accepts **both** and builds the identical `BDSLiteUNet` (2,198,003 parameters). See `docs/config_of_record.md`.

## Only added notes are non-verbatim

Apart from the clearly-marked `ADDED NOTE` comment lines (and this README), the snapshots are the verbatim resolved configurations from the training runs.
