# Large-dependency preservation and portability inventory

Written UTC: 2026-11-10T20:30:53.892270+00:00. Read-only hashing; original files were neither copied nor modified. No model, training or inference was executed.

Existing local files hashed: **10067**, totaling **18,360,963,870 bytes** (17.100 GiB by logical path size). Missing references: **9**. Distinct inodes hashed: 10067; path-size totals can count aliases/hardlinks more than once.

Training-checkpoint references examined: 526. Experiment C outcome/checkpoint paths were excluded entirely. Shared dataset inputs may also have supported C training, but reading those original inputs does not inspect C outcomes.

| Role | Files | Logical bytes |
|---|---:|---:|
| external pinned pretrained ResNet34 encoder weights | 1 | 87,275,112 |
| frozen teacher probability cache | 4 | 2,861,738,496 |
| nonblinded completed training checkpoint | 508 | 8,813,317,373 |
| official ISIC source archive | 4 | 2,620,128,304 |
| processed dataset/preprocessing input | 9538 | 1,923,271,049 |
| raw Synapse held-out H5 | 12 | 2,055,233,536 |

## Preservation scope

- Every existing manifest row has an independently computed SHA-256 over actual local bytes. The status local_only_not_in_statistical_archive is deliberate: no large dependency was copied into the selected statistical archive or Git. This manifest proves current identity and local availability, not backup completeness, continued availability or training integrity.
- The raw image/annotation access and redistribution terms remain with their providers. A lawfully controlled independent backup/retrieval plan is still needed; a Git tag or manifest cannot recover missing bytes.
- Raw ACDC NIfTI and other raw source families outside the explicitly requested paths are not exhaustively inventoried here. Processed ACDC is included, but its original third-party conversion and the earlier trainer remain unrecoverable provenance dependencies.
- The external encoder weights are present in the local Hugging Face cache at the pinned repository/revision. Their digest matches scripts/kd/common.py. Future retrieval still requires the pinned object and permitted provider access; offline teacher creation fails if that cache is absent.
- Python portability remains incomplete: .venv/lib/python3.14/site-packages/_bds_frozen_overlay.pth resolves packages from /home/jc/bds-followup/.venv-kd, /home/jc/bds-followup/.venv, and external ACDC tools. pyproject minimum versions and revision/env_freeze.txt do not establish a complete, validated environment lock. This pass neither installed nor altered packages.
- Prediction maps are not part of this requested large-dependency subset; the final statistical archive/evaluation-record preservation pass must define separately which completed-run predictions it actually preserves.

## Missing references

- `recovered/followup/results/gsl/runs/phase16_matched_gsl_acdc_seed1/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_acdc_seed2/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_acdc_seed3/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_isic2018_seed1/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_isic2018_seed2/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_isic2018_seed3/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_synapse_seed1/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_synapse_seed2/best.pt`: missing historical completed-run checkpoint reference
- `recovered/followup/results/gsl/runs/phase16_matched_gsl_synapse_seed3/best.pt`: missing historical completed-run checkpoint reference
