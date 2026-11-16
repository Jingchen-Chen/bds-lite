# Protocols

Each study was specified before the runs or analyses it governs; the documents say when and what was already known.
Files here are frozen. They cite each other by their original `revision/…` and `preregistration/…` paths, which
`../evidence/path_map.csv` maps to the current ones. `DEVIATIONS.md` logs deviations across studies, and
`queue_plan.yaml` gives the order in which the queues were run.

| Folder | Study | Protocol and amendments | Run queue |
|---|---|---|---|
| `ablation/` | Training adapter, diagnostics and ablations | `TRAINING_ADAPTER_PROTOCOL.md`, `revision_analysis_plan.md`, `ABLATION_PROTOCOL.md` (+ amendments 1, 2) | `queue.yaml`, `queue_withdrawn.yaml` (withdrawn runs) |
| `comparators/` | GSL retrain, Boundary Loss, LB-UNet | `COMPARATOR_PROTOCOL.md` | `queue_b.yaml` |
| `sensitivity/` | Sensitivity and Tier B | `SENSITIVITY_PROTOCOL.md` | `queue_tierb.yaml` |
| `expansion/` | Expansion experiments | `EXPANSION_PROTOCOL.md`, `NARRATIVE_CRITERIA_ADDENDUM.md` | `queue_expansion.yaml` |
| `confirmation/` | Experiment C | `CONFIRMATION_PROTOCOL.md`, `CONFIRMATION_SCOPE_AMENDMENT_20261107.md` | `queue_confirmation.yaml`, `queue_confirmation_acdc_synapse.yaml` |
| `kd_isic2018/` | External-teacher distillation, ISIC2018 | `KD_COMPARATOR_PROTOCOL.md`, `KD_COMPARATOR_AMENDMENT_1.md`, `DECISIONS_AND_DEVIATIONS.md` | `queue_kd_phase1.yaml`, `queue_kd_phase2a.yaml`, `queue_kd_phase2b.yaml` |
| `kd_acdc_synapse/` | External-teacher distillation, ACDC and Synapse | `KD_MULTICLASS_PROTOCOL.md`, `DECISIONS_AND_DEVIATIONS.md` | `queue_kd_mc.yaml` |
| `acdc_confirm/` | Preregistered ACDC confirmatory study | `acdc_confirmation.md` (+ `.ots`), `AMENDMENT_1.md`, `PROTOCOL_LOG.md`, `DEVIATION_CHECKLIST.md`, `TIMESTAMP_MANIFEST.txt` (+ `.ots`) | `queue_acdc_confirm.yaml` |
| `followup/` | Follow-up controls E1–E3 | `followup_registration.md` (+ `.ots`), `AMENDMENT_1.md`, `PROTOCOL_LOG.md`, `TIMESTAMP_MANIFEST.txt` (+ `.ots`) | `queue_followup_e1e2.yaml`, `queue_followup_e3.yaml` |
| `historical_reevaluation/` | Re-evaluation of historical arms | `HISTORICAL_REEVALUATION_PROTOCOL.md` | — |
| `prediction_recovery/` | Predictions of fixed checkpoints | `prediction_recovery_protocol.md`, `main_prediction_protocol.md` | `prediction_recovery_queue.json`, `main_prediction_queue.json` |
| `gate_identity/` | Gate identity intervention | `gate_identity_protocol.md`, `gate_identity_statistics_protocol.md` | `gate_identity_queue.json` |
| `volume/` | Volume-level evaluation | `volume_evaluation_protocol.md`, `volume_statistics_protocol.md` | — |

Results are under `../results/`: statistics of the ablation, comparator, sensitivity, expansion and confirmation
studies are in `../results/tables/families/`, and the other studies have a folder of the same name.
