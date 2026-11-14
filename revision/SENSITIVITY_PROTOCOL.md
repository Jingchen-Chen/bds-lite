# WP3 sensitivity + Tier-B queue protocol (frozen before training)

Written 2026-11-02 before any run in revision/queue_tierb.yaml started; hashed into
its sensitivity configs. Same adapter, locked protocol, seeds and selection rule as
ABLATION_PROTOCOL.md. M2 (aux boundary only) keeps its ABLATION_PROTOCOL.md
definition and config; it was moved unchanged from queue.yaml to this Tier-B queue
so that Tier-A comparators (queue_b.yaml) run first.

- alpha sensitivity: M6 full recipe with gate scale alpha in {0.1, 0.5, 1.0}
  (0 = M5, 0.25 = existing M6), ISIC2018 and Synapse, seeds 1/2/3.
- lambda sensitivity: M6 with lambda_b = lambda_distill in {0.01, 0.2}
  (0.05 = existing M6), ISIC2018, seeds 1/2/3.

These describe robustness of conclusions to alpha/lambda. No value is selected from
these runs using test data; alpha=0.25 and lambda=0.05 were fixed before the original
matched study. Any statement about a "better" value uses validation data only and is
labelled post hoc.
