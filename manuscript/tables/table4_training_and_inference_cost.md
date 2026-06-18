# Table 4. Training and inference resource profile

One local environment (RTX 5060 Laptop GPU). Inference latency at batch 1;
training-step profiling at batch 8, 224 x 224, AMP, 10 warmup + 40 timed
forward/backward iterations, **no optimizer step, weight update, or checkpoint
write**. Deploy parameters for BDS-Lite are the retained segmentation graph
after removing the 269,889-parameter boundary branch. All values from
`results/submission_blocker_resolution_v1/profiling/training_cost_combined_summary.csv`.

| Method | Params (total) | Deploy params | Inf. GFLOPs | Inf. latency (ms) | Inf. peak (MB) | Train step (ms) | Train peak (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 1,927,042 | 1,927,042 | 15.970 | 1.93 | 76.17 | 39.62 ± 3.98 | 600.29 |
| BDS-Lite | 2,198,003 | 1,928,114 | 16.079 | 2.40 | 77.21 | 57.17 ± 4.71 | 873.57 |
| U-Net+GSL | 1,927,042 | 1,927,042 | 15.970 | 1.97 | 76.17 | 52.32 ± 1.09 | 600.29 |

Allowed interpretation only: the BDS-Lite deploy graph is close to the U-Net
backbone in parameters (+1,072, ≈0.056%) and FLOPs (≈+0.68%). These local
measurements **do not** establish time-to-convergence, total operational cost,
or cross-hardware portability. The GSL CPU distance transform is part of its
timed step, so its local timing is implementation- and hardware-specific.
Forbidden wording (training-efficient, deployment-ready, faster,
hardware-friendly, clinical deployment, lower total training cost, more
efficient than GSL, time-to-convergence advantage, hardware-independent
efficiency) is not used.
