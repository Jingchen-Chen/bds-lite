# Surface-error distribution findings (descriptive)

33 saved-test cohorts; 149420 image/class rows; 52981 valid surfaces. Saved HD95/missingness joins pass; maximum rounding delta=7.63e-06px. Independent KDTree checks: 10. Empties remain separate, without invented distances.

| Dataset | Family | CDF <=2px | Tail >20px | Spurious share of predicted tail |
|---|---|---:|---:|---:|
| acdc | attention_reference | 0.8436 | 0.0207 | 0.9930 |
| acdc | legacy_gsl | 0.8583 | 0.0177 | 0.9721 |
| acdc | main_p1b_full | 0.8632 | 0.0231 | 0.9663 |
| acdc | main_p1b_unet | 0.8483 | 0.0191 | 0.9634 |
| isic2018 | F | 0.3144 | 0.1413 | 0.2656 |
| isic2018 | S | 0.2786 | 0.1445 | 0.2840 |
| isic2018 | U | 0.3109 | 0.1331 | 0.2536 |
| isic2018 | attention_reference | 0.2812 | 0.1391 | 0.2840 |
| isic2018 | legacy_gsl | 0.3297 | 0.1285 | 0.2262 |
| synapse | attention_reference | 0.7941 | 0.0394 | 0.9222 |
| synapse | legacy_gsl | 0.7859 | 0.0374 | 0.8901 |
| synapse | main_p1b_full | 0.7798 | 0.0471 | 0.9054 |
| synapse | main_p1b_unet | 0.7728 | 0.0455 | 0.8690 |

CDFs give each eligible image/class equal weight within a seed, then average seeds. Point-pooled CDFs are also saved. Eligibility differs between methods when masks are empty; these curves are descriptive rather than matched-unit effect estimates. Attention uses seed2026; legacy GSL retains its separate provenance caveat.

For the main full recipe versus U-Net, all three datasets have more probability within2px alongside more tail probability beyond20px in these conditional curves. This mixture does not support describing every change as a uniform contour shift. ACDC/Synapse remote predicted-surface points frequently belong to components with no GT intersection. The spurious-share column concerns only the predicted-to-GT direction; it is not a fraction of all bidirectional errors or causal attribution of an HD95 difference. ISIC has a different tail pattern. Never generalize this association to an established mechanism.

The20px cutoff was chosen for this exploratory descriptive analysis; it is not a preregistered clinical threshold. All distances use the processed224 grid. Numeric Synapse classes remain unrelabelled. No physical or clinical conclusion is inferred. Sources: surface_cdf/*.csv and validation/surface_cdf.json.
