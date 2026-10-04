# Local README Reproduction Report

Source: `experiments/joint_softcomp/README.md`, transcribed line by line from IMG_0144.MOV. The original result tables came from earlier server runs, rather than measurements from this Mac run.

This run used REPS=2 as allowed by the source, retaining sample sizes, 1000 epochs, model hyperparameters, and postprocessing. Experiments I/III called main() in the original modules instead of using the internal Buck build, with at most two concurrent processes. Experiment II ran as one single-threaded process alongside I/III. All computation used the CPU.

Validation: all 82 Experiment I/III tasks and 24 Experiment II tasks exited with code 0. All 106 JSON files were readable and contained finite floating-point values. Experiment I/III wall time was 13.39 minutes; Experiment II took 7.49 minutes. The batches overlapped, so these durations cannot be added.

## Execution Scope and Differences

- Experiment I: 3 β values; SoftComp M=0/1/2/4/8; JointSoftComp M=1/2/4/8; Cox, true-model, no-covariate, and population-minimizer references. Training n=200, test n=50000, and 2 repetitions per setting.
- Experiment III: ρ=0/0.2/0.5/0.8; case3, constant, and depcens (without rerunning depcens at ρ=0); 2 repetitions for each model. Training n=5000, test n=1000, τ=20, and 100 evaluation points. Also ran 8 no-covariate AJ checks with training n=10000 and the theoretical integration table.
- Experiment II: Case II/III × censored/uncensored × JointSoftComp/SoftComp/SoftComp-noaug × 2 repetitions; fitting n=4500, validation n=500, and test n=1000. NeuralFG was not run because its source was missing.
- The standalone Experiment II launcher skipped unavailable baseline imports in memory and scheduled only existing SoftComp specs. Data generation, training, prediction, postprocessing, and metrics used the original code. Missing models were neither fabricated nor substituted, and original files were unchanged.
- The README specifies an evaluation grid through the 97.5% event-time quantile with fixed times added. The transcribed Case II prepare_data calls build_evaluation_time_grid with its default 90%, while Case III already uses 97.5% and fixed times. This run retained source behavior, so Experiment II is not an exact reproduction of the original table. The matching prepare_data/grid implementation was still needed.
- The README states that censored-data C_td and IBS use IPCW. The compute_ctd implementation counts Antolini comparable pairs without IPCW; only compute_ibs uses training-set censoring KM weights. The source text and code were retained without changing metric definitions.
- The original tests/test_joint_softcomp.py had not yet been supplied, so the original unit tests specified in the README were not run.
- Two repetitions serve only as preliminary validation and do not replace the full 200/10/5-repetition statistical experiments specified in the README.

## Experiment I


beta = 0  (test event fraction 0.397)
| Method | M | n | Brier(10) | AUC(10) | IPA | MSE vs truth |
|---|---|---|---|---|---|---|
| True model |  | 1 | 0.2393 | 0.500 | -0.000 | 0.0000 |
| Cox |  | 2 | 0.2410 (0.0016) | 0.503 (0.000) | -0.001 (0.002) | 0.0020 (0.0018) |
| No covariate |  | 2 | 0.2408 (0.0020) | 0.500 (0.000) | 0.000 (0.000) | 0.0017 (0.0022) |
| SoftComp | 0 | 2 | 0.6034 (0.0000) | 0.500 (0.000) | -1.506 (0.020) | 0.3679 (0.0000) |
| SoftComp, limit | 0 | 1 | 0.6034 | 0.500 | -1.521 | 0.3679 |
| SoftComp | 1 | 2 | 0.3298 (0.0071) | 0.500 (0.005) | -0.370 (0.018) | 0.0923 (0.0072) |
| SoftComp, limit | 1 | 1 | 0.3220 | 0.500 | -0.346 | 0.0846 |
| SoftComp | 2 | 2 | 0.2606 (0.0052) | 0.502 (0.004) | -0.082 (0.013) | 0.0222 (0.0053) |
| SoftComp, limit | 2 | 1 | 0.2545 | 0.500 | -0.064 | 0.0160 |
| SoftComp | 4 | 2 | 0.2400 (0.0008) | 0.500 (0.004) | 0.003 (0.011) | 0.0005 (0.0006) |
| SoftComp, limit | 4 | 1 | 0.2414 | 0.500 | -0.009 | 0.0018 |
| SoftComp | 8 | 2 | 0.2658 (0.0047) | 0.499 (0.005) | -0.104 (0.028) | 0.0254 (0.0045) |
| SoftComp, limit | 8 | 1 | 0.2730 | 0.500 | -0.141 | 0.0325 |
| JointSoftComp | 1 | 2 | 0.2410 (0.0012) | 0.503 (0.000) | -0.001 (0.003) | 0.0019 (0.0015) |
| JointSoftComp | 2 | 2 | 0.2406 (0.0009) | 0.503 (0.000) | 0.001 (0.005) | 0.0015 (0.0011) |
| JointSoftComp | 4 | 2 | 0.2416 (0.0002) | 0.503 (0.000) | -0.004 (0.007) | 0.0025 (0.0004) |
| JointSoftComp | 8 | 2 | 0.2415 (0.0003) | 0.503 (0.000) | -0.003 (0.007) | 0.0024 (0.0004) |

beta = log 1.5  (test event fraction 0.408)
| Method | M | n | Brier(10) | AUC(10) | IPA | MSE vs truth |
|---|---|---|---|---|---|---|
| True model |  | 1 | 0.2270 | 0.640 | 0.060 | 0.0000 |
| Cox |  | 2 | 0.2282 (0.0006) | 0.640 (0.000) | 0.059 (0.001) | 0.0014 (0.0007) |
| No covariate |  | 2 | 0.2424 (0.0008) | 0.500 (0.000) | 0.000 (0.000) | 0.0154 (0.0009) |
| SoftComp | 0 | 2 | 0.5916 (0.0000) | 0.500 (0.000) | -1.440 (0.008) | 0.3690 (0.0000) |
| SoftComp, limit | 0 | 1 | 0.5916 | 0.500 | -1.448 | 0.3690 |
| SoftComp | 1 | 2 | 0.3141 (0.0119) | 0.640 (0.000) | -0.296 (0.045) | 0.0891 (0.0121) |
| SoftComp, limit | 1 | 1 | 0.3083 | 0.640 | -0.276 | 0.0833 |
| SoftComp | 2 | 2 | 0.2501 (0.0046) | 0.640 (0.000) | -0.032 (0.016) | 0.0240 (0.0047) |
| SoftComp, limit | 2 | 1 | 0.2440 | 0.640 | -0.010 | 0.0177 |
| SoftComp | 4 | 2 | 0.2342 (0.0037) | 0.640 (0.000) | 0.034 (0.018) | 0.0067 (0.0034) |
| SoftComp, limit | 4 | 1 | 0.2343 | 0.640 | 0.030 | 0.0068 |
| SoftComp | 8 | 2 | 0.2662 (0.0140) | 0.640 (0.000) | -0.098 (0.061) | 0.0376 (0.0137) |
| SoftComp, limit | 8 | 1 | 0.2699 | 0.640 | -0.117 | 0.0413 |
| JointSoftComp | 1 | 2 | 0.2292 (0.0023) | 0.638 (0.002) | 0.055 (0.007) | 0.0022 (0.0023) |
| JointSoftComp | 2 | 2 | 0.2293 (0.0022) | 0.639 (0.001) | 0.054 (0.006) | 0.0023 (0.0021) |
| JointSoftComp | 4 | 2 | 0.2296 (0.0025) | 0.639 (0.001) | 0.053 (0.007) | 0.0027 (0.0023) |
| JointSoftComp | 8 | 2 | 0.2297 (0.0024) | 0.639 (0.001) | 0.052 (0.007) | 0.0027 (0.0023) |

beta = log 2  (test event fraction 0.425)
| Method | M | n | Brier(10) | AUC(10) | IPA | MSE vs truth |
|---|---|---|---|---|---|---|
| True model |  | 1 | 0.2050 | 0.729 | 0.161 | 0.0000 |
| Cox |  | 2 | 0.2062 (0.0008) | 0.729 (0.000) | 0.160 (0.000) | 0.0014 (0.0011) |
| No covariate |  | 2 | 0.2453 (0.0010) | 0.500 (0.000) | 0.000 (0.000) | 0.0391 (0.0011) |
| SoftComp | 0 | 2 | 0.5749 (0.0000) | 0.500 (0.000) | -1.344 (0.009) | 0.3735 (0.0000) |
| SoftComp, limit | 0 | 1 | 0.5749 | 0.500 | -1.352 | 0.3735 |
| SoftComp | 1 | 2 | 0.2907 (0.0088) | 0.729 (0.000) | -0.185 (0.031) | 0.0871 (0.0091) |
| SoftComp, limit | 1 | 1 | 0.2860 | 0.729 | -0.170 | 0.0824 |
| SoftComp | 2 | 2 | 0.2320 (0.0039) | 0.729 (0.000) | 0.054 (0.012) | 0.0272 (0.0043) |
| SoftComp, limit | 2 | 1 | 0.2257 | 0.729 | 0.077 | 0.0209 |
| SoftComp | 4 | 2 | 0.2242 (0.0058) | 0.729 (0.000) | 0.086 (0.027) | 0.0179 (0.0054) |
| SoftComp, limit | 4 | 1 | 0.2202 | 0.729 | 0.099 | 0.0139 |
| SoftComp | 8 | 2 | 0.2637 (0.0124) | 0.729 (0.000) | -0.075 (0.055) | 0.0559 (0.0120) |
| SoftComp, limit | 8 | 1 | 0.2609 | 0.729 | -0.067 | 0.0532 |
| JointSoftComp | 1 | 2 | 0.2095 (0.0040) | 0.729 (0.000) | 0.146 (0.013) | 0.0044 (0.0042) |
| JointSoftComp | 2 | 2 | 0.2088 (0.0034) | 0.729 (0.000) | 0.149 (0.010) | 0.0038 (0.0036) |
| JointSoftComp | 4 | 2 | 0.2096 (0.0043) | 0.729 (0.000) | 0.145 (0.014) | 0.0046 (0.0045) |
| JointSoftComp | 8 | 2 | 0.2096 (0.0042) | 0.729 (0.000) | 0.145 (0.014) | 0.0046 (0.0043) |


## Experiment II: Three Models with Available Source


Case 2
| Method | n | Censored MSE | C_td | IBS | Uncensored MSE | C_td | IBS |
|---|---|---|---|---|---|---|---|
| JointSoftComp | 2 | 1.97 (0.06) | 0.739 (0.010) | 0.055 (0.003) | 1.57 (0.08) | 0.741 (0.011) | 0.052 (0.004) |
| SoftComp (manuscript) | 2 | 7.99 (0.73) | 0.747 (0.002) | 0.059 (0.003) | 14.70 (0.81) | 0.742 (0.005) | 0.061 (0.004) |
| SoftComp, no augmentation | 2 | 21.36 (2.60) | 0.745 (0.002) | 0.071 (0.003) | 63.19 (3.97) | 0.724 (0.005) | 0.144 (0.002) |

Case 3
| Method | n | Censored MSE | C_td | IBS | Uncensored MSE | C_td | IBS |
|---|---|---|---|---|---|---|---|
| JointSoftComp | 2 | 1.14 (0.00) | 0.679 (0.012) | 0.124 (0.004) | 0.78 (0.06) | 0.684 (0.006) | 0.122 (0.003) |
| SoftComp (manuscript) | 2 | 3.08 (0.55) | 0.694 (0.013) | 0.127 (0.004) | 4.28 (0.10) | 0.682 (0.007) | 0.124 (0.003) |
| SoftComp, no augmentation | 2 | 8.64 (0.84) | 0.687 (0.014) | 0.128 (0.004) | 61.32 (0.51) | 0.671 (0.003) | 0.162 (0.002) |


## Experiment III

| Test | ρ | Model | n | MSE × 1000 | C_td | IBS |
|---|---:|---|---:|---:|---:|---:|
| case3 | 0% | joint | 2 | 1.054 | 0.6792 | 0.1276 |
| case3 | 0% | softcomp | 2 | 5.784 | 0.6787 | 0.1322 |
| case3 | 20% | joint | 2 | 1.225 | 0.6812 | 0.1277 |
| case3 | 20% | softcomp | 2 | 6.591 | 0.6807 | 0.1329 |
| case3 | 50% | joint | 2 | 1.762 | 0.6763 | 0.1283 |
| case3 | 50% | softcomp | 2 | 11.087 | 0.6787 | 0.1375 |
| case3 | 80% | joint | 2 | 5.508 | 0.6677 | 0.1316 |
| case3 | 80% | softcomp | 2 | 25.275 | 0.6722 | 0.1518 |
| constant | 0% | joint | 2 | 0.736 | 0.6672 | 0.1419 |
| constant | 0% | softcomp | 2 | 7.141 | 0.6611 | 0.1480 |
| constant | 20% | joint | 2 | 0.715 | 0.6649 | 0.1424 |
| constant | 20% | softcomp | 2 | 11.040 | 0.6638 | 0.1515 |
| constant | 50% | joint | 2 | 1.430 | 0.6644 | 0.1432 |
| constant | 50% | softcomp | 2 | 21.199 | 0.6649 | 0.1610 |
| constant | 80% | joint | 2 | 10.547 | 0.6560 | 0.1536 |
| constant | 80% | softcomp | 2 | 44.910 | 0.6653 | 0.1844 |
| depcens | 20% | joint | 2 | 1.431 | 0.6794 | 0.1281 |
| depcens | 20% | softcomp | 2 | 7.226 | 0.6703 | 0.1335 |
| depcens | 50% | joint | 2 | 2.127 | 0.6772 | 0.1286 |
| depcens | 50% | softcomp | 2 | 11.595 | 0.6516 | 0.1381 |
| depcens | 80% | joint | 2 | 7.513 | 0.6601 | 0.1335 |
| depcens | 80% | softcomp | 2 | 27.918 | 0.5781 | 0.1545 |

See `censoring_summary.txt` for complete AJ errors, biases, and censoring fractions, and `population_targets.txt` for the theoretical integration table.

## Reevaluation of Saved Case II Weights

No retraining was performed. SoftComp PAV/simplex postprocessing was added, with censored test data and recovered original metrics. The saved weights used all 5000 training subjects, fixed test seed 140000, and model seeds 0/1/2, differing from the Experiment II protocol.

| Model | MSE × 1000 | C_td | IPCW IBS |
|---|---:|---:|---:|
| JointSoftComp | 1.766 | 0.7428 | 0.0518 |
| SoftComp | 7.821 | 0.7454 | 0.0563 |

## Outstanding Items at the Time of This Run

- `baseline_models/neural_fine_gray.py`: complete NeuralFG in Experiment II.
- `baseline_models/dsm.py`, `baseline_models/fine_gray.py`: supply missing source required by the original entry-point imports.
- `tests/test_joint_softcomp.py`: run the original tests.
- A version of prepare_data in `experiments/case2_v4/run.py` and related grid functions matching the README 97.5% grid.
- The original project dependency list, to confirm library versions against the original server.

Audit records: `manifest.json` and `uncensored_manifest.json` retain commands, timings, and source SHA-256 hashes at startup; `logs/` retains per-task logs.
