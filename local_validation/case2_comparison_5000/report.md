# Case II v4: Model Comparison with 5,000 Training Subjects

This independent diagnostic used the copied original models and data-generation code. The hashes of all 61 original Python files were unchanged before and after the run. Missing evaluation code and postprocessing were not implemented for this diagnostic.

## Configuration

- 5,000 training subjects and 1,000 independent test subjects; 3 covariates and 3 competing risks.
- Training censoring parameter 0.5; complete simulated test events retained for evaluation. Parameter-generation seed=42, training seeds=130000/130001/130002, and fixed test seed=140000.
- Each model: 1,000 epochs, batch=256, 32 hidden units, 1 residual block, dropout=0, Adam, lr=0.001, cosine scheduling, and no early stopping. CPU with 2 threads.
- JointSoftComp: n_times=4, weight_decay=0.001, and a 1,000-point integration grid.
- SoftComp: n_aug=2, aug_weight=0.5, weight_decay=0.003; raw predictions without isotonic/simplex postprocessing.
- Evaluation at 61 equally spaced time points over 0–18. Both models used identical data for each seed. Model-initialization seeds were 0/1/2; test results were not used for tuning.

## Summary (Mean ± Sample Standard Deviation over 3 Runs)

| Metric | JointSoftComp | SoftComp Raw Output |
|---|---:|---:|
| MSE against True CIF | 0.000542 ± 0.000063 | 0.004072 ± 0.000164 |
| Brier | 0.031838 ± 0.000166 | 0.035335 ± 0.000283 |
| Training Time (s) | 15.944778 ± 0.150950 | 22.377884 ± 0.118279 |
| Prediction Time (s) | 0.190542 ± 0.000618 | 0.010351 ± 0.001208 |

Lower MSE and Brier scores are better. Brier uses complete simulated test events and is averaged over subjects, causes, and equally spaced time points; it is not the original IPCW IBS. Training time includes fit() but excludes startup, data generation, evaluation, and saving outputs.

The complete comparison script took 116.95 seconds, excluding initial Python/PyTorch imports and subsequent report generation.

JointSoftComp reduced mean MSE by 86.7%, and Brier by 9.9%; paired MSE wins: 3/3; paired Brier wins: 3/3.

The test Brier score of the analytic true-risk predictions was 0.031404; the empirical marginal reference using complete training events scored 0.037034. This marginal reference uses latent true events for censored training subjects and serves only as a simulation diagnostic.

## Per-Run Results

| seed | Model | MSE | Brier | Training Time (s) | Test Subjects with Decreasing CIFs |
|---|---|---:|---:|---:|---:|
| 0 | JointSoftComp | 0.000609 | 0.032030 | 15.86 | 0/1000 |
| 0 | SoftComp_raw | 0.004227 | 0.035656 | 22.34 | 1000/1000 |
| 1 | JointSoftComp | 0.000533 | 0.031727 | 15.85 | 0/1000 |
| 1 | SoftComp_raw | 0.004089 | 0.035119 | 22.51 | 997/1000 |
| 2 | JointSoftComp | 0.000485 | 0.031758 | 16.12 | 0/1000 |
| 2 | SoftComp_raw | 0.003900 | 0.035230 | 22.28 | 999/1000 |

## Scope of Interpretation

- This compares the models under the configurations above. In particular, SoftComp lacks the original experiment postprocessing, so the comparison does not establish the final relative performance of the complete manuscript methods.
- There are only 3 training repetitions and one fixed test set. Standard deviations reflect training-sample and model randomness, not changes in the test set or data distribution.
- Checks cover only the specified 0–18 time grid and do not establish accuracy at later times.
- At the time of this run, the original full experiment entry points could not run directly because evaluation code, baseline models, or third-party dependencies were missing.

## Reproduction

```sh
cd /Users/kevin/Projects/competing_risks
.venv/bin/python local_validation/case2_comparison_5000/run.py
.venv/bin/python local_validation/case2_comparison_5000/summarize.py
```

Rerunning overwrites results in the same directory. results.json stores raw metrics, summary.json stores aggregates, and each .pt file stores weights, per-epoch losses, and test predictions.
