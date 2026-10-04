# Small-Scale Runtime Validation

Conclusion: the available core code completed data generation, training, and prediction, but this does not establish completion of full experiments or achievement of manuscript performance targets. SHA256 hashes of all 61 copied Python source files were unchanged before and after the run.

## Setup

- CPU, 2 threads, random seeds 0/1/2, 300 epochs per model per run, and 12 training runs in total.
- Simple setting: one risk, no censoring, 256 training samples, 2048 independent test samples, and 21 time points over 0–10.
- Case II v4: 3 competing risks, 512 training samples, 512 independent test samples, and 25 time points over 0–18; training censoring parameter 0.5, with complete simulated test events.
- Used the available CRSoftNet and JointSoftComp, reducing the JointSoftComp integration grid to 200. SoftComp used raw outputs without the isotonic/simplex postprocessing that was missing at the time. This is not a comparison of the formal manuscript configurations.

## Results (Means over 3 Runs)

| Setting | Model | MSE against Truth before Training | MSE against Truth after Training | Brier | Reference Brier |
|---|---|---:|---:|---:|---:|
| single_event_uncensored | JointSoftComp | 0.098176 | 0.010438 | 0.146756 | 0.176175 |
| single_event_uncensored | SoftComp_raw | 0.130971 | 0.043109 | 0.178309 | 0.176175 |
| case2_v4_censored | JointSoftComp | 0.142013 | 0.007687 | 0.039149 | 0.038655 |
| case2_v4_censored | SoftComp_raw | 0.120528 | 0.008223 | 0.040376 | 0.038655 |

MSE measures squared error between predicted CIFs and the analytic simulation truth. Brier uses complete simulated test-event indicators, averaged over subjects, causes, and equally spaced time points. It is not the original evaluation module IPCW IBS. Lower values are better.
The reference prediction is the empirical training-set event probability without covariates. Under censoring, it uses complete training events retained by the simulator, making it a diagnostic reference with extra information rather than a baseline directly applicable to real censored data. It was not fitted on test data.

- All 6 JointSoftComp training runs produced finite outputs, with no decreasing CIF steps, no increasing survival steps, and initial CIFs of 0. The probability-sum error was below 1e-6.
- Raw SoftComp outputs had decreasing CIF steps: 2236 in the simple setting and 3941 in the three-risk setting. Raw outputs should not be treated as fully postprocessed curves.
- JointSoftComp had lower Brier than the empirical marginal reference in the simple setting. In the censored three-risk setting, its mean Brier was slightly higher than the reference with complete training-event information, so a performance advantage cannot be claimed for that setting.
- Training losses decreased for all models, but the models use different losses, so their numerical loss values cannot be compared directly.

## Full Entry-Point Blockers at the Time of This Run

- Importing experiments/joint_softcomp/simple.py failed because competing_risks.evaluation was missing.
- Importing experiments/case2_v4/run.py first failed because pandas was missing. Static inspection also found missing baseline_models/dsm.py, fine_gray.py, neural_fine_gray.py, and evaluation source.
- This run imported the complete core model and data modules directly without fabricating missing modules or modifying copied code.

## Rerunning

```sh
cd /Users/kevin/Projects/competing_risks
.venv/bin/python local_validation/small_experiment/run.py
```

Results: results.json; full log: run.log; dependency versions: environment.txt. Rerunning overwrites results in the same directory.
