# SoftComp and JointSoftComp: Competing Risks

PyTorch implementations of SoftComp (`CRSoftNet`) and JointSoftComp for
competing-risk prediction with residual feed-forward neural networks.

## Models

### SoftComp (`CRSoftNet`)

We model the cumulative incidence function (CIF) via a continuous-time logistic link:

```
log(F_k(t|x) / S(t|x)) = mu_k(x, t),   k = 1, ..., K
```

A shared neural network outputs K logits. A zero is prepended for the survival class, then softmax ensures `F_1 + F_2 + ... + F_K + S = 1` automatically. Training uses negative log-likelihood loss with **time augmentation**: for each sample, we randomly sample additional time points `t < Y_i` and add survival labels (delta=0) to provide supervision across the full time range.

At **inference time**, predicted CIFs are post-processed with **isotonic regression** (Pool-Adjacent-Violators) along the time axis to enforce monotonicity in `t`. The (K+1)-class softmax does not architecturally guarantee `∂F_k/∂t ≥ 0`; isotonic regression is the L2-optimal projection onto the cone of non-decreasing functions and resolves residual violations in the data-sparse tail. See `evaluation/postprocess.py`.

### JointSoftComp

`crsoft_model/joint_softcomp.py` extends the same network with an explicit
censoring state. A `(K+2)`-class softmax predicts the observed states: at risk,
each of the K causes, and censored. Training samples times from the empirical
training event-time distribution and minimizes observed-state cross-entropy.
Prediction uses an Aalen–Johansen recursion to recover monotone CIFs and a
survival curve satisfying `S + sum_k F_k = 1`.

| Model | Training supervision | CIF prediction | Default supplementary configuration |
|---|---|---|---|
| SoftComp | Event/censoring labels at observed time plus survival time augmentation | Direct CIF output, then PAV and simplex postprocessing in the supplementary runners | Width 32, 1 residual block, 1,000 epochs, M=2, augmentation weight 0.5, weight decay 0.003 |
| JointSoftComp | Observed-state labels, including censoring, at sampled times | Aalen–Johansen reconstruction | Width 32, 1 residual block, 1,000 epochs, M=4, weight decay 0.001 |

Both use learning rate 0.001 and batch size 256. `M` has different roles:
SoftComp augments survival labels; JointSoftComp samples times for its loss.
The standard six-method Case II/III runners do not include JointSoftComp;
use the supplementary runners below to compare it with SoftComp and NeuralFG.

This local checkout runs with Python. Some files and datasets from the original
project have not yet been copied; the tree and commands below describe the files
currently available. The result tables below use the completed local experiments in
`local_validation/readme_full/`. Original result tables and ablation notes are
kept in [the historical archive](../historical_results.md).

## Project Structure

```text
competing_risks/
├── .gitignore
├── README.md
├── requirements.txt
├── baseline_models/
│   ├── __init__.py
│   ├── cs_cox.py
│   ├── deephit.py
│   ├── dsm.py
│   ├── fine_gray.py
│   └── neural_fine_gray.py
├── crsoft_model/
│   ├── __init__.py
│   ├── crsoft.py
│   ├── functional.py
│   └── joint_softcomp.py
├── data/
│   ├── __init__.py
│   ├── case1.py
│   ├── case1_v3.py
│   ├── case2.py
│   ├── case2_v3.py
│   ├── case2_v4.py
│   ├── case3.py
│   ├── case3_interaction.py
│   ├── case3_v4.py
│   ├── case3_v5.py
│   ├── synthetic.py
│   └── utils.py
├── docs/
│   └── historical_results.md
├── evaluation/
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── postprocess.py
│   ├── simulation.py
│   ├── survival.py
│   └── visualize.py
├── experiments/
│   ├── case1/
│   │   ├── __init__.py
│   │   └── run.py
│   ├── case1_v2/
│   │   ├── __init__.py
│   │   └── run.py
│   ├── case1_v3/
│   │   ├── __init__.py
│   │   ├── formal.py
│   │   ├── plot_cif_uncertainty.py
│   │   └── run.py
│   ├── case2/
│   │   └── run.py
│   ├── case2_v2/
│   │   └── run.py
│   ├── case2_v3/
│   │   ├── __init__.py
│   │   ├── backfill_fine_gray.py
│   │   ├── formal.py
│   │   ├── plot_survival.py
│   │   └── run.py
│   ├── case2_v4/
│   │   ├── __init__.py
│   │   ├── formal.py
│   │   ├── plot_cif_uncertainty.py
│   │   └── run.py
│   ├── case3/
│   │   ├── __init__.py
│   │   ├── run.py
│   │   └── sweep.py
│   ├── case3_v2/
│   │   ├── backfill_checkpoints.py
│   │   ├── development.py
│   │   ├── plot_cif_uncertainty.py
│   │   └── run.py
│   ├── case3_v4/
│   │   ├── __init__.py
│   │   ├── formal.py
│   │   ├── plot_cif_uncertainty.py
│   │   └── run.py
│   ├── case3_v5/
│   │   ├── __init__.py
│   │   ├── formal.py
│   │   ├── plot_cif_uncertainty.py
│   │   └── run.py
│   ├── joint_softcomp/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── analyze_simple.py
│   │   ├── analyze_uncensored.py
│   │   ├── censoring_levels.py
│   │   ├── population_targets.py
│   │   ├── simple.py
│   │   └── uncensored.py
│   ├── synthetic/
│   │   ├── __init__.py
│   │   ├── brier_sweep.py
│   │   └── run.py
│   ├── __init__.py
│   └── runner.py
├── tests/                         # Unit tests
└── local_validation/              # Local experiment runners and audit artifacts
    ├── small_experiment/
    ├── case2_comparison_5000/
    ├── readme_reproduction/
    └── readme_full/
```

The tree omits generated outputs, `.venv/`, `__pycache__/`, and the contents of
`tests/` and `local_validation/`. PBC/Framingham loaders and experiment folders,
`data/synthetic_comprisk.csv`, paper figures, and standalone experiment-plan
documents have not yet been copied into this checkout.

## Running

### Local setup

Use Python 3.12 (the version used by the local validation environment).
Start with a fresh clone (or run `git pull` in an existing checkout). From the
`competing_risks/` project root, create and activate a virtual environment, install
the pinned runtime dependencies, and make the parent directory importable:

```bash
git clone https://github.com/KevinLuo41/competing_risks.git
cd competing_risks
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"

# Check the environment and run the complete unit test suite
python -m pip check
python -m unittest discover -s tests -t .. -v
```

Run the remaining commands from this directory in the same shell. The source
uses relative imports, so use the full `competing_risks` package name. The parent
path lets Python load this directory as a namespace package; an editable install
is not required. `requirements.txt` pins the seven direct runtime dependencies
used by the validated Python 3.12 environment. No external datasets are needed
for the unit tests or the simulated Case I/II/III experiments.

### Run an experiment

```bash
# Inspect the execution plan without training
python -m competing_risks.experiments.case1.run --list

# Train/load all models and evaluate
python -m competing_risks.experiments.case1.run
python -m competing_risks.experiments.case2.run

# Six-method, K=p=3 simulations
python -m competing_risks.experiments.case2_v4.run
python -m competing_risks.experiments.case3_v4.run
python -m competing_risks.experiments.case3_v5.run
```

For a small Case II v4 check before a full run:

```bash
python -m competing_risks.experiments.case2_v4.run \
  --models SoftComp --n-train 100 --n-test 50 --epochs 2 --cpu-threads 1 \
  --output /tmp/competing_risks_case2_smoke.json
```

Case III v4 uses a gated two-phase protocol. Run the development gate before
starting or resuming the formal 50 paired replicates:

```bash
python -m competing_risks.experiments.case3_v4.formal --phase development --resume
python -m competing_risks.experiments.case3_v4.formal --phase formal --resume
```

Use `--help` on each module to inspect its own options. The older `case3` and
`case3_v2` modules remain available for historical reproduction.

### Synthetic benchmark data

The loader and experiment code are present, but the required
`data/synthetic_comprisk.csv` is not included yet. After supplying that file:

```bash
python -m competing_risks.experiments.synthetic.run
python -m competing_risks.experiments.synthetic.brier_sweep --help
```

PBC and Framingham cannot currently be run because their loaders and experiment
code have not been copied.

### Device and output

`CRSoftNet.fit()` selects CUDA when available and CPU otherwise. The current
code does not automatically select Apple MPS; it uses CPU on a Mac without CUDA.
To check the active environment:

```bash
python -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

Standard runners save metrics, checkpoints, and plots under
`experiments/<case>/outputs/`; runners with `--output` or `--output-dir` can use a
custom destination. Full runs and formal replicate suites are much longer than
the small check above; the result tables below report the local CPU experiments.

### Run JointSoftComp

From the project root after the setup above, run a single simple-setting task:

```bash
python -c 'from competing_risks.experiments.joint_softcomp.simple import main; main()' \
  joint --beta 0.6931471806 --m 4 --reps 1 --threads 1 \
  --out-dir /tmp/joint_softcomp/simple
```

The three supplementary experiment modules define `main()` without an automatic
module entry point, so the commands call `main()` explicitly. See the
[JointSoftComp experiment guide](../../README.md) for all
three experiments, their parameters, and the complete local tables.

To reproduce the full run using the saved task plan (200 / 10 / 5 repetitions):

```bash
python local_validation/readme_full/run_suite.py main 4
python local_validation/readme_full/run_suite.py uncensored 2
python local_validation/readme_full/audit_results.py
```

The driver resumes successful task records in `local_validation/readme_full/`.
The auditor checks repetition coverage and finite outputs, then rebuilds the
summaries. Each training task uses one Torch thread.

### Tests and JointSoftComp

```bash
python -m unittest competing_risks.tests.test_joint_softcomp
```

Run the complete suite with `python -m unittest discover -s tests -t .. -v`.
The current suite contains 116 tests, including the restored Case I v3 modules.

See [JointSoftComp: model, loss, and supplementary experiments](../../README.md)
for its local setup and experiment commands.

### Classical Fine-Gray baseline

`baseline_models/fine_gray.py` implements the classical proportional
subdistribution-hazards model directly with Fine-Gray IPCW risk sets. The implementation solves the weighted partial likelihood with NumPy and stores only
the fitted coefficients, baseline subdistribution hazards, and convergence
diagnostics in checkpoints.

Fine-Gray is the required sixth baseline for Case II v3. Its dedicated backfill
runner reuses saved subjects, censoring, fitting/validation split, and evaluation
grid for formal replicates 0--9. Run it only after generating those artifacts
with the Case II v3 formal runner; the original saved replicates are not included
locally. It does not retrain the first five methods:

```bash
python -m competing_risks.experiments.case2_v3.backfill_fine_gray
```

The original configuration hash remains the base identity for the paired data
and completed checkpoints; `config.json`, replicate manifests, and aggregate
outputs record the required sixth method. Fine-Gray also has its own method
configuration hash, stored in its checkpoints and completion markers. Its hash
payload includes an implementation version that must be bumped with future
algorithm changes, preventing silent reuse of an older fit.

Separate per-cause Fine-Gray regressions do not define a joint native survival
curve. `predict_cif_survival()` returns the unmodified implied survival
`1 - sum_k CIF_k`; it is intentionally not clamped or simplex-projected so the
negative-survival diagnostics can reveal incoherent multi-cause predictions.

### Case II v4: K=3 formal 50

Case II v4 is a new, isolated experiment rather than an overwrite or extension
of the K=8 Case II v3 outputs. It uses K=p=3 and runs DeepHit, DSM, cs-Cox,
NeuralFG, SoftComp, and Fine-Gray on the same 50 paired replicates:

```bash
python -m competing_risks.experiments.case2_v4.formal --phase development --resume
python -m competing_risks.experiments.case2_v4.formal --phase formal --resume
```

The [historical result table](../historical_results.md#case-ii-v4-strong-static-nonlinearity) describes the original formal run. The
standalone protocol document and paper figure are not included locally. Running
the command above produces new local artifacts; Case II v3 results must not be
mixed with this protocol.

### Cache CLI flags (case1, case2, and synthetic)

```text
python -m competing_risks.experiments.case1.run [flags]
  --retrain MODEL ...   # force retrain (also clears that model's eval cache)
  --reeval  MODEL ...   # force re-eval (re-predicts + re-evaluates; keeps model.pt)
  --list                # show cache state and exit
```

Model names are case-insensitive: `deephit dsm cs-cox neural-fg crsoft`.

Examples:
```bash
# Show what's cached, plan what would happen on a normal run
python -m competing_risks.experiments.case1.run --list

# Iterate CRSoft hyperparameters: edit _train_crsoft() in case1/run.py, then:
python -m competing_risks.experiments.case1.run --retrain crsoft

# Re-eval one baseline (e.g. after modifying compute_ibs)
python -m competing_risks.experiments.case1.run --reeval deephit
```

### Two-Tier Caching for Fast Iteration

The `case1`, `case2`, and `synthetic` runners use `experiments/runner.py`, which provides a uniform `(train | load) × (eval | load_eval)` runner over all models. There are two cache tiers per (case, model):

| Tier | Path | What it stores | When skipped |
|---|---|---|---|
| **Model** | `outputs/models/<key>.pt` | Trained model weights / fit state | Loaded if present; training otherwise |
| **Eval** | `outputs/eval_cache/<key>.pt` | Predicted CIF tensor + metrics dict | Loaded if present; predict + eval otherwise (skips O(n²) C^td) |

**CLI** (for these three runners):
```text
python -m competing_risks.experiments.case1.run [flags]
  --retrain MODEL ...   # force retrain (clears both model + eval caches)
  --reeval  MODEL ...   # force re-eval (re-predicts + re-evaluates; keeps model.pt)
  --list                # show cache state and exit
```

Model names are case-insensitive: `deephit dsm cs-cox neural-fg crsoft`.

The `--list` view prints a planning table so you can see what each model will do before any work starts:
```
Execution plan:
  model        model action             eval action
  ------------ ------------------------ ------------------------
  DeepHit      load model.pt            load eval_cache.pt
  ...
  CRSoft       TRAIN (no cache)         EVAL (iteration target)
```

### CRSoft Hyperparameter Iteration

The recommended loop:

1. **Iteration mode (default)** — set `is_iteration_target=True` on the CRSoft `ModelSpec`. The eval cache is then never read for CRSoft (always re-evaluated), while baselines' metrics load instantly from cache.
2. **Edit `_train_crsoft()`** in `experiments/<case>/run.py` to change hyperparameters (one block clearly delimited at the top of the function).
3. **Re-run with `--retrain crsoft`** to wipe the stale model+eval and retrain. Baselines stay cached; CRSoft retrains and re-evaluates.
4. **Settle**: when the config is final, set `is_iteration_target=False` on the CRSoft spec and run `--retrain crsoft` once more so CRSoft's eval cache is also persisted.

For a systematic grid sweep, create a temporary `search.py` in the experiment directory that loads the same data split as `run.py`, iterates through configs with manual `model.fit()` loops, and prints a comparison table. Delete the script after the sweep is recorded.

---

## Local experimental results

These are the measured results of the **2026-10-03 local CPU run**, regenerated
from its per-task JSON outputs. All **878 tasks** exited successfully: 600
experiment-I chunks, 160 experiment-II fits, and 118 experiment-III tasks.
Repetition coverage and the 20 censored/uncensored data pairs were checked.

Tables report **mean (sample SD)**. **Bold marks the best unrounded mean** within
the same setting and metric: MSE, Brier, and IBS are lower-is-better; C_td, AUC,
and IPA are higher-is-better. Close rounded values do not imply a tie or a
statistically significant difference. Oracle/theoretical rows are reference
values and excluded from rankings of fitted models. Bias is a diagnostic and
is not ranked.

### Experiment I: one covariate, no censoring

The data use `x ~ N(0,1)`, one event, and hazard `0.05 exp(βx)`. Each fitted
method has **200 repetitions**, with 200 training subjects and a fixed 50,000
subject test set. The table shows the 10-year Brier score. For JointSoftComp,
M=4 is the default; all tested M values are included.

| Method | β=0: Brier ↓ | β=log 1.5: Brier ↓ | β=log 2: Brier ↓ |
|---|---|---|---|
| SoftComp, M=0 | 0.6034 (0.0000) | 0.5916 (0.0000) | 0.5749 (0.0000) |
| SoftComp, M=1 | 0.3182 (0.0105) | 0.3076 (0.0119) | 0.2887 (0.0143) |
| SoftComp, M=2 (default) | 0.2556 (0.0055) | 0.2457 (0.0066) | 0.2288 (0.0082) |
| SoftComp, M=4 | 0.2419 (0.0018) | 0.2349 (0.0027) | 0.2219 (0.0040) |
| SoftComp, M=8 | 0.2726 (0.0056) | 0.2698 (0.0064) | 0.2622 (0.0076) |
| JointSoftComp, M=1 | 0.2431 (0.0031) | 0.2308 (0.0031) | 0.2082 (0.0026) |
| JointSoftComp, M=2 | 0.2431 (0.0032) | 0.2306 (0.0029) | 0.2082 (0.0026) |
| JointSoftComp, M=4 (default) | 0.2432 (0.0031) | 0.2306 (0.0028) | 0.2082 (0.0025) |
| JointSoftComp, M=8 | 0.2432 (0.0032) | 0.2307 (0.0028) | 0.2080 (0.0024) |
| Cox | 0.2412 (0.0021) | **0.2288 (0.0020)** | **0.2064 (0.0017)** |
| No covariate | **0.2406 (0.0020)** | 0.2430 (0.0021) | 0.2457 (0.0020) |
| True model (oracle reference) | 0.2393 | 0.2270 | 0.2050 |

The no-covariate model has the lowest fitted-model Brier at β=0, and Cox has the
lowest at both nonzero β values. JointSoftComp is close to Cox and improves on
SoftComp when β is nonzero; at β=0, SoftComp with M=4 has a lower Brier than
JointSoftComp. This run does not establish one method as best in every setting.
AUC, IPA, MSE, and the numerical SoftComp limits are in the
[detailed experiment-I results](../../README.md#35-实验结果).

### Experiment II: censored vs. uncensored Case II/III

This comparison uses **Case II v4** (`K=p=3`) and **Case III v5** (`K=3, p=4`),
not the older Case III v4 design in the historical archive. Each cell has **10
paired repetitions**, with 5,000 training subjects (4,500 fit / 500 validation)
and 1,000 test subjects. Removing censoring preserves the subjects, latent event
times, causes, and evaluation grid. SoftComp includes the supplied PAV/simplex
postprocessing; NeuralFG uses its validation early stopping.

**MSE values are multiplied by 1,000.** C_td is the supplied Antolini comparable-pair
statistic; only IBS uses censoring-KM IPCW. Compare C_td and IBS within the same
censoring condition. The two conditions share the true-CIF MSE target and grid.

| Case | Method | Censored MSE ↓ | C_td ↑ | IBS ↓ | Uncensored MSE ↓ | C_td ↑ | IBS ↓ |
|---|---|---|---|---|---|---|---|
| II v4 | JointSoftComp | **2.28 (0.15)** | **0.7499 (0.0116)** | **0.0620 (0.0027)** | **1.72 (0.13)** | **0.7396 (0.0096)** | **0.0611 (0.0029)** |
| II v4 | SoftComp | 8.12 (0.42) | 0.7496 (0.0096) | 0.0659 (0.0026) | 14.29 (0.39) | 0.7388 (0.0098) | 0.0697 (0.0033) |
| II v4 | SoftComp-noaug | 20.33 (0.94) | 0.7467 (0.0093) | 0.0760 (0.0026) | 59.49 (2.44) | 0.7207 (0.0106) | 0.1465 (0.0036) |
| II v4 | NeuralFG | 5.58 (1.27) | 0.7263 (0.0106) | 0.0639 (0.0020) | 3.35 (0.56) | 0.7320 (0.0104) | 0.0619 (0.0029) |
| III v5 | JointSoftComp | **1.12 (0.26)** | 0.6809 (0.0080) | **0.1235 (0.0027)** | **0.81 (0.11)** | 0.6776 (0.0051) | **0.1213 (0.0020)** |
| III v5 | SoftComp | 3.20 (0.41) | **0.6866 (0.0081)** | 0.1273 (0.0021) | 4.31 (0.15) | **0.6782 (0.0056)** | 0.1235 (0.0018) |
| III v5 | SoftComp-noaug | 9.22 (0.60) | 0.6825 (0.0078) | 0.1288 (0.0022) | 61.14 (2.38) | 0.6690 (0.0057) | 0.1618 (0.0015) |
| III v5 | NeuralFG | 3.44 (0.50) | 0.6528 (0.0122) | 0.1266 (0.0024) | 2.24 (0.25) | 0.6607 (0.0067) | 0.1235 (0.0022) |

JointSoftComp has the lowest MSE and IBS in all four case/condition settings and
the highest C_td in Case II. SoftComp has the highest C_td in Case III under
both conditions. JointSoftComp's censored MSE is 2.28 versus SoftComp's 8.12 in
Case II, and 1.12 versus 3.20 in Case III. Removing censoring reduces
JointSoftComp's MSE but increases SoftComp's MSE in these runs.

The supplementary grid extends to the 97.5th percentile of observed test event
times. Case III adds its four fixed evaluation times; Case II defines no extra
fixed times in the supplied source. This differs from the original Case II
formal runner's 90th-percentile grid. The local library versions are also
recorded separately, so these tables replace the previous numbers without
claiming exact replication of the original server environment.

### Experiment III: censoring over a fixed time horizon

Follow-up ends at **τ=20**. The target censoring fraction is
`P(C < min(T, τ)) = ρ`, with ρ in {0, 20%, 50%, 80%}. Each simulation setting
has **5 seeds**, 5,000 training subjects, and 1,000 uncensored test subjects.
Evaluation uses 100 equally spaced times in `(0,20]`. `case3` uses the Case III
v5 DGP with independent censoring; `depcens` uses covariate-dependent censoring;
`constant` uses covariate-dependent constant cause-specific hazards. At ρ=0,
`depcens` duplicates `case3` and is not run separately.

**MSE values are multiplied by 1,000.** Signed bias is `mean(predicted CIF -
true CIF)` at t=20 for each cause.

| Setting | ρ | Method | MSE ↓ | C_td ↑ | IBS ↓ | Bias at t=20 (causes 1 / 2 / 3) |
|---|---|---|---|---|---|---|
| case3 | 0% | SoftComp | 5.97 (0.18) | 0.6829 (0.0049) | 0.1310 (0.0021) | -0.095 / -0.097 / -0.097 |
| case3 | 0% | JointSoftComp | **1.01 (0.18)** | **0.6844 (0.0054)** | **0.1261 (0.0021)** | +0.001 / -0.001 / -0.001 |
| case3 | 20% | SoftComp | 7.00 (0.39) | 0.6835 (0.0046) | 0.1320 (0.0019) | -0.113 / -0.118 / -0.115 |
| case3 | 20% | JointSoftComp | **1.20 (0.19)** | **0.6852 (0.0046)** | **0.1261 (0.0023)** | +0.000 / +0.002 / +0.005 |
| case3 | 50% | SoftComp | 11.05 (0.21) | **0.6850 (0.0079)** | 0.1360 (0.0026) | -0.141 / -0.148 / -0.131 |
| case3 | 50% | JointSoftComp | **1.73 (0.29)** | 0.6822 (0.0068) | **0.1266 (0.0025)** | -0.005 / -0.009 / +0.004 |
| case3 | 80% | SoftComp | 25.83 (4.87) | 0.6736 (0.0077) | 0.1512 (0.0071) | -0.170 / -0.190 / -0.206 |
| case3 | 80% | JointSoftComp | **6.07 (0.92)** | **0.6761 (0.0083)** | **0.1311 (0.0024)** | -0.060 / -0.022 / -0.050 |
| depcens | 20% | SoftComp | 7.45 (0.30) | 0.6749 (0.0043) | 0.1323 (0.0020) | -0.117 / -0.119 / -0.115 |
| depcens | 20% | JointSoftComp | **1.36 (0.20)** | **0.6843 (0.0051)** | **0.1264 (0.0022)** | -0.004 / -0.001 / +0.006 |
| depcens | 50% | SoftComp | 13.10 (1.58) | 0.6543 (0.0035) | 0.1382 (0.0024) | -0.147 / -0.156 / -0.148 |
| depcens | 50% | JointSoftComp | **2.32 (0.39)** | **0.6812 (0.0049)** | **0.1272 (0.0025)** | -0.002 / -0.008 / +0.000 |
| depcens | 80% | SoftComp | 27.75 (2.70) | 0.5938 (0.0243) | 0.1531 (0.0041) | -0.195 / -0.183 / -0.203 |
| depcens | 80% | JointSoftComp | **8.19 (2.61)** | **0.6681 (0.0100)** | **0.1327 (0.0030)** | +0.002 / +0.009 / -0.030 |
| constant | 0% | SoftComp | 7.23 (0.18) | 0.6652 (0.0062) | 0.1493 (0.0021) | -0.147 / -0.106 / -0.090 |
| constant | 0% | JointSoftComp | **0.62 (0.18)** | **0.6682 (0.0021)** | **0.1429 (0.0018)** | +0.001 / +0.000 / -0.002 |
| constant | 20% | SoftComp | 11.35 (0.66) | **0.6674 (0.0056)** | 0.1535 (0.0022) | -0.186 / -0.131 / -0.112 |
| constant | 20% | JointSoftComp | **0.67 (0.10)** | 0.6663 (0.0032) | **0.1432 (0.0017)** | +0.002 / +0.001 / -0.004 |
| constant | 50% | SoftComp | 20.93 (1.88) | **0.6690 (0.0053)** | 0.1631 (0.0027) | -0.235 / -0.166 / -0.149 |
| constant | 50% | JointSoftComp | **1.21 (0.28)** | 0.6661 (0.0031) | **0.1439 (0.0019)** | -0.003 / +0.015 / -0.034 |
| constant | 80% | SoftComp | 48.24 (6.35) | **0.6667 (0.0039)** | 0.1904 (0.0082) | -0.330 / -0.239 / -0.200 |
| constant | 80% | JointSoftComp | **10.03 (1.77)** | 0.6585 (0.0054) | **0.1534 (0.0020)** | -0.131 / -0.047 / -0.105 |

JointSoftComp has the lowest MSE and IBS in all 11 simulation settings. The C_td
winner varies, including SoftComp for all three nonzero censoring levels of the
constant-hazard setting. At 80% censoring, both methods have larger errors;
JointSoftComp's constant-hazard MSE is 10.03 versus SoftComp's 48.24. The ranking
uses this run's observed means and does not assert statistical significance.

The eight constant-input checks use 10,000 training subjects and compare each
learned model with nonparametric AJ and the true marginal CIF. The following
values are the maximum absolute error on `[0,20]` (one check per cell; no SD):

| ρ | SoftComp | JointSoftComp | Nonparametric AJ reference |
|---|---|---|---|
| 0% | 0.1022 | 0.0117 | **0.0069** |
| 20% | 0.1148 | 0.0081 | **0.0057** |
| 50% | 0.1507 | 0.0206 | **0.0092** |
| 80% | 0.1878 | 0.0748 | **0.0359** |

Nonparametric AJ has the smallest error in all four checks; JointSoftComp has
less error than SoftComp. The numerical population-target table is available in
[the detailed guide](../../README.md#55-实验结果).

### Earlier Case II diagnostic

A separate earlier comparison used three training seeds, a fixed test set, and
raw SoftComp predictions without PAV/simplex postprocessing. Its mean (SD) CIF
MSE was **0.000542 (0.000063)** for JointSoftComp versus 0.004072 (0.000164) for
raw SoftComp; complete-event Brier was **0.031838 (0.000166)** versus
0.035335 (0.000283). That Brier is not IPCW IBS, and this diagnostic is not pooled
with the full supplementary runs. See the
[diagnostic report](../../local_validation/case2_comparison_5000/report.md).

### Result artifacts

| Artifact | Contents |
|---|---|
| [Run protocol](../../local_validation/readme_full/protocol.txt) | Seeds, evaluation conventions, source changes, and environment limits |
| [Environment lock](../../local_validation/readme_full/requirements.lock.txt) | Full local dependency versions |
| [Audit](../../local_validation/readme_full/audit.json) | Completed task counts, finite-value checks, and summary hashes |
| [Paired-data checks](../../local_validation/readme_full/all_pair_checks.json) | Shared subjects and grids for all 20 case/replicate pairs |
| [Experiment I summary](../../local_validation/readme_full/simple_summary.txt) | Brier, AUC, IPA, and true-CIF MSE |
| [Experiment II summary](../../local_validation/readme_full/uncensored_summary.txt) | Censored/uncensored mean (SD) by method |
| [Experiment III summary](../../local_validation/readme_full/censoring_summary.txt) | Censoring levels, metrics, horizon bias, and AJ checks |
| [Local vs. original comparison](../../local_validation/readme_full/readme_comparison.txt) | The original values, local measurements, and their differences |
| [Historical archive](../historical_results.md) | Original simulation, real-world, and ablation tables |

Per-task outputs are under `local_validation/readme_full/{simple,censored,
uncensored,censoring}/`; `records/` and `logs/` preserve commands, exit codes,
and stdout/stderr.
