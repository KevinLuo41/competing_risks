# JointSoftComp: Model, Loss, and Supplementary Experiments

This document describes JointSoftComp's model and loss, its implementation, and three supplementary experiments, including their code, commands, results, and analysis. All paths are relative to the project root, `competing_risks/`. The result tables report complete local CPU runs from 2026-10-03: 200 repetitions per setting in Experiment I, 10 in Experiment II, and 5 in Experiment III.

The run completed 878 tasks. Raw JSON outputs, commands, and audit records are available in
[`local_validation/readme_full/`](local_validation/readme_full/).
Table entries are means from the raw results, with sample standard deviations in parentheses.
**Bold marks the best unrounded mean among fitted methods for the same setting and metric.**
Lower MSE/Brier/IBS and higher C_td/AUC/IPA are better. Similar displayed values do not imply ties or statistically significant differences. The true model and population minimizers are references and are excluded from fitted-model rankings. Signed bias is a diagnostic and is not ranked.

## 1. Local Setup

Install Git and Miniconda (or Anaconda), then run the following commands in your terminal.
These instructions apply to macOS / Linux. Run all experiment commands from the project root.

### 1.1 Clone the Repository

```bash
git clone https://github.com/KevinLuo41/competing_risks.git
cd competing_risks
```

### 1.2 Create and Activate a Conda Environment

```bash
conda create -n competing-risks python=3.12 pip -y
conda activate competing-risks
```

If `conda activate` reports that your shell has not been initialized, run `conda init`, reopen your terminal, return to `competing_risks/`, and run `conda activate competing-risks` again.

### 1.3 Install Dependencies and Set the Import Path

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"
```

The project uses relative imports under the `competing_risks` package name, so its parent directory must be on `PYTHONPATH`. Use the same terminal for the Quick Start commands below. After opening a new terminal, activate the Conda environment, enter the project root, and run the `export` command again.
The Full Experiments Bash scripts set the import path automatically.
Direct dependency versions are pinned in `requirements.txt`; the complete validated environment is recorded in the [dependency lock](local_validation/readme_full/requirements.lock.txt).

### 1.4 Verify the Installation

```bash
python -m unittest competing_risks.tests.test_joint_softcomp
```

An output of `Ran 5 tests` followed by `OK` confirms that installation validation passed. The tests check observed-state labels, CIF recovery through Aalen–Johansen, monotone predictions whose probabilities sum to 1, and finite losses during short training runs. No additional datasets are required.

## 2. Model and Loss

- **Idea**: Following the joint survival super learner of Munch & Gerds (2026), treat censoring as a separate state. Each subject's observed state η(t) ∈ {at risk, causes 1..K, censored} is then known at every time t.
- **Model**: Use SoftComp's residual feed-forward network with input [x; t] and K + 1 output logits. Prepend an at-risk logit fixed at 0 and apply a (K + 2)-class softmax to obtain observed-state probabilities.
- **Loss**: In each minibatch, sample M = 4 times per subject from the empirical distribution of training event times and minimize observed-state cross-entropy at those times. All labels are observed, and the loss is strictly proper without estimating the censoring distribution. There is no time augmentation, auxiliary Brier term, or postprocessing.
- **Prediction**: Recover CIFs from observed-state probabilities through the Aalen–Johansen recursion ΔΛ_k = [ΔP_k]₊ / P_0(t-). The resulting CIFs are monotone and satisfy S + Σ_k F_k = 1.
- **Default configuration**: Width 32, 1 residual block, 4 times per subject, learning rate 1e-3, weight decay 1e-3, batch size 256, and 1000 epochs, shared across all six datasets.
- **Comparison: manuscript SoftComp**: A (K + 1)-class softmax directly outputs (S, F_1, …, F_K). Its loss combines Eq. (5), cross-entropy at the observed time, with time augmentation: M times per subject drawn from Unif(0, Y_i), labeled as survival with weight 0.5; the manuscript default is M = 2. Predictions undergo PAV and simplex postprocessing. Its population target depends on the censoring distribution and M and is not the CIF.

### 2.1 Code

| File | Class / Function | Purpose |
|---|---|---|
| `crsoft_model/joint_softcomp.py` | `JointSoftComp` | JointSoftComp training and CIF prediction |
| | `observed_state_labels` | Construct observed-state labels η(t) from (Y, Δ) |
| | `aalen_johansen_from_observed` | Recover CIFs and survival from observed-state probabilities through AJ recursion |
| `crsoft_model/crsoft.py` (existing) | `CRSoftNet` | Manuscript SoftComp; also the parent class of `JointSoftComp`, providing its network structure and training loop |
| `evaluation/postprocess.py` (existing) | `isotonic_project_cif`, `enforce_cif_simplex` | PAV and simplex postprocessing for manuscript SoftComp |
| `evaluation/survival.py` (existing) | `evaluate_cif_metrics` | C_td (Antolini) and IPCW IBS |
| `evaluation/simulation.py` (existing) | `compute_mse_accuracy` | MSE against the true CIF |

Complete the setup in Section 1 before running this example. The caller supplies `p`, `K`, and the data tensors:

```python
from competing_risks.crsoft_model.joint_softcomp import JointSoftComp

model = JointSoftComp(input_dim=p, num_causes=K)  # Default width 32, 1 block
model.fit(X, Y, Delta)  # Train with the default configuration
cif, survival = model.predict_cif_survival_grid(X_test, times)
```

### 2.2 Background Documents

The background documents are saved as HTML files. Download them from the repository and open them in a browser:

- [SoftComp: censoring dependence of the current loss, and Option A](docs/references/softcomp_censoring_dependence_option_a.html)
- [JointSoftComp: two follow-up tests](docs/references/jointsoftcomp_two_follow_up_tests.html)

The HTML documents retain the original historical results. The tables below report local reproduction results.

## 3. Experiment I: Simplest Setting (One Predictor, No Censoring)

### 3.1 Setup and Methods

- **Data**: x ∼ N(0, 1), a single event type with hazard 0.05·exp(βx), and β ∈ {0, log 1.5, log 2}. There is no censoring. Training n = 200, with 200 repetitions. The test set contains 50,000 uncensored subjects generated with a fixed seed.
- **Metrics**: Brier score at 10 years, AUC(10) with cases T ≤ 10 and controls T > 10, IPA = 1 − Brier / Brier(no-covariate model), and MSE between predicted and true 10-year risks.
- **Methods**:
  - SoftComp with the manuscript loss and postprocessing, M = 0, 1, 2 (manuscript default), 4, or 8 augmentation times, with weight 0.5.
  - JointSoftComp with M = 1, 2, 4 (default), or 8 sampled times per subject.
  - Cox with one covariate and a Breslow baseline, the true model, and the no-covariate model using the training set's marginal risk.
  - The population minimizer of the SoftComp loss, the function learned in the limit of infinite training data: π(10 | x) = 1 / (1 + 0.5·M·e^z·E₁(z)), where z = 10·0.05·e^{βx} and E₁ is the exponential integral. It is computed numerically without training.

### 3.2 Code

| Role | File | Functions |
|---|---|---|
| Data generation | `experiments/joint_softcomp/simple.py` | `simulate` (generate x and T), `true_risk` (true 10-year risk) |
| Models | `simple.py` | `predict_softcomp` (`CRSoftNet` with postprocessing), `predict_joint` (`JointSoftComp`), `predict_cox`, `softcomp_limit` and `scaled_exp1` (population minimizer) |
| Evaluation | `simple.py` | `interpolate` (interpolate predictions from the x grid to test subjects), `auc`, `evaluate`, `summarize` |
| Entry point | `main` in `simple.py` | Run one (β, method, M) combination and write `simple_b{beta}_{method}_m{M}_r{rep_start}.json`; `reference` outputs Cox, the true model, the no-covariate model, and population minimizers for each M |
| Result summary | `experiments/joint_softcomp/analyze_simple.py` | Merge repetition batches and report Brier, AUC, IPA, and MSE by β |

### 3.3 Quick Start

Run two reference repetitions to check data generation, reference-model evaluation, and JSON output:

```bash
python -m competing_risks.experiments.joint_softcomp.simple \
    reference \
    --beta 0 \
    --m 0 \
    --reps 2 \
    --out-dir /tmp/joint_softcomp
```

Output: `/tmp/joint_softcomp/simple_b0.0000_reference_m0_r0.json`.

### 3.4 Full Experiments

```bash
bash scripts/run_joint_softcomp.sh
```

The script runs all three β values, five M values for SoftComp, four M values for JointSoftComp, and the reference models, with 200 repetitions per setting. It summarizes the results automatically. Outputs are written to `/tmp/joint_softcomp/full/simple/`, with the summary in `summary.txt`. See the [script](scripts/run_joint_softcomp.sh) for the complete loops.

### 3.5 Results

The three tables below group all fitted-model configurations and the true-model reference by β.
Each fitted configuration has 200 repetitions, with the same fixed test set of 50,000 subjects.

**β = 0** (200 repetitions per fitted method; mean and sample standard deviation):

| Method | Brier(10) ↓ | AUC(10) ↑ | IPA ↑ | MSE vs truth ↓ |
|---|---|---|---|---|
| SoftComp, M=0 | 0.6034 (0.0000) | **0.5000 (0.0009)** | -1.5075 (0.0203) | 0.3679 (0.0000) |
| SoftComp, M=1 | 0.3182 (0.0105) | 0.4999 (0.0029) | -0.3222 (0.0444) | 0.0806 (0.0106) |
| SoftComp, M=2 (default) | 0.2556 (0.0055) | 0.4993 (0.0029) | -0.0622 (0.0235) | 0.0170 (0.0056) |
| SoftComp, M=4 | 0.2419 (0.0018) | 0.4990 (0.0028) | -0.0052 (0.0102) | 0.0023 (0.0017) |
| SoftComp, M=8 | 0.2726 (0.0056) | 0.4988 (0.0029) | -0.1331 (0.0249) | 0.0322 (0.0055) |
| JointSoftComp, M=1 | 0.2431 (0.0031) | 0.4992 (0.0029) | -0.0102 (0.0109) | 0.0035 (0.0029) |
| JointSoftComp, M=2 | 0.2431 (0.0032) | 0.4996 (0.0030) | -0.0104 (0.0110) | 0.0036 (0.0030) |
| JointSoftComp, M=4 (default) | 0.2432 (0.0031) | 0.4997 (0.0030) | -0.0104 (0.0104) | 0.0036 (0.0029) |
| JointSoftComp, M=8 | 0.2432 (0.0032) | 0.4998 (0.0030) | -0.0106 (0.0106) | 0.0037 (0.0030) |
| Cox | 0.2412 (0.0021) | 0.5000 (0.0028) | -0.0023 (0.0029) | 0.0019 (0.0021) |
| No covariate | **0.2406 (0.0020)** | 0.5000 (0.0000) | **0.0000 (0.0000)** | **0.0013 (0.0020)** |
| True model (oracle reference) | 0.2393 | 0.5000 | -0.0000 | 0.0000 |

**β = log 1.5** (200 repetitions per fitted method; mean and sample standard deviation):

| Method | Brier(10) ↓ | AUC(10) ↑ | IPA ↑ | MSE vs truth ↓ |
|---|---|---|---|---|
| SoftComp, M=0 | 0.5916 (0.0000) | 0.5095 (0.0340) | -1.4343 (0.0204) | 0.3690 (0.0000) |
| SoftComp, M=1 | 0.3076 (0.0119) | 0.6393 (0.0030) | -0.2658 (0.0509) | 0.0825 (0.0121) |
| SoftComp, M=2 (default) | 0.2457 (0.0066) | 0.6391 (0.0038) | -0.0113 (0.0285) | 0.0195 (0.0068) |
| SoftComp, M=4 | 0.2349 (0.0027) | 0.6386 (0.0047) | 0.0332 (0.0119) | 0.0074 (0.0026) |
| SoftComp, M=8 | 0.2698 (0.0064) | 0.6380 (0.0072) | -0.1103 (0.0259) | 0.0411 (0.0062) |
| JointSoftComp, M=1 | 0.2308 (0.0031) | 0.6388 (0.0045) | 0.0504 (0.0095) | 0.0035 (0.0029) |
| JointSoftComp, M=2 | 0.2306 (0.0029) | 0.6386 (0.0060) | 0.0510 (0.0088) | 0.0034 (0.0027) |
| JointSoftComp, M=4 (default) | 0.2306 (0.0028) | 0.6384 (0.0073) | 0.0512 (0.0085) | 0.0033 (0.0026) |
| JointSoftComp, M=8 | 0.2307 (0.0028) | 0.6380 (0.0076) | 0.0507 (0.0086) | 0.0035 (0.0027) |
| Cox | **0.2288 (0.0020)** | **0.6398 (0.0000)** | **0.0585 (0.0036)** | **0.0017 (0.0019)** |
| No covariate | 0.2430 (0.0021) | 0.5000 (0.0000) | 0.0000 (0.0000) | 0.0158 (0.0020) |
| True model (oracle reference) | 0.2270 | 0.6398 | 0.0603 | 0.0000 |

**β = log 2** (200 repetitions per fitted method; mean and sample standard deviation):

| Method | Brier(10) ↓ | AUC(10) ↑ | IPA ↑ | MSE vs truth ↓ |
|---|---|---|---|---|
| SoftComp, M=0 | 0.5749 (0.0000) | 0.5239 (0.0615) | -1.3396 (0.0190) | 0.3735 (0.0000) |
| SoftComp, M=1 | 0.2887 (0.0143) | 0.7291 (0.0002) | -0.1750 (0.0594) | 0.0851 (0.0143) |
| SoftComp, M=2 (default) | 0.2288 (0.0082) | 0.7291 (0.0004) | 0.0687 (0.0342) | 0.0240 (0.0083) |
| SoftComp, M=4 | 0.2219 (0.0040) | 0.7290 (0.0011) | 0.0967 (0.0169) | 0.0156 (0.0039) |
| SoftComp, M=8 | 0.2622 (0.0076) | 0.7290 (0.0010) | -0.0672 (0.0309) | 0.0545 (0.0074) |
| JointSoftComp, M=1 | 0.2082 (0.0026) | 0.7291 (0.0002) | 0.1525 (0.0081) | 0.0031 (0.0025) |
| JointSoftComp, M=2 | 0.2082 (0.0026) | 0.7291 (0.0004) | 0.1529 (0.0076) | 0.0031 (0.0024) |
| JointSoftComp, M=4 (default) | 0.2082 (0.0025) | 0.7291 (0.0007) | 0.1528 (0.0075) | 0.0031 (0.0023) |
| JointSoftComp, M=8 | 0.2080 (0.0024) | 0.7291 (0.0008) | 0.1534 (0.0074) | 0.0030 (0.0023) |
| Cox | **0.2064 (0.0017)** | **0.7291 (0.0000)** | **0.1599 (0.0043)** | **0.0014 (0.0017)** |
| No covariate | 0.2457 (0.0020) | 0.5000 (0.0000) | 0.0000 (0.0000) | 0.0392 (0.0020) |
| True model (oracle reference) | 0.2050 | 0.7291 | 0.1614 | 0.0000 |

Brier scores of the SoftComp loss population minimizers, computed numerically rather than through training:

| M | β=0 | β=log 1.5 | β=log 2 |
|---|---|---|---|
| 0 | 0.6034 | 0.5916 | 0.5749 |
| 1 | 0.3220 | 0.3083 | 0.2860 |
| 2 | 0.2545 | 0.2440 | 0.2257 |
| 4 | 0.2414 | 0.2343 | 0.2202 |
| 8 | 0.2730 | 0.2699 | 0.2609 |

### 3.6 Analysis

- **The best Brier score depends on the setting.** At β=0, the no-covariate model performs best (0.2406); at nonzero β, Cox performs best (0.2288 / 0.2064). JointSoftComp with default M=4 scores 0.2432 / 0.2306 / 0.2082. At β=0, SoftComp with M=4 has a lower score of 0.2419, so JointSoftComp does not outperform SoftComp in every setting.
- **SoftComp is sensitive to the amount of augmentation.** Its Brier scores are 0.6034 / 0.5916 / 0.5749 at M=0, fall to 0.2419 / 0.2349 / 0.2219 at M=4, and increase again at M=8. JointSoftComp's M=1/2/4/8 controls the number of sampled loss-evaluation times; its Brier scores differ by less than 0.0003 within each β setting.
- **Discrimination and calibration should be considered separately.** At β=log 2, many methods have AUC values near 0.7291, while their Brier scores and MSE against truth differ substantially. Best-value markings compare means and do not establish statistical significance.

## 4. Experiment II: Censored vs. Uncensored Data (Cases II/III)

The local results compare only the four methods actually run below.

### 4.1 Setup and Methods

- **Data**: Case II v4 (K = p = 3) and Case III v5 (K = 3, p = 4), using the first 10 formal-seed repetitions. Training seeds are 130000 + r / 330000 + r, and test seeds are 140000 + r / 340000 + r. There are 5000 training subjects (4500 for fitting, 500 for validation) and 1000 test subjects.
- **Conditions**: The censored condition follows the manuscript setup: exponential censoring calibrated to P(C ≤ median T) = 0.5. The uncensored condition retains the same subjects' covariates, true event times, and causes, removing censoring from both training and test sets. Both conditions use the same evaluation grid: censored-test event-time quantiles up to 97.5%, with the four fixed times defined in the Case III source added. No additional fixed times are supplied by the Case II source, so none are added. The original Case II formal runner uses a 90% grid; this supplementary experiment uses 97.5%.
- **Methods**: JointSoftComp with its default configuration; SoftComp with the manuscript configuration, M = 2; SoftComp-noaug with the same configuration but M = 0; and NeuralFG with its manuscript configuration, the best baseline in Cases II/III.
- **Metrics**: MSE against the true CIF (×10⁻³), C_td, and IBS. The test sets differ across conditions. C_td uses the source's Antolini comparable-pair statistic; only IBS uses IPCW from the censoring KM estimate. Compare C_td and IBS within each condition. MSE is computed against the true CIF on the same grid and can be compared across conditions.

### 4.2 Code

| Role | File | Functions |
|---|---|---|
| Data generation | `experiments/case2_v4/run.py`, `experiments/case3_v5/run.py` (existing) | `prepare_data` (added `censor_rate` parameter, default 0.5) |
| | `data/case2_v4.py`, `data/case3_v5.py` (existing) | `generate_parameters`, `generate_data`, `compute_cif` (true CIF) |
| | `data/utils.py` | `solve_inverse_cdf`, `assign_causes_and_censor`, `generate_test_observations` (no censoring when `censor_rate = 0`) |
| | `evaluation/survival.py` (existing) | `build_evaluation_time_grid` |
| | `experiments/joint_softcomp/uncensored.py` | `prepare` (generate the same subjects without censoring and retain the censored condition's evaluation grid) |
| Models | `uncensored.py` | `build_method`: construct JointSoftComp directly; obtain SoftComp, SoftComp-noaug, and NeuralFG from `build_specs` in `run.py` |
| | `baseline_models/neural_fine_gray.py` (existing) | `NeuralFG` |
| Evaluation | `evaluation/simulation.py`, `evaluation/survival.py` (existing) | `compute_mse_accuracy`, `compute_dist`, `evaluate_cif_metrics` |
| Entry point | `run`, `main` in `uncensored.py` | Run one (case, method, repetition, condition) combination and write `case{c}_rep{r}_{method}.json` |
| Result summary | `experiments/joint_softcomp/analyze_uncensored.py` | Pair the same repetitions across conditions and report MSE, C_td, and IBS |

### 4.3 Quick Start

Run one uncensored JointSoftComp repetition for Case II:

```bash
python -m competing_risks.experiments.joint_softcomp.uncensored \
    --case 2 \
    --method JointSoftComp \
    --replicate 0 \
    --uncensored \
    --threads 1 \
    --out-dir /tmp/joint_softcomp/quick_uncensored
```

Output: `/tmp/joint_softcomp/quick_uncensored/case2_rep00_JointSoftComp.json`.
This command uses the full default training configuration for one case / method / condition.

### 4.4 Full Experiments

```bash
bash scripts/run_joint_softcomp_uncensored.sh
```

The script runs Cases II / III, all four methods, and both censored / uncensored conditions, with 10 paired repetitions per setting. It summarizes the results automatically. Outputs are written to the `censored/` and `uncensored/` subdirectories of `/tmp/joint_softcomp/full/uncensored/`, with the summary in `summary.txt`. See the [script](scripts/run_joint_softcomp_uncensored.sh) for the complete loops.

### 4.5 Results

Each setting has 10 paired repetitions. Entries are means (sample standard deviations).
**MSE values are multiplied by 1,000.** Bold marks the best metric within the same case and censoring condition.

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

### 4.6 Analysis

- **JointSoftComp achieves the lowest MSE and IBS in all four case/condition combinations.** Under censoring, its MSE is 2.28 / 1.12, compared with 8.12 / 3.20 for SoftComp.
- **The best C_td depends on the case.** JointSoftComp has the highest mean in both Case II conditions; SoftComp has the highest mean in both Case III conditions. In uncensored Case III, both previously displayed as 0.678 at three decimal places. Four decimal places distinguish 0.6782 from 0.6776, so they should not be marked as tied for first.
- **Removing censoring increases SoftComp's MSE.** Case II changes from 8.12 → 14.29, and Case III from 3.20 → 4.31. Without augmentation, the changes are 20.33 → 59.49 and 9.22 → 61.14. JointSoftComp and NeuralFG have lower MSE without censoring.

Paired data and evaluation grids were checked for all 20 case/repetition pairs. The original environment's dependency versions are unknown, and no additional fixed evaluation times were provided for Case II. These are local measurements, not a claim of exact agreement with the original server. See the [run protocol](local_validation/readme_full/protocol.txt) for conventions and fixes.

## 5. Experiment III: Censoring Levels over a Fixed Time Horizon

The empirical results below come from local runs.

### 5.1 Setup and Methods

- **Censoring definition**: Fix τ = 20. Censoring times follow an exponential distribution, with the rate calibrated to a fraction ρ censored before τ: P(C < min(T, τ)) = ρ. Follow-up ends at τ, so Y = min(T, C, τ). Use ρ ∈ {0, 20%, 50%, 80%}; at ρ = 0, data are complete on [0, τ]. The realized training censoring fraction is saved in each raw result's `censored_before_tau_train` field.
- **Four tests**:
  - `aj-check`: Case III data (n = 10000), with only a constant model input to estimate marginal CIFs, compared with nonparametric AJ estimates and the true marginal CIFs.
  - `case3`: Case III DGP with independent censoring.
  - `depcens`: Case III DGP with covariate-dependent censoring; multiply the rate by e^{0.8 x₁}. At ρ = 0 this matches `case3`, so it is not run again.
  - `constant`: Constant cause-specific hazards, K = 3 and p = 4, with λ_k(x) = exp(a_k + b_kᵀx), baseline rates 0.05 / 0.035 / 0.025, and independent censoring.
- **Sample sizes and repetitions**: 5000 training subjects and 1000 test subjects, with data seeds 30000 + s / 40000 + s for s = 0, …, 4. The test set is uncensored, and metrics are evaluated at 100 equally spaced points in (0, τ]. C_td and IBS therefore do not require IPCW, and results can be compared directly across ρ values.
- **Methods**: Manuscript SoftComp (`softcomp`, M = 2, augmentation weight 0.5, weight decay 3e-3, with postprocessing) versus JointSoftComp (`joint`, default configuration).
- **Metrics**: MSE against the true CIF (×10⁻³), C_td, IBS, and mean cause-specific bias F̂_k − F_k at t = 20.
- **Theoretical table**: Population minimizers of each training term, computed by numerical integration for a constant-hazard example: λ₁ = 0.10, λ₂ = 0.05, cause 1, 0.5·M = 1, and τ = 20.

### 5.2 Code

| Role | File | Functions |
|---|---|---|
| Data generation | `experiments/joint_softcomp/censoring_levels.py` | `simulate`, `_event_times`, `_censoring_times` (calibrate the censoring rate to ρ), `_constant_hazards`, `constant_cif` |
| | `data/case3_v5.py`, `data/utils.py` (existing) | `generate_parameters`, `compute_cif`, `solve_inverse_cdf` |
| Models | `censoring_levels.py` | `train` (`CRSoftNet` or `JointSoftComp`), `predict` (postprocess SoftComp predictions) |
| Evaluation | `censoring_levels.py` | `run_simulation`, `_horizon_bias`, `aalen_johansen` (nonparametric AJ estimate), `run_aj_check` |
| | `evaluation/simulation.py`, `evaluation/survival.py` (existing) | `compute_mse_accuracy`, `evaluate_cif_metrics` |
| Entry point | `main` in `censoring_levels.py` | Run one (test, method, ρ, seed) combination and write `{test}_rho{rho}_{method}_seed{s}.json` |
| Result summary | `summarize` subcommand in `censoring_levels.py` | Summarize results by (test, ρ, method) |
| Theoretical table | `experiments/joint_softcomp/population_targets.py` (standard library only) | `censoring_rate`, `loss_targets`, `aalen_johansen_from_observed`, `print_table` |

### 5.3 Quick Start

Use two epochs to check training and evaluation with constant hazards and 50% censoring:

```bash
python -m competing_risks.experiments.joint_softcomp.censoring_levels \
    constant \
    --method joint \
    --rho 0.5 \
    --seed 0 \
    --epochs 2 \
    --threads 1 \
    --out-dir /tmp/joint_softcomp/quick_censoring
```

Output: `/tmp/joint_softcomp/quick_censoring/constant_rho0.5_joint_seed0.json`.
Two training epochs only check the workflow. The full experiment below uses the default 1,000 epochs.

### 5.4 Full Experiments

```bash
bash scripts/run_joint_softcomp_censoring.sh
```

The script runs four censoring levels, both methods, and the Case III / constant-hazard / covariate-dependent-censoring settings, with 5 repetitions per setting. It also runs the no-covariate AJ checks, result summary, and theoretical table. Outputs are written to `/tmp/joint_softcomp/full/censoring/`, with the summary in `summary.txt` and the theoretical table in `population_targets.txt`. See the [script](scripts/run_joint_softcomp_censoring.sh) for the complete loops.

### 5.5 Results

**Theoretical table**: Population minimizers for cause 1. The true F₁ is 0.352 at t = 5 and 0.633 at t = 20.

| ρ | t | Eq. (5) | Eq. (5) + augmentation | Brier term | JointSoftComp |
|---|---|---|---|---|---|
| 0 | 5 / 20 | 0.667 / 0.667 | 0.386 / 0.500 | 0.352 / 0.633 | 0.352 / 0.633 |
| 20% | 5 / 20 | 0.530 / 0.530 | 0.327 / 0.419 | 0.324 / 0.518 | 0.352 / 0.633 |
| 50% | 5 / 20 | 0.333 / 0.333 | 0.230 / 0.285 | 0.259 / 0.332 | 0.352 / 0.633 |
| 80% | 5 / 20 | 0.133 / 0.133 | 0.109 / 0.125 | 0.130 / 0.133 | 0.352 / 0.633 |

**Local empirical results**: 5 repetitions per setting, with means and sample standard deviations. MSE values are multiplied by 1,000.

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

**No-covariate check**: One run per setting, with 10,000 training subjects. Entries show the maximum absolute difference from the true marginal CIF on `[0,20]`. The nonparametric AJ estimate is also included in this comparison.

| ρ | SoftComp | JointSoftComp | Nonparametric AJ reference |
|---|---|---|---|
| 0% | 0.1022 | 0.0117 | **0.0069** |
| 20% | 0.1148 | 0.0081 | **0.0057** |
| 50% | 0.1507 | 0.0206 | **0.0092** |
| 80% | 0.1878 | 0.0748 | **0.0359** |

### 5.6 Analysis

- **JointSoftComp has lower MSE and IBS in all 11 settings.** In Case III with independent censoring, its MSE at ρ=0/20%/50%/80% is 1.01 / 1.20 / 1.73 / 6.07, compared with 5.97 / 7.00 / 11.05 / 25.83 for SoftComp.
- **JointSoftComp does not always lead in C_td.** SoftComp has higher means for constant hazards at ρ=20%/50%/80%. With covariate-dependent censoring at ρ=80%, JointSoftComp scores 0.6681, compared with 0.5938 for SoftComp.
- **Heavy censoring still increases error.** With constant hazards and ρ=80%, JointSoftComp's MSE is 10.03, compared with 48.24 for SoftComp. Its three cause-specific biases at t=20 are approximately -0.131 / -0.047 / -0.105, so it should not be described as approximately unbiased at every censoring level.
- **Nonparametric AJ has the lowest error in the no-covariate check.** JointSoftComp's maximum absolute difference is lower than SoftComp's at all four censoring levels but higher than nonparametric AJ's. At ρ=80%, the values are 0.0748, 0.1878, and 0.0359, respectively.

All raw outputs, summaries, the environment lock, and audit records are stored in
[`local_validation/readme_full/`](local_validation/readme_full/).
Original server results and ablation records are retained in the [historical archive](docs/historical_results.md).
