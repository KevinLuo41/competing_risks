# JointSoftComp: 模型、loss 与补充实验

本文档说明 JointSoftComp 的模型和 loss、实现它的代码，以及三个补充实验（用到的代码、运行方式、结果和分析）。文中路径都相对于本地项目根目录 `competing_risks/`。结果表保留原项目的实验记录，不代表本地已完成相同次数的运行；内部报告链接可能需要原环境的访问权限。

## 1. 模型和 loss

完整的公式和推导见 https://www.internalfb.com/intern/px/p/dSK3v/ 。

- **思路**: 按 Munch & Gerds (2026) 的 joint survival super learner，把删失当作一个独立状态。这样每个样本在任何时刻 t 的观测状态 η(t) ∈ {在险, 原因 1..K, 删失} 都已知。
- **模型**: 沿用 SoftComp 的残差前馈网络（输入 [x; t]），输出 K + 1 个 logit，补上固定为 0 的在险 logit 后做 (K + 2) 类 softmax，得到观测状态概率。
- **Loss**: 每个 minibatch 里，对每个样本从训练集事件时间的经验分布中抽 M = 4 个时刻，最小化这些时刻上观测状态的交叉熵。所有标签都可观测，loss 严格 proper，不需要估计删失分布；没有 time augmentation、辅助 Brier 项和后处理。
- **预测**: 用 Aalen–Johansen 递推 ΔΛ_k = [ΔP_k]₊ / P_0(t-) 由观测状态概率还原 CIF；得到的 CIF 单调，且 S + Σ_k F_k = 1。
- **默认配置**: 宽 32，1 个残差块，每人 4 个时刻，学习率 1e-3、weight decay 1e-3、batch 256、1000 epochs；六个数据集共用。
- **对照：论文版 SoftComp**: (K + 1) 类 softmax 直接输出 (S, F_1, …, F_K)。Loss 为式 (5)（观测时刻上的交叉熵）加 time augmentation（每人从 Unif(0, Y_i) 抽 M 个时刻标为存活，权重 0.5；论文默认 M = 2），预测后做 PAV 和 simplex 后处理。它的总体目标依赖删失分布和 M，不是 CIF。

### 1.1 代码

| 文件 | 类 / 函数 | 作用 |
|---|---|---|
| `crsoft_model/joint_softcomp.py` | `JointSoftComp` | JointSoftComp 模型：训练与预测 CIF |
| | `observed_state_labels` | 由 (Y, Δ) 构造观测状态标签 η(t) |
| | `aalen_johansen_from_observed` | 由观测状态概率经 AJ 递推得到 CIF 和生存函数 |
| `crsoft_model/crsoft.py`（原有） | `CRSoftNet` | 论文版 SoftComp；也是 `JointSoftComp` 的父类（网络结构和训练循环） |
| `evaluation/postprocess.py`（原有） | `isotonic_project_cif`、`enforce_cif_simplex` | 论文版 SoftComp 的 PAV 和 simplex 后处理 |
| `evaluation/survival.py`（原有） | `evaluate_cif_metrics` | C_td (Antolini) 和 IPCW IBS |
| `evaluation/simulation.py`（原有） | `compute_mse_accuracy` | 对真实 CIF 的 MSE |

先完成第 2 节的本地环境设置，再运行以下示例（`p`、`K` 和数据张量由调用者提供）：

```python
from competing_risks.crsoft_model.joint_softcomp import JointSoftComp

model = JointSoftComp(input_dim=p, num_causes=K)  # 默认宽 32、1 层
model.fit(X, Y, Delta)  # 默认配置训练
cif, survival = model.predict_cif_survival_grid(X_test, times)
```

单元测试（`tests/test_joint_softcomp.py`：标签构造；固定 τ = 20、ρ = 0/20%/50%/80% 时 AJ 还原真 CIF；预测单调且和为 1；训练时刻来自事件时间；训练 loss 有限）：

```bash
python -m unittest competing_risks.tests.test_joint_softcomp
```

## 2. 本地运行环境

在 `competing_risks/` 根目录运行，后续命令也使用同一个 shell：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy pandas scipy scikit-learn matplotlib lifelines
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"

simple() {
  python -c 'from competing_risks.experiments.joint_softcomp.simple import main; main()' "$@"
}
uncensored() {
  python -c 'from competing_risks.experiments.joint_softcomp.uncensored import main; main()' "$@"
}
censoring() {
  python -c 'from competing_risks.experiments.joint_softcomp.censoring_levels import main; main()' "$@"
}

# 查看参数，不训练
simple --help
uncensored --help
censoring --help
```

这三个文件定义了 `main()`，目前没有 `if __name__ == "__main__"` 入口，
所以上面的 shell 函数显式调用 `main()` 并传入参数。其余汇总脚本可以直接用
`python -m` 运行。模型和训练逻辑沿用现有源码。

每个实验用 `REPS` 控制重复次数。以下循环按顺序运行，适合本地直接执行；
先调小 `REPS` 检查流程，再运行完整重复次数。原文的耗时来自 72 核 devserver，
不作为本机耗时估计。

## 3. 实验一：最简单设定（单预测变量、无删失）

### 3.1 实验设置和方法

- **数据**: x ∼ N(0, 1)；单一事件，hazard 为 0.05·exp(βx)，β ∈ {0, log 1.5, log 2}；无删失。训练 n = 200，重复 200 次；测试集为 5 万人的无删失数据（固定种子）。
- **指标**: 10 年的 Brier score、AUC(10)（cases 为 T ≤ 10，controls 为 T > 10）、IPA = 1 − Brier / Brier(无协变量模型)、预测与真实 10 年风险的 MSE。
- **方法**:
  - SoftComp（论文版 loss 与后处理），增广时刻数 M = 0、1、2（论文默认）、4、8，权重 0.5；
  - JointSoftComp，每人 M = 1、2、4（默认）、8 个时刻；
  - Cox（单协变量，Breslow 基线）、真实模型、无协变量模型（训练集的边际风险）；
  - SoftComp loss 的总体极小值，即训练数据无穷多时网络收敛到的函数：π(10 | x) = 1 / (1 + 0.5·M·e^z·E₁(z))，z = 10·0.05·e^{βx}，E₁ 为指数积分。直接数值计算，不需要训练。

### 3.2 用到的代码

| 角色 | 文件 | 函数 |
|---|---|---|
| 生成数据 | `experiments/joint_softcomp/simple.py` | `simulate`（生成 x 和 T）、`true_risk`（真实 10 年风险） |
| 模型 | `simple.py` | `predict_softcomp`（`CRSoftNet` 加后处理）、`predict_joint`（`JointSoftComp`）、`predict_cox`、`softcomp_limit` 与 `scaled_exp1`（总体极小值） |
| 评估 | `simple.py` | `interpolate`（x 网格上的预测插值到测试集）、`auc`、`evaluate`、`summarize` |
| 运行入口 | `simple.py` 的 `main` | 一次运行一个 (β, 方法, M)，输出 `simple_b{β}_{方法}_m{M}_r{起始重复}.json`；`reference` 输出 Cox、真实模型、无协变量模型和各 M 的总体极小值 |
| 输出结果 | `experiments/joint_softcomp/analyze_simple.py` | 合并各段重复，按 β 输出 Brier、AUC、IPA、MSE 的表 |

### 3.3 运行代码

```bash
OUT=/tmp/joint_softcomp/simple
REPS=200  # 首次检查可改成 1 或 2
for b in 0 0.4054651081 0.6931471806; do
  simple reference --beta "$b" --m 0 --reps "$REPS" --out-dir "$OUT"
  for m in 0 1 2 4 8; do
    simple softcomp --beta "$b" --m "$m" --reps "$REPS" --out-dir "$OUT"
  done
  for m in 1 2 4 8; do
    simple joint --beta "$b" --m "$m" --reps "$REPS" --out-dir "$OUT"
  done
done
python -m competing_risks.experiments.joint_softcomp.analyze_simple "$OUT"
```

`--rep-start` 可以把重复拆成几段并行跑（例如每段 `--reps 50`）；汇总脚本会自动合并。

### 3.4 实验结果

10 年 Brier score（200 次重复的均值）：

| 方法 | β = 0 | β = log 1.5 | β = log 2 |
|---|---|---|---|
| SoftComp, M = 0（无 augmentation） | 0.603 | 0.592 | 0.575 |
| SoftComp, M = 1 | 0.318 | 0.308 | 0.289 |
| SoftComp, M = 2（论文默认） | 0.256 | 0.246 | 0.229 |
| SoftComp, M = 4 | 0.242 | 0.235 | 0.222 |
| SoftComp, M = 8 | 0.275 | 0.270 | 0.262 |
| JointSoftComp | 0.243 | 0.231 | 0.208 |
| Cox | 0.241 | 0.229 | 0.206 |
| 真实模型 | 0.239 | 0.227 | 0.205 |
| 无协变量 | 0.241 | 0.243 | 0.246 |

SoftComp 的总体极小值（训练数据无穷多时）的 Brier，M = 0 / 1 / 2 / 4 / 8：

| β | 0 | log 1.5 | log 2 |
|---|---|---|---|
| Brier | 0.603 / 0.322 / 0.255 / 0.241 / 0.273 | 0.592 / 0.308 / 0.244 / 0.234 / 0.270 | 0.575 / 0.286 / 0.226 / 0.220 / 0.261 |

JointSoftComp 在 M = 1、2、4、8 时的 Brier 相差不超过 0.0003。

AUC(10)：β = 0 时所有方法都约为 0.500；β = log 1.5 时真实模型为 0.640，Cox、JointSoftComp 和 M ≥ 1 的 SoftComp 为 0.638–0.640，SoftComp M = 0 为 0.509；β = log 2 时除 SoftComp M = 0 (0.524) 外都是 0.729。

### 3.5 分析

- **SoftComp 的结果完全由 M 决定。** 没有 augmentation 时（M = 0），式 (5) 在无删失数据上只见到事件标签，对所有人预测风险为 1，Brier 高达 0.58–0.60。M 增大时 Brier 先降后升，因为总体极小值随 M 移动，在 M = 2 和 M = 4 之间越过真实风险（例如 x = 0 时，M = 0/1/2/4/8 对应的 10 年风险为 1.00/0.68/0.52/0.35/0.21，真值为 0.39）。
- **augmentation 不降方差，也不是纠偏。** 每个 M 下网络拟合的结果都几乎等于总体极小值（例如 β = log 2、M = 2 时为 0.229 对 0.226），所以 M 改变的是 loss 收敛到的位置，偏差随 M 时大时小；Brier 高于 0.25 意味着比“所有人预测 50%”还差，M = 0、1、8 在三个 β 下都是如此，β = 0 时 M = 2 也是。
- **JointSoftComp 不用 time augmentation。** 它的 M 是积分 loss 的 Monte Carlo 点，只影响梯度噪声，结果不随 M 变化。它在三个 β 下都优于任何 M 的 SoftComp、与 Cox、真实模型接近；β = 0 时与无协变量模型相当（0.243 对 0.241）。
- **AUC 区分不出方法。** 只有一个预测变量时，所有对 x 单调的预测都有相同的 AUC(10)；augmentation 在 AUC 上的“提升”只是把 M = 0 的退化值 0.5 恢复到正常水平，方法之间的差别体现在 Brier（校准）上。

## 4. 实验二：有删失 vs. 无删失（Case II/III）

图文版见 https://www.internalfb.com/intern/px/p/dVrW5/ 中的测试一（包含全部 8 个方法）。

### 4.1 实验设置和方法

- **数据**: 论文 Case II (K = p = 3) 和 Case III (K = 3, p = 4)；正式种子的前 10 次重复（训练种子 130000 + r / 330000 + r，测试种子 140000 + r / 340000 + r）；训练 5000（4500 拟合、500 验证），测试 1000。
- **两种条件**: 有删失为论文设置（指数删失，校准为 P(C ≤ median T) = 0.5）；无删失版本保留同一批对象的协变量、真实事件时间和原因，只去掉训练集和测试集里的删失。两种条件使用同一个评估网格（有删失测试集事件时间分位数至 97.5%，外加论文的固定时刻）。
- **方法**: JointSoftComp（默认配置）；SoftComp（论文配置，M = 2）；SoftComp-noaug（同一配置但 M = 0）；NeuralFG（论文配置，Case II/III 中最好的基线）。
- **指标**: 对真实 CIF 的 MSE (×10⁻³)、C_td、IBS。两种条件下测试集不同（有删失时 C_td 和 IBS 用 IPCW），所以 C_td 和 IBS 只在同一条件内比较；MSE 都对真实 CIF、在同一网格上计算，可以跨条件比较。

### 4.2 用到的代码

| 角色 | 文件 | 函数 |
|---|---|---|
| 生成数据 | `experiments/case2_v4/run.py`、`experiments/case3_v5/run.py`（原有） | `prepare_data`（新增 `censor_rate` 参数，默认 0.5） |
| | `data/case2_v4.py`、`data/case3_v5.py`（原有） | `generate_parameters`、`generate_data`、`compute_cif`（真实 CIF） |
| | `data/utils.py` | `solve_inverse_cdf`、`assign_causes_and_censor`、`generate_test_observations`（`censor_rate = 0` 时不删失） |
| | `evaluation/survival.py`（原有） | `build_evaluation_time_grid` |
| | `experiments/joint_softcomp/uncensored.py` | `prepare`（无删失时生成同一批对象，并沿用有删失版本的评估网格） |
| 模型 | `uncensored.py` | `build_method`：JointSoftComp 直接构建；SoftComp、SoftComp-noaug 和 NeuralFG 取自 `run.py` 的 `build_specs` |
| | `baseline_models/neural_fine_gray.py`（原有） | `NeuralFG` |
| 评估 | `evaluation/simulation.py`、`evaluation/survival.py`（原有） | `compute_mse_accuracy`、`compute_dist`、`evaluate_cif_metrics` |
| 运行入口 | `uncensored.py` 的 `run`、`main` | 一次运行一个 (case, 方法, 重复, 条件)，输出 `case{c}_rep{r}_{方法}.json` |
| 输出结果 | `experiments/joint_softcomp/analyze_uncensored.py` | 配对两种条件下的同一批重复，输出 MSE、C_td、IBS 的表 |

### 4.3 运行代码

```bash
OUT=/tmp/joint_softcomp/uncensored
REPS=10  # 首次检查可改成 1 或 2
for r in $(seq 0 $((REPS - 1))); do
  for c in 2 3; do
    for m in JointSoftComp SoftComp SoftComp-noaug NeuralFG; do
      uncensored --case "$c" --method "$m" --replicate "$r" --out-dir "$OUT/censored"
      uncensored --case "$c" --method "$m" --replicate "$r" --out-dir "$OUT/uncensored" --uncensored
    done
  done
done
python -m competing_risks.experiments.joint_softcomp.analyze_uncensored "$OUT/censored" "$OUT/uncensored" "$REPS"
```

### 4.4 实验结果

10 次重复的均值；MSE 单位为 10⁻³。

| Case | 方法 | 有删失 MSE | C_td | IBS | 无删失 MSE | C_td | IBS |
|---|---|---|---|---|---|---|---|
| II | **JointSoftComp** | **2.10** | **0.750** | **0.052** | **1.65** | **0.740** | **0.052** |
| II | SoftComp | 8.20 | **0.750** | 0.056 | 15.10 | 0.739 | 0.061 |
| II | SoftComp-noaug | 21.48 | 0.747 | 0.068 | 65.22 | 0.721 | 0.147 |
| II | NeuralFG | 4.89 | 0.729 | 0.054 | 2.98 | 0.732 | **0.052** |
| III | **JointSoftComp** | **1.13** | 0.681 | **0.123** | **0.82** | **0.678** | **0.121** |
| III | SoftComp | 3.20 | **0.687** | 0.127 | 4.30 | **0.678** | 0.123 |
| III | SoftComp-noaug | 9.22 | 0.682 | 0.129 | 61.07 | 0.669 | 0.162 |
| III | NeuralFG | 3.36 | 0.649 | 0.127 | 2.17 | 0.661 | 0.123 |

### 4.5 分析

- **JointSoftComp 的优势不依赖删失。** 两种条件下它的 MSE 和 IBS 都是第一，C_td 第一或与第一持平（唯一例外是 Case III 有删失时 SoftComp 高 0.006）。有删失时，它的 MSE 约为 SoftComp 的 1/4（Case II）和 1/3（Case III）。
- **SoftComp 的表现受删失左右。** 去掉删失后它的 MSE 反而变差（8.20 → 15.10、3.20 → 4.30），不带 augmentation 时更明显；而 JointSoftComp 和 NeuralFG 都因信息更完整而变好。
- **time augmentation 在无删失时作用更大，但这暴露的是式 (5) 的缺陷。** 没有删失时式 (5) 从不出现“存活”标签，总体极小值是 hazard 占比 λ_k(t) / λ(t)，S ≡ 0；augmentation 是唯一的“存活”信息来源（去掉后 MSE 为 65.2 / 61.1），它是在给一个不以 CIF 为目标的 loss 打补丁。

## 5. 实验三：固定时间范围的删失水平

图文版见 https://www.internalfb.com/intern/px/p/dVrW5/ 中的测试二。

### 5.1 实验设置和方法

- **删失定义**: 固定 τ = 20。删失时间服从指数分布，速率校准为在 τ 之前被删失的比例为 ρ，即 P(C < min(T, τ)) = ρ；随访到 τ 截止，Y = min(T, C, τ)。ρ ∈ {0, 20%, 50%, 80%}；ρ = 0 时 [0, τ] 上的数据完整。训练集实测的 τ 前删失比例：Case III 为 0.000 / 0.200 / 0.503 / 0.801，常数 hazard 为 0.000 / 0.198 / 0.501 / 0.800。
- **四组检验**:
  - `aj-check`：Case III 数据（n = 10000），模型只输入常数（即估计边际 CIF），与非参数 AJ 估计和真实边际 CIF 比较；
  - `case3`：Case III DGP，独立删失；
  - `depcens`：Case III DGP，协变量依赖删失（速率乘 e^{0.8 x₁}；ρ = 0 时与 `case3` 相同，不重复运行）；
  - `constant`：常数 cause-specific hazard，K = 3、p = 4，λ_k(x) = exp(a_k + b_kᵀx)，基线速率 0.05 / 0.035 / 0.025，独立删失。
- **样本与重复**: 训练 5000、测试 1000，数据种子 30000 + s / 40000 + s，s = 0, …, 4。测试集不删失，指标在 (0, τ] 上 100 个等距点计算，所以 C_td 和 IBS 不需要 IPCW，不同 ρ 可以直接比较。
- **方法**: 论文版 SoftComp（`softcomp`，M = 2、权重 0.5、weight decay 3e-3、含后处理）vs. JointSoftComp（`joint`，默认配置）。
- **指标**: 对真实 CIF 的 MSE (×10⁻³)、C_td、IBS，以及 t = 20 处每类事件的平均偏差 F̂_k − F_k。
- **理论表**: 常数 hazard 例子（λ₁ = 0.10、λ₂ = 0.05，看原因 1；0.5·M = 1；τ = 20）下各训练项的总体极小值，由数值积分得到。

### 5.2 用到的代码

| 角色 | 文件 | 函数 |
|---|---|---|
| 生成数据 | `experiments/joint_softcomp/censoring_levels.py` | `simulate`、`_event_times`、`_censoring_times`（按 ρ 校准删失速率）、`_constant_hazards`、`constant_cif` |
| | `data/case3_v5.py`、`data/utils.py`（原有） | `generate_parameters`、`compute_cif`、`solve_inverse_cdf` |
| 模型 | `censoring_levels.py` | `train`（`CRSoftNet` 或 `JointSoftComp`）、`predict`（SoftComp 做后处理） |
| 评估 | `censoring_levels.py` | `run_simulation`、`_horizon_bias`、`aalen_johansen`（非参数 AJ 估计）、`run_aj_check` |
| | `evaluation/simulation.py`、`evaluation/survival.py`（原有） | `compute_mse_accuracy`、`evaluate_cif_metrics` |
| 运行入口 | `censoring_levels.py` 的 `main` | 一次运行一个 (检验, 方法, ρ, 种子)，输出 `{检验}_rho{ρ}_{方法}_seed{s}.json` |
| 输出结果 | `censoring_levels.py` 的 `summarize` 子命令 | 按 (检验, ρ, 方法) 汇总的表 |
| 理论表 | `experiments/joint_softcomp/population_targets.py`（只用标准库） | `censoring_rate`、`loss_targets`、`aalen_johansen_from_observed`、`print_table` |

### 5.3 运行代码

```bash
OUT=/tmp/joint_softcomp/censoring
REPS=5  # 首次检查可改成 1 或 2
for rho in 0 0.2 0.5 0.8; do
  for m in softcomp joint; do
    censoring aj-check --method "$m" --rho "$rho" --seed 0 --n-train 10000 --threads 2 --out-dir "$OUT"
    for s in $(seq 0 $((REPS - 1))); do
      censoring case3 --method "$m" --rho "$rho" --seed "$s" --threads 2 --out-dir "$OUT"
      censoring constant --method "$m" --rho "$rho" --seed "$s" --threads 2 --out-dir "$OUT"
      if [ "$rho" != 0 ]; then
        censoring depcens --method "$m" --rho "$rho" --seed "$s" --threads 2 --out-dir "$OUT"
      fi
    done
  done
done
censoring summarize --out-dir "$OUT"
python -m competing_risks.experiments.joint_softcomp.population_targets
```

### 5.4 实验结果

**理论表**：原因 1 的总体极小值（真实 F₁：t = 5 时 0.352，t = 20 时 0.633）。

| ρ | t | 式 (5) | 式 (5) + augmentation | Brier 项 | JointSoftComp |
|---|---|---|---|---|---|
| 0 | 5 / 20 | 0.667 / 0.667 | 0.386 / 0.500 | 0.352 / 0.633 | 0.352 / 0.633 |
| 20% | 5 / 20 | 0.530 / 0.530 | 0.327 / 0.419 | 0.324 / 0.518 | 0.352 / 0.633 |
| 50% | 5 / 20 | 0.333 / 0.333 | 0.230 / 0.285 | 0.259 / 0.332 | 0.352 / 0.633 |
| 80% | 5 / 20 | 0.133 / 0.133 | 0.109 / 0.125 | 0.130 / 0.133 | 0.352 / 0.633 |

**实证结果**（5 次重复的均值）：

| 检验 | ρ | 方法 | MSE | C_td | IBS | t = 20 的偏差（原因 1 / 2 / 3） |
|---|---|---|---|---|---|---|
| case3 | 0 | SoftComp | 5.97 | 0.683 | 0.131 | -0.095 / -0.097 / -0.097 |
| | | JointSoftComp | 1.04 | 0.684 | 0.126 | 0.000 / 0.000 / -0.001 |
| | 20% | SoftComp | 7.03 | 0.683 | 0.132 | -0.113 / -0.118 / -0.115 |
| | | JointSoftComp | 1.22 | 0.685 | 0.126 | 0.000 / 0.003 / 0.005 |
| | 50% | SoftComp | 11.03 | 0.686 | 0.136 | -0.141 / -0.148 / -0.131 |
| | | JointSoftComp | 1.76 | 0.682 | 0.127 | -0.007 / -0.010 / 0.006 |
| | 80% | SoftComp | 25.81 | 0.674 | 0.151 | -0.170 / -0.190 / -0.206 |
| | | JointSoftComp | 6.06 | 0.676 | 0.131 | -0.058 / -0.025 / -0.047 |
| depcens | 20% | SoftComp | 7.45 | 0.675 | 0.132 | -0.117 / -0.119 / -0.115 |
| | | JointSoftComp | 1.33 | 0.685 | 0.126 | -0.003 / 0.000 / 0.003 |
| | 50% | SoftComp | 13.17 | 0.654 | 0.138 | -0.148 / -0.157 / -0.148 |
| | | JointSoftComp | 2.30 | 0.681 | 0.127 | 0.000 / -0.010 / 0.000 |
| | 80% | SoftComp | 27.83 | 0.594 | 0.153 | -0.195 / -0.183 / -0.204 |
| | | JointSoftComp | 8.41 | 0.667 | 0.133 | 0.007 / 0.005 / -0.030 |
| constant | 0 | SoftComp | 7.23 | 0.665 | 0.149 | -0.147 / -0.107 / -0.089 |
| | | JointSoftComp | 0.62 | 0.668 | 0.143 | 0.001 / 0.000 / -0.002 |
| | 20% | SoftComp | 11.34 | 0.667 | 0.154 | -0.186 / -0.131 / -0.112 |
| | | JointSoftComp | 0.68 | 0.666 | 0.143 | 0.002 / 0.002 / -0.004 |
| | 50% | SoftComp | 20.94 | 0.669 | 0.163 | -0.236 / -0.166 / -0.149 |
| | | JointSoftComp | 1.18 | 0.666 | 0.144 | -0.001 / 0.016 / -0.035 |
| | 80% | SoftComp | 48.29 | 0.667 | 0.190 | -0.330 / -0.239 / -0.200 |
| | | JointSoftComp | 9.86 | 0.659 | 0.153 | -0.131 / -0.046 / -0.103 |

**无协变量检验**：[0, 20] 上与真实边际 CIF 的最大绝对差。

| ρ | SoftComp | JointSoftComp | 非参数 AJ 估计 |
|---|---|---|---|
| 0 | 0.102 | 0.012 | 0.007 |
| 20% | 0.114 | 0.008 | 0.006 |
| 50% | 0.150 | 0.021 | 0.009 |
| 80% | 0.188 | 0.084 | 0.036 |

### 5.5 分析

- **SoftComp 在 ρ = 0 时就有偏，偏差随 ρ 系统性变大。** t = 20 处 Case III 的平均偏差从 -0.10 变到 -0.19，常数 hazard 原因 1 从 -0.15 变到 -0.33。这与理论表一致：式 (5) 和 time augmentation 的目标即使在无删失时也不是 CIF，删失越重偏得越多。
- **JointSoftComp 在 ρ ≤ 50% 时近似无偏。** 三类事件的偏差基本在 0.01 以内（例外是常数 hazard、ρ = 50% 时的原因 2 和 3：0.016 和 -0.035）；MSE 只随删失加重而温和上升（Case III：1.04 → 1.76），IBS 在所有设定下都更好。理论表中它在所有 ρ 下都等于真值。
- **C_td 两者接近。** C_td 只看排序，对整体的系统性偏移不敏感，所以 SoftComp 的偏差主要体现在 MSE 和 IBS 上。例外是协变量依赖删失、ρ = 80% 时：不同的人被扭曲的程度不同，排序被打乱，SoftComp 降到 0.594，JointSoftComp 为 0.667。
- **ρ = 80% 时误差主要来自信息不足。** 此时训练集中 τ 时仍在观测的人只有 0%–0.3%，τ 附近几乎没有数据；JointSoftComp 的误差也变大，但 MSE 仍比 SoftComp 小 3–5 倍。非参数 AJ 估计本身的最大误差也从 0.007 升到 0.036，说明这是可识别性的极限，而不是 loss 的偏差。
