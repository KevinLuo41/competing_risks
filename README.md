# JointSoftComp: 模型、loss 与补充实验

本文档说明 JointSoftComp 的模型和 loss、实现它的代码，以及三个补充实验（用到的代码、运行方式、结果和分析）。文中路径都相对于本地项目根目录 `competing_risks/`。结果表已替换为 2026-10-03 本地 CPU 完整运行的数据：实验一每组 200 次、实验二每组 10 次、实验三每组 5 次。

本次共有 878 个成功任务，原始 JSON、运行命令和审计位于
[`local_validation/readme_full/`](local_validation/readme_full/)。
下表均值来自原始结果，括号内为样本标准差。**粗体表示同一设定、同一指标中
未四舍五入均值的最佳值**；MSE/Brier/IBS 越小越好，C_td/AUC/IPA 越大越好。
显示值接近不表示并列或差异显著。真实模型及总体极小值是参照，不参与拟合模型排名；
有符号偏差只作为诊断，不排名。

## 1. 模型和 loss

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
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

这三个实验文件定义了 `main()`，目前没有 `if __name__ == "__main__"` 入口，
所以上面的 shell 函数显式调用 `main()` 并传入参数。其余汇总脚本可以直接用
`python -m` 运行。模型和训练逻辑沿用现有源码。

每个实验用 `REPS` 控制重复次数。以下循环按顺序运行，适合本地直接执行；
先调小 `REPS` 检查流程，再运行完整重复次数。本次任务使用 CPU、每任务一个
Torch 线程；本地环境版本见[完整依赖锁](local_validation/readme_full/requirements.lock.txt)。

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
  simple reference --beta "$b" --m 0 --reps "$REPS" --threads 1 --out-dir "$OUT"
  for m in 0 1 2 4 8; do
    simple softcomp --beta "$b" --m "$m" --reps "$REPS" --threads 1 --out-dir "$OUT"
  done
  for m in 1 2 4 8; do
    simple joint --beta "$b" --m "$m" --reps "$REPS" --threads 1 --out-dir "$OUT"
  done
done
python -m competing_risks.experiments.joint_softcomp.analyze_simple "$OUT"
```

`--rep-start` 可以把重复拆成几段并行跑（例如每段 `--reps 50`）；汇总脚本会自动合并。

### 3.4 实验结果

以下三张表按 β 分组，展示所有拟合模型配置及真实模型参照。
每个拟合配置均有 200 次重复，测试集固定为 50,000 人。

**β = 0**（每个拟合方法 200 次重复，均值及样本标准差）：

| 方法 | Brier(10) ↓ | AUC(10) ↑ | IPA ↑ | MSE vs truth ↓ |
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
| 真实模型（oracle 参照） | 0.2393 | 0.5000 | -0.0000 | 0.0000 |

**β = log 1.5**（每个拟合方法 200 次重复，均值及样本标准差）：

| 方法 | Brier(10) ↓ | AUC(10) ↑ | IPA ↑ | MSE vs truth ↓ |
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
| 真实模型（oracle 参照） | 0.2270 | 0.6398 | 0.0603 | 0.0000 |

**β = log 2**（每个拟合方法 200 次重复，均值及样本标准差）：

| 方法 | Brier(10) ↓ | AUC(10) ↑ | IPA ↑ | MSE vs truth ↓ |
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
| 真实模型（oracle 参照） | 0.2050 | 0.7291 | 0.1614 | 0.0000 |

SoftComp loss 总体极小值的 Brier（数值计算，不是训练结果）：

| M | β=0 | β=log 1.5 | β=log 2 |
|---|---|---|---|
| 0 | 0.6034 | 0.5916 | 0.5749 |
| 1 | 0.3220 | 0.3083 | 0.2860 |
| 2 | 0.2545 | 0.2440 | 0.2257 |
| 4 | 0.2414 | 0.2343 | 0.2202 |
| 8 | 0.2730 | 0.2699 | 0.2609 |

### 3.5 分析

- **Brier 最佳方法随设定变化。** β=0 时无协变量模型最好（0.2406）；β 非零时 Cox 最好（0.2288 / 0.2064）。JointSoftComp 默认 M=4 为 0.2432 / 0.2306 / 0.2082；β=0 时 SoftComp M=4 的 0.2419 更低，不能说 JointSoftComp 在所有设定都优于 SoftComp。
- **SoftComp 对 augmentation 数量敏感。** M=0 时 Brier 为 0.6034 / 0.5916 / 0.5749；M=4 时降至 0.2419 / 0.2349 / 0.2219，再增加到 M=8 会回升。JointSoftComp 的 M=1/2/4/8 是 loss 的采样时刻数，Brier 在每个 β 下相差不到 0.0003。
- **排序与校准应分开看。** β=log 2 时许多方法的 AUC 都接近 0.7291，但 Brier 和对真值的 MSE 明显不同。最佳值标记只比较均值，不代表统计显著性。

## 4. 实验二：有删失 vs. 无删失（Case II/III）

本地此表只比较下面实际运行的四种方法。

### 4.1 实验设置和方法

- **数据**: Case II v4 (K = p = 3) 和 Case III v5 (K = 3, p = 4)；正式种子的前 10 次重复（训练种子 130000 + r / 330000 + r，测试种子 140000 + r / 340000 + r）；训练 5000（4500 拟合、500 验证），测试 1000。
- **两种条件**: 有删失为论文设置（指数删失，校准为 P(C ≤ median T) = 0.5）；无删失版本保留同一批对象的协变量、真实事件时间和原因，只去掉训练集和测试集里的删失。两种条件使用同一个评估网格：有删失测试集事件时间分位数至 97.5%；Case III 加入源码定义的四个固定时刻。Case II 源码未提供额外固定时刻，因此没有添加。Case II 原正式 runner 使用 90% 网格，本次补充实验使用 97.5%。
- **方法**: JointSoftComp（默认配置）；SoftComp（论文配置，M = 2）；SoftComp-noaug（同一配置但 M = 0）；NeuralFG（论文配置，Case II/III 中最好的基线）。
- **指标**: 对真实 CIF 的 MSE (×10⁻³)、C_td、IBS。两种条件下测试集不同（C_td 使用源码的 Antolini 可比对统计量，只有 IBS 使用删失 KM 的 IPCW），所以 C_td 和 IBS 只在同一条件内比较；MSE 都对真实 CIF、在同一网格上计算，可以跨条件比较。

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
      uncensored --case "$c" --method "$m" --replicate "$r" --threads 1 --out-dir "$OUT/censored"
      uncensored --case "$c" --method "$m" --replicate "$r" --threads 1 --out-dir "$OUT/uncensored" --uncensored
    done
  done
done
python -m competing_risks.experiments.joint_softcomp.analyze_uncensored "$OUT/censored" "$OUT/uncensored" "$REPS"
```

### 4.4 实验结果

每组 10 次配对重复，均值（样本标准差）。**MSE 数值乘以 1,000**。
粗体分别在同一个 Case 和删失条件内选择最佳指标。

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

### 4.5 分析

- **JointSoftComp 在四个 Case/条件中都取得最低 MSE 和 IBS。** 有删失时它的 MSE 为 2.28 / 1.12，而 SoftComp 为 8.12 / 3.20。
- **C_td 的最佳方法因 Case 而异。** Case II 两种条件都由 JointSoftComp 取得最高均值；Case III 两种条件都由 SoftComp 取得最高均值。无删失 Case III 原先三位小数都显示 0.678，四位小数可看出 0.6782 对 0.6776，不能标成并列第一。
- **去掉删失后，SoftComp 的 MSE 增大。** Case II 为 8.12 → 14.29，Case III 为 3.20 → 4.31；不带 augmentation 时为 20.33 → 59.49、9.22 → 61.14。JointSoftComp 和 NeuralFG 的 MSE 则降低。

本次核对了全部 20 个 Case/重复的配对数据与网格。原运行环境依赖版本未知，
Case II 固定时刻也未提供，故这些数字是本地测量值，不声称与原服务器完全一致。
详细约定和修复记录见[运行协议](local_validation/readme_full/protocol.txt)。

## 5. 实验三：固定时间范围的删失水平

下面的实证数字来自本地运行。

### 5.1 实验设置和方法

- **删失定义**: 固定 τ = 20。删失时间服从指数分布，速率校准为在 τ 之前被删失的比例为 ρ，即 P(C < min(T, τ)) = ρ；随访到 τ 截止，Y = min(T, C, τ)。ρ ∈ {0, 20%, 50%, 80%}；ρ = 0 时 [0, τ] 上的数据完整。实际删失比例保存在每个原始结果的 `censored_before_tau_train` 字段中。
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
    censoring aj-check --method "$m" --rho "$rho" --seed 0 --n-train 10000 --threads 1 --out-dir "$OUT"
    for s in $(seq 0 $((REPS - 1))); do
      censoring case3 --method "$m" --rho "$rho" --seed "$s" --threads 1 --out-dir "$OUT"
      censoring constant --method "$m" --rho "$rho" --seed "$s" --threads 1 --out-dir "$OUT"
      if [ "$rho" != 0 ]; then
        censoring depcens --method "$m" --rho "$rho" --seed "$s" --threads 1 --out-dir "$OUT"
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

**本地实证结果**（每组 5 次重复，均值及样本标准差；MSE 数值乘以 1,000）：

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

**无协变量检验**：每组一次运行，10,000 个训练对象；显示 `[0,20]` 上
与真实边际 CIF 的最大绝对差。非参数 AJ 也参与此表比较。

| ρ | SoftComp | JointSoftComp | Nonparametric AJ reference |
|---|---|---|---|
| 0% | 0.1022 | 0.0117 | **0.0069** |
| 20% | 0.1148 | 0.0081 | **0.0057** |
| 50% | 0.1507 | 0.0206 | **0.0092** |
| 80% | 0.1878 | 0.0748 | **0.0359** |

### 5.5 分析

- **JointSoftComp 在全部 11 个设定中 MSE 和 IBS 都更低。** Case III 独立删失时，ρ=0/20%/50%/80% 的 MSE 为 1.01 / 1.20 / 1.73 / 6.07，而 SoftComp 为 5.97 / 7.00 / 11.05 / 25.83。
- **C_td 并非总由 JointSoftComp 领先。** 常数 hazard 的 ρ=20%/50%/80% 都是 SoftComp 的均值更高；协变量依赖删失的 ρ=80% 则是 JointSoftComp 的 0.6681 高于 SoftComp 的 0.5938。
- **重删失仍然增加误差。** 常数 hazard、ρ=80% 时 JointSoftComp 的 MSE 为 10.03，SoftComp 为 48.24；JointSoftComp 的 t=20 三类偏差约 -0.131 / -0.047 / -0.105，因此不能把它描述为所有删失水平都近似无偏。
- **无协变量检查中，非参数 AJ 的误差最小。** JointSoftComp 的最大绝对差在四个删失水平都低于 SoftComp，但高于非参数 AJ；ρ=80% 时分别为 0.0748、0.1878、0.0359。

本次所有原始输出、汇总、环境锁和审计都保存在
[`local_validation/readme_full/`](local_validation/readme_full/)。
旧的服务器结果与消融记录见[历史文档](docs/historical_results.md)。
