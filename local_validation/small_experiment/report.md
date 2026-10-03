# 小规模运行验证

结论：现有核心代码能完成数据生成、训练和预测，但尚不能据此确认完整实验或论文性能目标实现。复制的 61 个 Python 源文件运行前后 SHA256 均未改变。

## 设置

- CPU、2 线程、随机种子 0/1/2，每个模型每次训练 300 epochs，共 12 次训练。
- 简单场景：单个风险，无删失，256 条训练样本、2048 条独立测试样本，21 个时间点（0–10）。
- Case II v4：3 类竞争风险，512 条训练样本、512 条独立测试样本，25 个时间点（0–18）；训练删失参数 0.5，测试使用模拟完整事件。
- 使用现有 CRSoftNet 与 JointSoftComp；JointSoftComp 积分网格缩至 200。SoftComp 为原始输出，未执行尚缺失的 isotonic/simplex 后处理。这不是论文正式配置对比。

## 结果（3 次均值）

| 场景 | 模型 | 训练前真值 MSE | 训练后真值 MSE | Brier | 参考 Brier |
|---|---|---:|---:|---:|---:|
| single_event_uncensored | JointSoftComp | 0.098176 | 0.010438 | 0.146756 | 0.176175 |
| single_event_uncensored | SoftComp_raw | 0.130971 | 0.043109 | 0.178309 | 0.176175 |
| case2_v4_censored | JointSoftComp | 0.142013 | 0.007687 | 0.039149 | 0.038655 |
| case2_v4_censored | SoftComp_raw | 0.120528 | 0.008223 | 0.040376 | 0.038655 |

MSE 是预测 CIF 与模拟解析真值的均方差。Brier 对完整模拟测试事件指示变量计算，在人、原因、等距时间点上平均；不是原 evaluation 模块的 IPCW IBS。越小越好。
参考预测是不使用协变量的训练集经验事件概率。在删失场景中，这个参考使用模拟器保存的完整训练事件，是具有额外信息的诊断参照，不是可直接用于真实删失数据的基线。它没有使用测试数据拟合。

- JointSoftComp 的 6 次训练输出均有限，CIF 无下降步，生存概率无上升步，起点 CIF 为 0；概率总和误差小于 1e-6。
- SoftComp 原始输出存在 CIF 下降步，简单场景合计 2236 步，三风险场景合计 3941 步。不能把原始输出视为已通过完整后处理的曲线。
- 简单场景 JointSoftComp 比经验边际参考的 Brier 更低；三风险删失场景的平均 Brier 则略高于具有完整训练事件信息的参考，不能宣称该场景性能优势。
- 所有模型训练损失均下降，但不同模型使用不同损失，损失数值不能直接横向比较。

## 完整入口的当前阻碍

- 实际导入 experiments/joint_softcomp/simple.py 失败：缺少 competing_risks.evaluation。
- 实际导入 experiments/case2_v4/run.py 首先失败于缺少 pandas；静态检查还发现 baseline_models/dsm.py、fine_gray.py、neural_fine_gray.py 和 evaluation 源码缺失。
- 本次直接导入完整的核心模型和数据模块，没有伪造缺失模块或修复复制代码。

## 重跑

```sh
cd /Users/kevin/Projects/competing_risks
.venv/bin/python local_validation/small_experiment/run.py
```

结果：results.json；完整日志：run.log；依赖版本：environment.txt。复跑会覆盖同目录结果。
