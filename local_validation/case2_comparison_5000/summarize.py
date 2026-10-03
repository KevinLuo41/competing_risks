from pathlib import Path
import json
import statistics
import torch

out = Path(__file__).resolve().parent
p = json.loads((out / 'results.json').read_text())
assert len(p['runs']) == 6 and p['source_unchanged'], 'Comparison not complete'
summary = {}
for method in ['JointSoftComp', 'SoftComp_raw']:
    rr = [r for r in p['runs'] if r['method'] == method]
    summary[method] = {key: {'mean': statistics.mean(r[key] for r in rr),
                           'sd': statistics.stdev(r[key] for r in rr)}
                       for key in ['mse_true', 'brier', 'seconds', 'predict_seconds',
                                   'marginal_brier', 'oracle_brier']}
    summaries = []
    for r in rr:
        d = torch.load(out / f"{method}_seed{r['seed']}.pt", map_location='cpu', weights_only=True)
        cif, survival = d['cif'], d['survival']
        summaries.append({'seed': r['seed'],
            'mse_per_cause': ((cif - d['truth']) ** 2).mean(dim=(0, 2)).tolist(),
            'subjects_with_decreasing_cif': int(((cif[:, :, 1:] - cif[:, :, :-1]) < -1e-6).any(dim=2).any(dim=1).sum()),
            'max_mass_error': r['validity']['max_mass_error'],
            'cif_decreasing_steps': r['validity']['decreasing_cif_steps'],
            'survival_increasing_steps': r['validity']['increasing_survival_steps']})
    summary[method]['diagnostics'] = summaries
    summary[method]['all_finite'] = all(r['validity']['finite'] for r in rr)
j, s = summary['JointSoftComp'], summary['SoftComp_raw']
summary['relative_improvement_joint'] = {key: 1 - j[key]['mean'] / s[key]['mean'] for key in ['mse_true','brier','seconds']}
summary['paired_joint_wins'] = {key: sum(next(r for r in p['runs'] if r['seed']==seed and r['method']=='JointSoftComp')[key] < next(r for r in p['runs'] if r['seed']==seed and r['method']=='SoftComp_raw')[key] for seed in [0,1,2]) for key in ['mse_true','brier']}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
lines = ['# Case II v4：5,000 样本模型比较', '',
'本次独立诊断调用复制的原模型与数据生成代码。61 个原 Python 文件运行前后哈希一致；未补写缺失的 evaluation 或后处理。', '',
'## 配置', '',
'- 训练 5,000 人；独立测试 1,000 人；3 个协变量、3 类风险。',
'- 训练删失参数 0.5；测试保留完整模拟事件用于评估。参数生成 seed=42，训练 seeds=130000/130001/130002，固定测试 seed=140000。',
'- 每个模型 1,000 epochs；batch=256；32 个隐藏单元、1 个残差块、dropout=0；Adam、lr=0.001、余弦调度、无早停。CPU 2 线程。',
'- JointSoftComp：n_times=4，weight_decay=0.001，积分网格 1,000 点。',
'- SoftComp：n_aug=2，aug_weight=0.5，weight_decay=0.003；原始预测，无 isotonic/simplex 后处理。',
'- 评估 0–18 的 61 个等距时间点。两模型每个种子使用完全相同的数据；模型初始化 seed 为 0/1/2，没有利用测试结果调参。', '',
'## 汇总（3 次平均 ± 样本标准差）', '',
'| 指标 | JointSoftComp | SoftComp 原始输出 |', '|---|---:|---:|']
for key, label in [('mse_true','真值 CIF MSE'),('brier','Brier'),('seconds','纯训练秒数'),('predict_seconds','预测秒数')]:
    lines.append(f"| {label} | {j[key]['mean']:.6f} ± {j[key]['sd']:.6f} | {s[key]['mean']:.6f} ± {s[key]['sd']:.6f} |")
lines += ['', 'MSE 和 Brier 越小越好。Brier 使用完整模拟测试事件，在人、风险类别和等距时间点上平均，不是原项目 IPCW IBS。训练耗时包含 fit()，不含启动、数据生成、评估或存盘。', '',
f"整个比较脚本耗时 {p['wall_seconds']:.2f} 秒（不含最初 Python/PyTorch 导入与后续报告生成）。", '',
f"JointSoftComp 平均 MSE 降低 {summary['relative_improvement_joint']['mse_true']:.1%}，Brier 降低 {summary['relative_improvement_joint']['brier']:.1%}；配对 MSE 胜出 {summary['paired_joint_wins']['mse_true']}/3 次，Brier 胜出 {summary['paired_joint_wins']['brier']}/3 次。", '',
f"解析真值预测的测试 Brier 为 {j['oracle_brier']['mean']:.6f}；完整训练事件的经验边际参考 Brier 为 {j['marginal_brier']['mean']:.6f}。这个边际参考使用删失训练者的潜在真实事件，仅为模拟诊断参照。", '',
'## 每次结果', '', '| seed | 模型 | MSE | Brier | 训练秒 | 有下降 CIF 的测试人数 |', '|---|---|---:|---:|---:|---:|']
for r in p['runs']:
    diagnostic = next(d for d in summary[r['method']]['diagnostics'] if d['seed']==r['seed'])
    lines.append(f"| {r['seed']} | {r['method']} | {r['mse_true']:.6f} | {r['brier']:.6f} | {r['seconds']:.2f} | {diagnostic['subjects_with_decreasing_cif']}/1000 |")
lines += ['', '## 解释范围', '',
'- 这比较的是当前模型的上述配置，尤其 SoftComp 缺少原实验后处理，不能当作完整论文方法的最终优劣结论。',
'- 只有 3 个训练重复和一个固定测试集；标准差仅反映训练样本及模型随机性，未涵盖更换测试集或数据分布的影响。',
'- 检查仅覆盖 0–18 的指定时间网格，不声称更晚时域同样准确。',
'- 原完整实验入口仍因缺失 evaluation 和基线模型/第三方依赖而不可直接运行。', '',
'## 复现', '', '```sh', 'cd /Users/kevin/Projects/competing_risks',
'.venv/bin/python local_validation/case2_comparison_5000/run.py',
'.venv/bin/python local_validation/case2_comparison_5000/summarize.py', '```', '',
'复跑会覆盖同目录结果。results.json 保存原始指标，summary.json 保存汇总，每个 .pt 文件保存权重、逐 epoch 损失及测试预测。']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
