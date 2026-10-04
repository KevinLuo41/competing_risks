"""Audit outputs and write a local run report; never overwrite copied README."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    paired=json.loads((HERE/'uncensored_manifest.json').read_text())
    assert manifest.get('finished') and paired.get('finished'), 'Runs still active'
    assert len(manifest['results'])==82 and len(paired['runs'])==24
    assert all(r['returncode']==0 for r in manifest['results']+paired['runs'])
    assert all(r['returncode']==0 for r in manifest['summaries']) and paired['summary_returncode']==0
    paths=list((HERE/'simple').glob('*.json'))+list((HERE/'censoring').glob('*.json'))+list((HERE/'uncensored_available').glob('*/*.json'))
    assert len(paths)==106
    def finite(obj):
        if isinstance(obj,float): assert math.isfinite(obj)
        elif isinstance(obj,dict):
            for value in obj.values(): finite(value)
        elif isinstance(obj,list):
            for value in obj: finite(value)
    for p in paths:
        record=json.loads(p.read_text()); finite(record)
        metrics=record.get('result',{}).get('metrics',{})
        for key,value in metrics.items():
            if key.startswith('Ctd_'): assert 0 <= value <= 1
            if key.startswith(('IBS_','MSE_')): assert value >= 0
    groups=defaultdict(list)
    for p in (HERE/'censoring').glob('*.json'):
        r=json.loads(p.read_text())
        if r['command']!='aj-check':groups[r['command'],r['rho'],r['method']].append(r['result']['metrics'])
    report=[
        '# README 本地复现实验记录', '',
        '原文：`experiments/joint_softcomp/README.md`，由 IMG_0144.MOV 逐行转录。原文中的结果表是此前服务器上的结果，不是此次 Mac 的实测结果。', '',
        '本次按原文允许的方式设 REPS=2，保留样本量、1000 epochs、模型超参数和后处理。实验一/三通过原模块的 main() 启动，替代内部 Buck 构建；最多并行两个进程。实验二单进程、单线程，与一/三并行运行。CPU 执行。', '',
        f'验证：82 个实验一/三任务、24 个实验二任务全部退出码为 0；106 份 JSON 均可读取，浮点值均有限。实验一/三墙钟耗时 {(manifest["finished"]-manifest["started"])/60:.2f} 分钟，实验二 {(paired["finished"]-paired["started"])/60:.2f} 分钟（两批重叠，不能相加）。', '',
        '## 执行范围与差异', '',
        '- 实验一：3 个 β；SoftComp M=0/1/2/4/8；JointSoftComp M=1/2/4/8；Cox、真实模型、无协变量、总体极小值参考。训练 200、测试 50000，每组 2 次。',
        '- 实验三：ρ=0/0.2/0.5/0.8；case3、constant、depcens（ρ=0 不重复）；两模型各 2 次。训练 5000、测试 1000、τ=20、100 个评估点。另跑了 8 个无协变量 AJ 检验（训练 10000）和理论积分表。',
        '- 实验二：Case II/III × 有删失/无删失 × JointSoftComp/SoftComp/SoftComp-noaug × 2 次，训练 4500、验证 500、测试 1000。NeuralFG 源码缺失，未运行。',
        '- 实验二的独立启动器在内存中略过不可用基线的 import，只调度已有 SoftComp spec；数据生成、训练、预测、后处理和指标函数都来自原代码。没有伪造或替代缺失模型，原文件未修改。',
        '- README 写评估时间网格取事件时间 97.5% 分位数并加入固定时刻；已转录的 Case II prepare_data 调用的是默认 90% 的 build_evaluation_time_grid；Case III 已使用 97.5% 分位数并加入固定时刻。本次保留代码行为，因此实验二不是原文表格的严格复现。需补充对应版本的 prepare_data/grid 代码。',
        '- README 称有删失时 C_td 和 IBS 用 IPCW；当前 compute_ctd 实现是未加 IPCW 的 Antolini 可比较对计数，compute_ibs 才使用训练集删失 KM 权重。保留原文与源代码，没有擅自改指标。',
        '- 原始 tests/test_joint_softcomp.py 尚未提供，未执行 README 的原版单元测试。',
        '- 两次重复只用于初步验证，不能代替 README 的 200/10/5 次完整统计实验。', '',
        '## 实验一', '', (HERE/'simple_summary.txt').read_text(), '',
        '## 实验二：已具备源码的三个模型', '', (HERE/'uncensored_available_summary.txt').read_text(), '',
        '## 实验三', '',
        '| 检验 | ρ | 模型 | n | MSE × 1000 | C_td | IBS |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for (kind,rho,method),runs in sorted(groups.items()):
        assert len(runs)==2
        values=[statistics.mean(r[k] for r in runs) for k in ('MSE_overall','Ctd_overall','IBS_overall')]
        report.append(f'| {kind} | {rho:.0%} | {method} | {len(runs)} | {values[0]*1000:.3f} | {values[1]:.4f} | {values[2]:.4f} |')
    report += ['', '完整 AJ 误差、偏差和删失比例见 `censoring_summary.txt`；理论积分表见 `population_targets.txt`。', '',
        '## 旧 Case II 权重复评', '',
        '没有重新训练；补上 SoftComp PAV/simplex，使用有删失测试数据和恢复的原始指标。旧权重使用全部 5000 人训练、固定测试种子 140000、模型种子 0/1/2，与实验二协议不同。', '',
        '| 模型 | MSE × 1000 | C_td | IPCW IBS |', '|---|---:|---:|---:|']
    old=json.loads((HERE/'saved_case2_metrics.json').read_text())
    for m,v in old['means'].items():report.append(f'| {m} | {v["MSE_overall"]*1000:.3f} | {v["Ctd_overall"]:.4f} | {v["IBS_overall"]:.4f} |')
    report += ['', '## 仍需补充', '',
        '- `baseline_models/neural_fine_gray.py`：完成实验二的 NeuralFG。',
        '- `baseline_models/dsm.py`、`baseline_models/fine_gray.py`：解除原入口集中 import 的源码缺口。',
        '- `tests/test_joint_softcomp.py`：运行原版测试。',
        '- 与 README 97.5% 网格一致版本的 `experiments/case2_v4/run.py` 的 `prepare_data` 及相关网格函数。',
        '- 原项目依赖清单：确认库版本与原服务器一致。', '',
        '可复查记录：`manifest.json`、`uncensored_manifest.json` 保留调用命令、耗时和启动时源文件 SHA-256；`logs/` 保留每个任务的日志。', '']
    (HERE/'REPORT.md').write_text('\n'.join(report))
    (HERE/'audit.json').write_text(json.dumps(dict(json_files=len(paths),all_tasks_success=True,all_floats_finite=True,final_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'evaluation').glob('*.py')}),indent=2))
    print('Wrote REPORT.md; 106 JSON files validated')

if __name__=='__main__':main()
