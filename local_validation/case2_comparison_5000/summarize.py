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
lines = ['# Case II v4: Model Comparison with 5,000 Training Subjects', '',
'This independent diagnostic used the copied original models and data-generation code. The hashes of all 61 original Python files were unchanged before and after the run. Missing evaluation code and postprocessing were not implemented for this diagnostic.', '',
'## Configuration', '',
'- 5,000 training subjects and 1,000 independent test subjects; 3 covariates and 3 competing risks.',
'- Training censoring parameter 0.5; complete simulated test events retained for evaluation. Parameter-generation seed=42, training seeds=130000/130001/130002, and fixed test seed=140000.',
'- Each model: 1,000 epochs, batch=256, 32 hidden units, 1 residual block, dropout=0, Adam, lr=0.001, cosine scheduling, and no early stopping. CPU with 2 threads.',
'- JointSoftComp: n_times=4, weight_decay=0.001, and a 1,000-point integration grid.',
'- SoftComp: n_aug=2, aug_weight=0.5, weight_decay=0.003; raw predictions without isotonic/simplex postprocessing.',
'- Evaluation at 61 equally spaced time points over 0–18. Both models used identical data for each seed. Model-initialization seeds were 0/1/2; test results were not used for tuning.', '',
'## Summary (Mean ± Sample Standard Deviation over 3 Runs)', '',
'| Metric | JointSoftComp | SoftComp Raw Output |', '|---|---:|---:|']
for key, label in [('mse_true','MSE against True CIF'),('brier','Brier'),('seconds','Training Time (s)'),('predict_seconds','Prediction Time (s)')]:
    lines.append(f"| {label} | {j[key]['mean']:.6f} ± {j[key]['sd']:.6f} | {s[key]['mean']:.6f} ± {s[key]['sd']:.6f} |")
lines += ['', 'Lower MSE and Brier scores are better. Brier uses complete simulated test events and is averaged over subjects, causes, and equally spaced time points; it is not the original IPCW IBS. Training time includes fit() but excludes startup, data generation, evaluation, and saving outputs.', '',
f"The complete comparison script took {p['wall_seconds']:.2f} seconds, excluding initial Python/PyTorch imports and subsequent report generation.", '',
f"JointSoftComp reduced mean MSE by {summary['relative_improvement_joint']['mse_true']:.1%}, and Brier by {summary['relative_improvement_joint']['brier']:.1%}; paired MSE wins: {summary['paired_joint_wins']['mse_true']}/3; paired Brier wins: {summary['paired_joint_wins']['brier']}/3.", '',
f"The test Brier score of the analytic true-risk predictions was {j['oracle_brier']['mean']:.6f}; the empirical marginal reference using complete training events scored {j['marginal_brier']['mean']:.6f}. This marginal reference uses latent true events for censored training subjects and serves only as a simulation diagnostic.", '',
'## Per-Run Results', '', '| seed | Model | MSE | Brier | Training Time (s) | Test Subjects with Decreasing CIFs |', '|---|---|---:|---:|---:|---:|']
for r in p['runs']:
    diagnostic = next(d for d in summary[r['method']]['diagnostics'] if d['seed']==r['seed'])
    lines.append(f"| {r['seed']} | {r['method']} | {r['mse_true']:.6f} | {r['brier']:.6f} | {r['seconds']:.2f} | {diagnostic['subjects_with_decreasing_cif']}/1000 |")
lines += ['', '## Scope of Interpretation', '',
'- This compares the models under the configurations above. In particular, SoftComp lacks the original experiment postprocessing, so the comparison does not establish the final relative performance of the complete manuscript methods.',
'- There are only 3 training repetitions and one fixed test set. Standard deviations reflect training-sample and model randomness, not changes in the test set or data distribution.',
'- Checks cover only the specified 0–18 time grid and do not establish accuracy at later times.',
'- At the time of this run, the original full experiment entry points could not run directly because evaluation code, baseline models, or third-party dependencies were missing.', '',
'## Reproduction', '', '```sh', 'cd /Users/kevin/Projects/competing_risks',
'.venv/bin/python local_validation/case2_comparison_5000/run.py',
'.venv/bin/python local_validation/case2_comparison_5000/summarize.py', '```', '',
'Rerunning overwrites results in the same directory. results.json stores raw metrics, summary.json stores aggregates, and each .pt file stores weights, per-epoch losses, and test predictions.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
