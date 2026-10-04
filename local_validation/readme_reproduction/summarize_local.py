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
        '# Local README Reproduction Report', '',
        'Source: `experiments/joint_softcomp/README.md`, transcribed line by line from IMG_0144.MOV. The original result tables came from earlier server runs, rather than measurements from this Mac run.', '',
        'This run used REPS=2 as allowed by the source, retaining sample sizes, 1000 epochs, model hyperparameters, and postprocessing. Experiments I/III called main() in the original modules instead of using the internal Buck build, with at most two concurrent processes. Experiment II ran as one single-threaded process alongside I/III. All computation used the CPU.', '',
        f'Validation: all 82 Experiment I/III tasks and 24 Experiment II tasks exited with code 0. All 106 JSON files were readable and contained finite floating-point values. Experiment I/III wall time was {(manifest["finished"]-manifest["started"])/60:.2f} minutes; Experiment II took {(paired["finished"]-paired["started"])/60:.2f} minutes. The batches overlapped, so these durations cannot be added.', '',
        '## Execution Scope and Differences', '',
        '- Experiment I: 3 β values; SoftComp M=0/1/2/4/8; JointSoftComp M=1/2/4/8; Cox, true-model, no-covariate, and population-minimizer references. Training n=200, test n=50000, and 2 repetitions per setting.',
        '- Experiment III: ρ=0/0.2/0.5/0.8; case3, constant, and depcens (without rerunning depcens at ρ=0); 2 repetitions for each model. Training n=5000, test n=1000, τ=20, and 100 evaluation points. Also ran 8 no-covariate AJ checks with training n=10000 and the theoretical integration table.',
        '- Experiment II: Case II/III × censored/uncensored × JointSoftComp/SoftComp/SoftComp-noaug × 2 repetitions; fitting n=4500, validation n=500, and test n=1000. NeuralFG was not run because its source was missing.',
        '- The standalone Experiment II launcher skipped unavailable baseline imports in memory and scheduled only existing SoftComp specs. Data generation, training, prediction, postprocessing, and metrics used the original code. Missing models were neither fabricated nor substituted, and original files were unchanged.',
        '- The README specifies an evaluation grid through the 97.5% event-time quantile with fixed times added. The transcribed Case II prepare_data calls build_evaluation_time_grid with its default 90%, while Case III already uses 97.5% and fixed times. This run retained source behavior, so Experiment II is not an exact reproduction of the original table. The matching prepare_data/grid implementation was still needed.',
        '- The README states that censored-data C_td and IBS use IPCW. The compute_ctd implementation counts Antolini comparable pairs without IPCW; only compute_ibs uses training-set censoring KM weights. The source text and code were retained without changing metric definitions.',
        '- The original tests/test_joint_softcomp.py had not yet been supplied, so the original unit tests specified in the README were not run.',
        '- Two repetitions serve only as preliminary validation and do not replace the full 200/10/5-repetition statistical experiments specified in the README.', '',
        '## Experiment I', '', (HERE/'simple_summary.txt').read_text(), '',
        '## Experiment II: Three Models with Available Source', '', (HERE/'uncensored_available_summary.txt').read_text(), '',
        '## Experiment III', '',
        '| Test | ρ | Model | n | MSE × 1000 | C_td | IBS |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for (kind,rho,method),runs in sorted(groups.items()):
        assert len(runs)==2
        values=[statistics.mean(r[k] for r in runs) for k in ('MSE_overall','Ctd_overall','IBS_overall')]
        report.append(f'| {kind} | {rho:.0%} | {method} | {len(runs)} | {values[0]*1000:.3f} | {values[1]:.4f} | {values[2]:.4f} |')
    report += ['', 'See `censoring_summary.txt` for complete AJ errors, biases, and censoring fractions, and `population_targets.txt` for the theoretical integration table.', '',
        '## Reevaluation of Saved Case II Weights', '',
        'No retraining was performed. SoftComp PAV/simplex postprocessing was added, with censored test data and recovered original metrics. The saved weights used all 5000 training subjects, fixed test seed 140000, and model seeds 0/1/2, differing from the Experiment II protocol.', '',
        '| Model | MSE × 1000 | C_td | IPCW IBS |', '|---|---:|---:|---:|']
    old=json.loads((HERE/'saved_case2_metrics.json').read_text())
    for m,v in old['means'].items():report.append(f'| {m} | {v["MSE_overall"]*1000:.3f} | {v["Ctd_overall"]:.4f} | {v["IBS_overall"]:.4f} |')
    report += ['', '## Outstanding Items at the Time of This Run', '',
        '- `baseline_models/neural_fine_gray.py`: complete NeuralFG in Experiment II.',
        '- `baseline_models/dsm.py`, `baseline_models/fine_gray.py`: supply missing source required by the original entry-point imports.',
        '- `tests/test_joint_softcomp.py`: run the original tests.',
        '- A version of prepare_data in `experiments/case2_v4/run.py` and related grid functions matching the README 97.5% grid.',
        '- The original project dependency list, to confirm library versions against the original server.', '',
        'Audit records: `manifest.json` and `uncensored_manifest.json` retain commands, timings, and source SHA-256 hashes at startup; `logs/` retains per-task logs.', '']
    (HERE/'REPORT.md').write_text('\n'.join(report))
    (HERE/'audit.json').write_text(json.dumps(dict(json_files=len(paths),all_tasks_success=True,all_floats_finite=True,final_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'evaluation').glob('*.py')}),indent=2))
    print('Wrote REPORT.md; 106 JSON files validated')

if __name__=='__main__':main()
