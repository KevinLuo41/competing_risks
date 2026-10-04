"""Local execution of README experiments 1 and 3, with REPS=2.

Only launch paths, output paths, parallelism, and REPS differ from the README.
Original experimental source is imported unchanged. Missing experiment 2 and
unit-test source is reported explicitly, rather than replaced with stand-ins.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREFIX = 'competing_risks.experiments.joint_softcomp.'
REPS = 2
ENV = dict(os.environ, PYTHONPATH=str(ROOT.parent), MPLBACKEND='Agg')


def execute(task):
    name, module, args = task
    command = [sys.executable, '-c', f'from {PREFIX}{module} import main; main()', *map(str, args)]
    started = time.time()
    with (OUT / 'logs' / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    return dict(name=name, command=command, returncode=result.returncode, seconds=time.time()-started)


def main():
    (OUT/'logs').mkdir(exist_ok=True)
    tasks = []
    for bi, beta in enumerate((0, 0.4054651081, 0.6931471806)):
        for method, ms in [('reference', [0]), ('softcomp', [0,1,2,4,8]), ('joint', [1,2,4,8])]:
            for m in ms:
                tasks.append((f'simple_b{bi}_{method}_m{m}', 'simple', [method,'--beta',beta,'--m',m,'--reps',REPS,'--out-dir',OUT/'simple']))
    for rho in (0, 0.2, 0.5, 0.8):
        for method in ('softcomp','joint'):
            for kind, seed in [('aj-check',0)] + [(kind,seed) for seed in range(REPS) for kind in ('case3','constant','depcens') if kind != 'depcens' or rho != 0]:
                args=[kind,'--method',method,'--rho',rho,'--seed',seed,'--threads',2,'--out-dir',OUT/'censoring']
                if kind=='aj-check': args += ['--n-train',10000]
                tasks.append((f'{kind}_rho{rho}_{method}_s{seed}', 'censoring_levels', args))
    manifest = dict(reps=REPS, workers=2, tasks=len(tasks), started=time.time(), source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('crsoft_model','data','evaluation','experiments/joint_softcomp') for p in (ROOT/folder).glob('*.py')}, results=[])
    def save(): (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(execute,t) for t in tasks]):
            r=future.result(); manifest['results'].append(r); save()
            print(f"{len(manifest['results'])}/{len(tasks)} {r['name']} exit={r['returncode']} {r['seconds']:.1f}s",flush=True)
    for name, command in [
        ('simple_summary',[sys.executable,str(ROOT/'experiments/joint_softcomp/analyze_simple.py'),str(OUT/'simple')]),
        ('censoring_summary',[sys.executable,'-c',f'from {PREFIX}censoring_levels import main; main()','summarize','--out-dir',str(OUT/'censoring')]),
        ('population_targets',[sys.executable,str(ROOT/'experiments/joint_softcomp/population_targets.py')]),
    ]:
        with (OUT/(name+'.txt')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        manifest.setdefault('summaries',[]).append(dict(name=name,returncode=result.returncode))
    manifest['finished']=time.time(); save()

if __name__ == '__main__': main()
