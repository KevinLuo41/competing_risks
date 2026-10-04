"""Full README experiments, resumable by verified output and per-task records."""
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
ENV = dict(os.environ, PYTHONPATH=str(ROOT.parent), MPLBACKEND='Agg', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
PREFIX = 'competing_risks.experiments.joint_softcomp.'

def tasks(group):
    if group == 'main':
        for rho in (0, .2, .5, .8):
            for method in ('softcomp', 'joint'):
                for kind, seed in [('aj-check', 0)] + [(k,s) for s in range(5) for k in ('case3','constant','depcens') if k != 'depcens' or rho != 0]:
                    name=f'{kind}_rho{rho:g}_{method}_seed{seed}'
                    args=[kind,'--method',method,'--rho',rho,'--seed',seed,'--threads',1,'--out-dir',OUT/'censoring']
                    if kind=='aj-check': args += ['--n-train',10000]
                    yield name,'censoring_levels',args,OUT/'censoring'/f'{name}.json'
        for start in range(0,200,10):
            for beta in (0,.4054651081,.6931471806):
                for method,ms in [('reference',[0]),('softcomp',[0,1,2,4,8]),('joint',[1,2,4,8])]:
                    for m in ms:
                        name=f'simple_b{beta:.4f}_{method}_m{m}_r{start}'
                        yield name,'simple',[method,'--beta',beta,'--m',m,'--rep-start',start,'--reps',10,'--threads',1,'--out-dir',OUT/'simple'],OUT/'simple'/f'{name}.json'
    else:
        for r in range(10):
            for case in (2,3):
                for method in ('JointSoftComp','SoftComp','SoftComp-noaug','NeuralFG'):
                    for condition in ('censored','uncensored'):
                        name=f'case{case}_rep{r:02d}_{method}_{condition}'
                        args=['--case',case,'--method',method,'--replicate',r,'--threads',1,'--out-dir',OUT/condition]
                        if condition=='uncensored':args+=['--uncensored']
                        yield name,'uncensored',args,OUT/condition/f'case{case}_rep{r:02d}_{method}.json'

def execute(task):
    name,module,args,path=task
    record=OUT/'records'/f'{name}.json'
    if record.exists() and path.exists() and json.loads(record.read_text()).get('returncode')==0:
        json.loads(path.read_text())
        return name,'cached'
    cmd=[sys.executable,'-c',f'from {PREFIX}{module} import main; main()',*map(str,args)]
    start=time.time()
    with (OUT/'logs'/f'{name}.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        record.write_text(json.dumps(dict(name=name,pid=p.pid,started=start,command=cmd,output=str(path)),indent=2))
        code=p.wait()
    data=json.loads(record.read_text());data.update(returncode=code,seconds=time.time()-start)
    record.write_text(json.dumps(data,indent=2))
    return name,code

if __name__=='__main__':
    group=sys.argv[1];workers=int(sys.argv[2])
    for folder in ('records','logs'): (OUT/folder).mkdir(exist_ok=True)
    plan=list(tasks(group))
    manifest=dict(group=group,tasks=len(plan),workers=workers,started=time.time(),pid=os.getpid(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('crsoft_model','data','evaluation','experiments/joint_softcomp','baseline_models','experiments/case2_v4','experiments/case3_v5') for p in (ROOT/folder).glob('*.py')})
    (OUT/f'{group}_manifest.json').write_text(json.dumps(manifest,indent=2))
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(name,code) in enumerate(pool.map(execute,plan),1):
            print(f'{i}/{len(plan)} {name} {code}',flush=True)
            if code not in (0,'cached'):failures.append(name)
    manifest.update(finished=time.time(),failures=failures)
    (OUT/f'{group}_manifest.json').write_text(json.dumps(manifest,indent=2))
    sys.exit(bool(failures))
