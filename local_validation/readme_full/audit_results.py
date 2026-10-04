"""Require complete README coverage before generating final summaries."""
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from run_suite import ROOT, OUT, ENV, PREFIX, tasks

def finite(value):
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,dict): return all(finite(v) for v in value.values())
    if isinstance(value,list): return all(finite(v) for v in value)
    return True

def main():
    missing=[];failures=[];invalid=[];counts=collections.Counter();coverage=collections.defaultdict(list)
    paired=collections.defaultdict(list);censoring=collections.defaultdict(list)
    source_changes={}
    for group in ('main','uncensored'):
        manifest=json.loads((OUT/f'{group}_manifest.json').read_text())
        changed=[f for f,h in manifest['source_sha256'].items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
        source_changes[group]=changed
        allowed={'experiments/joint_softcomp/analyze_simple.py'}
        if group=='main':allowed.add('experiments/joint_softcomp/uncensored.py')
        for f in set(changed)-allowed:invalid.append((group,'source changed: '+f))
        for name,module,args,path in tasks(group):
            record=OUT/'records'/f'{name}.json'
            if not record.exists(): missing.append(name);continue
            info=json.loads(record.read_text())
            if 'returncode' not in info: missing.append(name);continue
            if info['returncode']!=0: failures.append(name);continue
            try:
                expected=[sys.executable,'-c',f'from {PREFIX}{module} import main; main()',*map(str,args)]
                assert info['command']==expected, 'command differs from plan'
                result=json.loads(path.read_text())
                assert finite(result), 'nonfinite value'
                if module=='simple':
                    runs=result['cox']['runs'] if result['method']=='reference' else result['runs']
                    assert len(runs)==10
                    if result['method']=='reference':assert len(result['null']['runs'])==10
                    assert result['n_train']==200 and result['n_test']==50000
                    key=(round(result['beta'],4),result['method'],result['m'])
                    coverage[key].extend(range(result['rep_start'],result['rep_start']+len(runs)))
                elif module=='uncensored':
                    assert result['method'] in ('JointSoftComp','SoftComp','SoftComp-noaug','NeuralFG')
                    key=(result['case'],result['method'],result['censoring'])
                    paired[key].append(result['replicate'])
                else:
                    key=(result['command'],result['rho'],result['method'])
                    censoring[key].append(result['seed'])
                counts[module]+=1
            except Exception as e:invalid.append((name,str(e)))
    complete=not (missing or failures or invalid)
    if complete:
        assert len(coverage)==30 and all(sorted(v)==list(range(200)) for v in coverage.values())
        assert len(paired)==16 and all(sorted(v)==list(range(10)) for v in paired.values())
        assert len(censoring)==30
        assert all(sorted(v)==([0] if k[0]=='aj-check' else list(range(5))) for k,v in censoring.items())
        assert dict(counts)==dict(simple=600,censoring_levels=118,uncensored=160)
    audit=dict(complete=complete,checked_at=time.time(),counts=dict(counts),pending=missing,failures=failures,invalid=invalid,all_floats_finite=not invalid,source_changes=source_changes)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='pending'},indent=2));print('pending',len(missing))
    if not complete:return 1
    commands={
        'simple_summary':[sys.executable,str(ROOT/'experiments/joint_softcomp/analyze_simple.py'),str(OUT/'simple')],
        'uncensored_summary':[sys.executable,str(ROOT/'experiments/joint_softcomp/analyze_uncensored.py'),str(OUT/'censored'),str(OUT/'uncensored'),'10'],
        'censoring_summary':[sys.executable,'-c',f'from {PREFIX}censoring_levels import main; main()','summarize','--out-dir',str(OUT/'censoring')],
        'population_targets':[sys.executable,str(ROOT/'experiments/joint_softcomp/population_targets.py')],
    }
    for name,cmd in commands.items():
        with (OUT/f'{name}.txt').open('w') as f:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(OUT/'compare_readme.py')],check=True)
    report=['Full README experiment results','', 'Coverage verified: 200 / 10 / 5 repeats; all four experiment-II methods.', 'See protocol.txt for environment, evaluation conventions and source discrepancies.','']
    for name in commands:report += [name,'',(OUT/f'{name}.txt').read_text(),'']
    report += ['Comparison with README','',(OUT/'readme_comparison.txt').read_text()]
    (OUT/'RESULTS.txt').write_text('\n'.join(report))
    audit['summaries_complete']=True
    audit['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [OUT/'RESULTS.txt',OUT/'readme_comparison.json',OUT/'readme_comparison.txt',
                  *[OUT/f'{name}.txt' for name in commands]]}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print('Wrote RESULTS.txt')
    return 0

if __name__=='__main__':sys.exit(main())
