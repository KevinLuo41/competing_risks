"""Execute REPS=2 paired Case II/III runs for the three available methods."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def main():
    manifest={'started':time.time(),'reps':2,'threads':1,'missing_methods':['NeuralFG'],'grid_warning':'Case II source default 90%, README describes 97.5% plus fixed times; source retained.','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('case2_v4','case3_v5','joint_softcomp') for p in (ROOT/'experiments'/d).glob('*.py')},'runs':[]}
    for rep in range(2):
        for case in (2,3):
            for method in ('JointSoftComp','SoftComp','SoftComp-noaug'):
                for condition in ('censored','uncensored'):
                    dest=OUT/'uncensored_available'/condition
                    command=[sys.executable,str(OUT/'uncensored_available.py'),'--case',str(case),'--method',method,'--replicate',str(rep),'--threads','1','--out-dir',str(dest)]
                    if condition=='uncensored':command+=['--uncensored']
                    start=time.time()
                    with (OUT/'logs'/f'case{case}_{method}_{rep}_{condition}.log').open('w') as log:
                        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    manifest['runs'].append(dict(case=case,method=method,replicate=rep,condition=condition,returncode=result.returncode,seconds=time.time()-start,command=command))
                    (OUT/'uncensored_manifest.json').write_text(json.dumps(manifest,indent=2))
                    print(f"{len(manifest['runs'])}/24 Case {case} {method} rep={rep} {condition} exit={result.returncode}",flush=True)
    with (OUT/'uncensored_available_summary.txt').open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'experiments/joint_softcomp/analyze_uncensored.py'),str(OUT/'uncensored_available/censored'),str(OUT/'uncensored_available/uncensored'),'2'],stdout=log,stderr=subprocess.STDOUT)
    manifest['summary_returncode']=result.returncode
    manifest['finished']=time.time()
    (OUT/'uncensored_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
