"""Verify every experiment-II paired replicate before interpreting its results."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from competing_risks.experiments.joint_softcomp.uncensored import prepare
from competing_risks.evaluation import build_evaluation_time_grid

torch.set_num_threads(1)
checks=[]
for case in (2,3):
    for replicate in range(10):
        a,b=prepare(case,replicate,False),prepare(case,replicate,True)
        keys=('X_train_full','T_train_true','X_test','eval_times')
        same={k:bool(torch.equal(getattr(a,k),getattr(b,k))) for k in keys}
        assert all(same.values())
        assert (b.Delta_train_full>0).all() and (b.Delta_test>0).all()
        assert torch.equal(a.Delta_train_full[a.Delta_train_full>0],b.Delta_train_full[a.Delta_train_full>0])
        assert torch.equal(a.Delta_test[a.Delta_test>0],b.Delta_test[a.Delta_test>0])
        assert (len(a.X_train_fit),len(a.X_val),len(a.X_test))==(4500,500,1000)
        expected=build_evaluation_time_grid(a.Y_test,a.Delta_test,100,97.5)
        assert torch.isin(expected,a.eval_times).all()
        checks.append(dict(case=case,replicate=replicate,paired_equal=same,n_fit=4500,n_val=500,n_test=1000,n_grid=len(a.eval_times),time_min=float(a.eval_times.min()),time_max=float(a.eval_times.max())))
Path(__file__).with_name('all_pair_checks.json').write_text(json.dumps(checks,indent=2))
print('PASS: 20 case/replicate pairs, no censoring in complete condition, shared subjects/grid and observed causes preserved')
