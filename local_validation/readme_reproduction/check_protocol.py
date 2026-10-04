"""Check source-run protocol: paired subjects, common grid, default cap."""
import json
from pathlib import Path
import torch
from uncensored_available import load_case

def main():
    torch.set_num_threads(1)
    load_case('case2_v4');load_case('case3_v5')
    from competing_risks.experiments.joint_softcomp.uncensored import prepare
    checks=[]
    for case in (2,3):
        a,b=prepare(case,0,False),prepare(case,0,True)
        same={key:bool(torch.equal(getattr(a,key),getattr(b,key))) for key in ('X_train_full','T_train_true','X_test','eval_times')}
        assert all(same.values())
        assert bool((b.Delta_train_full>0).all() and (b.Delta_test>0).all())
        assert len(a.X_train_fit)==4500 and len(a.X_val)==500 and len(a.X_test)==1000
        checks.append(dict(case=case,paired_equal=same,n_fit=4500,n_val=500,n_test=1000,n_times=len(a.eval_times),time_min=float(a.eval_times.min()),time_max=float(a.eval_times.max()),test_event_90pct=float(torch.quantile(a.Y_test[a.Delta_test>0],.9)),test_event_975pct=float(torch.quantile(a.Y_test[a.Delta_test>0],.975)),train_censor_fraction=float((a.Delta_train_full==0).float().mean()),test_censor_fraction=float((a.Delta_test==0).float().mean())))
    (Path(__file__).parent/'protocol_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
