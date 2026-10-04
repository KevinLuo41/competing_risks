"""Re-evaluate existing checkpoints with recovered competing-risk metrics.

This is a diagnostic, not README experiment 2: existing weights fitted all 5000
subjects, use model seeds 0/1/2, and the fixed test seed is 140000.
No model is retrained. Test observations are censored at the same 0.5 setting as
training. The recovered grid helper's default (90th percentile) is retained.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from competing_risks.crsoft_model import CRSoftNet
from competing_risks.crsoft_model.joint_softcomp import JointSoftComp
from competing_risks.data import case2_v4
from competing_risks.evaluation import build_evaluation_time_grid, compute_mse_accuracy, evaluate_cif_metrics
from competing_risks.evaluation.postprocess import enforce_cif_simplex, isotonic_project_cif

def main():
    torch.set_num_threads(2)
    params=case2_v4.generate_parameters(K=3,p=3,seed=42)
    truth=functools.partial(case2_v4.compute_cif,params=params)
    test=case2_v4.generate_data(n=1000,seed=140000,params=params,censor_rate=0.5)
    times=build_evaluation_time_grid(test['Y'],test['Delta'],n_grid=100)
    records=[]
    for seed in range(3):
        train=case2_v4.generate_data(n=5000,seed=130000+seed,params=params,censor_rate=0.5)
        for method in ('JointSoftComp','SoftComp_raw'):
            path=ROOT/'local_validation/case2_comparison_5000'/f'{method}_seed{seed}.pt'
            ckpt=torch.load(path,map_location='cpu',weights_only=True)
            kwargs={k:ckpt[k] for k in ('input_dim','num_causes','hidden_dim','num_blocks','dropout')}
            model=JointSoftComp(**kwargs,grid_size=ckpt['grid_size']) if method=='JointSoftComp' else CRSoftNet(**kwargs)
            model.load_state_dict(ckpt['state_dict']);model.eval()
            cif,_=model.predict_cif_survival_grid(test['X'],times)
            if method=='SoftComp_raw':cif=enforce_cif_simplex(isotonic_project_cif(cif))
            metrics=compute_mse_accuracy(cif,test['X'],times,truth,3)
            metrics.update(evaluate_cif_metrics(cif,test['Y'],test['Delta'],train['Y'],train['Delta'],times,3))
            records.append(dict(seed=seed,method=method.replace('_raw',''),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics))
    means={m:{k:sum(r['metrics'][k] for r in records if r['method']==m)/3 for k in ('MSE_overall','Ctd_overall','IBS_overall')} for m in ('JointSoftComp','SoftComp')}
    output=dict(description=__doc__,test_censored_fraction=float((test['Delta']==0).float().mean()),eval_times=times.tolist(),runs=records,means=means)
    (Path(__file__).parent/'saved_case2_metrics.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(means,indent=2))
if __name__=='__main__':main()
