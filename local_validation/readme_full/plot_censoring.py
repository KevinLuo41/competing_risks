"""Plot verified full experiment III results; bars show SD across five seeds."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
for ax,kind,title,rhos in zip(axes,('case3','depcens','constant'),('Case III: independent censoring','Case III: dependent censoring','Constant cause-specific hazards'),([0,.2,.5,.8],[.2,.5,.8],[0,.2,.5,.8])):
    for method,label,color,marker in [('softcomp','SoftComp','#b84a36','s'),('joint','JointSoftComp','#196b9c','o')]:
        means=[];sds=[]
        for rho in rhos:
            paths=list((OUT/'censoring').glob(f'{kind}_rho{rho:g}_{method}_seed*.json'))
            assert len(paths)==5
            vals=[json.loads(p.read_text())['result']['metrics']['MSE_overall']*1000 for p in paths]
            means.append(statistics.mean(vals));sds.append(statistics.stdev(vals))
        ax.errorbar([r*100 for r in rhos],means,yerr=sds,label=label,color=color,marker=marker,capsize=3,lw=2)
    ax.set(title=title,xlabel='Censoring before time 20 (%)',ylabel='True CIF MSE × 1000',ylim=(0,None))
    ax.set_xticks([0,20,50,80]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle('Experiment III · full run · mean ± SD over 5 seeds',fontsize=14)
fig.savefig(OUT/'experiment_III.png',dpi=180)
fig.savefig(OUT/'experiment_III.pdf')
plt.close(fig)
