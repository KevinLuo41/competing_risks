"""Compare full-run means to the rounded tables in the supplied README."""
import collections
import json
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent

def main():
    assert json.loads((OUT/'audit.json').read_text())['complete']
    rows=[]
    def add(experiment,setting,method,metric,values,reference):
        mean=statistics.mean(values)
        rows.append(dict(experiment=experiment,setting=setting,method=method,metric=metric,n=len(values),mean=mean,sd=statistics.stdev(values) if len(values)>1 else 0,readme=reference,difference=mean-reference))
    simple=collections.defaultdict(list)
    for p in (OUT/'simple').glob('*.json'):
        d=json.loads(p.read_text());beta=round(d['beta'],4)
        if d['method']=='reference':
            for method in ('cox','null'):simple[(beta,method,0)].extend(r['brier'] for r in d[method]['runs'])
            simple[(beta,'oracle',0)]=[d['oracle']['brier']]
        else:simple[(beta,d['method'],d['m'])].extend(r['brier'] for r in d['runs'])
    expected_simple={
        ('softcomp',0):[.603,.592,.575],('softcomp',1):[.318,.308,.289],
        ('softcomp',2):[.256,.246,.229],('softcomp',4):[.242,.235,.222],
        ('softcomp',8):[.275,.270,.262],('joint',4):[.243,.231,.208],
        ('cox',0):[.241,.229,.206],('oracle',0):[.239,.227,.205],('null',0):[.241,.243,.246]}
    for (method,m),targets in expected_simple.items():
        for beta,reference in zip((0,.4055,.6931),targets):
            add('I',f'beta={beta}, M={m}',method,'Brier(10)',simple[(beta,method,m)],reference)
    expected_ii={
        (2,'JointSoftComp'):[2.10,.750,.052,1.65,.740,.052],
        (2,'SoftComp'):[8.20,.750,.056,15.10,.739,.061],
        (2,'SoftComp-noaug'):[21.48,.747,.068,65.22,.721,.147],
        (2,'NeuralFG'):[4.89,.729,.054,2.98,.732,.052],
        (3,'JointSoftComp'):[1.13,.681,.123,.82,.678,.121],
        (3,'SoftComp'):[3.20,.687,.127,4.30,.678,.123],
        (3,'SoftComp-noaug'):[9.22,.682,.129,61.07,.669,.162],
        (3,'NeuralFG'):[3.36,.649,.127,2.17,.661,.123]}
    for (case,method),refs in expected_ii.items():
        for ci,condition in enumerate(('censored','uncensored')):
            data=[json.loads(p.read_text())['result']['metrics'] for p in (OUT/condition).glob(f'case{case}_rep*_{method}.json')]
            assert len(data)==10
            for j,(key,scale) in enumerate((('MSE_overall',1000),('Ctd_overall',1),('IBS_overall',1))):
                add('II',f'case={case}, {condition}',method,key,[d[key]*scale for d in data],refs[3*ci+j])
    refs_iii={
        ('case3',0):[[5.97,.683,.131],[1.04,.684,.126]],
        ('case3',.2):[[7.03,.683,.132],[1.22,.685,.126]],
        ('case3',.5):[[11.03,.686,.136],[1.76,.682,.127]],
        ('case3',.8):[[25.81,.674,.151],[6.06,.676,.131]],
        ('depcens',.2):[[7.45,.675,.132],[1.33,.685,.126]],
        ('depcens',.5):[[13.17,.654,.138],[2.30,.681,.127]],
        ('depcens',.8):[[27.83,.594,.153],[8.41,.667,.133]],
        ('constant',0):[[7.23,.665,.149],[.62,.668,.143]],
        ('constant',.2):[[11.34,.667,.154],[.68,.666,.143]],
        ('constant',.5):[[20.94,.669,.163],[1.18,.666,.144]],
        ('constant',.8):[[48.29,.667,.190],[9.86,.659,.153]]}
    for (kind,rho),refs in refs_iii.items():
        for mi,method in enumerate(('softcomp','joint')):
            data=[json.loads(p.read_text())['result']['metrics'] for p in (OUT/'censoring').glob(f'{kind}_rho{rho:g}_{method}_seed*.json')]
            assert len(data)==5
            for j,(key,scale) in enumerate((('MSE_overall',1000),('Ctd_overall',1),('IBS_overall',1))):
                add('III',f'{kind}, rho={rho:g}',method,key,[d[key]*scale for d in data],refs[mi][j])
    (OUT/'readme_comparison.json').write_text(json.dumps(rows,indent=2))
    text=['Local full-run means vs rounded README tables','MSE_overall values are multiplied by 1000. SD is across repetitions.',
          'Differences are descriptive, not pass/fail tolerances. See protocol.txt for grid/metric and environment differences.',
          '| Experiment | Setting | Method | Metric | n | Local mean | SD | README | Difference |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:text.append(f"| {r['experiment']} | {r['setting']} | {r['method']} | {r['metric']} | {r['n']} | {r['mean']:.5f} | {r['sd']:.5f} | {r['readme']:.5f} | {r['difference']:+.5f} |")
    (OUT/'readme_comparison.txt').write_text('\n'.join(text)+'\n')
    print('Compared',len(rows),'reported means')

if __name__=='__main__':main()
