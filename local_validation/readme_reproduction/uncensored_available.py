"""Run the three available methods from README experiment 2.

Original files remain unchanged. In-memory loading omits the unavailable baseline
import and restricts build_specs to the existing SoftComp spec. There are no
replacement baseline implementations. Data preparation, training, prediction,
postprocessing and metrics are executed from the recovered source unchanged.
WARNING: Case II source uses a 90% grid cap, versus 97.5% stated in the README.
"""
import ast
import importlib
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))


def load_case(case):
    name='competing_risks.experiments.'+case+'.run'
    path=ROOT/'experiments'/case/'run.py'
    importlib.import_module(name.rsplit('.',1)[0])
    tree=ast.parse(path.read_text(),filename=str(path))
    removed=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='baseline_models']
    assert len(removed)==1
    tree.body=[n for n in tree.body if n not in removed]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    exec(compile(tree,str(path),'exec'),module.__dict__)
    module.build_specs=lambda data,config,model_seed:[module._softcomp_spec(data,config,model_seed)]
    return module

if __name__=='__main__':
    if '--method' not in sys.argv or sys.argv[sys.argv.index('--method')+1] not in ('JointSoftComp','SoftComp','SoftComp-noaug'):
        raise SystemExit('Only the three recovered model variants are supported')
    load_case('case2_v4');load_case('case3_v5')
    from competing_risks.experiments.joint_softcomp.uncensored import main
    main()
