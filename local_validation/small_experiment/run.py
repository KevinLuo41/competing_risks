"""Local diagnostic harness; not transcribed source or the full paper experiment."""
from pathlib import Path
import hashlib
import importlib
import json
import platform
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
import torch
from crsoft_model import CRSoftNet
from crsoft_model.joint_softcomp import JointSoftComp
from data import case2_v4

OUT = Path(__file__).resolve().parent
EPOCHS = 300
SEEDS = [0, 1, 2]
torch.set_num_threads(2)

def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('data', 'crsoft_model', 'baseline_models', 'experiments')
            for p in sorted((ROOT / folder).rglob('*.py'))}

def labels(t, cause, times, k):
    return torch.stack([((t[:, None] <= times) & (cause[:, None] == c)).float()
                        for c in range(1, k + 1)], dim=1)

def make_data(scenario, seed):
    if scenario == 'single_event_uncensored':
        def simulate(n, s):
            gen = torch.Generator().manual_seed(s)
            x = torch.randn(n, 1, generator=gen)
            t = torch.empty(n).exponential_(1.0, generator=gen) / (0.05 * torch.exp(x[:, 0]))
            return {'X': x, 'Y': t, 'Delta': torch.ones(n, dtype=torch.long), 'T_true': t,
                    'epsilon_true': torch.ones(n, dtype=torch.long)}
        train, test = simulate(256, 700000 + seed), simulate(2048, 999999)
        times = torch.linspace(0, 10, 21)
        truth = (1 - torch.exp(-0.05 * torch.exp(test['X']) * times)).unsqueeze(1)
        k = 1
    else:
        params = case2_v4.generate_parameters(K=3, p=3, seed=42)
        train = case2_v4.generate_data(n=512, seed=130000 + seed, params=params, censor_rate=0.5)
        test = case2_v4.generate_data(n=512, seed=140000, params=params, censor_rate=0.0)
        times = torch.linspace(0, 18, 25)
        truth = torch.stack([case2_v4.compute_cif(test['X'], t, params)[0] for t in times], dim=-1)
        k = 3
    return train, test, times, truth, k

def diagnostics(cif, survival):
    return {'finite': bool(torch.isfinite(cif).all() and torch.isfinite(survival).all()),
            'min_probability': float(torch.min(cif.min(), survival.min())),
            'max_probability': float(torch.max(cif.max(), survival.max())),
            'max_mass_error': float((cif.sum(1) + survival - 1).abs().max()),
            'decreasing_cif_steps': int(((cif[:, :, 1:] - cif[:, :, :-1]) < -1e-6).sum()),
            'increasing_survival_steps': int(((survival[:, 1:] - survival[:, :-1]) > 1e-6).sum()),
            'max_cif_at_zero': float(cif[:, :, 0].abs().max())}

def run_one(scenario, seed, method):
    train, test, times, truth, k = make_data(scenario, seed)
    target = labels(test['T_true'], test['epsilon_true'], times, k)
    # Complete-data training marginal is a diagnostic reference, using latent training
    # outcomes for the censored scenario. No test data is used to fit this reference.
    marginal = labels(train['T_true'], train['epsilon_true'], times, k).mean(0, keepdim=True)
    torch.manual_seed(seed)
    kwargs = dict(input_dim=train['X'].shape[1], num_causes=k, hidden_dim=32, num_blocks=1)
    model = JointSoftComp(**kwargs, grid_size=200) if method == 'JointSoftComp' else CRSoftNet(**kwargs)
    before, _ = model.predict_cif_survival_grid(test['X'], times)
    start = time.perf_counter()
    extra = {'n_times': 4, 'weight_decay': 1e-3} if method == 'JointSoftComp' else {'n_aug': 2, 'aug_weight': 0.5, 'weight_decay': 3e-3}
    loss = model.fit(train['X'], train['Y'], train['Delta'], epochs=EPOCHS,
                     batch_size=256, lr=1e-3, device='cpu', verbose=False, **extra)
    elapsed = time.perf_counter() - start
    cif, survival = model.predict_cif_survival_grid(test['X'], times)
    assert cif.shape == truth.shape
    result = dict(scenario=scenario, seed=seed, method=method, n_train=len(train['X']),
                  n_test=len(test['X']), epochs=len(loss), seconds=elapsed,
                  censor_fraction=float((train['Delta'] == 0).float().mean()),
                  loss_first10=statistics.mean(loss[:10]), loss_last10=statistics.mean(loss[-10:]),
                  mse_true_before=float(((before - truth) ** 2).mean()),
                  mse_true=float(((cif - truth) ** 2).mean()),
                  brier=float(((cif - target) ** 2).mean()),
                  oracle_brier=float(((truth - target) ** 2).mean()),
                  marginal_brier=float(((marginal - target) ** 2).mean()),
                  marginal_mse_true=float(((marginal - truth) ** 2).mean()),
                  validity=diagnostics(cif, survival))
    if k == 1:
        result['brier_at_10'] = float(((cif[:, 0, -1] - target[:, 0, -1]) ** 2).mean())
        result['marginal_brier_at_10'] = float(((marginal[:, 0, -1] - target[:, 0, -1]) ** 2).mean())
    return result

before_hash = source_hashes()
payload = {'config': {'epochs': EPOCHS, 'seeds': SEEDS, 'device': 'cpu', 'threads': 2,
    'torch': torch.__version__, 'python': platform.python_version(),
    'description': 'Reduced independent diagnostic; raw SoftComp predictions, no missing postprocessing reconstructed. Brier uses complete simulated test outcomes and uniform time-grid averaging, not IPCW IBS. Censored-case marginal reference uses complete latent TRAIN outcomes.'},
    'source_hashes': before_hash, 'entrypoint_imports': {}, 'runs': []}
for name in ['competing_risks.experiments.joint_softcomp.simple', 'competing_risks.experiments.case2_v4.run']:
    try:
        importlib.import_module(name)
        payload['entrypoint_imports'][name] = 'ok'
    except Exception:
        payload['entrypoint_imports'][name] = traceback.format_exc()
for scenario in ['single_event_uncensored', 'case2_v4_censored']:
    for seed in SEEDS:
        for method in ['JointSoftComp', 'SoftComp_raw']:
            result = run_one(scenario, seed, method)
            payload['runs'].append(result)
            print(json.dumps(result), flush=True)
            (OUT / 'results.json').write_text(json.dumps(payload, indent=2))
payload['source_unchanged'] = before_hash == source_hashes()
(OUT / 'results.json').write_text(json.dumps(payload, indent=2))
print('DONE source_unchanged=', payload['source_unchanged'], flush=True)
