"""Independent counted rerun: no modifications to transcribed source files."""
from pathlib import Path
import hashlib
import json
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data import case2_v4
from crsoft_model import CRSoftNet
from crsoft_model.joint_softcomp import JointSoftComp
OUT = Path(__file__).resolve().parent
original = json.loads((OUT / 'results.json').read_text())
torch.set_num_threads(2)
params = case2_v4.generate_parameters(K=3, p=3, seed=42)
train = case2_v4.generate_data(n=5000, seed=130000, params=params, censor_rate=0.5)
results = []
for name, cls in [('JointSoftComp', JointSoftComp), ('SoftComp_raw', CRSoftNet)]:
    torch.manual_seed(0)
    model = cls(input_dim=3, num_causes=3, hidden_dim=32, num_blocks=1)
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    counters = dict(optimizer_steps=0, backward_calls=0, train_batches=0,
                    subject_visits=0, network_forward_calls=0, network_rows=0)
    def forward_hook(module, inputs, output):
        counters['network_forward_calls'] += 1
        counters['network_rows'] += len(inputs[0])
    def grad_hook(grad):
        counters['backward_calls'] += 1
        return grad
    fhandle = model.register_forward_hook(forward_hook)
    ghandle = next(model.parameters()).register_hook(grad_hook)
    original_train_step = model._train_step
    def counted_train_step(*args, **kwargs):
        counters['train_batches'] += 1
        counters['subject_visits'] += len(args[0])
        return original_train_step(*args, **kwargs)
    model._train_step = counted_train_step
    original_adam_step = torch.optim.Adam.step
    def counted_adam_step(optimizer, *args, **kwargs):
        value = original_adam_step(optimizer, *args, **kwargs)
        counters['optimizer_steps'] += 1
        return value
    torch.optim.Adam.step = counted_adam_step
    extra = {'n_times': 4, 'weight_decay': 0.001} if name == 'JointSoftComp' else {'n_aug': 2, 'aug_weight': 0.5, 'weight_decay': 0.003}
    start = time.perf_counter()
    try:
        losses = model.fit(train['X'], train['Y'], train['Delta'], epochs=1000,
                           batch_size=256, lr=0.001, verbose=False, device='cpu', **extra)
    finally:
        torch.optim.Adam.step = original_adam_step
        fhandle.remove()
        ghandle.remove()
    elapsed = time.perf_counter() - start
    saved = torch.load(OUT / f'{name}_seed0.pt', map_location='cpu', weights_only=True)
    delta = sum(float((v.detach() - initial[k]).square().sum()) for k, v in model.named_parameters()) ** 0.5
    error = max(float((v.cpu() - saved['state_dict'][k]).abs().max()) for k,v in model.state_dict().items())
    result = dict(model=name, seconds_with_audit_hooks=elapsed, epochs=len(losses),
                  parameter_count=sum(v.numel() for v in model.parameters()),
                  parameter_change_l2=delta, device=str(next(model.parameters()).device),
                  first_loss=losses[0], last_loss=losses[-1],
                  max_parameter_difference_vs_previous=error,
                  max_loss_difference_vs_previous=max(abs(a-b) for a,b in zip(losses,saved['losses'])),
                  counters=counters)
    assert len(losses)==1000
    assert counters['optimizer_steps']==counters['backward_calls']==counters['train_batches']==20000
    assert counters['subject_visits']==5000000 and delta > 0
    assert all(torch.isfinite(v).all() for v in model.parameters())
    results.append(result)
    print(json.dumps(result), flush=True)
source_unchanged = all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in original['source_hashes'].items())
assert source_unchanged
(OUT/'training_audit.json').write_text(json.dumps({'source_unchanged':source_unchanged,'runs':results}, indent=2))
