"""Check new loss against the official criterion, including gradients."""
import importlib.util
import sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'scripts/evaluation'))
from run_scannet40_scratch import official_dice
spec=importlib.util.spec_from_file_location('upstream_criterion', '/cluster/work/igp_psr/nedela/AGILE3D/models/criterion.py')
upstream=importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
criterion=upstream.SetCriterion({}, [])
torch.manual_seed(42)
for classes in (2, 5, 11):
    a=torch.randn(37, classes, requires_grad=True)
    b=a.detach().clone().requires_grad_()
    target=torch.randint(classes, (37,)); weights=torch.rand(37)+.8
    actual=official_dice(a, target, weights)
    expected=(criterion.multiclass_dice_loss(b,target)*weights).mean()
    torch.testing.assert_close(actual,expected)
    actual.backward(); expected.backward()
    torch.testing.assert_close(a.grad,b.grad)
print('PASS: official Dice values and gradients for 2, 5, 11 classes',flush=True)

import numpy as np
from scratch_clicks import candidates, _original
rng=np.random.default_rng(42)
for classes in (2,5,11):
    for case in range(4):
        target=rng.integers(classes,size=2000)
        pred=rng.integers(classes,size=2000)
        xyz=rng.normal(size=(2000,3))
        if case==1: xyz=np.round(xyz)
        if case==2: pred=target.copy()
        if case==3: target[:]=1; pred[:]=0
        assert candidates(pred,target,xyz)==_original(pred,target,xyz,method='kdtree')
print('PASS: parallel KD-tree matches exact reference, including ties and edge cases',flush=True)
parameter=torch.nn.Parameter(torch.ones(()))
optimizer=torch.optim.AdamW([parameter],lr=1e-4)
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[1000],gamma=.1)
for epoch in range(1,1101):
    assert abs(optimizer.param_groups[0]['lr']-(1e-4 if epoch<=1000 else 1e-5))<1e-12
    optimizer.step(); scheduler.step()
print('PASS: AGILE3D LR stays at 1e-4 for 1000 full epochs, then 1e-5',flush=True)
