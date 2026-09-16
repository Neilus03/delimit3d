import json, hashlib
from pathlib import Path
import torch
root=Path('/cluster/work/igp_psr/nedela/scratch_baselines_20260916')
run=root/'maskonly_smoke/litept_mask3d_ca_scratch_600_seed42'
status=json.loads((run/'status.json').read_text())
assert status['epoch']==2
initial=json.loads((run/'initialization_report.json').read_text())
assert initial['backbone_source']=='scratch' and initial['backbone_checkpoint'] is None
assert not initial['backbone_frozen']
assert initial['training_protocol']['classification_head'] is False
assert initial['training_protocol']['classification_loss'] is False
assert initial['training_protocol']['semantic_loss'] is False
checkpoint=torch.load(run/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
assert not any('class_embed_head' in k for k in checkpoint['model_state_dict'])
assert checkpoint['backbone_state_sha256'] != initial['backbone_state_sha256']
assert checkpoint['decoder_state_sha256'] != initial['decoder_state_sha256']
assert (run/'eval_epoch_0002.json').exists()
result=dict(passed=True,mask_only=True,classification_head=False,semantic_loss=False,
    mask3d_decoder_file_sha256=hashlib.sha256((root/'mask3d_maskonly_decoder_random_seed42.pt').read_bytes()).hexdigest(),
    status=status)
(root/'maskonly_preflight_passed.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
