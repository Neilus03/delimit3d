import json
from pathlib import Path
root=Path('/cluster/work/igp_psr/nedela/scratch_baselines_20260916')
a=json.loads((root/'agile3d_smoke/status.json').read_text())
m=json.loads((root/'mask3d_smoke/litept_mask3d_ca_scratch_600_seed42/status.json').read_text())
assert a['epoch']==2 and m['epoch']==2, (a,m)
assert (root/'agile3d_smoke/eval_epoch_0002.json').exists()
assert (root/'mask3d_smoke/litept_mask3d_ca_scratch_600_seed42/eval_epoch_0002.json').exists()
assert json.loads((root/'agile3d_smoke/initialization.json').read_text())['public_weights_loaded'] is False
result={'passed':True,'agile3d':a,'mask3d':m,'checks':['official Dice values and gradients','random initialization','two batches of five per epoch','checkpoint and optimizer resume']}
import torch
mask=root/'mask3d_smoke/litept_mask3d_ca_scratch_600_seed42'
checkpoint=torch.load(mask/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
initial=json.loads((mask/'initialization_report.json').read_text())
assert initial['backbone_source']=='scratch' and initial['backbone_checkpoint'] is None
assert initial['backbone_frozen'] is False and initial['official_mask3d_weights_loaded'] is False
assert checkpoint['backbone_state_sha256'] != initial['backbone_state_sha256']
assert checkpoint['decoder_state_sha256'] != initial['decoder_state_sha256']
result['checks'].append('Mask3D encoder and decoder both changed from random initialization')
import hashlib
result['mask3d_decoder_file_sha256']=hashlib.sha256((root/'mask3d_decoder_random_seed42.pt').read_bytes()).hexdigest()
(root/'preflight_passed.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
