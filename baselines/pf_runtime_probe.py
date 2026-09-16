import os,sys,time
from pathlib import Path
root=Path('/home/nedela/scratch_baselines_20260916')
sys.path[:0]=[str(root/'source/src'),str(root/'runtime'),str(root/'litept')]
import torch,flash_attn,pointrope
from delimit3d.models.litept_wrapper import LitePTBackbone
print('runtime',torch.__version__,flash_attn.__file__,pointrope.__file__,flush=True)
assert torch.cuda.device_count()==1
model=LitePTBackbone(litept_root=root/'litept',in_channels=6,grid_size=.05,litept_variant='litept_s_star',multi_scale=False,voxel_reduce='representative',representative_sampling='first',cache_training_voxelization=False).cuda().train()
x=torch.rand(20000,3,device='cuda')*5
f=torch.rand(20000,6,device='cuda')
with torch.autocast('cuda',dtype=torch.float16):
 y=model(x,f,torch.tensor([4000,8000,12000,16000,20000],device='cuda'))
 loss=y.scene_tokens.float().square().mean()
loss.backward()
assert torch.isfinite(loss)
print('PASS native 4090 forward/backward',y.scene_tokens.shape,float(loss),flush=True)
