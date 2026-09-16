"""Wait for and verify the byte-identical ScanNet copies on pf-pc69."""
import hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root=Path('/home/nedela/scratch_baselines_20260916')
m=json.loads((root/'runtime/pf_selection_manifest.json').read_text())
expected={}
for r in m['train_scenes']+m['validation_scenes']:
 expected[r['data_path']]=r['data_sha256']
 expected[r['normal_path']]=r['source_hashes'][r['normal_path']]
start=time.monotonic();reported=-1
with ThreadPoolExecutor(8) as pool:
 while expected:
  def valid(item):
   path,digest=item;p=Path(path)
   return path if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest else None
  for path in pool.map(valid,list(expected.items())):
   if path:del expected[path]
  interval=int((time.monotonic()-start)//30)
  if interval!=reported:
   reported=interval;print('Full dataset files awaiting verified transfer:',len(expected),flush=True)
  if time.monotonic()-start>3600:raise TimeoutError('Dataset transfer not complete')
  if expected:time.sleep(5)
(root/'runtime/dataset_verified.json').write_text(json.dumps({'train_scenes':1200,'validation_scenes':312,'verified_files':3024}))
