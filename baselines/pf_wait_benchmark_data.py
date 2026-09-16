import hashlib,json,random,time
from pathlib import Path
root=Path('/home/nedela/scratch_baselines_20260916')
m=json.loads((root/'runtime/pf_selection_manifest.json').read_text())
order=list(range(1200));random.Random(43).shuffle(order)
expected={}
for i in order[:100]:
 r=m['train_scenes'][i]
 expected[r['data_path']]=r['data_sha256']
 expected[r['normal_path']]=r['source_hashes'][r['normal_path']]
start=time.monotonic();reported=-1
while expected:
 for path,digest in list(expected.items()):
  p=Path(path)
  if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest: del expected[path]
 if int((time.monotonic()-start)//30)!=reported:
  reported=int((time.monotonic()-start)//30);print('Benchmark files still transferring:',len(expected),flush=True)
 if time.monotonic()-start>1800:raise TimeoutError('Dataset transfer not ready')
 if expected:time.sleep(2)
print('Benchmark subset hashes verified',flush=True)
