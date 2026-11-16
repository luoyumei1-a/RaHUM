"""Download pinned official PointLLM files, with bounded retry and SHA256 checks."""
import argparse,concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
REPO='RunsenXu/PointLLM_7B_v1.2'
REVISION='37d8c15aaff8e7e05f04729dcb6960d5758e9f86'
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'checkpoints'/'PointLLM_7B_v1.2'
SPECS=[
 ('pytorch_model-00001-of-00003.bin',9878038738,'d88c71f870f1f0232872eb937d8fe506d3e57405be3a751396d3b11708ed6b07'),
 ('pytorch_model-00002-of-00003.bin',9894801014,'271f703595b7af549b715ca7503e297320b1c9c0fb3d33b4125930ca974c0990'),
 ('pytorch_model-00003-of-00003.bin',7312083996,'4ea5cc85e04b434a4105d79d9bb0a9193fe37cb5e77f0fdfee3edd6c161293a7'),
 ('tokenizer.model',499723,'9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'),
]
def download(spec,verify_only=False):
 name,expected_size,expected_sha=spec
 p=OUT/name; start=time.time(); next_print=256*1024**2
 for attempt in range(1,6):
  offset=p.stat().st_size if p.exists() else 0
  if offset==expected_size:break
  if verify_only:raise RuntimeError(f'{name}: absent or incomplete in verify-only mode')
  if offset>expected_size:raise ValueError(f'{name}: local file larger than expected')
  url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
  headers={'User-Agent':'RaHUM-Research-Reproducibility/1.0'}
  if offset:headers['Range']=f'bytes={offset}-'
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as r:
    if offset and r.status!=206:raise RuntimeError(f'Expected HTTP206 for resume, got {r.status}')
    with p.open('ab' if offset else 'wb') as f:
     while chunk:=r.read(4*1024**2):
      f.write(chunk);offset+=len(chunk)
      if offset>=next_print:
       print(f'{name}: {offset}/{expected_size} bytes, {offset/1024**2/(time.time()-start):.2f} MiB/s',flush=True)
       next_print=offset+256*1024**2
  except Exception as e:
   print(f'{name}: attempt {attempt} {type(e).__name__}: {e}',flush=True)
   if attempt==5:raise
  if p.stat().st_size==expected_size:break
 if p.stat().st_size!=expected_size:raise RuntimeError(f'{name}: incomplete {p.stat().st_size}/{expected_size}')
 sha=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
 if sha!=expected_sha:raise RuntimeError(f'{name}: SHA256 mismatch {sha}')
 result={'file':name,'bytes':expected_size,'sha256':sha,'elapsed_s':time.time()-start,'source_repo':REPO,'source_revision':REVISION}
 (ROOT/'evidence'/f'{name}.receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print(f'{name}: VERIFIED {sha}',flush=True)
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=OUT)
 parser.add_argument('--workers',type=int,default=3)
 parser.add_argument('--verify-only',action='store_true',help='Check existing weights only; never download missing bytes.')
 args=parser.parse_args();OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
 for n in ['config.json','generation_config.json','pytorch_model.bin.index.json','special_tokens_map.json','tokenizer_config.json','added_tokens.json']:
  (OUT/n).write_bytes((ROOT/'evidence'/n).read_bytes())
 (ROOT/'evidence').mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
  results=list(ex.map(lambda spec:download(spec,args.verify_only),SPECS))
 (ROOT/'evidence'/'official_download_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
 print('OFFICIAL_POINTLLM_DOWNLOAD_COMPLETE',flush=True)
