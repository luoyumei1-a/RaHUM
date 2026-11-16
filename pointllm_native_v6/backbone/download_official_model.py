"""Download the pinned publisher PointLLM 7B model under its original license.

No model weights are redistributed with RaHUM source. This command requires
approximately 27.1 GB for the original three FP32 model shards.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path

REPO='RunsenXu/PointLLM_7B_v1.2'
REVISION='37d8c15aaff8e7e05f04729dcb6960d5758e9f86'
FILES=[
 ('pytorch_model-00001-of-00003.bin',9878038738,'d88c71f870f1f0232872eb937d8fe506d3e57405be3a751396d3b11708ed6b07'),
 ('pytorch_model-00002-of-00003.bin',9894801014,'271f703595b7af549b715ca7503e297320b1c9c0fb3d33b4125930ca974c0990'),
 ('pytorch_model-00003-of-00003.bin',7312083996,'4ea5cc85e04b434a4105d79d9bb0a9193fe37cb5e77f0fdfee3edd6c161293a7'),
 ('tokenizer.model',499723,'9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'),
 ('config.json',868,'e308892ee8b0a49602d4d3f8e7a1c75e8d57db7178ebb121ca8e3b36c0e76506'),
 ('generation_config.json',137,'fd7ff399e5568cc21a0a8414f43df88ef7c424995b9b97a90563165d2cf79efd'),
 ('pytorch_model.bin.index.json',42089,'d7e596fb272adfb91f41c70216cb2ca86ef0632bc57feaeecb8cffd844e178aa'),
 ('special_tokens_map.json',435,'2ea88203b955648951a454e73a73967ff77ba4e79ae8ceb4400d4ab2cb040777'),
 ('tokenizer_config.json',778,'60e65829650cc502396bdcf2fd0103b5509093a997d371511ff71abb73377d5c'),
 ('added_tokens.json',79,'024d44caddbf1ca0cdf469cd16b4069ef99fe7883473240ecbaf865b5cca4a35'),
]


def fetch(spec,directory,verify_only):
    name,size,expected=spec;path=directory/name;started=time.time()
    for attempt in range(5):
        offset=path.stat().st_size if path.exists() else 0
        if offset==size:break
        if verify_only:raise RuntimeError(f'{name}: absent or incomplete in verify-only mode')
        if offset>size:raise ValueError(f'{name}: local file larger than pinned publisher size')
        headers={'User-Agent':'RaHUM-Reproducibility/1.0'}
        if offset:headers['Range']=f'bytes={offset}-'
        request=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',headers=headers)
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                if offset and response.status!=206:raise RuntimeError('Publisher did not honor partial-file resume')
                with path.open('ab' if offset else 'wb') as handle:
                    while chunk:=response.read(4*1024**2):handle.write(chunk)
        except Exception as error:
            # Redirected signed URLs/cookies never belong in public logs.
            print(f'{name}: attempt {attempt+1} failed ({type(error).__name__})',flush=True)
            if attempt==4:raise RuntimeError(f'{name}: publisher download failed after bounded retry') from None
    if path.stat().st_size!=size:raise RuntimeError(f'{name}: downloaded size mismatch')
    with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual!=expected:raise RuntimeError(f'{name}: pinned publisher SHA256 mismatch')
    print(f'{name}: SHA256 verified',flush=True)
    return {'file':name,'bytes':size,'sha256':actual,'elapsed_s':time.time()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=3:parser.error('--workers must be 1 to 3')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        files=list(executor.map(lambda spec:fetch(spec,args.output_dir,args.verify_only),FILES))
    receipt={'official_model':REPO,'official_revision':REVISION,'publisher_url':f'https://huggingface.co/{REPO}/tree/{REVISION}',
             'model_license':'CC BY-NC 4.0; retain publisher model-card terms','files':files}
    (args.output_dir/'download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
