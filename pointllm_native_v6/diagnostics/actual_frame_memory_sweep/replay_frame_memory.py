"""Strict fixed-checkpoint replay for frame-resolved, real-input K sensitivity.

No training, language decoding or semantic correctness evaluation is performed.
The original three-frame TRAIN feature window and two-update checkpoint are used.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

def sha(p):
 with open(p,'rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--release',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 r=a.release
 cpfile=r/'backbone/rgb_runner_01/inference.pt'
 cachefile=r/'backbone/nuscenes_real_rgb_probe/adapter_inputs_v1/actual_radar_rgb_adapter_inputs.npz'
 source=r/'backbone/rgb_runner_01_source_frozen/adapter.py'
 assert sha(cpfile)=='f39fc35d0d766c86485f3c630448e63554f7dac5b75a969da1ee7a2b6b2bfa76'
 assert sha(cachefile)=='01f2e9c765fb4efef8f4df8f77e70adfb6bf4c73f4cf5683139d7615520720a0'
 cp=torch.load(cpfile,map_location='cpu',weights_only=True)
 assert sha(source)==cp['metadata']['source_sha256']['adapter.py']
 spec=importlib.util.spec_from_file_location('rahum_native_v9_replay',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 with np.load(cachefile,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 radar=torch.from_numpy(arrays['radar_tokens']);ref=torch.from_numpy(arrays['reference_tokens'])
 context={k.removeprefix('context_'):torch.from_numpy(v) for k,v in arrays.items() if k.startswith('context_')}
 assert not any(any(t in k.lower() for t in ['label','category','ground_truth','claim']) for k in context)
 torch.set_num_threads(2);rows=[];aggregate=[]
 for k in [1,2,4,8,12,16,24,32]:
  cfg=dict(cp['adapter_config']);cfg['local_memory']=k
  model=mod.PointLLMTokenAdapter(mod.AdapterConfig(**cfg)).eval();model.load_state_dict(cp['adapter_state_dict'],strict=True)
  with torch.inference_mode():out=model(radar,ref,context)
  all_entropy=[]
  for t,fr in enumerate(out['sequence_outputs']):
   mask=context['radar_mask'][:,t].bool()&fr['reference_available']
   w=fr['utility_weights'][...,0][mask];att=fr['attention'][mask]
   assert torch.isfinite(att).all() and torch.isfinite(w).all()
   assert torch.allclose(att.sum(-1),torch.ones_like(w),atol=1e-6)
   all_entropy.extend((-(att*att.clamp_min(1e-30).log()).sum(-1)).tolist())
   rows.append({'local_K':k,'frame':t+1,'supported_groups':int(mask.sum()),'radar_utility_mean':float(w.mean()),'attention_entropy_nats_mean':float((-(att*att.clamp_min(1e-30).log()).sum(-1)).mean()),'attention_max_mean':float(att.amax(-1).mean()),'reference_attention_rank_mean':att.mean(0).tolist()})
  aggregate.append({'local_K':k,'supported_groups':len(all_entropy),'entropy_mean':float(np.mean(all_entropy)),'entropy_p10':float(np.quantile(all_entropy,.1)),'entropy_p90':float(np.quantile(all_entropy,.9))})
 receipt={'status':'completed_strict_restore_frame_resolved_memory_replay','scope':__doc__.strip(),'checkpoint_sha256':sha(cpfile),'feature_cache_sha256':sha(cachefile),'adapter_source_sha256':sha(source),'script_sha256':sha(__file__),'completed_training_updates':cp['metadata'].get('completed_steps'),'checkpoint_trained_local_K':cp['adapter_config']['local_memory'],'input_shape':list(radar.shape),'rows':rows,'aggregate':aggregate}
 a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'rows':len(rows),'output':str(a.output)}))
if __name__=='__main__':main()
