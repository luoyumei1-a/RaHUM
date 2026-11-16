"""Faithful official PointLLM-7B-v1.2 inference with NF4 Llama on CPU.

The official PointBERT/projector weights are loaded strictly. Its 513 feature
vectors replace the same <point_patch> token positions as the upstream model.
Llama uses the official complete fine-tuned checkpoint with CPU NF4 quantization;
this is a disclosed execution condition, not the author's full-precision score.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,platform,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from easydict import EasyDict
from transformers import BitsAndBytesConfig,LlamaConfig,LlamaForCausalLM,LlamaTokenizer,TextStreamer
if __package__:
 from .vendor_pointbert import PointTransformer
else:
 from vendor_pointbert import PointTransformer
ROOT=Path(__file__).resolve().parent
OFFICIAL_REPO='RunsenXu/PointLLM_7B_v1.2'
OFFICIAL_REV='37d8c15aaff8e7e05f04729dcb6960d5758e9f86'
SYSTEM="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

def sha256(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare_cloud(arr,*,seed=7588,npoints=8192,geometry_only=False):
 arr=np.asarray(arr)
 if arr.ndim!=2 or arr.shape[1]<3:raise ValueError('Expected measured Nx3, Nx5 radar, or Nx6 XYZRGB array.')
 finite=np.isfinite(arr[:,:3]).all(1);original_n=len(arr);arr=arr[finite]
 if not len(arr):raise ValueError('No finite measured XYZ rows.')
 rng=np.random.default_rng(seed)
 # A fixed traceable measured-row sample, with replacement only when sparse.
 idx=rng.choice(len(arr),size=npoints,replace=len(arr)<npoints)
 measured=np.asarray(arr[idx],dtype=np.float32)
 xyz=measured[:,:3].copy();centroid=xyz.mean(0);xyz-=centroid
 scale=float(np.linalg.norm(xyz,axis=1).max())
 if scale<=1e-8:raise ValueError('Degenerate measured cloud cannot be unit-sphere normalized.')
 xyz/=scale
 if arr.shape[1]==6 and not geometry_only:
  rgb=measured[:,3:6].copy()
  if rgb.min()<0 or rgb.max()>1:raise ValueError('Measured RGB must already be in [0,1]; no attribute reinterpretation.')
  color_mode='measured_rgb'
 else:
  rgb=np.zeros_like(xyz);color_mode='missing_rgb_zero_channels_as_official_ModelNet_geometry_condition'
 pc=np.concatenate([xyz,rgb],axis=1)
 receipt={'original_rows':original_n,'finite_measured_rows':len(arr),'model_rows':npoints,'replacement_sampling':len(arr)<npoints,'measured_index_sha256':hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(),'normalization_centroid':centroid.tolist(),'normalization_radius':scale,'color_mode':color_mode,'input_sha256':hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),'random_seed':seed}
 return torch.from_numpy(pc).unsqueeze(0),receipt

class OfficialPointLLMCPU:
 def __init__(self,model_dir,threads=2):
  torch.set_num_threads(threads);torch.set_num_interop_threads(1)
  self.model_dir=Path(model_dir);c=json.loads((self.model_dir/'config.json').read_text())
  self.tokenizer=LlamaTokenizer.from_pretrained(self.model_dir,legacy=True)
  added=self.tokenizer.add_tokens(['<point_patch>','<point_start>','<point_end>'],special_tokens=True)
  self.ids={t:self.tokenizer.convert_tokens_to_ids(t) for t in ['<point_patch>','<point_start>','<point_end>']}
  assert self.ids=={'<point_patch>':32000,'<point_start>':32001,'<point_end>':32002},self.ids
  cfg=EasyDict(trans_dim=384,depth=12,drop_path_rate=.1,cls_dim=40,num_heads=6,group_size=32,num_group=512,encoder_dims=256,point_dims=6,projection_hidden_layer=2,projection_hidden_dim=[1024,2048],use_max_pool=False)
  self.backbone=PointTransformer(cfg,use_max_pool=False).eval()
  self.projector=torch.nn.Sequential(torch.nn.Linear(384,1024),torch.nn.GELU(),torch.nn.Linear(1024,2048),torch.nn.GELU(),torch.nn.Linear(2048,4096)).eval()
  wmap=json.loads((self.model_dir/'pytorch_model.bin.index.json').read_text())['weight_map']
  relevant={k:v for k,v in wmap.items() if k.startswith('model.point_backbone.') or k.startswith('model.point_proj.')}
  pstate={};bstate={}
  for shard_name in sorted(set(relevant.values())):
   state=torch.load(self.model_dir/shard_name,map_location='cpu',mmap=True,weights_only=True)
   for key in [k for k,v in relevant.items() if v==shard_name]:
    if key.startswith('model.point_backbone.'):bstate[key.removeprefix('model.point_backbone.')]=state[key]
    else:pstate[key.removeprefix('model.point_proj.')]=state[key]
   del state
  self.backbone.load_state_dict(bstate,strict=True);self.projector.load_state_dict(pstate,strict=True)
  del bstate,pstate;gc.collect()
  # The exact official Llama parameter names match HF. Non-Llama weights are
  # intentionally loaded above and reported as unexpected by the Llama reader.
  lc=LlamaConfig(**c);lc.model_type='llama';lc.architectures=['LlamaForCausalLM'];lc.use_cache=True
  quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=False)
  self.language,info=LlamaForCausalLM.from_pretrained(self.model_dir,config=lc,quantization_config=quant,device_map={'':'cpu'},dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',output_loading_info=True)
  self.language.eval()
  if info.get('missing_keys'):raise RuntimeError(f"Unloaded language parameters: {info['missing_keys']}")
  unexpected=[k for k in info.get('unexpected_keys',[]) if not (k.startswith('model.point_backbone.') or k.startswith('model.point_proj.') or k.endswith('.rotary_emb.inv_freq'))]
  if unexpected:raise RuntimeError(f'Unexpected unaccounted weights: {unexpected}')
  self.load_receipt={'official_model':OFFICIAL_REPO,'official_revision':OFFICIAL_REV,'source_config_sha256':sha256(self.model_dir/'config.json'),'point_backbone_parameter_count':sum(p.numel() for p in self.backbone.parameters()),'projector_parameter_count':sum(p.numel() for p in self.projector.parameters()),'point_state_strict':True,'language_missing_keys':info.get('missing_keys',[]),'non_language_or_legacy_buffer_keys':len(info.get('unexpected_keys',[])),'language_quantization':'bitsandbytes NF4 4-bit weights, BF16 compute, CPU','point_compute':'FP32 CPU','torch_version':torch.__version__,'threads':threads,'platform':platform.platform()}

 @torch.inference_mode()
 def generate(self,cloud,question,seed=7588,max_new_tokens=80):
  torch.manual_seed(seed)
  point_started=time.time()
  features=self.projector(self.backbone(cloud.to(torch.float32)))
  point_encode_s=time.time()-point_started
  assert tuple(features.shape)==(1,513,4096),features.shape
  query='<point_start>'+'<point_patch>'*513+'<point_end>\n'+question
  prompt=SYSTEM+' USER: '+query+' ASSISTANT:'
  enc=self.tokenizer(prompt,return_tensors='pt');input_ids=enc.input_ids
  positions=(input_ids[0]==self.ids['<point_patch>']).nonzero().flatten()
  assert len(positions)==513
  emb=self.language.get_input_embeddings()(input_ids)
  emb[0,positions,:]=features[0].to(emb.dtype)
  started=time.time()
  print(json.dumps({'event':'language_generation_started','point_encode_s':point_encode_s,'prompt_tokens':input_ids.shape[1],'max_new_tokens':max_new_tokens}),flush=True)
  streamer=TextStreamer(self.tokenizer,skip_prompt=True,skip_special_tokens=True)
  out=self.language.generate(input_ids=input_ids,inputs_embeds=emb,attention_mask=enc.attention_mask,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,pad_token_id=0,eos_token_id=2,streamer=streamer)
  generated=out[0,input_ids.shape[1]:]
  response=self.tokenizer.decode(generated,skip_special_tokens=True).strip()
  return {'prompt':question,'conversation_template':'official vicuna_v1_1','raw_response':response,'generated_token_ids':generated.tolist(),'generated_tokens':len(generated),'prompt_tokens':input_ids.shape[1],'point_tokens':513,'point_encode_s':point_encode_s,'language_generation_s':time.time()-started,'seed':seed,'max_new_tokens':max_new_tokens,'decoding':'greedy','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'point_feature_sha256':hashlib.sha256(features.numpy().tobytes()).hexdigest()}

def read_cloud(p,key=None):
 p=Path(p)
 if p.suffix=='.npy':return np.load(p,allow_pickle=False)
 if p.suffix=='.npz':
  z=np.load(p,allow_pickle=False)
  if key:return z[key]
  for k in ['points','xyz','radar_xyz','pc_xyziv_ti','point_cloud']:
   if k in z:return z[k]
  raise KeyError('Specify --key for NPZ cloud.')
 if p.suffix=='.mat':
  from scipy.io import loadmat
  z=loadmat(p)
  return z[key or 'pc_xyziv_ti']
 raise ValueError('Supported measured cloud containers: .npy/.npz/.mat')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,default=ROOT/'checkpoints/PointLLM_7B_v1.2');ap.add_argument('--cloud',type=Path,required=True);ap.add_argument('--key');ap.add_argument('--question',default='What object is represented by this point cloud? Describe only what its geometry supports.');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,default=7588);ap.add_argument('--max-new-tokens',type=int,default=64);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--geometry-only',action='store_true');args=ap.parse_args()
 arr=read_cloud(args.cloud,args.key);cloud,preprocess=prepare_cloud(arr,seed=args.seed,geometry_only=args.geometry_only)
 t=time.time();model=OfficialPointLLMCPU(args.model_dir,args.threads);load_s=time.time()-t
 result=model.generate(cloud,args.question,seed=args.seed,max_new_tokens=args.max_new_tokens)
 result.update({'source_cloud':str(args.cloud.resolve()),'source_file_sha256':sha256(args.cloud),'preprocessing':preprocess,'model_load_s':load_s,'model':model.load_receipt,'scope':'actual model generation; no claim correctness or hallucination rate is inferred without independently annotated ground truth'})
 args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
