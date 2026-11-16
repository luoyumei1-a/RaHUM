"""Actual official PointLLM components with a trainable sensory-prefix path.

The 7B language parameters are frozen NF4 weights. Grad-enabled language CE
still differentiates into the corrected point tokens and optional projector.
Native PointBERT group count is adapted to 96 (97 tokens including CLS), with
strict preservation of every pretrained learned tensor. No point classifier
or generated-answer rewrite stands in for the language decoder.
"""
from __future__ import annotations
import contextlib,gc,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from easydict import EasyDict
from transformers import BitsAndBytesConfig,LlamaConfig,LlamaForCausalLM,LlamaTokenizer
try:
    from .vendor_pointbert import PointTransformer
    from .vendor_pointbert.dvae import Group,knn_point
    from .vendor_pointbert import misc
except ImportError:
    from vendor_pointbert import PointTransformer
    from vendor_pointbert.dvae import Group,knn_point
    from vendor_pointbert import misc

MODEL_REPO='RunsenXu/PointLLM_7B_v1.2'
MODEL_REVISION='37d8c15aaff8e7e05f04729dcb6960d5758e9f86'
SYSTEM='A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'


def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tensor_hash_update(h,tensor):
    t=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    h.update(memoryview(t.numpy()).cast('B'))


def parameter_fingerprint(module):
    h=hashlib.sha256()
    for name,value in module.named_parameters():
        h.update((name+str(tuple(value.shape))+str(value.dtype)).encode());tensor_hash_update(h,value)
    return h.hexdigest()


class CapturingGroup(Group):
    """Same official FPS/KNN operations, recording the actual neighbor indices.

    This adds no parameters and does not change the returned point-group data.
    The recorded indices tie physical attributes/coordinates to actual tokens.
    """
    def forward(self,data):
        batch,n,channels=data.shape
        xyz=data[:,:,:3]
        center=misc.fps(xyz,self.num_group)
        idx=knn_point(self.group_size,xyz,center)
        self.last_center=center.detach()
        self.last_neighbor_indices=idx.detach()
        base=torch.arange(batch,device=xyz.device).view(-1,1,1)*n
        flat=(idx+base).reshape(-1)
        group_xyz=xyz.reshape(batch*n,3)[flat].reshape(batch,self.num_group,self.group_size,3).contiguous()-center.unsqueeze(2)
        if channels>3:
            other=data[:,:,3:].reshape(batch*n,channels-3)[flat].reshape(batch,self.num_group,self.group_size,channels-3).contiguous()
            group=torch.cat([group_xyz,other],dim=-1)
        else:group=group_xyz
        return group,center


class PointLLMBackboneRuntime:
    def __init__(self,checkpoint_dir,*,groups=96,threads=2,device='cpu',train_projector=True,language_cache_dir=None,progress=None):
        self.checkpoint_dir=Path(checkpoint_dir);self.groups=groups;self.point_tokens=groups+1
        self.device=torch.device(device);self.progress=progress or (lambda event:None)
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA was requested but is not available; no CPU stand-in is substituted.')
        torch.set_num_threads(threads);torch.set_num_interop_threads(1)
        cfg_json=json.loads((self.checkpoint_dir/'config.json').read_text())
        self.tokenizer=LlamaTokenizer.from_pretrained(self.checkpoint_dir,legacy=True)
        self.tokenizer.add_tokens(['<point_patch>','<point_start>','<point_end>'],special_tokens=True)
        self.ids={s:self.tokenizer.convert_tokens_to_ids(s) for s in ['<point_patch>','<point_start>','<point_end>']}
        assert self.ids=={'<point_patch>':32000,'<point_start>':32001,'<point_end>':32002}
        pcfg=EasyDict(trans_dim=384,depth=12,drop_path_rate=.1,cls_dim=40,num_heads=6,group_size=32,num_group=groups,encoder_dims=256,point_dims=6,projection_hidden_layer=2,projection_hidden_dim=[1024,2048],use_max_pool=False)
        self.point_encoder=PointTransformer(pcfg,use_max_pool=False)
        self.point_encoder.group_divider=CapturingGroup(groups,32)
        self.projector=nn.Sequential(nn.Linear(384,1024),nn.GELU(),nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,4096))
        mapping=json.loads((self.checkpoint_dir/'pytorch_model.bin.index.json').read_text())['weight_map']
        selected={k:v for k,v in mapping.items() if k.startswith('model.point_backbone.') or k.startswith('model.point_proj.')}
        bs,ps={},{}
        for name in sorted(set(selected.values())):
            state=torch.load(self.checkpoint_dir/name,map_location='cpu',mmap=True,weights_only=True)
            for key,shard in selected.items():
                if shard!=name:continue
                if key.startswith('model.point_backbone.'):bs[key.removeprefix('model.point_backbone.')]=state[key]
                else:ps[key.removeprefix('model.point_proj.')]=state[key]
            del state
        self.point_encoder.load_state_dict(bs,strict=True);self.projector.load_state_dict(ps,strict=True)
        del bs,ps;gc.collect()
        self.point_encoder.requires_grad_(False).eval().to(self.device)
        self.projector.requires_grad_(train_projector).to(self.device)
        lc=LlamaConfig(**cfg_json);lc.model_type='llama';lc.architectures=['LlamaForCausalLM'];lc.use_cache=False
        language_path=self.checkpoint_dir;cache_provenance=None
        if language_cache_dir is not None:
            language_path=Path(language_cache_dir)
            cache_provenance=json.loads((language_path/'official_canonical_cache_provenance.json').read_text())
            if cache_provenance.get('official_model')!=MODEL_REPO or cache_provenance.get('official_revision')!=MODEL_REVISION or cache_provenance.get('source_config_sha256')!=file_sha(self.checkpoint_dir/'config.json'):
                raise ValueError('Private canonical language cache has different official source provenance')
            cached=LlamaConfig.from_pretrained(language_path)
            for key in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','vocab_size']:
                if getattr(cached,key)!=getattr(lc,key):raise ValueError('Cached language architecture differs: '+key)
            if not getattr(cached,'quantization_config',None):raise ValueError('Private cache does not declare its NF4 quantization')
            lc=cached;lc.use_cache=False
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=False)
        self.progress({'stage':'official_7B_load_started','groups':groups,'point_tokens':self.point_tokens})
        started=time.time()
        self.language,loading=LlamaForCausalLM.from_pretrained(language_path,config=lc,quantization_config=quant,device_map={'':str(self.device)},dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',output_loading_info=True)
        if loading.get('missing_keys'):raise RuntimeError('Official language weights missing: '+str(loading['missing_keys']))
        unexpected=[k for k in loading.get('unexpected_keys',[]) if not (k.startswith('model.point_backbone.') or k.startswith('model.point_proj.') or k.endswith('.rotary_emb.inv_freq'))]
        if unexpected:raise RuntimeError('Unaccounted official checkpoint tensors: '+str(unexpected))
        self.language.requires_grad_(False)
        self.language.config.use_cache=False
        self.enable_training_path()
        self.require_canonical_quantization()
        if cache_provenance is not None:
            if self.frozen_fingerprint()!=cache_provenance['frozen_language_and_pointencoder_fingerprint']:
                raise RuntimeError('Reloaded private cache is not identical to the saved canonical frozen weights/quantstates')
        self.receipt={'official_model':MODEL_REPO,'official_revision':MODEL_REVISION,'source_config_sha256':file_sha(self.checkpoint_dir/'config.json'),'pointbert_strict_load':True,'projector_strict_load':True,'language_missing_keys':loading.get('missing_keys',[]),'adapted_point_groups':groups,'point_tokens':self.point_tokens,'official_default_point_groups':512,'point_input_channels':6,'missing_color':'zero channels, not measured color','point_encoder_parameter_count':sum(p.numel() for p in self.point_encoder.parameters()),'projector_parameter_count':sum(p.numel() for p in self.projector.parameters()),'language_frozen':True,'projector_trainable':train_projector,'language_precision':'canonical NF4 weights/BF16 compute','point_encoder_projector_precision':'FP32','language_load_s':time.time()-started,'torch':torch.__version__,'device':str(self.device),'threads':threads,'checkpointing':'non-reentrant; use_cache=False during training'}
        self.receipt['private_canonical_language_cache_used']=cache_provenance is not None
        if cache_provenance is not None:self.receipt['private_cache_provenance_sha256']=file_sha(language_path/'official_canonical_cache_provenance.json')
        self.progress({'stage':'official_7B_load_complete','seconds':self.receipt['language_load_s']})

    def enable_training_path(self):
        self.language.train()
        self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.language.config.use_cache=False
        if float(getattr(self.language.config,'attention_dropout',0.))!=0.:
            raise RuntimeError('The deterministic probe expects official zero attention dropout.')
        self.point_encoder.eval()

    def quant_modules(self):
        for name,module in self.language.named_modules():
            state=getattr(getattr(module,'weight',None),'quant_state',None)
            if state is not None:yield name,module,state

    def require_canonical_quantization(self):
        packed=[name for name,_,state in self.quant_modules() if getattr(state,'packing_format_for_cpu',False)]
        if packed:raise RuntimeError('CPU inference-packed quantization cannot be used for the trainable-prefix path: '+str(packed[:3]))

    def frozen_fingerprint(self):
        h=hashlib.sha256()
        for label,module in [('point_encoder',self.point_encoder),('language',self.language)]:
            for name,param in module.named_parameters():
                h.update((label+name+str(tuple(param.shape))+str(param.dtype)).encode());tensor_hash_update(h,param)
        def visit(state):
            for key in ['absmax','code','offset','blocksize','quant_type','dtype','shape']:
                value=getattr(state,key,None);h.update(key.encode())
                if torch.is_tensor(value):tensor_hash_update(h,value)
                else:h.update(str(value).encode())
            if getattr(state,'state2',None) is not None:visit(state.state2)
        for name,_,state in self.quant_modules():h.update(name.encode());visit(state)
        return h.hexdigest()

    @torch.no_grad()
    def encode_measured_points(self,points,*,attributes=None,source_row_ids=None,seed=7588,point_budget=96):
        arr=np.asarray(points,dtype=np.float32)
        if arr.ndim!=2 or arr.shape[1]<3:raise ValueError('Measured points must be an Nx3-or-more array.')
        finite=np.isfinite(arr[:,:3]).all(1);original_n=len(arr);finite_rows=np.nonzero(finite)[0];arr=arr[finite]
        if not len(arr):raise ValueError('No finite measured points')
        if source_row_ids is None:source_row_ids=np.arange(original_n,dtype=np.int64)
        original_ids=np.asarray(source_row_ids)[finite]
        attrs=np.asarray(attributes,dtype=np.float32)[finite] if attributes is not None else None
        if attrs is not None and (attrs.shape!=(len(arr),2) or not np.isfinite(attrs).all()):raise ValueError('Radar attributes must be finite dataset-declared physical channels Nx2; preserve their source schema/units.')
        rng=np.random.default_rng(seed);idx=rng.choice(len(arr),point_budget,replace=len(arr)<point_budget)
        measured=arr[idx,:3].copy();centroid=measured.mean(0);normalized=measured-centroid
        radius=float(np.linalg.norm(normalized,axis=1).max())
        if radius<=1e-8:raise ValueError('Measured cloud is geometrically degenerate.')
        normalized/=radius
        model_points=np.concatenate([normalized,np.zeros_like(normalized)],axis=-1)
        torch.manual_seed(seed)
        features=self.point_encoder(torch.from_numpy(model_points)[None].to(self.device))
        if tuple(features.shape)!=(1,self.point_tokens,384):raise RuntimeError('Native PointBERT token contract broken: '+str(features.shape))
        grouping=self.point_encoder.group_divider
        neighbor=grouping.last_neighbor_indices[0].cpu().numpy()
        centers=grouping.last_center[0].cpu().numpy()*radius+centroid
        grouped_source_ids=original_ids[idx][neighbor]
        pooled_attrs=attrs[idx][neighbor].mean(1) if attrs is not None else None
        metadata={'centers':torch.from_numpy(centers),'attributes':torch.from_numpy(pooled_attrs) if pooled_attrs is not None else None,'mask':torch.ones(self.groups,dtype=torch.bool),'group_source_row_ids':grouped_source_ids,'group_unique_source_counts':np.asarray([len(np.unique(row)) for row in grouped_source_ids]),'source_row_indices':finite_rows[idx],'neighbor_indices':neighbor,'normalization_centroid':centroid.tolist(),'normalization_radius':radius,'original_rows':original_n,'finite_measured_rows':len(arr),'distinct_source_rows_sampled':len(np.unique(original_ids[idx])),'model_rows':point_budget,'replacement_sampling':len(arr)<point_budget,'sampling_index_sha256':hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(),'feature_sha256':hashlib.sha256(features.cpu().numpy().tobytes()).hexdigest(),'measured_xyz_sha256':hashlib.sha256(np.ascontiguousarray(arr[:,:3]).tobytes()).hexdigest(),'seed':seed}
        return features.detach(),metadata

    def build_language_sample(self,question,response=None):
        query='<point_start>'+'<point_patch>'*self.point_tokens+'<point_end>\n'+question
        prompt=SYSTEM+' USER: '+query+' ASSISTANT:'
        prompt_ids=self.tokenizer(prompt,return_tensors='pt').input_ids
        if response is not None:
            response_ids=self.tokenizer(response,add_special_tokens=False,return_tensors='pt').input_ids
            ids=torch.cat([prompt_ids,response_ids,torch.tensor([[2]])],dim=1)
            labels=torch.full_like(ids,-100);labels[:,prompt_ids.shape[1]:]=ids[:,prompt_ids.shape[1]:]
        else:ids=prompt_ids;labels=None
        positions=(ids[0]==32000).nonzero().flatten()
        if len(positions)!=self.point_tokens or not torch.equal(positions,torch.arange(positions[0],positions[0]+self.point_tokens)):
            raise RuntimeError('Prompt patch tokens are not a consecutive native group-token span.')
        return {'input_ids':ids.to(self.device),'labels':labels.to(self.device) if labels is not None else None,'attention_mask':torch.ones_like(ids,device=self.device),'start':int(positions[0]),'end':int(positions[-1])+1,'prompt_length':prompt_ids.shape[1],'question':question,'response':response}

    def point_embeddings(self,corrected_tokens,sample):
        if corrected_tokens.ndim!=3 or tuple(corrected_tokens.shape[1:])!=(self.point_tokens,384):raise ValueError('Corrected tokens must preserve CLS+all native PointBERT groups.')
        base=self.language.get_input_embeddings()(sample['input_ids']).detach()
        projected=self.projector(corrected_tokens.to(torch.float32)).to(base.dtype)
        merged=torch.cat([base[:,:sample['start']],projected,base[:,sample['end']:]],dim=1)
        if merged.shape[:2]!=sample['input_ids'].shape:raise RuntimeError('Embedding/mask/label sequence lengths differ after point replacement.')
        return merged,projected

    def language_forward(self,corrected_tokens,sample):
        self.require_canonical_quantization()
        merged,projected=self.point_embeddings(corrected_tokens,sample)
        outputs=self.language(inputs_embeds=merged,attention_mask=sample['attention_mask'],labels=sample['labels'],use_cache=False,return_dict=True)
        return outputs,projected

    @contextlib.contextmanager
    def canonical_inference(self):
        was_training=self.language.training
        self.language.eval()
        # Only the quantized Linear modules stay in train mode; this blocks
        # irreversible CPU inference packing without enabling decoder dropout.
        for _,module,_ in self.quant_modules():module.train(True)
        self.require_canonical_quantization()
        try:yield
        finally:
            self.language.train(was_training)
            self.require_canonical_quantization()

    @torch.no_grad()
    def response_logits(self,corrected_tokens,sample,*,all_response_positions=False):
        with self.canonical_inference():
            output,_=self.language_forward(corrected_tokens,sample)
            if all_response_positions:result=output.logits[:,sample['prompt_length']-1:,:].detach().float().cpu().clone()
            else:result=output.logits[:,sample['prompt_length']-1,:].detach().float().cpu().clone()
            del output;return result

    @torch.no_grad()
    def generate(self,corrected_tokens,question,*,seed=7588,max_new_tokens=32):
        sample=self.build_language_sample(question)
        with self.canonical_inference():
            merged,_=self.point_embeddings(corrected_tokens,sample)
            torch.manual_seed(seed);started=time.time()
            output=self.language.generate(input_ids=sample['input_ids'],inputs_embeds=merged,attention_mask=sample['attention_mask'],do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,pad_token_id=0,eos_token_id=2)
            ids=output[0,sample['input_ids'].shape[1]:].cpu().tolist()
        return {'raw_response':self.tokenizer.decode(ids,skip_special_tokens=True).strip(),'generated_token_ids':ids,'question':question,'max_new_tokens':max_new_tokens,'seed':seed,'greedy':True,'point_tokens':self.point_tokens,'generation_s':time.time()-started,'model':self.receipt}
