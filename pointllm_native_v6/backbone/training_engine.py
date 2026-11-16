"""Shared actual PointLLM prefix training, restore and generation functions."""
from __future__ import annotations
import gc,hashlib,json,math,time
import random
from dataclasses import asdict
from pathlib import Path
import torch
try:
    from .adapter import AdapterConfig,PointLLMTokenAdapter,compute_auxiliary_losses
    from .runtime import parameter_fingerprint
    from .cached_data import sha,canonical_hash
except ImportError:
    from adapter import AdapterConfig,PointLLMTokenAdapter,compute_auxiliary_losses
    from runtime import parameter_fingerprint
    from cached_data import sha,canonical_hash


def create_adapter(dataset,variant,device):
    if variant=='plain_projector':return None
    if variant!='rahumpointllm':raise ValueError('Unknown matched backbone variant')
    config=dict(dataset.spec['adapter_config'])
    if config.get('groups',96)!=96 or config.get('token_dim',384)!=384:raise ValueError('Native PointLLM adapter budget/dimension mismatch')
    return PointLLMTokenAdapter(AdapterConfig(**config)).to(device)


def forward_tokens(variant,adapter,radar,reference,context):
    if variant=='plain_projector':
        # No reference, reference-category auxiliary head or semantic classifier
        # is evaluated by the plain native PointLLM projector baseline.
        return radar[:,-1],None
    output=adapter(radar,reference,context)
    return output['corrected_tokens'],output


def gradient_norms(module):
    if module is None:return {}
    result={name:float(p.grad.detach().float().norm()) for name,p in module.named_parameters() if p.grad is not None}
    if not all(math.isfinite(value) for value in result.values()):raise RuntimeError('Nonfinite parameter gradient')
    return result


def grouped_gradients(values):
    groups={'discrepancy_network':['discrepancy.'],'utility':['utility.'],'feature_expert_MLPs':['experts.'],'expert_router':['expert_router.'],'output_residual':['output_residual.','cls_residual.'],'rgb_global_reconstruction':['rgb_global_reconstruction.'],'rgb_global_residual':['rgb_global_residual.'],'all_adapter':['']}
    return {key:math.sqrt(sum(value*value for name,value in values.items() if any(name.startswith(prefix) for prefix in prefixes))) for key,prefixes in groups.items()}


def auxiliary_targets(target,device):
    category=target.get('category_id')
    verified=bool(target.get('category_verified',False)) and category is not None
    result={'category_ids':torch.tensor([category if category is not None else 0],device=device,dtype=torch.long),
            'category_mask':torch.tensor([verified and target.get('corrected_category_mask',True)],device=device),
            'radar_category_mask':torch.tensor([verified and target.get('radar_category_mask',False)],device=device),
            'reference_query_category_mask':torch.tensor([verified and target.get('reference_query_category_mask',False)],device=device)}
    # A full-scene multi-object nuScenes annotation is not broadcast as a
    # single category label to every radar or reference query.
    if target.get('support_targets') is not None:
        if not target.get('independent_support_annotation_source'):raise ValueError('Physical group support targets need their own independent truth source')
        result['support_targets']=torch.tensor(target['support_targets'],device=device,dtype=torch.float32)
        result['support_mask']=torch.tensor(target['support_mask'],device=device,dtype=torch.bool)
    return result


def actual_language_step(runtime,adapter,variant,optimizer,radar,reference,context,question,target,*,aux_weight=0.,step=1,progress=None):
    """LM-only backward evidence is collected before any auxiliary gradients."""
    progress=progress or (lambda event:None);started=time.time()
    optimizer.zero_grad(set_to_none=True);runtime.enable_training_path()
    if adapter is not None:adapter.train()
    sample=runtime.build_language_sample(question,target['response'])
    corrected,output=forward_tokens(variant,adapter,radar,reference,context)
    progress({'stage':'actual_LM_forward','step':step,'point_tokens':corrected.shape[1]})
    lm,projected=runtime.language_forward(corrected,sample)
    if not projected.requires_grad:raise RuntimeError('Real language loss has no trainable prefix/projector graph')
    if not bool(torch.isfinite(lm.loss)):raise RuntimeError('Nonfinite actual language CE')
    aux={}
    if variant=='rahumpointllm' and aux_weight>0:aux=compute_auxiliary_losses(output,auxiliary_targets(target,runtime.device))
    loss_value=float(lm.loss.detach());aux_values={key:float(value.detach()) for key,value in aux.items()}
    progress({'stage':'actual_LM_backward_only','step':step,'loss':loss_value})
    lm.loss.backward(retain_graph=bool(aux))
    adapter_grads=gradient_norms(adapter);projector_grads=gradient_norms(runtime.projector)
    if variant=='rahumpointllm' and not any(value>0 for value in adapter_grads.values()):raise RuntimeError('LM CE did not reach actual adapter parameters')
    if any(p.requires_grad for p in runtime.projector.parameters()) and not any(value>0 for value in projector_grads.values()):raise RuntimeError('LM CE did not reach trainable official projector')
    if any(p.grad is not None for module in [runtime.language,runtime.point_encoder] for p in module.parameters()):raise RuntimeError('Frozen 7B/PointBERT parameters received gradients')
    if adapter is not None and any(p.grad is not None for name,p in adapter.named_parameters() if name.startswith(('category_head.','radar_category_head.','reference_query_category_head.'))):raise RuntimeError('Auxiliary classifier gradients contaminated LM-only evidence')
    diagnostic={'step':step,'language_loss':loss_value,'auxiliary_losses':aux_values,'pure_LM_adapter_gradients':adapter_grads,'pure_LM_projector_gradients':projector_grads,'pure_LM_gradient_groups':grouped_gradients(adapter_grads),'full_input_length':sample['input_ids'].shape[1],'supervised_response_tokens':int((sample['labels']!=-100).sum())}
    if output is not None:
        diagnostic.update({'physical_G_logprior_std':float(output['log_geometric_prior'].detach().std()),'temporal_feature_norm':float(output['temporal_features'].detach().norm())})
    if aux:
        progress({'stage':'separate_auxiliary_backward','step':step});(aux_weight*sum(aux.values())).backward()
    optimizer.step();runtime.require_canonical_quantization()
    diagnostic['elapsed_s']=time.time()-started
    del lm,projected,corrected,output,aux;gc.collect()
    return diagnostic


def save_adaptation(path,runtime,adapter,variant,metadata,optimizer=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Normalize string subclasses such as torch.TorchVersion to plain JSON
    # primitives so weights_only checkpoint restores need no custom objects.
    meta=json.loads(json.dumps(metadata));official=json.loads(json.dumps(runtime.receipt))
    state={'schema_version':1,'variant':variant,'adapter_config':asdict(adapter.config) if adapter is not None else None,'adapter_state_dict':adapter.state_dict() if adapter is not None else None,'projector_state_dict':runtime.projector.state_dict(),'metadata':meta,'official_model':official,'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,'torch_rng_state':torch.random.get_rng_state(),'python_rng_state':random.getstate(),'cuda_rng_states':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
    return sha(path)


def restore_adaptation(path,runtime,*,expected_variant=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('schema_version')!=1:raise ValueError('Unknown reusable training checkpoint schema; engineering probe files use their dedicated restore script')
    variant=checkpoint['variant']
    if expected_variant is not None and expected_variant!=variant:raise ValueError('Checkpoint/model variant mismatch')
    if checkpoint['official_model']['official_revision']!=runtime.receipt['official_revision'] or checkpoint['official_model']['point_tokens']!=97:raise ValueError('Checkpoint baseline/model backbone mismatch')
    runtime.projector.load_state_dict(checkpoint['projector_state_dict'],strict=True)
    adapter=None
    if variant=='rahumpointllm':
        adapter=PointLLMTokenAdapter(AdapterConfig(**checkpoint['adapter_config'])).to(runtime.device)
        adapter.load_state_dict(checkpoint['adapter_state_dict'],strict=True)
    elif variant!='plain_projector':raise ValueError('Unknown restored variant')
    return adapter,checkpoint


def generation_protocol_hash(protocol_file,*,max_new_tokens,seed):
    return canonical_hash({'method_contract_sha256':sha(protocol_file),'native_point_token_budget':97,'greedy':True,'max_new_tokens':max_new_tokens,'seed':seed})


def generate_record(runtime,adapter,variant,dataset,record,*,checkpoint_sha256,protocol_sha256,seed,max_new_tokens):
    # Deliberately no dataset.targets call or sidecar content read here.
    result={'case_id':record['case_id'],'model_id':variant,'seed':seed,'split':record['split'],'scene_id':record['scene_id'],'checkpoint_sha256':checkpoint_sha256,'protocol_sha256':protocol_sha256,'prompt':record['prompt'],'source_case_sha256':dataset.source_case_sha(record),'point_token_budget':97,'ground_truth_sidecar_read_for_generation':False}
    try:
        if not record.get('input_available',True):raise ValueError('Recorded input unavailable; no synthetic point cloud or future sensor fill is constructed')
        radar,reference,context=dataset.features(record,runtime.device)
        if adapter is not None:adapter.eval()
        with torch.no_grad():corrected,_=forward_tokens(variant,adapter,radar,reference,context)
        generated=runtime.generate(corrected,record['prompt'],seed=seed,max_new_tokens=max_new_tokens)
        result.update(generated);result['execution_status']='generated';result['raw_response_sha256']=hashlib.sha256(result['raw_response'].encode()).hexdigest()
    except Exception as error:
        result.update({'execution_status':'input_or_generation_error','raw_response':'','generated_token_ids':[],'error_type':type(error).__name__,'error':str(error),'max_new_tokens':max_new_tokens})
    return result
