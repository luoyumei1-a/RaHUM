"""Finite actual PointLLM adapter/projector training on real encoder caches.

No default 50-epoch run, test-based selection or classifier-to-text substitution.
Every update uses actual response-token CE through the complete frozen 7B.
"""
import argparse,json,random,resource,time,traceback
from pathlib import Path
from dataclasses import asdict

SOURCE_NAMES=['runtime.py','adapter.py','cached_data.py','training_engine.py','train_pointllm.py']


def validate_resume_metadata(state,*,variant,adapter_config,metadata,source_sha256,projector_trainable,max_steps):
    """Pure preflight: reject a changed experiment before opening an output dir."""
    if state.get('schema_version')!=1 or state.get('variant')!=variant:raise ValueError('Resume checkpoint schema/variant mismatch')
    if state.get('adapter_config')!=adapter_config:raise ValueError('Resume adapter config changed')
    if state.get('optimizer_state_dict') is None:raise ValueError('Inference-only exports cannot resume optimizer state')
    meta=state.get('metadata',{})
    for key,value in metadata.items():
        if meta.get(key)!=value:raise ValueError('Resume contract changed: '+key)
    if meta.get('source_sha256')!=source_sha256:raise ValueError('Resume implementation source changed; use the checkpoint frozen source')
    saved_trainable=meta.get('projector_trainable',state.get('official_model',{}).get('projector_trainable'))
    if saved_trainable!=projector_trainable:raise ValueError('Resume projector freeze/train status changed')
    completed=meta.get('completed_steps')
    if not isinstance(completed,int) or not 0<=completed<=max_steps:raise ValueError('Resume completed-step budget is invalid or exceeds --max-steps')
    for field in ['torch_rng_state','python_rng_state']:
        if state.get(field) is None:raise ValueError('Resume random state missing: '+field)
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--language-cache-dir',type=Path)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variant',choices=['plain_projector','rahumpointllm'],required=True)
    p.add_argument('--max-steps',type=int,required=True)
    p.add_argument('--seed',type=int,default=7588)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--aux-weight',type=float,default=0.)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--freeze-projector',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--export-inference',action='store_true')
    a=p.parse_args()
    if a.max_steps<1:raise SystemExit('--max-steps must be a positive explicitly chosen budget')
    if a.variant=='plain_projector' and (a.aux_weight!=0 or a.freeze_projector):raise SystemExit('Plain baseline trains only the same LM CE through its official projector')
    if a.output_dir.exists():raise SystemExit('--output-dir must be a new directory; for --resume select a new output directory so previous receipts/checkpoints are not overwritten')
    resume_preflight=None
    if a.resume:
        import torch
        try:
            from .cached_data import CachedRealDataset,sha
            from .adapter import AdapterConfig
        except ImportError:
            from cached_data import CachedRealDataset,sha
            from adapter import AdapterConfig
        pre_dataset=CachedRealDataset(a.dataset)
        pre_sources={name:sha(Path(__file__).resolve().parent/name) for name in SOURCE_NAMES}
        pre_state=torch.load(a.resume,map_location='cpu',weights_only=True)
        completed=validate_resume_metadata(pre_state,variant=a.variant,
            adapter_config=asdict(AdapterConfig(**pre_dataset.spec['adapter_config'])) if a.variant=='rahumpointllm' else None,
            metadata={'dataset_manifest_sha256':sha(a.dataset),'method_contract_sha256':sha(a.protocol),'seed':a.seed,'learning_rate':a.learning_rate,'aux_weight':a.aux_weight},
            source_sha256=pre_sources,projector_trainable=not a.freeze_projector,max_steps=a.max_steps)
        resume_preflight={'resume_from_sha256':sha(a.resume),'resumed_completed_steps':completed,'resume_guards':'source/config/variant/projector/dataset/protocol/optimizer/RNG/step budget checked before output creation'}
        del pre_state,pre_dataset
    a.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.time();receipt={'status':'started','scope':'Finite real-data PointLLM language-objective training; not a held-out hallucination metric','variant':a.variant,'max_steps':a.max_steps,'step_records':[],'selection_policy':'fixed_update_budget_latest; no test loss or test outputs select a checkpoint'}
    if resume_preflight:receipt.update(resume_preflight)
    def save():
        receipt['elapsed_s']=time.time()-started;receipt['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temp=a.output_dir/'training_receipt.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(a.output_dir/'training_receipt.json')
    def progress(event):print(json.dumps({**event,'elapsed_s':time.time()-started}),flush=True);receipt['stage']=event.get('stage',receipt.get('stage'));save()
    save()
    try:
        import torch
        try:
            from .runtime import PointLLMBackboneRuntime
            from .cached_data import CachedRealDataset,sha
            from .training_engine import create_adapter,actual_language_step,save_adaptation,restore_adaptation
        except ImportError:
            from runtime import PointLLMBackboneRuntime
            from cached_data import CachedRealDataset,sha
            from training_engine import create_adapter,actual_language_step,save_adaptation,restore_adaptation
        dataset=CachedRealDataset(a.dataset);train=sorted(dataset.split('train'),key=lambda r:r['case_id'])
        train=[r for r in train if r.get('input_available',True)]
        if not train:raise ValueError('No actual available training cases; missing inputs are not fabricated')
        protocol=json.loads(a.protocol.read_text());protocol_sha=sha(a.protocol)
        source_paths={name:Path(__file__).resolve().parent/name for name in SOURCE_NAMES}
        sources={name:sha(path) for name,path in source_paths.items()}
        receipt.update({'dataset_manifest_sha256':sha(a.dataset),'method_contract_sha256':protocol_sha,'seed':a.seed,'learning_rate':a.learning_rate,'aux_weight':a.aux_weight,'available_train_cases':len(train),'train_scenes':sorted({r['scene_id'] for r in train}),'input_target_sidecars_separate':True,'source_sha256':sources})
        # Validate source binding before paying for a 7B load. This read is a
        # training-target operation, not a generation operation.
        for record in train:dataset.targets(record)
        torch.manual_seed(a.seed)
        runtime=PointLLMBackboneRuntime(a.checkpoint_dir,groups=96,threads=a.threads,device=a.device,train_projector=not a.freeze_projector,language_cache_dir=a.language_cache_dir,progress=progress)
        adapter=create_adapter(dataset,a.variant,runtime.device)
        offset=0;resume_state=None
        if a.resume:
            if sha(a.resume)!=resume_preflight['resume_from_sha256']:raise ValueError('Resume checkpoint changed after preflight')
            adapter,resume_state=restore_adaptation(a.resume,runtime,expected_variant=a.variant)
            meta=resume_state['metadata']
            for key,expected in [('dataset_manifest_sha256',sha(a.dataset)),('method_contract_sha256',protocol_sha),('seed',a.seed),('learning_rate',a.learning_rate),('aux_weight',a.aux_weight)]:
                if meta.get(key)!=expected:raise ValueError('Resume contract changed: '+key)
            offset=int(meta['completed_steps'])
        parameters=([p for p in runtime.projector.parameters() if p.requires_grad]+([] if adapter is None else [p for p in adapter.parameters() if p.requires_grad]))
        if not parameters:raise ValueError('No trainable prefix/projector parameters')
        optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate)
        if resume_state is not None:
            if resume_state.get('optimizer_state_dict') is None:raise ValueError('An inference-only export cannot promise exact optimizer resume')
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            torch.random.set_rng_state(resume_state['torch_rng_state']);random.setstate(resume_state['python_rng_state'])
            if resume_state.get('cuda_rng_states') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(resume_state['cuda_rng_states'])
        frozen_before=runtime.frozen_fingerprint();receipt['official_model']=runtime.receipt
        receipt['trainable_parameter_count']=sum(p.numel() for p in parameters)
        latest=a.output_dir/'latest.pt'
        if resume_state is not None and offset==a.max_steps:
            receipt['latest_checkpoint_sha256']=save_adaptation(latest,runtime,adapter,a.variant,resume_state['metadata'],optimizer)
        for step in range(offset,a.max_steps):
            epoch=step//len(train);order=list(train);random.Random(a.seed+epoch).shuffle(order);record=order[step%len(train)]
            radar,reference,context=dataset.features(record,runtime.device);target=dataset.targets(record)
            diagnostic=actual_language_step(runtime,adapter,a.variant,optimizer,radar,reference,context,record['prompt'],target,aux_weight=a.aux_weight,step=step+1,progress=progress)
            diagnostic['case_id']=record['case_id'];diagnostic['source_case_sha256']=dataset.source_case_sha(record)
            receipt['step_records'].append(diagnostic)
            meta={'completed_steps':step+1,'dataset_manifest_sha256':sha(a.dataset),'method_contract_sha256':protocol_sha,'seed':a.seed,'learning_rate':a.learning_rate,'aux_weight':a.aux_weight,'projector_trainable':not a.freeze_projector,'max_steps_requested':a.max_steps,'case_schedule':'sorted cases, per-epoch random.Random(seed+epoch).shuffle; batchsize1','latest_step':diagnostic,'source_sha256':sources,'checkpoint_selection':'latest fixed budget, never test-selected','engineering_scope':dataset.spec.get('scope')}
            receipt['latest_checkpoint_sha256']=save_adaptation(latest,runtime,adapter,a.variant,meta,optimizer)
            with (a.output_dir/'training_steps.jsonl').open('a') as f:f.write(json.dumps(diagnostic)+'\n')
            save()
        if runtime.frozen_fingerprint()!=frozen_before:raise RuntimeError('Frozen official 7B/PointBERT weights or quantstates changed')
        if {name:sha(path) for name,path in source_paths.items()}!=sources:raise RuntimeError('Implementation source changed during this training run')
        # The engineering RGB smoke may request a concrete per-module LM-only
        # connectivity check; this is not an accuracy or generalization claim.
        required=dataset.spec.get('required_second_step_LM_gradient_groups',[]) if a.variant=='rahumpointllm' else []
        if required:
            latest_diagnostic=receipt['step_records'][-1] if receipt['step_records'] else (resume_state or {}).get('metadata',{}).get('latest_step')
            if not latest_diagnostic or latest_diagnostic['step']<2:raise ValueError('The identity-initialized adapter needs at least two real steps for the requested internal-gradient check')
            groups=latest_diagnostic['pure_LM_gradient_groups']
            missing=[name for name in required if not (groups.get(name,0)>0)]
            if missing:raise RuntimeError('Actual LM CE did not reach requested modules: '+str(missing))
        receipt['frozen_language_pointbert_quantstates_unchanged']=True;receipt['implementation_sources_unchanged']=True
        receipt['updates_executed_this_run']=len(receipt['step_records'])
        if a.export_inference:
            loaded=torch.load(latest,map_location='cpu',weights_only=True)
            receipt['inference_checkpoint_sha256']=save_adaptation(a.output_dir/'inference.pt',runtime,adapter,a.variant,loaded['metadata'],optimizer=None)
        receipt['status']='completed_finite_training_no_heldout_performance_claim';progress({'stage':'finite_actual_training_completed','steps':a.max_steps,'status':receipt['status']})
    except Exception as error:
        receipt.update({'status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()});save();print(json.dumps(receipt,indent=2),flush=True);raise
    finally:save()


if __name__=='__main__':main()
