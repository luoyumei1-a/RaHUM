"""Restore a genuine PointLLM checkpoint and retain every raw generation.

The sensory manifest is read, but target sidecars are never opened here.
Claim judgments require a separate independently evidenced annotation file.
"""
from __future__ import annotations
import argparse,json,resource,time,traceback
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--language-cache-dir',type=Path)
    parser.add_argument('--adaptation-checkpoint',type=Path)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--variant',choices=['plain_projector','rahumpointllm','pretrained_native'],required=True)
    parser.add_argument('--split',choices=['train','validation','test'],required=True)
    parser.add_argument('--seed',type=int,default=7588)
    parser.add_argument('--max-new-tokens',type=int,default=32)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--compare-raw',type=Path)
    parser.add_argument('--compare-model-id')
    args=parser.parse_args()
    if args.max_new_tokens<1:parser.error('Choose a positive fixed generation token budget')
    if (args.variant=='pretrained_native')!=(args.adaptation_checkpoint is None):
        parser.error('Only pretrained_native omits --adaptation-checkpoint')
    if args.compare_model_id and not args.compare_raw:parser.error('--compare-model-id needs --compare-raw')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.time()
    receipt={'status':'started','split':args.split,'variant':args.variant,
             'scope':'TRAIN engineering generation, not a held-out performance measure' if args.split=='train' else 'Raw generated responses; claim scores require independent annotations',
             'ground_truth_sidecar_read_for_generation':False,'selection':'all manifest records in the requested split, including unavailable inputs',
             'seed':args.seed,'max_new_tokens':args.max_new_tokens,'records':[]}
    def save():
        receipt['elapsed_s']=time.time()-started
        receipt['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temp=args.output_dir/'generation_receipt.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(args.output_dir/'generation_receipt.json')
    def progress(event):
        print(json.dumps({**event,'elapsed_s':time.time()-started}),flush=True)
        receipt['stage']=event.get('stage',receipt.get('stage'));save()
    save()
    try:
        import torch
        try:
            from .runtime import PointLLMBackboneRuntime
            from .cached_data import CachedRealDataset,sha,canonical_hash
            from .training_engine import restore_adaptation,generate_record,generation_protocol_hash
            from .evaluation.claim_metrics import require_matched_comparison,read_jsonl
        except ImportError:
            from runtime import PointLLMBackboneRuntime
            from cached_data import CachedRealDataset,sha,canonical_hash
            from training_engine import restore_adaptation,generate_record,generation_protocol_hash
            from evaluation.claim_metrics import require_matched_comparison,read_jsonl
        dataset=CachedRealDataset(args.dataset)
        records=sorted(dataset.split(args.split),key=lambda item:item['case_id'])
        if not records:raise ValueError('No records in requested split')
        if args.adaptation_checkpoint:
            checkpoint_sha=sha(args.adaptation_checkpoint)
        else:
            # Identifies the exact official model plus execution convention,
            # rather than pretending a downloaded FP32 shard is an adaptation.
            checkpoint_sha=canonical_hash({'official_model':'RunsenXu/PointLLM_7B_v1.2','revision':'37d8c15aaff8e7e05f04729dcb6960d5758e9f86','native_tokens':97,'quantization':'NF4/BF16'})
        protocol_sha=generation_protocol_hash(args.protocol,max_new_tokens=args.max_new_tokens,seed=args.seed)
        source_paths={name:Path(__file__).resolve().parent/name for name in ['runtime.py','adapter.py','cached_data.py','training_engine.py','eval_pointllm.py']}
        sources={name:sha(path) for name,path in source_paths.items()}
        receipt.update({'dataset_manifest_sha256':sha(args.dataset),'method_contract_sha256':sha(args.protocol),
                        'generation_protocol_sha256':protocol_sha,'checkpoint_sha256':checkpoint_sha,
                        'n_manifest_records':len(records),'source_sha256':sources})
        torch.manual_seed(args.seed)
        runtime=PointLLMBackboneRuntime(args.checkpoint_dir,groups=96,threads=args.threads,device=args.device,train_projector=False,language_cache_dir=args.language_cache_dir,progress=progress)
        receipt['official_model']=runtime.receipt
        frozen_before=runtime.frozen_fingerprint()
        if args.adaptation_checkpoint:
            adapter,checkpoint=restore_adaptation(args.adaptation_checkpoint,runtime,expected_variant=args.variant)
            trained_sources=checkpoint['metadata'].get('source_sha256',{})
            # Adapters must not silently execute with a different definition.
            if args.variant=='rahumpointllm' and trained_sources.get('adapter.py')!=sources['adapter.py']:
                raise ValueError('Adapter source differs from the trained checkpoint; restore with its frozen implementation')
            receipt['restored_completed_steps']=checkpoint['metadata'].get('completed_steps')
            receipt['restored_checkpoint_metadata']=checkpoint['metadata']
        else:adapter=None
        model_variant='plain_projector' if args.variant=='pretrained_native' else args.variant
        raw_path=args.output_dir/'raw_generations.jsonl'
        for index,record in enumerate(records):
            progress({'stage':'actual_autoregressive_generation','case_id':record['case_id'],'index':index+1,'n':len(records)})
            generated=generate_record(runtime,adapter,model_variant,dataset,record,checkpoint_sha256=checkpoint_sha,protocol_sha256=protocol_sha,seed=args.seed,max_new_tokens=args.max_new_tokens)
            generated['model_id']=args.variant
            ids=generated.get('generated_token_ids',[])
            generated['generated_tokens']=len(ids)
            if generated['execution_status']=='generated':
                generated['finish_reason']='eos' if ids and ids[-1]==2 else ('max_new_tokens' if len(ids)==args.max_new_tokens else 'generation_stopped')
            generated['execution_scope']=receipt['scope']
            with raw_path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(generated,ensure_ascii=False)+'\n')
            receipt['records'].append({'case_id':record['case_id'],'execution_status':generated['execution_status'],'raw_response_sha256':generated.get('raw_response_sha256'),'generated_tokens':len(ids)})
            save()
        if runtime.frozen_fingerprint()!=frozen_before:raise RuntimeError('Frozen 7B/PointBERT weights or quantstates changed during generation')
        if {name:sha(path) for name,path in source_paths.items()}!=sources:raise RuntimeError('Implementation changed during generation')
        if args.adaptation_checkpoint and sha(args.adaptation_checkpoint)!=checkpoint_sha:raise RuntimeError('Adaptation checkpoint changed during generation')
        receipt.update({'frozen_language_pointbert_quantstates_unchanged':True,'implementation_sources_unchanged':True,
                        'raw_generations_sha256':sha(raw_path),'n_generated':sum(r['execution_status']=='generated' for r in receipt['records']),
                        'n_input_or_generation_errors':sum(r['execution_status']!='generated' for r in receipt['records'])})
        if args.compare_raw:
            prior=read_jsonl(args.compare_raw)
            model=args.compare_model_id
            if model is None:
                models={record['model_id'] for record in prior}
                if len(models)!=1:raise ValueError('Specify --compare-model-id for a multi-model raw file')
                model=next(iter(models))
            if model==args.variant:raise ValueError('Matched comparison requires two distinct model IDs')
            paired=require_matched_comparison(prior+read_jsonl(raw_path),model,args.variant)
            receipt['matched_comparison']={'other_model_id':model,'this_model_id':args.variant,'paired_case_seed_count':paired,'other_raw_sha256':sha(args.compare_raw),'guard':'evaluation.claim_metrics.require_matched_comparison','claim_metric_computed':False}
        receipt['status']='completed_raw_generation_no_automatic_claim_score'
        progress({'stage':'generation_completed','n_manifest_records':len(records),'n_generated':receipt['n_generated']})
    except Exception as error:
        receipt.update({'status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()});save();raise
    finally:save()


if __name__=='__main__':main()
