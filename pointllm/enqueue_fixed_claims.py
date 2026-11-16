"""Queue only available crops from the fixed official-test pilot manifest."""
import argparse,json
from pathlib import Path
if __package__:
    from .pointllm_cpu_infer import ROOT,sha256
else:
    from pointllm_cpu_infer import ROOT,sha256

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--crop-dir',type=Path,required=True);p.add_argument('--queue-dir',type=Path,default=ROOT/'claim_queue');p.add_argument('--variant',choices=['raw','mitigated'],default='raw');args=p.parse_args()
    frozen=args.crop_dir/'fixed12_source_selection.json'
    selection=json.loads(frozen.read_text())
    protocol=json.loads((args.queue_dir/'frozen_pilot_protocol.json').read_text())
    for i,member in enumerate(selection['selected_members']):
        case_id=f'case{i:03d}'
        cloud=args.crop_dir/f'{case_id}_{args.variant}_measured_points.npz'
        gt=args.crop_dir/f'{case_id}_ground_truth.json'
        if not cloud.exists() or not gt.exists():continue
        dest=args.queue_dir/f'request_{case_id}_{args.variant}.json'
        if dest.exists():continue
        truth=json.loads(gt.read_text())
        if truth['source_member']!=member:raise ValueError('Crop differs from frozen sequence selection')
        request={'cloud_path':str(cloud.resolve()),'key':'points','geometry_only':True,'expected_source_sha256':sha256(cloud),'question':protocol['question'],'seed':protocol['seed'],'max_new_tokens':protocol['max_new_tokens'],'scope':protocol['ground_truth_scope'],'metadata':{'sample_id':case_id,'variant':args.variant,'ground_truth_file':str(gt.resolve()),'ground_truth_sha256':sha256(gt),'fixed_selection_file':str(frozen.resolve()),'fixed_selection_sha256':sha256(frozen),'selection_frozen_before_generation':True}}
        if args.variant=='mitigated':
            receipt=args.crop_dir/f'{case_id}_mitigation_receipt.json'
            if not receipt.exists():raise ValueError('No train/val-only mitigation receipt for '+case_id)
            request['metadata']['mitigation_receipt_file']=str(receipt.resolve())
            request['metadata']['mitigation_receipt_sha256']=sha256(receipt)
        dest.write_text(json.dumps(request,indent=2)+'\n')
        print(dest,flush=True)
