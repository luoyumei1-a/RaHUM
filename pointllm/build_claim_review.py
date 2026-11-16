"""Expose every preselected actual response for author semantic adjudication."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
if __package__:
    from .pointllm_cpu_infer import sha256
else:
    from pointllm_cpu_infer import sha256

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue-dir',type=Path,required=True);p.add_argument('--crop-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    selection=json.loads((a.crop_dir/'fixed12_source_selection.json').read_text())
    rows=[]
    for i,member in enumerate(selection['selected_members']):
        case=f'case{i:03d}';gtpath=a.crop_dir/f'{case}_ground_truth.json'
        gt=json.loads(gtpath.read_text()) if gtpath.exists() else {}
        for variant in ['raw','mitigated']:
            path=a.queue_dir/'results'/f'result_{case}_{variant}.json';err=path.with_suffix('.error.json')
            z=json.loads(path.read_text()) if path.exists() else {}
            fail=json.loads(err.read_text()) if err.exists() else {}
            tokens=z.get('generated_token_ids',[])
            finish='eos' if tokens and tokens[-1]==2 else ('max_new_tokens' if z and z.get('generated_tokens')==z.get('max_new_tokens') else 'other' if z else '')
            rows.append({'case_id':case,'variant':variant,'official_test_member':member,'frame':gt.get('frame'),'ground_truth_category':gt.get('ground_truth_class'),'gt_file':str(gtpath.resolve()),'gt_sha256':sha256(gtpath) if gtpath.exists() else '', 'execution_status':'generated' if z else ('input_or_execution_error' if fail else 'not_yet_recorded'),'raw_response':z.get('raw_response',''),'result_file':str(path.resolve()),'result_sha256':sha256(path) if path.exists() else '', 'source_file_sha256':z.get('source_file_sha256',''),'measured_input_rows':z.get('preprocessing',{}).get('finite_measured_rows',''),'token_count':z.get('generated_tokens',''),'token_limit':z.get('max_new_tokens',''),'at_token_limit':z.get('generated_tokens')==z.get('max_new_tokens') if z else '', 'finish_reason':finish,'point_encode_s':z.get('point_encode_s',''),'language_generation_s':z.get('language_generation_s',''),'category_claim_exact_text':'','category_adjudication_Supported_Contradicted_Unverifiable':'','category_granularity_exact_coarse_none':'','additional_unverifiable_attribute_claims':'','annotation_reason':'','error':fail.get('error','')})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a.output.with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'all_preselected_rows':len(rows),'generated':sum(r['execution_status']=='generated' for r in rows),'output':str(a.output)},indent=2))
