"""Bounded resident real-model worker to avoid reloading 27 GB per input.

Each request is an immutable JSON file. Only cloud/question fields enter the
model. Ground truth is provenance metadata and never appended to the prompt.
Create a STOP file to end the worker; it also exits after bounded idle time.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
if __package__:
    from .pointllm_cpu_infer import OfficialPointLLMCPU, ROOT, prepare_cloud, read_cloud, sha256
else:
    from pointllm_cpu_infer import OfficialPointLLMCPU, ROOT, prepare_cloud, read_cloud, sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queue-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=ROOT/'checkpoints/PointLLM_7B_v1.2')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--idle-seconds', type=int, default=1800)
    args=p.parse_args()
    args.queue_dir.mkdir(exist_ok=True, parents=True)
    output_dir=args.queue_dir/'results'; output_dir.mkdir(exist_ok=True)
    started=time.time(); model=OfficialPointLLMCPU(args.model_dir, args.threads)
    loaded_s=time.time()-started
    (args.queue_dir/'ready.json').write_text(json.dumps({'loaded_model':model.load_receipt,'model_load_s':loaded_s},indent=2)+'\n')
    idle_since=time.time()
    while time.time()-idle_since < args.idle_seconds and not (args.queue_dir/'STOP').exists():
        did_work=False
        for request_path in sorted(args.queue_dir.glob('request_*.json')):
            result_path=output_dir/request_path.name.replace('request_', 'result_', 1)
            error_path=result_path.with_suffix('.error.json')
            if result_path.exists() or error_path.exists(): continue
            did_work=True; idle_since=time.time()
            request=json.loads(request_path.read_text())
            try:
                cloud_path=Path(request['cloud_path'])
                source_hash=sha256(cloud_path)
                if request.get('expected_source_sha256') and source_hash != request['expected_source_sha256']:
                    raise ValueError('Measured input hash differs from frozen request')
                arr=read_cloud(cloud_path,request.get('key'))
                seed=int(request.get('seed',7588))
                cloud,preprocess=prepare_cloud(arr,seed=seed,geometry_only=request.get('geometry_only',True))
                print(json.dumps({'running_request':str(request_path),'rows':len(arr)}),flush=True)
                result=model.generate(cloud,request['question'],seed=seed,max_new_tokens=int(request.get('max_new_tokens',64)))
                result.update({'request_file':str(request_path.resolve()),'request_sha256':sha256(request_path),'source_cloud':str(cloud_path.resolve()),'source_file_sha256':source_hash,'preprocessing':preprocess,'model':model.load_receipt,'model_load_s':loaded_s,'metadata_not_passed_to_model':request.get('metadata',{}),'scope':request.get('scope','Actual generation; independently supported claim annotation is separate.')})
                temp=result_path.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(result_path)
                print(json.dumps({'saved':str(result_path),'response':result['raw_response'],'generation_s':result['language_generation_s']}),flush=True)
            except Exception as error:
                import traceback
                failure={'request_file':str(request_path),'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
                error_path.write_text(json.dumps(failure,indent=2)+'\n')
                print(json.dumps(failure),flush=True)
        if not did_work: time.sleep(1)
    (args.queue_dir/'stopped.json').write_text(json.dumps({'idle_timeout_seconds':args.idle_seconds,'explicit_stop':(args.queue_dir/'STOP').exists()})+'\n')


if __name__=='__main__': main()
