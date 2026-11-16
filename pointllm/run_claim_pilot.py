"""Resident PointLLM execution for a preselected real-data claim pilot.

This file executes generation and writes immutable raw-text receipts. It does
not infer scene absence from multipath/noise labels and does not automatically
adjudicate open-ended claims. A separate ground-truth manifest is retained for
supported/contradicted/unverifiable annotation after every output is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

if __package__:
    from .pointllm_cpu_infer import OfficialPointLLMCPU, ROOT, prepare_cloud, read_cloud, sha256
else:
    from pointllm_cpu_infer import OfficialPointLLMCPU, ROOT, prepare_cloud, read_cloud, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=ROOT/'checkpoints/PointLLM_7B_v1.2')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=7588)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--wait-for-input-seconds', type=int, default=0)
    args = p.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest['records']
    question = manifest.get('question', 'Describe the object represented by this point cloud. Focus on its object category and state uncertainty when the geometry is insufficient.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    protocol = {'manifest_sha256': manifest_hash, 'source_manifest': str(args.manifest.resolve()), 'number_of_preselected_records': len(records), 'question': question, 'seed': args.seed, 'maximum_new_tokens': args.max_new_tokens, 'all_records_retained': True, 'correctness_evaluation': 'separate independent main-target semantic ground truth; open-ended details remain unverifiable without annotation'}
    (args.output_dir/'execution_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    start = time.time()
    model = OfficialPointLLMCPU(args.model_dir, args.threads)
    load_s = time.time()-start
    for index, record in enumerate(records):
        sample_id = str(record['sample_id'])
        variant = str(record.get('variant', 'raw'))
        safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in sample_id+'__'+variant)
        dest = args.output_dir/(safe+'.json')
        cloud_path = Path(record['cloud_path'])
        if not cloud_path.is_absolute():
            cloud_path = args.manifest.parent/cloud_path
        deadline = time.time()+args.wait_for_input_seconds
        while not cloud_path.exists() and time.time() < deadline:
            time.sleep(min(2, max(0, deadline-time.time())))
        if not cloud_path.exists():
            failure = {'sample_id': sample_id, 'variant': variant, 'error': 'preselected input file missing', 'source_cloud': str(cloud_path), 'manifest_sha256': manifest_hash}
            dest.with_suffix('.error.json').write_text(json.dumps(failure, indent=2)+'\n')
            print(json.dumps(failure), flush=True)
            continue
        source_hash = sha256(cloud_path)
        if dest.exists():
            old = json.loads(dest.read_text())
            if old.get('manifest_sha256') != manifest_hash or old.get('source_file_sha256') != source_hash:
                raise RuntimeError('Refusing to reuse a result from changed inputs or protocol: '+str(dest))
            print(json.dumps({'reused_exact_receipt': str(dest)}), flush=True)
            continue
        arr = read_cloud(cloud_path, record.get('key'))
        cloud, preprocessing = prepare_cloud(arr, seed=args.seed, geometry_only=record.get('geometry_only', True))
        print(json.dumps({'running': sample_id, 'variant': variant, 'index': index, 'rows': len(arr)}), flush=True)
        result = model.generate(cloud, question, seed=args.seed, max_new_tokens=args.max_new_tokens)
        result.update({'sample_id': sample_id, 'variant': variant, 'source_cloud': str(cloud_path.resolve()), 'source_file_sha256': source_hash, 'preprocessing': preprocessing, 'manifest_sha256': manifest_hash, 'model_load_s': load_s, 'model': model.load_receipt, 'scope': manifest.get('scope', 'oracle annotated main-target crop; category and independently labeled attributes only'), 'ground_truth_metadata_not_passed_to_model': {k:v for k,v in record.items() if k not in ['cloud_path','key','geometry_only']}})
        temp = dest.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n')
        temp.replace(dest)
        print(json.dumps({'saved': str(dest), 'response': result['raw_response'], 'generation_s': result['language_generation_s']}), flush=True)


if __name__ == '__main__':
    main()
