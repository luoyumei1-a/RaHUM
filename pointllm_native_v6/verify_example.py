"""Verify the small real feature example without loading a language model.

Optionally pass --nuscenes-table-dir PATH/v1.0-mini to independently look up
the partial positive target in publisher sample_annotation/instance/category
tables. No generated response is assigned a truth value by this verifier.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nuscenes-table-dir',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/'backbone/nuscenes_real_rgb_probe'
    manifest=json.loads((root/'runner_dataset.json').read_text())
    assert manifest['measurement_data_kind']=='real' and manifest['point_token_budget']==97
    assert len(manifest['records'])==1
    record=manifest['records'][0]
    assert record['split']=='train'
    for field in ['feature_cache','input_receipt','target_sidecar']:
        assert digest(root/record[field])==record[field+'_sha256'],field
    receipt=json.loads((root/record['input_receipt']).read_text())
    assert receipt['assembled_input_sha256']==record['feature_cache_sha256']
    assert receipt['source_sample_tokens']==record['source_sample_tokens']
    with np.load(root/record['feature_cache'],allow_pickle=False) as cache:
        assert cache['radar_tokens'].shape==(1,3,97,384)
        assert cache['reference_tokens'].shape==(1,3,256,384)
        for name in cache.files:
            assert cache[name].dtype.kind in 'bifu'
            assert np.isfinite(cache[name]).all(),name
            assert not any(word in name for word in ['label','category','instance','claim','ground_truth'])
    target=json.loads((root/record['target_sidecar']).read_text())
    assert target['sample_token']==record['source_sample_tokens'][-1]
    assert target['evaluation_timestamp_us']==record['evaluation_timestamp_us']
    assert target['raw_window_manifest_sha256']==receipt['raw_window_sha256']
    assert target['evaluation_time_manifest_sha256']==receipt['evaluation_time_manifest_sha256']
    assert record['prompt']==target['question']
    assert not target['category_verified'] and not any(target['auxiliary_masks'].values())
    official_lookup=False
    if args.nuscenes_table_dir:
        tables={name:{row['token']:row for row in json.loads((args.nuscenes_table_dir/(name+'.json')).read_text())}
                for name in ['sample_annotation','instance','category']}
        for token in target['source_annotation_tokens']:
            annotation=tables['sample_annotation'][token]
            assert annotation['sample_token']==target['sample_token']
            instance=tables['instance'][annotation['instance_token']]
            assert tables['category'][instance['category_token']]['name']==target['source_category_name']
        official_lookup=True
    print(json.dumps({'status':'PASS_real_feature_hashes_shapes_and_target_separation',
        'source_sample_tokens':record['source_sample_tokens'],
        'feature_sha256':record['feature_cache_sha256'],
        'official_annotation_lookup_executed':official_lookup,
        'generated_claim_score_computed':False},indent=2))


if __name__=='__main__':main()
