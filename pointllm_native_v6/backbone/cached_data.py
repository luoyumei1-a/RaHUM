"""Real frozen-encoder caches and independent target sidecars.

Generation loads only measured feature/context artifacts and a generic prompt.
It never opens the target sidecar. Scene partitions and native token budget
are checked before any model operation.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

CONTEXT_KEYS={'radar_centers','reference_centers','radar_attributes','radar_mask','reference_mask','frame_timestamps','reference_timestamps','sequence_ids','calibration_valid','history_valid','projected_uv','reference_uv','projection_valid'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class CachedRealDataset:
    def __init__(self,path):
        self.path=Path(path).resolve();self.root=self.path.parent;self.spec=json.loads(self.path.read_text())
        if self.spec.get('schema_version')!=1 or not self.spec.get('frozen_before_training'):
            raise ValueError('A versioned, pre-frozen real-data manifest is required')
        if self.spec.get('point_token_budget')!=97:raise ValueError('All models must use the matched native97-token budget')
        if self.spec.get('measurement_data_kind')!='real':raise ValueError('Only source-provenanced real measurement caches are accepted')
        self.records=self.spec['records'];self.by_id={};scenes={};cache_splits={}
        if not self.records:raise ValueError('The manifest has no real records')
        for record in self.records:
            for field in ['case_id','scene_id','split','feature_cache','feature_cache_sha256','input_receipt','input_receipt_sha256','prompt']:
                if field not in record:raise ValueError('Missing record field: '+field)
            if record['case_id'] in self.by_id:raise ValueError('Duplicate case ID')
            if record['split'] not in {'train','validation','test'}:raise ValueError('Explicit train/validation/test scene split required')
            scene=record['scene_id'];split=record['split']
            if scene in scenes and scenes[scene]!=split:raise ValueError('A recording scene crosses train/validation/test partitions')
            scenes[scene]=split
            cache=record['feature_cache_sha256']
            if record.get('input_available',True):
                if not cache:raise ValueError('An available example requires an actual frozen feature cache')
                if cache in cache_splits and cache_splits[cache]!=split:raise ValueError('An encoder cache crosses data partitions')
                cache_splits[cache]=split
            for category in self.spec.get('class_vocabulary',[]):
                phrase=category.replace('_',' ').replace('.',' ')
                if re.search(r'\b'+re.escape(phrase)+r'\b',record['prompt'],flags=re.I):raise ValueError('A target-category name is forbidden in the generic input prompt')
            self.by_id[record['case_id']]=record

    def resolve(self,path):
        p=Path(path);return p if p.is_absolute() else self.root/p

    def split(self,name):return [r for r in self.records if r['split']==name]

    def source_case_sha(self,record):
        return canonical_hash({key:record[key] for key in ['case_id','scene_id','split','feature_cache_sha256','input_receipt_sha256','prompt']})

    def features(self,record,device='cpu'):
        import numpy as np
        import torch
        cache=self.resolve(record['feature_cache']);receipt=self.resolve(record['input_receipt'])
        if sha(cache)!=record['feature_cache_sha256'] or sha(receipt)!=record['input_receipt_sha256']:
            raise ValueError('Frozen real cache/encoder provenance changed')
        receipt_info=json.loads(receipt.read_text())
        assembled=receipt_info.get('assembled_input_sha256',receipt_info.get('assembled_cache_sha256'))
        if assembled!=record['feature_cache_sha256']:raise ValueError('Input receipt is not bound to this exact assembled sensory cache')
        if record.get('source_sample_tokens') is not None and record['source_sample_tokens']!=receipt_info.get('source_sample_tokens'):
            raise ValueError('Record source samples differ from the actual encoder/input receipt')
        with np.load(cache,allow_pickle=False) as z:
            keys=set(z.files);context_names={k[len('context_'):] for k in keys if k.startswith('context_')}
            if keys-{'radar_tokens','reference_tokens'}-{('context_'+k) for k in CONTEXT_KEYS}:
                raise ValueError('Only sensory tokens and whitelisted physical context may be in the feature cache')
            if context_names-CONTEXT_KEYS:raise ValueError('Unknown physical context key')
            arrays={k:z[k] for k in z.files}
        for key,value in arrays.items():
            if value.dtype.kind not in 'bifu' or (value.dtype.kind in 'f' and not np.isfinite(value).all()):
                raise ValueError('Feature/context arrays must be finite numeric observations, never text/GT objects: '+key)
        radar=torch.from_numpy(arrays['radar_tokens']).to(device=device,dtype=torch.float32)
        reference=torch.from_numpy(arrays['reference_tokens']).to(device=device,dtype=torch.float32)
        if radar.ndim!=4 or tuple(radar.shape[2:])!=(97,384) or radar.shape[0]!=1:raise ValueError('Expected batch1 real nativePointBERT[B,T,97,384]')
        if reference.ndim!=4 or reference.shape[:2]!=radar.shape[:2]:raise ValueError('Reference and radar windows do not correspond')
        context={k:torch.from_numpy(arrays['context_'+k]).to(device) for k in context_names}
        if any(any(term in k for term in ['label','category','ground_truth','supervision','claim','instance_id']) for k in context):raise ValueError('GT must remain outside forward physical context')
        return radar,reference,context

    def targets(self,record):
        # Called by training/validation objectives only, never generation.
        for key in ['target_sidecar','target_sidecar_sha256']:
            if key not in record:raise ValueError('Training requires a separate independently supported target sidecar')
        path=self.resolve(record['target_sidecar'])
        if sha(path)!=record['target_sidecar_sha256']:raise ValueError('Independent target sidecar changed')
        target=json.loads(path.read_text())
        if not target.get('response') or not target.get('independent_annotation_source') or not target.get('annotation_scope'):
            raise ValueError('Target needs real annotation provenance, scope and response')
        samples=record.get('source_sample_tokens')
        if samples and target.get('sample_token') is not None and target['sample_token']!=samples[-1]:
            raise ValueError('Target sidecar belongs to a different source query sample')
        if record.get('evaluation_timestamp_us') is not None and target.get('evaluation_timestamp_us')!=record['evaluation_timestamp_us']:
            raise ValueError('Target/query evaluation timestamps differ')
        receipt=json.loads(self.resolve(record['input_receipt']).read_text())
        for target_key,receipt_key in [('raw_window_manifest_sha256','raw_window_sha256'),('evaluation_time_manifest_sha256','evaluation_time_manifest_sha256')]:
            if target.get(target_key) is not None and target[target_key]!=receipt.get(receipt_key):
                raise ValueError('Independent target is bound to a different measured source window/time contract')
        return target

