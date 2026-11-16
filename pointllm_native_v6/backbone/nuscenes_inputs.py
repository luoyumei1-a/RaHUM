"""Measured nuScenes-mini -> native PointLLM/RGB adapter inputs.

Raw sensor/calibration inputs are separated from annotation-only supervision.
This module does not load a language model or claim to have run radar/RGB 7B.
Native PointBERT and RGB encoders are injected explicitly by their owners.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json
from typing import Callable,Mapping,Any
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _transform(T,xyz):
    return np.asarray(xyz,dtype=np.float64)@np.asarray(T,dtype=np.float64)[:3,:3].T+np.asarray(T,dtype=np.float64)[:3,3]


def load_manifest(path):
    path=Path(path);j=json.loads(path.read_text())
    if 'camera-anchor' not in j.get('scope','').lower() or not j.get('temporal_policy',{}).get('no_future_fill'):
        raise ValueError('Use the explicitly causal camera-anchored manifest, not nearest-keyframe pairing')
    return path.parent,j


def select_window(manifest,*,split='mini_train',length=3,scene=None,end_sample_token=None):
    """Fixed earliest eligible chronological measured window; never consult truth."""
    records=manifest['records'];by_scene={}
    for r in records:
        if r['official_mini_split']!=split or (scene is not None and r['scene_name']!=scene):continue
        by_scene.setdefault(r['scene_name'],[]).append(r)
    for name in sorted(by_scene):
        stream=sorted(by_scene[name],key=lambda r:r['frame_anchor_timestamp_us'])
        for end in range(length-1,len(stream)):
            chosen=stream[end-length+1:end+1]
            if end_sample_token is not None and chosen[-1]['sample_token']!=end_sample_token:continue
            if not all(r['measurement_available'] and r['measurement_file'] for r in chosen):continue
            if len({r['scene_token']for r in chosen})!=1:raise AssertionError('Cross-scene window')
            times=np.asarray([r['frame_anchor_timestamp_us']for r in chosen],np.int64)
            if not (np.diff(times)>0).all():raise AssertionError('Noncausal source order')
            return chosen
    raise ValueError('No requested eligible causal window; unavailable startup anchors are not future-filled')


def prepare_raw_window(manifest_path,output_dir,*,split='mini_train',length=3,scene=None,end_sample_token=None,point_policy='standard_sensor'):
    """Export true sensor arrays without reading any annotation file."""
    if point_policy not in {'standard_sensor','all_finite'}:raise ValueError('Unrecognized measurement-only point policy')
    root,manifest=load_manifest(manifest_path);chosen=select_window(manifest,split=split,length=length,scene=scene,end_sample_token=end_sample_token)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);frames=[]
    for n,r in enumerate(chosen):
        source=root/r['measurement_file']
        if sha(source)!=r['measurement_sha256']:raise ValueError('Measurement cache changed after source manifest')
        with np.load(source,allow_pickle=False)as z:a={k:z[k]for k in z.files}
        raw=a['radar_features18_sensor'];finite=np.isfinite(a['radar_xyz_global']).all(1)&np.isfinite(raw[:,[0,1,8,9,5]]).all(1)
        keep=finite & a['radar_standard_filter_mask'] if point_policy=='standard_sensor' else finite
        rows=np.flatnonzero(keep)
        if len(rows)<2:raise ValueError('Selected source window has fewer than two finite measured points')
        if not (a['radar_timestamp_us'][rows]<=r['frame_anchor_timestamp_us']).all():raise AssertionError('Future radar measurement in causal source')
        xyz=a['radar_xyz_global'][rows]
        sensor_xy=a['radar_xyz_sensor'][rows,:2]
        length_xy=np.linalg.norm(sensor_xy,axis=1)
        vr_comp=(raw[rows,8]*sensor_xy[:,0]+raw[rows,9]*sensor_xy[:,1])/np.maximum(length_xy,1e-12)
        # RCS remains its actual source field, not a fabricated RGB/log-amplitude channel.
        attrs=np.column_stack([vr_comp,raw[rows,5]]).astype(np.float32)
        points=np.column_stack([xyz,np.zeros_like(xyz)]).astype(np.float32)
        camera=r['camera'];camera_path=root/'data'/camera['filename']
        if sha(camera_path)!=camera['source_sha256']:raise ValueError('Actual RGB changed after manifest')
        global_from_camera=np.asarray(camera['T_global_from_ego_at_camera'])@np.asarray(camera['T_ego_at_camera_from_camera'])
        camera_from_global=np.linalg.inv(global_from_camera)
        rebuilt=_transform(camera_from_global,xyz)
        np.testing.assert_allclose(rebuilt,a['radar_xyz_camera'][rows],atol=1e-8,rtol=1e-9)
        f=out/f'frame{n}_radar_global.npz'
        if f.exists():raise FileExistsError('Refusing to overwrite fixed real input: '+str(f))
        np.savez_compressed(f,points=points,attributes=attrs,source_row_ids=rows,
                            raw_features18_sensor=raw[rows],radar_xyz_sensor=a['radar_xyz_sensor'][rows],
                            radar_sensor_id=a['radar_sensor_id'][rows],radar_native_source_row_id=a['radar_source_row_id'][rows],
                            radar_sample_data_token=a['radar_sample_data_token'][rows],radar_timestamp_us=a['radar_timestamp_us'][rows])
        frames.append({'frame_index':n,'sample_token':r['sample_token'],'recording_token':r['scene_token'],
                       'frame_anchor_timestamp_us':r['frame_anchor_timestamp_us'],'reference_timestamp_us':r['reference_timestamp_us'],
                       'radar_cloud':str(f.resolve()),'radar_sha256':sha(f),'point_key':'points','attribute_key':'attributes','source_row_key':'source_row_ids',
                       'rgb_path':str(camera_path.resolve()),'rgb_sha256':camera['source_sha256'],'image_size_wh':[camera['width'],camera['height']],
                       'camera_intrinsic':camera['camera_intrinsic'],'T_camera_from_global':camera_from_global.tolist(),
                       'source_measurement_npz':str(source.resolve()),'source_measurement_sha256':r['measurement_sha256'],
                       'raw_measured_rows':len(raw),'retained_measurement_rows':len(rows),
                       'radar_time_min_us':int(a['radar_timestamp_us'][rows].min()),'radar_time_max_us':int(a['radar_timestamp_us'][rows].max())})
    raw_manifest={'source':'officialnuScenesv1.0mini causal sensor/calibration data','manifest_sha256':sha(manifest_path),
                  'selection':'lexical first scene and earliest consecutive eligible measured camera-anchor window unless explicit scene/endtoken provided; independent of annotations/model outputs',
                  'source_split_for_audit_only':split,'reference_kind':'rgb_patch','coordinate_frame':'global metric frame, actual ego-pose motion compensated; actor motion not fabricated/compensated',
                  'radar_representation':'officialradarXYZ transformedtoglobal; originalradar elevation limitations remain; RGB channels neutralzeros, not measuredcolor',
                  'attribute_schema':['derived_radial_component_of_source_vx_comp_vy_comp_mps','source_rcs_native_unconverted'],
                  'attribute_caveat':'group mean radial components may mix sensor LOS directions; temporal difference is an empirical descriptor, not Cartesian velocity truth',
                  'point_policy':point_policy,'no_front_camera_projection_filter':True,'time_unit':'officialintegerUnixmicroseconds; preserveint64/float64precision',
                  'frames':frames,'ground_truth_files_opened_by_this_function':False,'data_redistribution':'localacquisitioncache only; do not include raw nuScenes RGB/PCD/derivedsensorcache in publiccode release'}
    _write_json(out/'raw_window_manifest.json',raw_manifest)
    return out/'raw_window_manifest.json'


def write_supervision_sidecar(manifest_path,raw_window_path):
    """Separate annotation read, never called by forward/input assembly."""
    root,manifest=load_manifest(manifest_path)
    raw=json.loads(Path(raw_window_path).read_text());by_token={r['sample_token']:r for r in manifest['records']}
    chosen=[by_token[f['sample_token']]for f in raw['frames']]
    out=Path(raw_window_path).parent
    sidecar=[]
    for r in chosen:
        gt=root/r['truth_file']
        if sha(gt)!=r['truth_sha256']:raise ValueError('Independent annotation sidecar changed')
        source_truth=json.loads(gt.read_text())
        sidecar.append({'sample_token':r['sample_token'],'source_truth_sha256':r['truth_sha256'],
                        'sample_timestamp_us':r['sample_timestamp_us'],'camera_anchor_timestamp_us':r['frame_anchor_timestamp_us'],
                        'annotation_time_offset_us':r['sample_annotation_time_delta_us'],
                        'annotations':source_truth['annotations']})
    _write_json(out/'supervision_only.json',{'scope':'officialannotations at their actual sample time, not exact camera-time or per-camera visibility truth',
                                           'never_forward_features_or_prompt':True,'no_pointwise_ghost_truth':True,
                                           'projection_is_not_visibility_or_absence':True,'frames':sidecar})
    return out/'supervision_only.json'


def write_evaluation_time_binding(manifest_path,raw_window_path):
    """Bind official sample-time evaluation using metadata only, never box content."""
    _,manifest=load_manifest(manifest_path);raw=json.loads(Path(raw_window_path).read_text())
    violations=[r['sample_token']for r in manifest['records']if r['camera']['timestamp_us']>r['sample_timestamp_us']]
    if violations:raise ValueError('Future camera relative to official sample time: '+str(violations[:5]))
    by_token={r['sample_token']:r for r in manifest['records']};frames=[]
    for f in raw['frames']:
        r=by_token[f['sample_token']];evaluation=int(r['sample_timestamp_us']);camera=int(r['camera']['timestamp_us'])
        if f['radar_time_max_us']>camera or camera>evaluation:raise ValueError('Future measurement at declared GT evaluation time')
        frames.append({'sample_token':f['sample_token'],'evaluation_timestamp_us':evaluation,'camera_anchor_timestamp_us':camera,
                       'camera_age_at_evaluation_us':evaluation-camera,'radar_latest_age_at_evaluation_us':evaluation-f['radar_time_max_us']})
    result={'raw_window_manifest_sha256':sha(raw_window_path),'source_causal_manifest_sha256':sha(manifest_path),
            'evaluation_time_policy':'officialsampletimestamp; allusedradarandcameraobservationsareat orbeforeit',
            'all404_source_camera_timestamps_checked':len(manifest['records']),'future_camera_violations':0,
            'GT_geometry_time':'officialsampletime, no inventedcamera-timeboxmotioncompensation','frames':frames}
    out=Path(raw_window_path).with_name('evaluation_time_manifest.json');_write_json(out,result);return out


def cache_actual_pointbert(raw_window_path,encode_measured_points:Callable,output_dir,*,point_budget=96,seed=7588):
    """Use actual native encoder callback; no LM is loaded by this function."""
    import torch
    raw=json.loads(Path(raw_window_path).read_text());out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    tokens=[];centers=[];attrs=[];masks=[];metadata=[]
    for j,f in enumerate(raw['frames']):
        path=Path(f['radar_cloud'])
        if sha(path)!=f['radar_sha256']:raise ValueError('Frozen measured input changed')
        with np.load(path,allow_pickle=False)as z:p=z['points'];a=z['attributes'];ids=z['source_row_ids']
        actual,info=encode_measured_points(p,attributes=a,source_row_ids=ids,seed=seed,point_budget=point_budget)
        if tuple(actual.shape)!=(1,97,384):raise ValueError('ActualPointBERT callback must return CLS+96native tokens')
        for name in ['centers','attributes','mask','neighbor_indices','source_row_indices']:
            if name not in info:raise ValueError('Actualgroup metadata missing '+name)
        tokens.append(actual.detach().float().cpu().numpy()[0]);centers.append(info['centers'].detach().float().cpu().numpy())
        attrs.append(info['attributes'].detach().float().cpu().numpy());masks.append(info['mask'].detach().bool().cpu().numpy())
        simple={k:v for k,v in info.items()if k not in ['centers','attributes','mask','group_source_row_ids','neighbor_indices','source_row_indices','group_unique_source_counts']}
        simple['actual_group_source_rows']=np.asarray(info['group_source_row_ids']).tolist()
        simple['actual_group_unique_counts']=np.asarray(info['group_unique_source_counts']).tolist();metadata.append(simple)
    cache=out/'actual_pointbert_tokens.npz'
    if cache.exists():raise FileExistsError('Refusing overwrite actual encoder cache')
    np.savez_compressed(cache,radar_tokens=np.stack(tokens)[None],radar_centers=np.stack(centers)[None],
                        radar_attributes=np.stack(attrs)[None],radar_mask=np.stack(masks)[None])
    _write_json(out/'actual_pointbert_tokens_receipt.json',{'source_raw_window_sha256':sha(raw_window_path),'encoder_callback':'actualofficialPointBERTgroups96withstrictpretrainedtensors; owner suppliescheckpoint/source receipt',
                                                          'point_budget':point_budget,'seed':seed,'cache_sha256':sha(cache),'frame_metadata':metadata,
                                                          'is_language_model_execution':False,'is_performance_experiment':False})
    return cache


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


def assemble_adapter_inputs(raw_window_path,pointbert_cache_path,rgb_result:Mapping[str,Any],output_dir=None):
    """Compose the actual group/RGB embeddings with true calibration and crop masks.

    ``rgb_result`` is the FrozenRGBEncoder.encode_sequence output, never GT.
    Missing reference masks are allowed; invalid UV slots are finite zeros with
    an independent false mask, so no NaN can leak through a masked cdist.
    """
    import torch
    raw=json.loads(Path(raw_window_path).read_text());frames=raw['frames'];t=len(frames)
    point_receipt_path=Path(pointbert_cache_path).with_name('actual_pointbert_tokens_receipt.json')
    if not point_receipt_path.exists():raise ValueError('Missing actualPointBERT cache provenance receipt')
    point_receipt=json.loads(point_receipt_path.read_text())
    if point_receipt['source_raw_window_sha256']!=sha(raw_window_path)or point_receipt['cache_sha256']!=sha(pointbert_cache_path):
        raise ValueError('PointBERT cache does not belong to this immutable measured window')
    evaluation_path=Path(raw_window_path).with_name('evaluation_time_manifest.json')
    if not evaluation_path.exists():raise ValueError('Explicit officialsample-time metadata binding is required')
    evaluation=json.loads(evaluation_path.read_text())
    if evaluation['raw_window_manifest_sha256']!=sha(raw_window_path):raise ValueError('Stale evaluation-time binding')
    if [f['sample_token']for f in evaluation['frames']]!=[f['sample_token']for f in frames]:raise ValueError('Evaluation timestamps belong to a different window')
    with np.load(pointbert_cache_path,allow_pickle=False)as z:p={k:z[k]for k in z.files}
    if p['radar_tokens'].shape!=(1,t,97,384):raise ValueError('Native real point-token cache shape mismatch')
    required=['reference_tokens','reference_uv','reference_mask','A_processed_from_original','processed_size_wh']
    for key in required:
        if key not in rgb_result:raise ValueError('Actual RGB encoder result missing '+key)
    ref=_numpy(rgb_result['reference_tokens']).astype(np.float32);uvref=_numpy(rgb_result['reference_uv']).astype(np.float32)
    refmask=_numpy(rgb_result['reference_mask']).astype(bool);affine=_numpy(rgb_result['A_processed_from_original']).astype(np.float64)
    if ref.shape[:2]!=(1,t)or ref.shape[-1]!=384 or uvref.shape!=(*ref.shape[:3],2)or refmask.shape!=ref.shape[:3]:raise ValueError('ActualRGBpatch token/UV/mask dimensions differ')
    if rgb_result.get('reference_has_cls',False):raise ValueError('This data interface expects explicit exclude-CLS RGB patch tokens')
    image_metadata=rgb_result.get('source_image_metadata')
    if image_metadata is None or len(image_metadata)!=t:raise ValueError('ActualRGB per-image source metadata is required')
    rgb_cache_receipt=rgb_result.get('_cache_receipt')
    if rgb_cache_receipt is not None:
        if rgb_cache_receipt['raw_window_manifest_sha256']!=sha(raw_window_path):raise ValueError('RGB cache belongs to a different measured window')
        if rgb_cache_receipt['sample_tokens']!=[f['sample_token']for f in frames]:raise ValueError('RGB cache source order differs')
    for j,(f,meta)in enumerate(zip(frames,image_metadata)):
        if meta['image_sha256']!=f['rgb_sha256']or [meta['native_width'],meta['native_height']]!=f['image_size_wh']:raise ValueError('Wrong RGB image/size in token cache')
        if hashlib.sha256(np.ascontiguousarray(ref[0,j]).tobytes()).hexdigest()!=meta['patch_tokens_sha256']:raise ValueError('RGB token bytes do not match their actual image-encoder receipt')
        if hashlib.sha256(np.ascontiguousarray(p['radar_tokens'][0,j]).tobytes()).hexdigest()!=point_receipt['frame_metadata'][j]['feature_sha256']:raise ValueError('PointBERT token bytes changed after actual encoder receipt')
    size=np.asarray(rgb_result['processed_size_wh']).reshape(-1,2)
    if len(size)==1:size=np.repeat(size,t,axis=0)
    if affine.shape!=(1,t,3,3)or len(size)!=t:raise ValueError('RGBpreprocessingaffine/size does not match frames')
    projected=[];valids=[];camera_centers=[]
    for j,f in enumerate(frames):
        xyz_cam=_transform(f['T_camera_from_global'],p['radar_centers'][0,j]);camera_centers.append(xyz_cam)
        positive=np.isfinite(xyz_cam).all(1)&(xyz_cam[:,2]>.1)
        native_uv=np.zeros((96,2),np.float64)
        homogeneous=xyz_cam[positive]@np.asarray(f['camera_intrinsic'],np.float64).T
        native_uv[positive]=homogeneous[:,:2]/homogeneous[:,2:3]
        pix=np.column_stack([native_uv,np.ones(96)])@affine[0,j].T
        processed_uv=pix[:,:2]/pix[:,2:3]
        width,height=f['image_size_wh'];pw,ph=size[j]
        inside=positive&(native_uv[:,0]>=0)&(native_uv[:,0]<width)&(native_uv[:,1]>=0)&(native_uv[:,1]<height)
        inside&=(processed_uv[:,0]>=0)&(processed_uv[:,0]<pw)&(processed_uv[:,1]>=0)&(processed_uv[:,1]<ph)
        native_uv[~inside]=0.;projected.append(native_uv.astype(np.float32));valids.append(inside)
    times=np.asarray([[f['evaluation_timestamp_us']for f in evaluation['frames']]],np.int64)
    reference_times=np.asarray([[f['reference_timestamp_us']for f in frames]],np.int64)
    context={'radar_centers':torch.from_numpy(p['radar_centers'].astype(np.float32)),
             'radar_attributes':torch.from_numpy(p['radar_attributes'].astype(np.float32)),
             'radar_mask':torch.from_numpy(p['radar_mask'].astype(bool)),
             'reference_mask':torch.from_numpy(refmask),'frame_timestamps':torch.from_numpy(times),
             'reference_timestamps':torch.from_numpy(reference_times),'sequence_ids':torch.zeros((1,t),dtype=torch.int64),
             'calibration_valid':torch.ones((1,t),dtype=torch.bool),'history_valid':torch.ones((1,t),dtype=torch.bool),
             'projected_uv':torch.from_numpy(np.stack(projected)[None]),'reference_uv':torch.from_numpy(uvref),
             'projection_valid':torch.from_numpy(np.stack(valids)[None])}
    # Derive RGB bandwidth from real encoder patch spacing, not labels/test scores.
    side=int(round(np.sqrt(ref.shape[2])))
    if side*side!=ref.shape[2]:raise ValueError('Expected recorded square RGB patchgrid for spacing-derivedsigma')
    grid=uvref.reshape(1,t,side,side,2)
    dx=np.linalg.norm(np.diff(grid,axis=3),axis=-1).mean();dy=np.linalg.norm(np.diff(grid,axis=2),axis=-1).mean()
    sigma=float(np.sqrt(dx*dy))
    arrays={'radar_tokens':torch.from_numpy(p['radar_tokens'].astype(np.float32)),
            'reference_tokens':torch.from_numpy(ref),'physical_context':context}
    for name,value in [('radar_tokens',arrays['radar_tokens']),('reference_tokens',arrays['reference_tokens']),*context.items()]:
        if torch.is_floating_point(value)and not torch.isfinite(value).all():raise ValueError('Non-finite forward input '+name)
    receipt={'scope':'actualmeasuredradar/nativePointBERTandactualRGBencoderinputassembly; not7BLMexecutionorperformance',
             'raw_window_sha256':sha(raw_window_path),'evaluation_time_manifest_sha256':sha(evaluation_path),'evaluation_time_policy':evaluation['evaluation_time_policy'],'source_camera_time_checks':evaluation['all404_source_camera_timestamps_checked'],'actual_pointbert_cache_sha256':sha(pointbert_cache_path),
             'actual_rgb_cache_sha256':rgb_cache_receipt['cache_sha256'] if rgb_cache_receipt is not None else None,
             'reference_kind':'rgb_patch','reference_has_cls':False,'native_point_tokens':97,'reference_patch_tokens':ref.shape[2],
             'spatial_sigma_native_pixels':sigma,'sigma_policy':'geometricmean of actualpatchgridspacing induced by encoderresize/crop; no label/test-score fitting',
             'history_sigma_m':1.,'geometry_dims':2,'projected_in_crop_counts':[int(x.sum())for x in valids],
             'unmatched_radar_groups_retained':True,'source_sample_tokens':[f['sample_token']for f in frames],
             'all_radar_sources_at_or_before_camera_anchor':all(f['radar_time_max_us']<=f['frame_anchor_timestamp_us']for f in frames),'all_camera_anchors_at_or_before_official_sampletime':all(frames[j]['frame_anchor_timestamp_us']<=evaluation['frames'][j]['evaluation_timestamp_us']for j in range(t)),
             'no_gt_forward_keys':True,'annotation_sidecar':'supervision_only.json; never openedbythisassembly',
             'projection_is_not_visibility_or_absence':True,'actor_motion_compensation':False}
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        numeric={'radar_tokens':arrays['radar_tokens'].numpy(),'reference_tokens':arrays['reference_tokens'].numpy()}
        numeric.update({'context_'+k:v.numpy()for k,v in context.items()})
        np.savez_compressed(out/'actual_radar_rgb_adapter_inputs.npz',**numeric)
        receipt['assembled_input_sha256']=sha(out/'actual_radar_rgb_adapter_inputs.npz')
        _write_json(out/'actual_radar_rgb_adapter_inputs_receipt.json',receipt)
    return arrays,receipt


def load_rgb_cache(path):
    """Read actual encoder cache only after strict whole-file/source receipt checks."""
    path=Path(path);receipt_path=path.with_name(path.stem+'_receipt.json')
    if not receipt_path.exists():raise ValueError('RGB cache lacks its actual-encoder provenance receipt')
    receipt=json.loads(receipt_path.read_text())
    if receipt['cache_sha256']!=sha(path):raise ValueError('RGB cache SHA differs from actual encoder receipt')
    with np.load(path,allow_pickle=False)as z:r={k:z[k]for k in z.files}
    r['reference_has_cls']=receipt['reference_has_cls'];r['source_image_metadata']=receipt['source_image_metadata'];r['_cache_receipt']=receipt
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'v6_nuscenes_mini/causal_scene_manifest.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--length',type=int,default=3);p.add_argument('--split',default='mini_train');p.add_argument('--scene');p.add_argument('--end-sample-token');p.add_argument('--point-policy',default='standard_sensor',choices=['standard_sensor','all_finite'])
    p=sub.add_parser('assemble');p.add_argument('--raw-window',type=Path,required=True);p.add_argument('--point-cache',type=Path,required=True);p.add_argument('--rgb-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        path=prepare_raw_window(args.manifest,args.output,split=args.split,length=args.length,scene=args.scene,end_sample_token=args.end_sample_token,point_policy=args.point_policy)
        write_supervision_sidecar(args.manifest,path);write_evaluation_time_binding(args.manifest,path);print(path)
    else:
        _,receipt=assemble_adapter_inputs(args.raw_window,args.point_cache,load_rgb_cache(args.rgb_cache),args.output);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
