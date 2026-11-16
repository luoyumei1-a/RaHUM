"""Re-pair real radar strictly at/before each front-camera output anchor.

This never substitutes future measurements at scene startup. Independent GT
retains its actual sample timestamp, including its offset from camera time.
"""
from pathlib import Path
import bisect
import collections
import json
import numpy as np
from nuscenes_data_interface import *

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
OUT = BASE / 'paired_causal'
MAXIMUM_PAST_AGE_US = 200_000

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    tables,ix=load_tables(DATA)
    original=json.loads((BASE/'paired_scene_manifest.json').read_text())
    streams=collections.defaultdict(list)
    for sd in tables['sample_data']:
        channel=ix['sensor'][ix['calibrated_sensor'][sd['calibrated_sensor_token']]['sensor_token']]['channel']
        if channel in RADAR_CHANNELS:
            streams[(ix['sample'][sd['sample_token']]['scene_token'],channel)].append(sd)
    times={}
    for key,rows in streams.items():
        rows.sort(key=lambda r:r['timestamp']);times[key]=[r['timestamp'] for r in rows]
    out=[];availability=collections.Counter();all_age=[];all_rows=all_filtered=all_front=all_front_filtered=0
    OUT.mkdir(exist_ok=True);(OUT/'measurements').mkdir(exist_ok=True)
    for old in original['records']:
        token=old['sample_token'];cam=ix['sample_data'][old['camera']['sample_data_token']];anchor=int(cam['timestamp']);sc=old['scene_token'];K=np.asarray(old['camera']['camera_intrinsic']);width=cam['width'];height=cam['height']
        columns=collections.defaultdict(list);radar=[];missing=[]
        for sensor_id,channel in enumerate(RADAR_CHANNELS):
            key=(sc,channel);i=bisect.bisect_right(times[key],anchor)-1
            sd=streams[key][i] if i>=0 else None
            reason='no_same_scene_observation_at_or_before_camera_anchor' if sd is None else 'past_observation_older_than_fixed_200ms_cap' if anchor-sd['timestamp']>MAXIMUM_PAST_AGE_US else None
            if reason:
                missing.append(channel);radar.append({'channel':channel,'sensor_index':sensor_id,'available':False,'reason':reason,'camera_anchor_timestamp_us':anchor,'raw_rows':0});continue
            assert sd['timestamp']<=anchor
            p,mask,receipt=load_radar_pcd(DATA/sd['filename']);n=len(p);chains=sensor_chain(ix,sd,cam)
            xyz_cam=transform_xyz(chains['T_camera_from_radar_sensor'],p[:,:3]);uv,view=project_camera(xyz_cam,K,width,height)
            np.testing.assert_allclose(transform_xyz(np.linalg.inv(chains['T_camera_from_radar_sensor']),xyz_cam),p[:,:3],atol=1e-8,rtol=1e-9)
            values={'radar_features18_sensor':p,'radar_xyz_sensor':p[:,:3],'radar_xyz_camera':xyz_cam,'radar_xyz_global':transform_xyz(chains['T_global_from_radar_sensor'],p[:,:3]),'radar_xyz_ego_at_camera':transform_xyz(chains['T_ego_at_camera_from_radar_sensor'],p[:,:3]),'radar_sensor_id':np.full(n,sensor_id,dtype=np.uint8),'radar_source_row_id':np.arange(n,dtype=np.int32),'radar_sample_data_token':np.full(n,sd['token'],dtype='U32'),'radar_timestamp_us':np.full(n,sd['timestamp'],dtype=np.int64),'radar_age_at_anchor_us':np.full(n,anchor-sd['timestamp'],dtype=np.int64),'radar_standard_filter_mask':mask,'radar_uv_native':uv,'radar_projection_in_front_image':view}
            for k,v in values.items():columns[k].append(v)
            previous=ix['sample_data'].get(sd['prev']);valid_previous=previous is not None and ix['sample'][previous['sample_token']]['scene_token']==sc and previous['timestamp']<sd['timestamp']
            all_age.append(anchor-sd['timestamp'])
            radar.append({'channel':channel,'sensor_index':sensor_id,'available':True,'sample_data_token':sd['token'],'source_sample_token':sd['sample_token'],'source_sample_scene_token':ix['sample'][sd['sample_token']]['scene_token'],'filename':sd['filename'],'source_sha256':receipt['file_sha256'],'timestamp_us':sd['timestamp'],'camera_time_delta_us':sd['timestamp']-anchor,'age_at_camera_anchor_us':anchor-sd['timestamp'],'ego_pose_token':sd['ego_pose_token'],'calibrated_sensor_token':sd['calibrated_sensor_token'],'raw_rows':n,'standard_filter_rows':int(mask.sum()),'projected_inside_front_rgb_rows':int(view.sum()),'standard_filter_and_projection_rows':int((mask&view).sum()),'pcd_schema':receipt,'rigid_transforms':{k:v.tolist() for k,v in chains.items()},'history_previous_same_scene_sample_data_token':previous['token'] if valid_previous else None})
        availability[len(RADAR_CHANNELS)-len(missing)]+=1
        measurement=None;measurement_sha=None;nr=nf=nv=nfv=0
        if columns:
            arrays={k:np.concatenate(v) for k,v in columns.items()}
            arrays.update({'camera_intrinsic':K,'camera_timestamp_us':np.asarray(anchor,dtype=np.int64),'frame_anchor_timestamp_us':np.asarray(anchor,dtype=np.int64),'camera_size_wh':np.asarray([width,height],dtype=np.int32)})
            assert np.all(arrays['radar_timestamp_us']<=anchor)
            nr=len(arrays['radar_timestamp_us']);nf=int(arrays['radar_standard_filter_mask'].sum());nv=int(arrays['radar_projection_in_front_image'].sum());nfv=int((arrays['radar_standard_filter_mask']&arrays['radar_projection_in_front_image']).sum())
            path=OUT/'measurements'/(token+'.npz');np.savez_compressed(path,**arrays);measurement=path.relative_to(BASE).as_posix();measurement_sha=file_sha(path)
        all_rows+=nr;all_filtered+=nf;all_front+=nv;all_front_filtered+=nfv
        keep={k:v for k,v in old.items() if k not in ('measurement_file','measurement_sha256','radar','raw_radar_rows','standard_filter_radar_rows','native_projected_radar_rows','standard_filter_projected_radar_rows')}
        record={**keep,'temporal_policy':'camera-anchored causal pair: every selected radar source timestamp is <= the actual CAM_FRONT timestamp, within the same source scene; no future sensor measurement is used.','frame_anchor_timestamp_us':anchor,'reference_timestamp_us':anchor,'sample_annotation_time_delta_us':old['sample_timestamp_us']-anchor,'annotation_temporal_scope':'Source boxes/classes remain at the original official sample timestamp; no fabricated claim of exact camera-time box annotation or object-motion compensation is made. Ego-motion compensation uses each actual radar/camera pose.','radar':radar,'missing_radar_channels':missing,'causal_radar_channels_available':5-len(missing),'measurement_available':measurement is not None and nr>0,'measurement_file':measurement,'measurement_sha256':measurement_sha,'raw_radar_rows':nr,'standard_filter_radar_rows':nf,'native_projected_radar_rows':nv,'standard_filter_projected_radar_rows':nfv,'no_gt_forward_context':'Do not pass annotation files/counts/category names, scene.description, or source benchmark split as model context; these exist for supervision/audit only.'}
        out.append(record)
        if record['measurement_available'] and not (BASE/'first_causal_real_pair_manifest.json').exists():
            write(BASE/'first_causal_real_pair_manifest.json',record)
    inherited={k:v for k,v in original.items() if k not in ('records','counts','radar_to_camera_timestamp_delta_us')}
    counts={'planned_camera_keyframes':len(out),'causal_measurement_keyframes':sum(r['measurement_available'] for r in out),'missing_past_radar_camera_keyframes':sum(not r['measurement_available'] for r in out),'available_radar_channel_histogram':dict(availability),'causal_mini_train_keyframes':sum(r['measurement_available'] and r['official_mini_split']=='mini_train' for r in out),'causal_mini_val_keyframes':sum(r['measurement_available'] and r['official_mini_split']=='mini_val' for r in out),'all_raw_radar_rows':all_rows,'standard_filter_radar_rows':all_filtered,'raw_projected_front_rows':all_front,'standard_filter_projected_front_rows':all_front_filtered}
    manifest={**inherited,'scope':'Complete official mini camera-anchor inventory. 404 keyframe anchors are retained; startup anchors without any past radar stay explicitly unavailable rather than receiving future radar. This is not full nuScenes trainval/test.','temporal_policy':{'output_anchor':'actual CAM_FRONT sample_data.timestamp','radar_selection':'latest real per-channel source timestamp <= camera anchor, same scene, from keyframes or sweeps','maximum_past_age_us':MAXIMUM_PAST_AGE_US,'reference_timestamp_rule':'CAM_FRONT reference_timestamp == frame_anchor_timestamp; all source radar timestamps <= this anchor','no_future_fill':True,'label_time_policy':'Original official sample-level annotations retain their actual timestamp and offset from camera anchor; not forward features.','absolute_time_precision':'Keep epoch timestamps as int64; subtract anchor in int64/float64 before converting relative seconds to float32.'},'records':out,'counts':counts,'radar_age_at_camera_anchor_us':{'min':min(all_age),'max':max(all_age),'median':float(np.median(all_age))},'annotation_offsets_us':{'min':min(r['sample_annotation_time_delta_us'] for r in out),'max':max(r['sample_annotation_time_delta_us'] for r in out),'median':float(np.median([r['sample_annotation_time_delta_us'] for r in out]))}}
    write(BASE/'causal_scene_manifest.json',manifest);write(BASE/'causal_pairing_summary.json',{k:v for k,v in manifest.items() if k not in ('records','taxonomy_classes','scenes')})
    # Mark the earlier nearest-keyframe export explicitly as a noncausal audit.
    original['temporal_policy']='Official nearest keyframe grouping is not guaranteed causal. Use causal_scene_manifest.json for causal model input.'
    write(BASE/'paired_scene_manifest.json',original)
    print(json.dumps(counts),flush=True)

if __name__=='__main__':
    main()
