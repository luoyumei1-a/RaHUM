"""Export 404 real mini pairs, measurement arrays and separate annotation truth."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np
from PIL import Image
from nuscenes_data_interface import *

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
OUT = BASE / 'paired'

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    tables, ix = load_tables(DATA)
    (OUT / 'measurements').mkdir(parents=True, exist_ok=True)
    (OUT / 'truth').mkdir(exist_ok=True)
    keyed = collections.defaultdict(dict)
    for sd in tables['sample_data']:
        if sd['is_key_frame']:
            channel = ix['sensor'][ix['calibrated_sensor'][sd['calibrated_sensor_token']]['sensor_token']]['channel']
            if channel in RADAR_CHANNELS or channel == 'CAM_FRONT':
                assert channel not in keyed[sd['sample_token']]
                keyed[sd['sample_token']][channel] = sd
    annotations = collections.defaultdict(list)
    for a in tables['sample_annotation']:
        annotations[a['sample_token']].append(a)
    records = []
    class_counts = collections.Counter()
    scene_counts = collections.Counter()
    raw_n = filtered_n = projected_n = filtered_projected_n = 0
    timestamp_deltas = []
    for sample in sorted(tables['sample'], key=lambda s:(ix['scene'][s['scene_token']]['name'],s['timestamp'])):
        st = sample['token']
        scene = ix['scene'][sample['scene_token']]
        split = 'mini_train' if scene['name'] in MINI_TRAIN else 'mini_val' if scene['name'] in MINI_VAL else None
        assert split is not None
        scene_counts[scene['name']] += 1
        assert set(keyed[st]) == set(RADAR_CHANNELS + ['CAM_FRONT'])
        cam = keyed[st]['CAM_FRONT']
        camera_file = DATA / cam['filename']
        assert camera_file.is_file()
        with Image.open(camera_file) as image:
            assert image.size == (cam['width'],cam['height'])
        ccal = ix['calibrated_sensor'][cam['calibrated_sensor_token']]
        cp = ix['ego_pose'][cam['ego_pose_token']]
        intrinsic = np.array(ccal['camera_intrinsic'],dtype=float)
        rows=[]; sensors=[]; source_rows=[]; source_tokens=[]; stamps=[]; standard=[]; cameras=[]; globals_=[]; egos=[]; uvs=[]; in_views=[]
        radar_records=[]
        for sensor_id,channel in enumerate(RADAR_CHANNELS):
            sd = keyed[st][channel]
            path = DATA / sd['filename']
            assert path.is_file()
            points, mask, pcd_meta = load_radar_pcd(path)
            chains = sensor_chain(ix,sd,cam)
            xyz = points[:,:3]
            xyz_cam = transform_xyz(chains['T_camera_from_radar_sensor'],xyz)
            xyz_global = transform_xyz(chains['T_global_from_radar_sensor'],xyz)
            xyz_ego = transform_xyz(chains['T_ego_at_camera_from_radar_sensor'],xyz)
            uv,in_view = project_camera(xyz_cam,intrinsic,cam['width'],cam['height'])
            # Reversibility and explicit four-link chain checks are geometry tests,
            # not labels or scene-support decisions.
            np.testing.assert_allclose(transform_xyz(np.linalg.inv(chains['T_camera_from_radar_sensor']),xyz_cam),xyz,rtol=1e-9,atol=1e-8)
            product = chains['T_camera_from_ego_at_camera'] @ chains['T_ego_at_camera_from_global'] @ chains['T_global_from_ego_at_radar'] @ chains['T_ego_at_radar_from_radar_sensor']
            np.testing.assert_allclose(product,chains['T_camera_from_radar_sensor'],rtol=1e-10,atol=1e-10)
            previous = ix['sample_data'].get(sd['prev'])
            same_scene_prev = previous is not None and ix['sample'][previous['sample_token']]['scene_token']==sample['scene_token'] and previous['timestamp']<sd['timestamp']
            delta = int(sd['timestamp']-cam['timestamp'])
            timestamp_deltas.append(delta)
            n=len(points)
            rows.append(points);sensors.append(np.full(n,sensor_id,dtype=np.uint8));source_rows.append(np.arange(n,dtype=np.int32));source_tokens.append(np.full(n,sd['token'],dtype='U32'));stamps.append(np.full(n,sd['timestamp'],dtype=np.int64));standard.append(mask);cameras.append(xyz_cam);globals_.append(xyz_global);egos.append(xyz_ego);uvs.append(uv);in_views.append(in_view)
            radar_records.append({'channel':channel,'sensor_index':sensor_id,'sample_data_token':sd['token'],'filename':sd['filename'],'source_sha256':pcd_meta['file_sha256'],'timestamp_us':sd['timestamp'],'camera_time_delta_us':delta,'ego_pose_token':sd['ego_pose_token'],'calibrated_sensor_token':sd['calibrated_sensor_token'],'raw_rows':n,'standard_filter_rows':int(mask.sum()),'projected_inside_front_rgb_rows':int(in_view.sum()),'standard_filter_and_projection_rows':int((mask&in_view).sum()),'pcd_schema':pcd_meta,'rigid_transforms':{k:v.tolist() for k,v in chains.items()},'history_previous_same_scene_sample_data_token':previous['token'] if same_scene_prev else None})
        full=np.concatenate(rows)
        masks=np.concatenate(standard);viewmask=np.concatenate(in_views)
        source_id=np.concatenate(source_rows)
        measure_file=OUT/'measurements'/(st+'.npz')
        np.savez_compressed(measure_file,radar_features18_sensor=full,radar_xyz_sensor=full[:,:3],radar_xyz_camera=np.concatenate(cameras),radar_xyz_global=np.concatenate(globals_),radar_xyz_ego_at_camera=np.concatenate(egos),radar_sensor_id=np.concatenate(sensors),radar_source_row_id=source_id,radar_sample_data_token=np.concatenate(source_tokens),radar_timestamp_us=np.concatenate(stamps),radar_standard_filter_mask=masks,radar_uv_native=np.concatenate(uvs),radar_projection_in_front_image=viewmask,camera_intrinsic=intrinsic,camera_timestamp_us=np.asarray(cam['timestamp'],dtype=np.int64),camera_size_wh=np.array([cam['width'],cam['height']],dtype=np.int32))
        raw_n+=len(full);filtered_n+=int(masks.sum());projected_n+=int(viewmask.sum());filtered_projected_n+=int((masks&viewmask).sum())
        t_camera_global=radar_records[0]['rigid_transforms']['T_camera_from_global']
        t_camera_global=np.asarray(t_camera_global)
        truth=[]
        for a in annotations[st]:
            cat=ix['category'][ix['instance'][a['instance_token']]['category_token']]['name']
            class_counts[cat]+=1
            center=transform_xyz(t_camera_global,np.array(a['translation'])[None])[0]
            corners=transform_xyz(t_camera_global,annotation_corners_global(a))
            corners_uv,_=project_camera(corners,intrinsic,cam['width'],cam['height'])
            valid=np.isfinite(corners_uv).all(axis=1)
            projected_bbox=None
            intersects=False
            if valid.any():
                low=corners_uv[valid].min(axis=0);high=corners_uv[valid].max(axis=0)
                projected_bbox=[float(low[0]),float(low[1]),float(high[0]),float(high[1])]
                intersects=bool(high[0]>=0 and high[1]>=0 and low[0]<cam['width'] and low[1]<cam['height'])
            truth.append({**a,'category_name':cat,'attribute_names':[ix['attribute'][t]['name'] for t in a['attribute_tokens']],'visibility_metadata':ix['visibility'].get(a['visibility_token']),'sample_timestamp_us':sample['timestamp'],'camera_timestamp_us':cam['timestamp'],'center_camera_m':center.tolist(),'rotation_matrix_camera_from_box':(t_camera_global[:3,:3]@quaternion_rotation(a['rotation'])).tolist(),'corners_camera_m':corners.tolist(),'projected_bbox_native_unclipped':projected_bbox,'projected_3d_box_intersects_front_image':intersects,'front_camera_visibility_independently_verified':None,'truth_scope':'Original class/box at sample timestamp; all-camera visibility code is not an independently verified CAM_FRONT occlusion label. Projection does not establish visibility or absence.'})
        truth_file=OUT/'truth'/(st+'.json')
        write_json(truth_file,{'sample_token':st,'scene_token':scene['token'],'scene_name':scene['name'],'official_mini_split':split,'sample_timestamp_us':sample['timestamp'],'annotations':truth,'annotation_truth_never_forward_features':True,'no_pointwise_ghost_labels_supplied':True})
        record={'sample_token':st,'scene_token':scene['token'],'scene_name':scene['name'],'official_mini_split':split,'sample_timestamp_us':sample['timestamp'],'previous_sample_token':sample['prev'] or None,'next_sample_token':sample['next'] or None,'measurement_file':measure_file.relative_to(BASE).as_posix(),'measurement_sha256':file_sha(measure_file),'camera':{'sample_data_token':cam['token'],'filename':cam['filename'],'source_sha256':file_sha(camera_file),'timestamp_us':cam['timestamp'],'sample_time_delta_us':cam['timestamp']-sample['timestamp'],'width':cam['width'],'height':cam['height'],'camera_intrinsic':intrinsic.tolist(),'calibrated_sensor_token':cam['calibrated_sensor_token'],'ego_pose_token':cam['ego_pose_token'],'T_ego_at_camera_from_camera':rigid(ccal['translation'],ccal['rotation']).tolist(),'T_global_from_ego_at_camera':rigid(cp['translation'],cp['rotation']).tolist()},'radar':radar_records,'truth_file':truth_file.relative_to(BASE).as_posix(),'truth_sha256':file_sha(truth_file),'number_of_annotations_all_surround':len(truth),'number_of_boxes_projected_to_front_view':sum(t['projected_3d_box_intersects_front_image'] for t in truth),'raw_radar_rows':len(full),'standard_filter_radar_rows':int(masks.sum()),'native_projected_radar_rows':int(viewmask.sum()),'standard_filter_projected_radar_rows':int((masks&viewmask).sum()),'source_point_key':'radar_sample_data_token plus radar_source_row_id; no class, box, label or instance enters measurement NPZ'}
        records.append(record)
        if len(records)==1:
            write_json(BASE/'first_real_pair_manifest.json',record)
            print('first_real_pair_ready',st,flush=True)
    scenes=[]
    for scene in tables['scene']:
        log=ix['log'][scene['log_token']]
        scenes.append({**scene,'official_mini_split':'mini_train' if scene['name'] in MINI_TRAIN else 'mini_val','actual_paired_keyframes':scene_counts[scene['name']],'location':log['location'],'source_log_token':log['token'],'date_captured':log['date_captured']})
    manifest={'dataset':'nuScenes v1.0-mini','scope':'Complete official mini, not full trainval/test benchmark. All five radar channels and CAM_FRONT are retained; LiDAR and other cameras are not used.','source_archive_sha256':json.loads((BASE/'download_receipt.json').read_text())['sha256'],'official_scene_splits':{'mini_train':MINI_TRAIN,'mini_val':MINI_VAL},'scene_split_source':'Official nuscenes-devkit splits.py; 8/2 mini visualization/debugging scenes.','scene_split_policy':'No sample crosses scene boundaries. Any tuning on mini_val makes it development data, not an untouched test set. For final holdout, split development scenes from mini_train before training.','records':records,'scenes':scenes,'metadata_table_counts':{k:len(v) for k,v in tables.items()},'taxonomy_classes':tables['category'],'actual_observed_annotation_categories':dict(class_counts),'sensor_channels':RADAR_CHANNELS,'measurement_feature_order':RADAR_FIELDS,'physical_units':{'xyz':'meters in original radar sensor coordinates; provided transforms produce meters in named frames','vx_vy_vx_comp_vy_comp':'m/s, documented by official devkit','rcs':'Raw radar cross-section feature; unit/calibration not explicitly specified in the inspected official devkit text, no conversion applied.','radar_timestamp_us':'Original integer Unix microseconds for each sample_data record, not inferred from frame number.','quality_and_state_fields':'Original encoded manufacturer/PCD values, not semantic truth labels.'},'counts':{'paired_keyframes':len(records),'scenes':len(scenes),'mini_train_keyframes':sum(r['official_mini_split']=='mini_train' for r in records),'mini_val_keyframes':sum(r['official_mini_split']=='mini_val' for r in records),'all_raw_radar_rows':raw_n,'standard_filter_radar_rows':filtered_n,'raw_projected_front_rows':projected_n,'standard_filter_projected_front_rows':filtered_projected_n,'annotated_boxes_all_surround':sum(class_counts.values())},'radar_to_camera_timestamp_delta_us':{'min':min(timestamp_deltas),'max':max(timestamp_deltas),'median':float(np.median(timestamp_deltas))},'image_preprocessing':'Native RGB and native-pixel UV only. Image encoder resize/crop affine must be separately recorded and applied to UV by the runtime.','visibility_and_absence':'Projection is geometric support only, not proof that an object is visually observable or absent; 3D annotations are limited to the source taxonomy and annotation policy.','data_redistribution':'Do not include raw nuScenes sensor files or this derived measurement cache in the public code archive; retain dataset attribution/license and source-acquisition instructions.'}
    write_json(BASE/'paired_scene_manifest.json',manifest)
    write_json(BASE/'pairing_summary.json',{k:v for k,v in manifest.items() if k not in ('records','taxonomy_classes','scenes')})
    print(json.dumps(manifest['counts']),flush=True)

if __name__=='__main__':
    main()
