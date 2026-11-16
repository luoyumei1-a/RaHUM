"""Small dependency-free nuScenes reader: NumPy, PCD, explicit rigid frames.

Only measured sensor values enter point arrays. Annotation tables are accessed
separately by manifest construction and must not be used as forward features.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

RADAR_CHANNELS = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
RADAR_FIELDS = ['x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp', 'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms', 'invalid_state', 'pdh0', 'vx_rms', 'vy_rms']
MINI_TRAIN = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
MINI_VAL = ['scene-0103', 'scene-0916']

def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def quaternion_rotation(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-6):
        raise ValueError('Source quaternion is not normalized.')
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def rigid(translation, rotation):
    out = np.eye(4, dtype=np.float64)
    out[:3,:3] = quaternion_rotation(rotation)
    out[:3,3] = np.asarray(translation, dtype=np.float64)
    return out

def transform_xyz(matrix, xyz):
    return np.asarray(xyz, dtype=np.float64) @ matrix[:3,:3].T + matrix[:3,3]

def load_tables(data_root):
    p = Path(data_root) / 'v1.0-mini'
    names = ['attribute', 'calibrated_sensor', 'category', 'ego_pose', 'instance', 'log', 'map', 'sample', 'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility']
    tables = {name: json.loads((p/(name+'.json')).read_text()) for name in names}
    index = {name: {r['token']: r for r in rows} for name, rows in tables.items()}
    return tables, index

def load_radar_pcd(path):
    """Read all raw PCD rows; retain hardware flags and expose a separate mask."""
    header = {}
    with Path(path).open('rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError('No PCD DATA header.')
            s = line.decode('ascii').strip()
            if not s or s.startswith('#'):
                continue
            key, *values = s.split()
            header[key] = values
            if key == 'DATA':
                if values != ['binary']:
                    raise ValueError('Only the actual binary nuScenes PCD format is supported.')
                break
        payload = f.read()
    fields = header['FIELDS']
    if fields != RADAR_FIELDS:
        raise ValueError('Unexpected radar field order: ' + repr(fields))
    sizes = [int(x) for x in header['SIZE']]
    counts = [int(x) for x in header['COUNT']]
    if counts != [1]*18:
        raise ValueError('Unexpected PCD count specification.')
    kinds = {'F':'f', 'I':'i', 'U':'u'}
    dtype = np.dtype([(name, '<'+kinds[kind]+str(size)) for name, size, kind in zip(fields, sizes, header['TYPE'])], align=False)
    n = int(header['POINTS'][0])
    declared_bytes = n*dtype.itemsize
    if len(payload) < declared_bytes:
        raise ValueError(f'PCD payload shorter than declared points: {len(payload)} vs {n}*{dtype.itemsize}')
    # The official reader consumes exactly the header's WIDTH records and
    # ignores any remaining bytes. Some official files contain zero padding.
    tail = payload[declared_bytes:]
    records = np.frombuffer(payload[:declared_bytes], dtype=dtype)
    matrix = np.column_stack([records[name] for name in fields]).astype(np.float64)
    # The official reader treats a NaN first x as an empty-cloud sentinel.
    if len(matrix) and np.isnan(matrix[0]).any():
        matrix = matrix[:0]
    finite_xyz = np.isfinite(matrix[:,:3]).all(axis=1)
    standard_mask = finite_xyz & (matrix[:,14] == 0) & np.isin(matrix[:,3], np.arange(7)) & (matrix[:,11] == 3)
    return matrix, standard_mask, {'fields': fields, 'sizes': sizes, 'types': header['TYPE'], 'count': counts, 'header_points': n, 'returned_points': len(matrix), 'packed_record_bytes': dtype.itemsize, 'payload_bytes':len(payload), 'trailing_bytes_after_declared_points':len(tail), 'trailing_bytes_all_zero':not any(tail), 'file_sha256': file_sha(path), 'empty_cloud_nan_sentinel': bool(n and len(matrix)==0), 'official_filter': {'invalid_state_keep':[0], 'dyn_prop_keep':list(range(7)), 'ambig_state_keep':[3]}, 'filter_not_semantic_truth': True}

def sensor_chain(index, radar_sd, camera_sd):
    rcal = index['calibrated_sensor'][radar_sd['calibrated_sensor_token']]
    ccal = index['calibrated_sensor'][camera_sd['calibrated_sensor_token']]
    rpose = index['ego_pose'][radar_sd['ego_pose_token']]
    cpose = index['ego_pose'][camera_sd['ego_pose_token']]
    ego_r_from_radar = rigid(rcal['translation'], rcal['rotation'])
    global_from_ego_r = rigid(rpose['translation'], rpose['rotation'])
    global_from_ego_c = rigid(cpose['translation'], cpose['rotation'])
    ego_c_from_camera = rigid(ccal['translation'], ccal['rotation'])
    global_from_radar = global_from_ego_r @ ego_r_from_radar
    camera_from_global = np.linalg.inv(ego_c_from_camera) @ np.linalg.inv(global_from_ego_c)
    camera_from_radar = camera_from_global @ global_from_radar
    return {
        'T_ego_at_radar_from_radar_sensor': ego_r_from_radar,
        'T_global_from_ego_at_radar': global_from_ego_r,
        'T_ego_at_camera_from_global': np.linalg.inv(global_from_ego_c),
        'T_camera_from_ego_at_camera': np.linalg.inv(ego_c_from_camera),
        'T_global_from_radar_sensor': global_from_radar,
        'T_ego_at_camera_from_radar_sensor': np.linalg.inv(global_from_ego_c) @ global_from_radar,
        'T_camera_from_global': camera_from_global,
        'T_camera_from_radar_sensor': camera_from_radar,
    }

def project_camera(camera_xyz, intrinsic, width, height):
    xyz = np.asarray(camera_xyz)
    uv = np.full((len(xyz),2), np.nan, dtype=np.float64)
    positive = np.isfinite(xyz).all(axis=1) & (xyz[:,2] > 0.1)
    homogeneous = xyz[positive] @ np.asarray(intrinsic).T
    uv[positive] = homogeneous[:,:2] / homogeneous[:,2:3]
    inside = positive & (uv[:,0] >= 0) & (uv[:,0] < width) & (uv[:,1] >= 0) & (uv[:,1] < height)
    return uv, inside

def annotation_corners_global(annotation):
    w, length, h = annotation['size']
    local = np.array([[sx*length/2, sy*w/2, sz*h/2] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    return transform_xyz(rigid(annotation['translation'], annotation['rotation']), local)
