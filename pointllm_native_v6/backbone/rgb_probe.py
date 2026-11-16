"""Download pinned official visual weights or encode one real paired image.

Downloading and static geometry checks do not load a Torch model. Actual
feature extraction is an explicit separate step, to avoid concurrent 7B RAM
pressure. This probe does not measure semantic hallucination performance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

REPO = 'facebook/dinov2-small'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
WEIGHT_BYTES = 88249960
WEIGHT_SHA = 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    start = time.time(); receipts = []
    for name in ['config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors']:
        path = directory / name
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            request = urllib.request.Request(url, headers={'User-Agent': 'RaHUM academic source reproducibility'})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as target:
                while block := response.read(4 * 1024**2): target.write(block)
                target.flush(); os.fsync(target.fileno())
            temporary.replace(path)
        digest = sha(path)
        if name == 'model.safetensors' and (path.stat().st_size != WEIGHT_BYTES or digest != WEIGHT_SHA):
            raise RuntimeError('Pinned official visual weight integrity mismatch')
        receipts.append({'file': name, 'bytes': path.stat().st_size, 'sha256': digest,
                         'official_source': url, 'official_repo': REPO, 'official_revision': REVISION})
    result = {'official_repo': REPO, 'official_revision': REVISION, 'license': 'Apache-2.0',
              'weight_lfs_sha256_verified': True, 'files': receipts, 'elapsed_s': time.time()-start,
              'visual_implementation_identity': 'v6 choice, not a recovered original HiRAC vision checkpoint'}
    (directory / 'download_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def geometry_check(output):
    import numpy as np
    from rgb_encoder import preprocessing_geometry, map_native_uv, project_camera_points, jsonable
    geometry = preprocessing_geometry(1600, 900, np.eye(3))
    assert (geometry['resize_width'], geometry['resize_height'], geometry['crop_left'], geometry['crop_top']) == (455, 256, 115, 16)
    native = geometry['patch_uv_native']
    model_uv, mask = map_native_uv(native, geometry, positive_depth_mask=np.ones(256, dtype=bool))
    error = float(np.max(np.abs(model_uv - geometry['patch_uv_model'])))
    assert error < 1e-10 and mask.all()
    # Native side regions are inside the original sensor but outside the crop.
    _, crop_mask = map_native_uv(np.asarray([[0., 450.], [1599., 450.], [800., 450.]]), geometry)
    assert crop_mask.tolist() == [False, False, True]
    xyz = np.asarray([[.1, .1, 1.], [.1, .1, -1.], [np.nan, 0., 1.]])
    projected = project_camera_points(xyz, np.eye(3), geometry)
    assert projected['positive_depth'].tolist() == [True, False, False]
    intrinsic = np.asarray([[1266.4, 0., 816.2], [0., 1266.4, 491.5], [0., 0., 1.]])
    metric = np.asarray([[.2, -.1, 5.], [-.3, .2, 8.]])
    calibrated_geometry = preprocessing_geometry(1600, 900, intrinsic)
    native_h = metric @ intrinsic.T
    native_uv = native_h[:, :2] / native_h[:, 2:]
    affine_uv, _ = map_native_uv(native_uv, calibrated_geometry)
    transformed_h = metric @ calibrated_geometry['camera_intrinsic_model'].T
    processed_uv = transformed_h[:, :2] / transformed_h[:, 2:]
    intrinsic_error = float(np.max(np.abs(affine_uv - processed_uv)))
    assert intrinsic_error < 1e-10
    result = {'status': 'PASS_geometry_contract_only', 'real_encoder_forward_executed': False,
              'patch_count': len(native), 'patch_roundtrip_max_abs_pixels': error,
              'A_K_projection_equivalence_max_abs_pixels': intrinsic_error,
              'native_side_regions_masked_out': True, 'negative_and_nonfinite_depth_rejected': True,
              'geometry': jsonable(geometry)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'geometry'}), flush=True)


def real_image_probe(args):
    import numpy as np
    from rgb_encoder import FrozenRGBEncoder, map_native_uv, jsonable
    manifest = json.loads(args.pair_manifest.read_text())
    if 'camera' not in manifest or not manifest.get('temporal_policy', '').startswith('camera-anchored causal'):
        raise ValueError('A real camera-anchored causal pair manifest is required')
    source_root = args.data_root
    image_path = source_root / manifest['camera']['filename']
    if sha(image_path) != manifest['camera']['source_sha256']:
        raise ValueError('Actual RGB does not match the immutable pair manifest')
    encoder = FrozenRGBEncoder(args.checkpoint_dir, threads=args.threads)
    started = time.time()
    result = encoder.encode(image_path, camera_intrinsic=manifest['camera']['camera_intrinsic'])
    payload = {'patch_tokens': result['tokens'][0].numpy(),
               'cls_token': result['cls_token'][0].numpy(),
               'reference_uv_native': result['reference_uv'][0].numpy(),
               'reference_uv_model': result['reference_uv_model'][0].numpy(),
               'reference_mask': result['reference_mask'][0].numpy(),
               'A_model_from_native': result['geometry']['A_model_from_native'],
               'camera_intrinsic_model': result['geometry']['camera_intrinsic_model']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'real_rgb_tokens.npz'
    np.savez_compressed(path, **payload)
    receipt = dict(result['receipt'])
    receipt.update({'status': 'PASS_actual_frozen_RGB_forward_not_semantic_performance',
        'sample_token': manifest['sample_token'], 'scene_name': manifest['scene_name'],
        'official_split': manifest['official_mini_split'],
        'camera_timestamp_us': manifest['camera']['timestamp_us'],
        'pair_manifest_sha256': sha(args.pair_manifest),
        'cache_sha256': sha(path), 'cache_file': path.name,
        'actual_encode_elapsed_s': time.time()-started,
        'geometry': jsonable(result['geometry']),
        'reference_uv_coordinate_system': 'native camera pixels',
        'scope': 'One actual RGB feature extraction and calibration/crop consistency check; not a language-model performance experiment'})
    (args.output_dir / 'real_rgb_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'geometry'}, indent=2), flush=True)


def real_window_probe(args):
    import numpy as np
    from rgb_encoder import FrozenRGBEncoder, jsonable
    window = json.loads(args.window_manifest.read_text())
    if window.get('reference_kind') != 'rgb_patch' or not window.get('frames'):
        raise ValueError('Expected a measurement-only genuine RGB window manifest')
    frames = window['frames']
    paths = [Path(frame['rgb_path']) for frame in frames]
    for path, frame in zip(paths, frames):
        if sha(path) != frame['rgb_sha256']:
            raise ValueError('Window image no longer matches its immutable source hash')
        if frame['radar_time_max_us'] > frame['frame_anchor_timestamp_us']:
            raise ValueError('A future radar sample entered the camera-anchored window')
    if len({frame['recording_token'] for frame in frames}) != 1:
        raise ValueError('A visual window cannot cross recording boundaries')
    encoder = FrozenRGBEncoder(args.checkpoint_dir, threads=args.threads)
    started = time.time()
    result = encoder.encode_sequence(paths, camera_intrinsics=[f['camera_intrinsic'] for f in frames])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / 'actual_rgb_tokens.npz'
    if cache.exists():
        raise FileExistsError('Refusing to overwrite an actual encoder cache: ' + str(cache))
    payload = {key: result[key].numpy() for key in
               ['reference_tokens', 'reference_uv', 'reference_mask', 'A_processed_from_original']}
    payload['processed_size_wh'] = np.asarray(result['processed_size_wh'], dtype=np.int64)
    np.savez_compressed(cache, **payload)
    receipt = {'status': 'PASS_actual_RGB_window_forward_not_semantic_performance',
               'reference_kind': 'rgb_patch', 'reference_has_cls': False,
               'raw_window_manifest_sha256': sha(args.window_manifest),
               'cache_sha256': sha(cache), 'cache_file': cache.name,
               'reference_tokens_shape': list(result['reference_tokens'].shape),
               'sample_tokens': [f['sample_token'] for f in frames],
               'actual_encode_elapsed_s': time.time()-started,
               'source_image_metadata': result['source_image_metadata'],
               'crop_geometry': jsonable(result['crop_geometry']),
               'scope': 'Real frozen image encoder only, no 7B language execution or performance claim'}
    (args.output_dir / 'actual_rgb_tokens_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['source_image_metadata', 'crop_geometry']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--pair-manifest', type=Path)
    parser.add_argument('--window-manifest', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('rgb_probe_result'))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.download_only: download(args.checkpoint_dir)
    elif args.geometry_only: geometry_check(args.output_dir / 'geometry_contract.json')
    elif args.window_manifest: real_window_probe(args)
    else:
        if not args.pair_manifest or not args.data_root: parser.error('--pair-manifest and --data-root are required for actual RGB forward')
        real_image_probe(args)
