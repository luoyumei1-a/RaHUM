"""Frozen real RGB patch features with an explicit camera-preprocessing map.

This is the v6 visual implementation, not a claim that the supplied HiRAC
paper specified DINOv2. The official pinned slow processor resizes the short
edge to 256 and center-crops 224 pixels. Geometry remains in native camera
pixels by default, with the exact resize/crop affine provided separately.
No point labels or object annotations are accepted by this encoder.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_REPO = 'facebook/dinov2-small'
MODEL_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
MODEL_SHA256 = 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
PATCH_SIZE = 14
SHORT_EDGE = 256
CROP_SIZE = (224, 224)


def file_sha256(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def preprocessing_geometry(width: int, height: int, camera_intrinsic=None) -> dict[str, Any]:
    """Match official integer resize/crop, with zero-based pixel centers.

    Pillow resize aligns image boundaries: u'=(u+0.5)*scale-0.5. The affine
    includes this subpixel term and the integer center-crop translation.
    Patch centers are the mean of their 14 pixel-center coordinates (6.5,
    20.5, ...), not their right-edge/boundary locations.
    """
    if width < 1 or height < 1:
        raise ValueError('Positive native image dimensions are required')
    if width <= height:
        resized_w, resized_h = SHORT_EDGE, int(SHORT_EDGE * height / width)
    else:
        resized_h, resized_w = SHORT_EDGE, int(SHORT_EDGE * width / height)
    crop_h, crop_w = CROP_SIZE
    top, left = (resized_h - crop_h) // 2, (resized_w - crop_w) // 2
    sx, sy = resized_w / width, resized_h / height
    affine = np.asarray([[sx, 0., .5 * sx - .5 - left],
                         [0., sy, .5 * sy - .5 - top], [0., 0., 1.]], dtype=np.float64)
    inverse = np.linalg.inv(affine)
    axis_x = np.arange(crop_w // PATCH_SIZE) * PATCH_SIZE + (PATCH_SIZE - 1) / 2
    axis_y = np.arange(crop_h // PATCH_SIZE) * PATCH_SIZE + (PATCH_SIZE - 1) / 2
    gx, gy = np.meshgrid(axis_x, axis_y, indexing='xy')
    uv_model = np.stack([gx, gy], axis=-1).reshape(-1, 2)
    uv_native_h = np.c_[uv_model, np.ones(len(uv_model))] @ inverse.T
    uv_native = uv_native_h[:, :2] / uv_native_h[:, 2:]
    # Raster support uses the same [0,width) convention as the paired reader.
    bounds_model = np.asarray([[0., 0.], [float(crop_w), 0.],
                               [float(crop_w), float(crop_h)], [0., float(crop_h)]])
    crop_native_h = np.c_[bounds_model, np.ones(4)] @ inverse.T
    result = {
        'native_width': width, 'native_height': height,
        'resize_width': resized_w, 'resize_height': resized_h,
        'crop_left': left, 'crop_top': top, 'model_width': crop_w, 'model_height': crop_h,
        'patch_size': PATCH_SIZE, 'patch_grid_height': crop_h // PATCH_SIZE,
        'patch_grid_width': crop_w // PATCH_SIZE,
        'A_model_from_native': affine, 'A_native_from_model': inverse,
        'patch_uv_model': uv_model, 'patch_uv_native': uv_native,
        'crop_native_polygon': crop_native_h[:, :2] / crop_native_h[:, 2:],
        'pixel_convention': 'zero-based pixel centers; Pillow half-pixel resize; integer center crop',
        'reference_uv_coordinate_system': 'native camera pixels',
        'model_uv_validity_bounds': '[0,224) in both model-pixel coordinates',
        'fov_scope': 'Only the official center crop is encoded; native-image side regions are not observed by these tokens.'
    }
    if camera_intrinsic is not None:
        intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError('Camera intrinsics must be a finite 3x3 matrix')
        result['camera_intrinsic_native'] = intrinsic
        result['camera_intrinsic_model'] = affine @ intrinsic
    return result


def map_native_uv(native_uv, geometry, *, positive_depth_mask=None):
    """Project native UV into the encoded crop and retain explicit validity."""
    uv = np.asarray(native_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError('Expected Nx2 native pixel coordinates')
    h = np.c_[uv, np.ones(len(uv))] @ np.asarray(geometry['A_model_from_native']).T
    model_uv = h[:, :2] / h[:, 2:]
    valid = (np.isfinite(uv).all(1) & np.isfinite(model_uv).all(1)
             & (uv[:, 0] >= 0) & (uv[:, 0] < geometry['native_width'])
             & (uv[:, 1] >= 0) & (uv[:, 1] < geometry['native_height'])
             & (model_uv[:, 0] >= 0) & (model_uv[:, 0] < geometry['model_width'])
             & (model_uv[:, 1] >= 0) & (model_uv[:, 1] < geometry['model_height']))
    if positive_depth_mask is not None:
        depth = np.asarray(positive_depth_mask, dtype=bool)
        if depth.shape != (len(uv),):
            raise ValueError('Depth-validity mask must have one entry per projection')
        valid &= depth
    return model_uv, valid


def project_camera_points(camera_xyz, camera_intrinsic, geometry):
    xyz = np.asarray(camera_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Expected Nx3 metric coordinates in the named camera frame')
    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    projected = xyz @ intrinsic.T
    depth_valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    native_uv = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    native_uv[depth_valid] = projected[depth_valid, :2] / projected[depth_valid, 2:]
    model_uv, valid = map_native_uv(native_uv, geometry, positive_depth_mask=depth_valid)
    return {'native_uv': native_uv, 'model_uv': model_uv, 'projection_valid': valid,
            'positive_depth': depth_valid}


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


class FrozenRGBEncoder:
    """Actual pinned official DINOv2; outputs exclude its CLS by default."""
    def __init__(self, checkpoint_dir, *, device='cpu', threads=2):
        # Deferred imports permit metadata/download/geometry checks without a
        # second Torch model while the official 7B resource probe is running.
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.directory = Path(checkpoint_dir)
        self.device = torch.device(device)
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; no substitute device is used')
        torch.set_num_threads(threads)
        weights = self.directory / 'model.safetensors'
        if file_sha256(weights) != MODEL_SHA256:
            raise RuntimeError('Official pinned DINOv2-small weight SHA256 mismatch')
        self.processor = AutoImageProcessor.from_pretrained(self.directory, local_files_only=True, use_fast=False)
        cfg = json.loads((self.directory / 'config.json').read_text())
        pcfg = json.loads((self.directory / 'preprocessor_config.json').read_text())
        if (cfg['hidden_size'], cfg['patch_size']) != (384, PATCH_SIZE):
            raise ValueError('The pinned official visual-token contract changed')
        if pcfg.get('size') != {'shortest_edge': SHORT_EDGE} or pcfg.get('crop_size') != {'height': 224, 'width': 224}:
            raise ValueError('Resize/crop differs from the recorded camera geometry')
        if not all(pcfg.get(k) for k in ('do_resize', 'do_center_crop', 'do_convert_rgb', 'do_rescale', 'do_normalize')):
            raise ValueError('The expected official image preprocessing flags changed')
        self.model, info = AutoModel.from_pretrained(self.directory, local_files_only=True,
            use_safetensors=True, dtype=torch.float32, output_loading_info=True,
            attn_implementation='sdpa')
        if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys'):
            raise RuntimeError('Non-strict official visual checkpoint loading: ' + str(info))
        self.model.requires_grad_(False).eval().to(self.device)
        self.receipt = {
            'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
            'model_weights_sha256': MODEL_SHA256, 'model_license': 'Apache-2.0',
            'implementation_scope': 'v6 visual implementation choice; source HiRAC paper did not identify its vision checkpoint',
            'feature_dim': 384, 'patch_tokens': 256, 'patch_grid': [16, 16],
            'patch_size': PATCH_SIZE, 'reference_has_cls': False,
            'frozen_parameter_count': sum(p.numel() for p in self.model.parameters()),
            'all_parameters_frozen': all(not p.requires_grad for p in self.model.parameters()),
            'precision': 'FP32', 'device': str(self.device), 'threads': threads,
            'config_sha256': file_sha256(self.directory / 'config.json'),
            'preprocessor_config_sha256': file_sha256(self.directory / 'preprocessor_config.json'),
            'strict_official_load': True, 'torch_version': torch.__version__,
            'geometry_default': 'native camera pixels; exact resize/crop affine and processed K also returned'
        }

    def encode(self, image_path, *, camera_intrinsic=None):
        torch = self.torch
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            rgb = source.convert('RGB')
        geometry = preprocessing_geometry(*rgb.size, camera_intrinsic=camera_intrinsic)
        processed = self.processor(images=rgb, return_tensors='pt')['pixel_values']
        if tuple(processed.shape) != (1, 3, 224, 224):
            raise RuntimeError('Official processor output disagrees with camera affine')
        # Cross-check actual slow-processor pixels against the recorded PIL
        # resize/crop geometry, including integer rounding and RGB channels.
        resized = rgb.resize((geometry['resize_width'], geometry['resize_height']), resample=Image.Resampling.BICUBIC)
        left, top = geometry['crop_left'], geometry['crop_top']
        cropped = resized.crop((left, top, left + 224, top + 224))
        arr = np.asarray(cropped).astype(np.float64) * float(self.processor.rescale_factor)
        arr = arr.astype(np.float32)
        arr = (arr - np.asarray(self.processor.image_mean, dtype=np.float32)) / np.asarray(self.processor.image_std, dtype=np.float32)
        expected = arr.transpose(2, 0, 1)[None]
        pixel_diff = float(np.max(np.abs(processed.numpy() - expected)))
        if pixel_diff > 2e-6:
            raise RuntimeError(f'Official processor pixels do not follow the recorded geometry: {pixel_diff}')
        with torch.inference_mode():
            hidden = self.model(pixel_values=processed.to(self.device)).last_hidden_state
        if tuple(hidden.shape) != (1, 257, 384) or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError('Actual DINOv2 token contract or finiteness check failed')
        tokens = hidden[:, 1:].detach().float().cpu().clone()
        cls = hidden[:, :1].detach().float().cpu().clone()
        receipt = dict(self.receipt)
        receipt.update({'image_sha256': file_sha256(image_path), 'native_width': rgb.width,
            'native_height': rgb.height, 'processed_pixels_sha256': hashlib.sha256(processed.numpy().tobytes()).hexdigest(),
            'processed_pixels_geometry_max_abs_error': pixel_diff,
            'patch_tokens_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest()})
        return {'tokens': tokens, 'cls_token': cls,
                'reference_uv': torch.from_numpy(geometry['patch_uv_native'].astype(np.float32))[None],
                'reference_uv_model': torch.from_numpy(geometry['patch_uv_model'].astype(np.float32))[None],
                'reference_mask': torch.ones((1, 256), dtype=torch.bool),
                'geometry': geometry, 'receipt': receipt}

    def encode_sequence(self, image_paths, *, camera_intrinsics=None):
        """Return [B=1,T,M,D] and native-pixel UV for the adapter reader."""
        torch = self.torch
        paths = list(image_paths)
        if not paths:
            raise ValueError('At least one actual image is required')
        intrinsics = [None] * len(paths) if camera_intrinsics is None else list(camera_intrinsics)
        if len(intrinsics) != len(paths):
            raise ValueError('One optional intrinsic matrix is required per actual image')
        encoded = [self.encode(path, camera_intrinsic=intrinsic) for path, intrinsic in zip(paths, intrinsics)]
        return {
            'reference_tokens': torch.stack([x['tokens'][0] for x in encoded], dim=0)[None],
            'reference_uv': torch.stack([x['reference_uv'][0] for x in encoded], dim=0)[None],
            'reference_mask': torch.stack([x['reference_mask'][0] for x in encoded], dim=0)[None],
            'reference_has_cls': False,
            'A_processed_from_original': torch.from_numpy(np.stack([x['geometry']['A_model_from_native'] for x in encoded]))[None],
            'processed_size_wh': [224, 224],
            'source_image_metadata': [x['receipt'] for x in encoded],
            'crop_geometry': [x['geometry'] for x in encoded],
            'reference_uv_coordinate_system': 'native camera pixels'
        }
