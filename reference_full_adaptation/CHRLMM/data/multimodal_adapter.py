"""Input adapter for CHR-LMM and VR-PCT multimodal sequence data.

The adapter keeps the model-facing contract independent from the file names
used by each data pipeline.  A sample is exposed as one synchronized radar and
vision frame while the source NPY files remain unchanged on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


DATA_LAYOUTS = {
    "chrlmm": {
        "radar": "list_all_mmwave.npy",
        "vision": "list_all_vision.npy",
    },
    "vrpct": {
        "radar": "list_all_ti.npy",
        "vision": "list_all_key2.npy",
    },
}

FRAME_MODES = {"first", "middle", "last", "all"}


@dataclass(frozen=True)
class MultimodalDataInfo:
    data_format: str
    split_root: Path
    num_samples: int
    sequence_length: int
    radar_tokens: int
    radar_feature_dim: int
    vision_tokens: int
    vision_feature_dim: int
    num_classes: int
    has_hallucination_targets: bool


def _load_npy(path: Path, mmap: bool = True):
    try:
        return np.load(
            path,
            mmap_mode="r" if mmap else None,
            allow_pickle=True,
        )
    except ValueError as exc:
        # Object arrays cannot be memory mapped.  Legacy CHR-LMM exports may
        # use object dtype even when every sample has the same shape.
        if mmap and "memory-mapped" in str(exc):
            return np.load(path, allow_pickle=True)
        raise


def _label_key(value) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_label_mapping(path: Optional[str]) -> Optional[Dict[str, int]]:
    """Load a raw-label to class-id mapping.

    Supported JSON forms are either ``{"raw": 0}`` or
    ``{"raw": "target-name"}``.  The latter is encoded to contiguous IDs in
    sorted target-name order.  A top-level ``label_to_id`` wrapper is accepted
    as well.
    """

    if not path:
        return None
    map_path = Path(path).expanduser().resolve()
    with map_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "label_to_id" in payload:
        payload = payload["label_to_id"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Label map must be a non-empty JSON object: {map_path}")

    keys = [_label_key(key) for key in payload]
    values = list(payload.values())
    if all(isinstance(value, (int, np.integer)) for value in values):
        mapping = {key: int(value) for key, value in zip(keys, values)}
        if min(mapping.values()) < 0:
            raise ValueError("Label-map class IDs must be non-negative.")
        return mapping
    if all(isinstance(value, str) for value in values):
        target_names = sorted(set(values))
        target_to_id = {name: index for index, name in enumerate(target_names)}
        return {
            key: target_to_id[value]
            for key, value in zip(keys, values)
        }
    raise ValueError("Label-map values must be all integers or all strings.")


def _resolve_split_root(data_path: str, split: str, data_format: str) -> Tuple[Path, str]:
    if data_format not in {*DATA_LAYOUTS, "auto"}:
        raise ValueError(
            f"Unknown data format {data_format!r}; expected auto, chrlmm, or vrpct."
        )

    data_root = Path(data_path).expanduser().resolve()
    candidates = [data_root / split, data_root]
    matches = []
    formats = DATA_LAYOUTS if data_format == "auto" else {data_format: DATA_LAYOUTS[data_format]}
    for candidate in candidates:
        for name, layout in formats.items():
            if (candidate / layout["radar"]).is_file() and (candidate / layout["vision"]).is_file():
                matches.append((candidate, name))

    unique_matches = list(dict.fromkeys(matches))
    if not unique_matches:
        expected = ", ".join(
            f"{layout['radar']} + {layout['vision']}"
            for layout in formats.values()
        )
        raise FileNotFoundError(
            f"No {data_format} multimodal split found under {data_root}. "
            f"Expected {expected} in either the root or its {split!r} directory."
        )
    if len(unique_matches) > 1:
        choices = ", ".join(f"{name}@{root}" for root, name in unique_matches)
        raise ValueError(
            f"Multiple data layouts match ({choices}); set data_format explicitly."
        )
    return unique_matches[0]


def _sample_shape(array, name: str) -> Tuple[int, ...]:
    if len(array) == 0:
        raise ValueError(f"{name} contains no samples.")
    shape = tuple(np.asarray(array[0]).shape)
    if not shape:
        raise ValueError(f"{name} sample 0 is scalar; a token array is required.")
    return shape


def _sequence_shape(shape: Tuple[int, ...], name: str) -> Tuple[int, int, int]:
    if len(shape) == 3:
        return shape
    if len(shape) == 2:
        tokens, feature_dim = shape
        return 1, tokens, feature_dim
    if len(shape) == 1 and name == "vision":
        return 1, 1, shape[0]
    raise ValueError(
        f"{name} samples must have shape [T, N, D] or [N, D], got {shape}."
    )


def _sanitize(array) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32).copy()
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_xyz(tokens: np.ndarray) -> np.ndarray:
    """Normalize valid XYZ rows while preserving all non-spatial channels."""

    tokens = _sanitize(tokens)
    if tokens.shape[-1] < 3:
        raise ValueError(f"XYZ normalization requires at least 3 channels, got {tokens.shape}.")
    valid = np.any(np.abs(tokens) > 1e-8, axis=1)
    if not np.any(valid):
        return tokens
    xyz = tokens[valid, :3]
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = np.sqrt(np.sum(xyz * xyz, axis=1)).max(initial=0.0)
    if scale > 1e-6:
        xyz = xyz / scale
    tokens[valid, :3] = xyz
    tokens[~valid, :3] = 0.0
    return tokens


def _resize_tokens(tokens: np.ndarray, token_count: int) -> np.ndarray:
    if token_count <= 0 or len(tokens) == token_count:
        return tokens
    if len(tokens) == 0:
        return np.zeros((token_count, tokens.shape[-1]), dtype=np.float32)
    if len(tokens) > token_count:
        indices = np.linspace(0, len(tokens) - 1, token_count, dtype=np.int64)
    else:
        indices = np.arange(token_count, dtype=np.int64) % len(tokens)
    return tokens[indices]


class MultimodalSequenceAdapter:
    """Load synchronized radar and vision sequences from either supported layout."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        data_format: str = "auto",
        frame_mode: str = "middle",
        sequence_length: int = 0,
        pointnum: int = 96,
        radar_feature_dim: int = 0,
        legacy_xyz_only: bool = False,
        label_map: Optional[str] = None,
        label_mapping: Optional[Mapping[str, int]] = None,
        mmap: bool = True,
    ):
        if frame_mode not in FRAME_MODES:
            raise ValueError(
                f"Unknown frame mode {frame_mode!r}; expected one of {sorted(FRAME_MODES)}."
            )
        self.split = split
        self.frame_mode = frame_mode
        self.pointnum = pointnum
        self.mmap = mmap
        self.split_root, self.data_format = _resolve_split_root(
            data_path, split, data_format
        )
        layout = DATA_LAYOUTS[self.data_format]
        self.radar = _load_npy(self.split_root / layout["radar"], mmap=mmap)
        self.vision = _load_npy(self.split_root / layout["vision"], mmap=mmap)

        label_path = self.split_root / "list_all_label.npy"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing labels: {label_path}")
        self.labels = _load_npy(label_path, mmap=mmap)
        self._validate_lengths()

        radar_shape = _sample_shape(self.radar, "radar")
        vision_shape = _sample_shape(self.vision, "vision")
        radar_seq, radar_tokens, source_radar_dim = _sequence_shape(radar_shape, "radar")
        self._vision_is_frame_matrix = (
            len(vision_shape) == 2
            and radar_seq > 1
            and vision_shape[0] == radar_seq
        )
        if self._vision_is_frame_matrix:
            vision_seq, vision_tokens, vision_dim = radar_seq, 1, vision_shape[1]
        else:
            vision_seq, vision_tokens, vision_dim = _sequence_shape(
                vision_shape, "vision"
            )
        if radar_seq != vision_seq:
            raise ValueError(
                "Radar and vision must be frame-synchronized: "
                f"sample 0 has {radar_seq} and {vision_seq} frames."
            )
        if sequence_length > 0 and radar_seq != sequence_length:
            raise ValueError(
                f"Expected {sequence_length} synchronized frames, found {radar_seq}."
            )
        if source_radar_dim < 3:
            raise ValueError(f"Radar input needs XYZ channels, got {radar_shape}.")
        if radar_feature_dim > source_radar_dim:
            raise ValueError(
                f"Requested {radar_feature_dim} radar channels, but data has {source_radar_dim}."
            )
        if radar_feature_dim > 0:
            self.radar_feature_dim = radar_feature_dim
        elif self.data_format == "chrlmm" and legacy_xyz_only:
            self.radar_feature_dim = 3
        else:
            self.radar_feature_dim = source_radar_dim

        explicit_mapping = (
            {_label_key(key): int(value) for key, value in label_mapping.items()}
            if label_mapping is not None
            else load_label_mapping(label_map)
        )
        self.label_to_id = self._build_label_mapping(explicit_mapping)
        self.label_ids = np.asarray(
            [self.label_to_id[_label_key(label)] for label in self.labels],
            dtype=np.int64,
        )
        self.num_classes = max(self.label_to_id.values(), default=-1) + 1

        self.weather = self._load_optional("list_all_weather.npy")
        if self.weather is None:
            self.weather = np.full(len(self.labels), "Unspecified", dtype=object)
        self.conversations = self._load_conversations()
        self.det_success = self._load_optional("list_all_detection_success.npy")
        self.rec_success = self._load_optional("list_all_recognition_success.npy")
        if (self.det_success is None) != (self.rec_success is None):
            raise ValueError(
                "Detection and recognition success arrays must either both exist or both be absent."
            )
        self.has_hallucination_targets = self.det_success is not None

        self.info = MultimodalDataInfo(
            data_format=self.data_format,
            split_root=self.split_root,
            num_samples=len(self.labels),
            sequence_length=radar_seq,
            radar_tokens=pointnum if pointnum > 0 else radar_tokens,
            radar_feature_dim=self.radar_feature_dim,
            vision_tokens=vision_tokens,
            vision_feature_dim=vision_dim,
            num_classes=self.num_classes,
            has_hallucination_targets=self.has_hallucination_targets,
        )

    def _validate_lengths(self):
        lengths = {
            "radar": len(self.radar),
            "vision": len(self.vision),
            "labels": len(self.labels),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Multimodal arrays have different sample counts: {lengths}")

    def _load_optional(self, filename: str):
        path = self.split_root / filename
        if not path.is_file():
            return None
        value = _load_npy(path, mmap=self.mmap)
        if len(value) != len(self.labels):
            raise ValueError(
                f"{filename} has {len(value)} entries; expected {len(self.labels)}."
            )
        return value

    def _build_label_mapping(
        self, explicit_mapping: Optional[Mapping[str, int]]
    ) -> Dict[str, int]:
        raw_keys = [_label_key(label) for label in self.labels]
        if explicit_mapping is not None:
            mapping = dict(explicit_mapping)
            missing = sorted(set(raw_keys) - set(mapping))
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Label map is missing source labels: {preview}")
            return mapping

        label_id_path = self.split_root / "list_all_label_id.npy"
        if label_id_path.is_file():
            label_ids = _load_npy(label_id_path, mmap=False).astype(np.int64)
            if len(label_ids) != len(raw_keys):
                raise ValueError(
                    f"list_all_label_id.npy has {len(label_ids)} entries; expected {len(raw_keys)}."
                )
            mapping: Dict[str, int] = {}
            for key, class_id in zip(raw_keys, label_ids):
                class_id = int(class_id)
                if key in mapping and mapping[key] != class_id:
                    raise ValueError(f"Label {key!r} maps to multiple class IDs.")
                mapping[key] = class_id
            return mapping

        return {key: index for index, key in enumerate(sorted(set(raw_keys)))}

    def _load_conversations(self):
        path = self.split_root / "conversations.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                conversations = json.load(handle)
            if len(conversations) != len(self.labels):
                raise ValueError(
                    f"conversations.json has {len(conversations)} entries; expected {len(self.labels)}."
                )
            return conversations
        return [
            [
                {
                    "from": "human",
                    "value": "<point>Describe the synchronized radar-vision observation.",
                },
                {
                    "from": "gpt",
                    "value": f"The observed class is {_label_key(label)}.",
                },
            ]
            for label in self.labels
        ]

    def __len__(self):
        return len(self.labels)

    def _frame_index(self, frame_count: int) -> int:
        if self.frame_mode == "first":
            return 0
        if self.frame_mode == "last":
            return frame_count - 1
        return frame_count // 2

    def _select_synchronized_frames(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        radar = np.asarray(self.radar[index])
        vision = np.asarray(self.vision[index])
        radar_frames = radar.shape[0] if radar.ndim == 3 else 1
        vision_frames = (
            vision.shape[0]
            if vision.ndim == 3 or self._vision_is_frame_matrix
            else 1
        )
        if radar_frames != vision_frames:
            raise ValueError(
                f"Sample {index} is not synchronized: radar has {radar_frames} frames, "
                f"vision has {vision_frames}."
            )

        if self.frame_mode == "all":
            if radar.ndim == 2:
                radar = radar[None, :, :]
            if vision.ndim == 3:
                pass
            elif self._vision_is_frame_matrix:
                vision = vision[:, None, :]
            elif vision.ndim == 2:
                vision = vision[None, :, :]
            elif vision.ndim == 1:
                vision = vision[None, None, :]
            if radar.ndim != 3 or vision.ndim != 3:
                raise ValueError(
                    f"Sample {index} must resolve to [T,N,D] arrays, got "
                    f"{radar.shape} and {vision.shape}."
                )
            return radar, vision

        frame_index = self._frame_index(radar_frames)
        if radar.ndim == 3:
            radar = radar[frame_index]
        if vision.ndim == 3:
            vision = vision[frame_index]
        elif self._vision_is_frame_matrix:
            vision = vision[frame_index][None, :]
        elif vision.ndim == 1:
            vision = vision[None, :]
        if radar.ndim != 2 or vision.ndim != 2:
            raise ValueError(
                f"Sample {index} must resolve to [N, D] arrays, got {radar.shape} and {vision.shape}."
            )
        return radar, vision

    def get_modalities(
        self,
        index: int,
        normalize_radar: bool = True,
        normalize_vrpct_vision: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        radar, vision = self._select_synchronized_frames(index)
        radar = _sanitize(radar[..., : self.radar_feature_dim])
        if radar.ndim == 3:
            radar = np.stack(
                [_resize_tokens(frame, self.pointnum) for frame in radar]
            )
            if normalize_radar:
                radar = np.stack([_normalize_xyz(frame) for frame in radar])
        else:
            radar = _resize_tokens(radar, self.pointnum)
            if normalize_radar:
                radar = _normalize_xyz(radar)

        vision = _sanitize(vision)
        if self.data_format == "vrpct" and normalize_vrpct_vision:
            if vision.ndim == 3:
                vision = np.stack([_normalize_xyz(frame) for frame in vision])
            else:
                vision = _normalize_xyz(vision)
        return radar, vision

    def hallucination_target(self, index: int) -> Tuple[float, float]:
        """Return (target, supervision mask) for the discrepancy-map loss."""

        if not self.has_hallucination_targets:
            return 0.0, 0.0
        hallucinated = not (
            bool(self.det_success[index]) and bool(self.rec_success[index])
        )
        return float(hallucinated), 1.0


def inspect_multimodal_data(
    data_path: str,
    split: str = "train",
    data_format: str = "auto",
    sequence_length: int = 0,
    pointnum: int = 96,
    radar_feature_dim: int = 0,
    legacy_xyz_only: bool = False,
    label_map: Optional[str] = None,
) -> MultimodalDataInfo:
    adapter = MultimodalSequenceAdapter(
        data_path=data_path,
        split=split,
        data_format=data_format,
        sequence_length=sequence_length,
        pointnum=pointnum,
        radar_feature_dim=radar_feature_dim,
        legacy_xyz_only=legacy_xyz_only,
        label_map=label_map,
    )
    return adapter.info
