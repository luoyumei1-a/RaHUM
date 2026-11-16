"""Convert VR-PCT MAT sequences into deterministic model-facing NPY arrays.

The source tree is read-only. Outputs follow the VR-PCT contract:
``list_all_ti.npy`` [S,T,N,5], ``list_all_key2.npy`` [S,T,32,3], and
``list_all_label.npy`` [S].
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.io import loadmat
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("scipy is required: python -m pip install scipy") from exc


FRAME_NUMBER = re.compile(r"(\d+)(?=\.mat$)")


@dataclass(frozen=True)
class SequenceSource:
    root: Path
    group: Path
    label: str
    files: Tuple[Path, ...]


def natural_key(value: str) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def frame_key(path: Path) -> Tuple[int, Tuple[object, ...]]:
    match = FRAME_NUMBER.search(path.name)
    frame = int(match.group(1)) if match else -1
    return frame, natural_key(path.name)


def discover_sequences(
    roots: Sequence[Path], sequence_length: int
) -> Iterable[SequenceSource]:
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"MAT root does not exist: {root}")
        groups = sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: natural_key(str(path.relative_to(root))),
        )
        for group in groups:
            files = tuple(sorted(group.glob("*.mat"), key=frame_key))
            if len(files) < sequence_length:
                continue
            label = group.parent.name
            yield SequenceSource(
                root=root,
                group=group,
                label=label,
                files=files[:sequence_length],
            )


def load_frame(path: Path) -> Dict[str, np.ndarray]:
    try:
        payload = loadmat(path)
    except NotImplementedError:
        try:
            import h5py
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                f"{path} is MATLAB v7.3; install h5py to read it."
            ) from exc
        with h5py.File(path, "r") as handle:
            payload = {
                key: np.asarray(handle[key]).T
                for key in ("pc_xyziv_ti", "pc_xyz_key")
                if key in handle
            }
    missing = {"pc_xyziv_ti", "pc_xyz_key"} - set(payload)
    if missing:
        raise KeyError(f"{path} is missing MAT fields: {sorted(missing)}")
    return payload


def clean_matrix(value: np.ndarray, columns: int, field: str, path: Path) -> np.ndarray:
    matrix = np.asarray(value)
    matrix = np.squeeze(matrix)
    if matrix.ndim != 2 or matrix.shape[1] < columns:
        raise ValueError(
            f"{path}: {field} needs shape [N,{columns}+], got {matrix.shape}."
        )
    matrix = np.asarray(matrix[:, :columns], dtype=np.float32)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if len(matrix) == 0:
        raise ValueError(f"{path}: {field} has no finite rows.")
    return matrix


def resample_points(
    points: np.ndarray, point_count: int, rng: np.random.Generator
) -> np.ndarray:
    count = len(points)
    if count >= point_count:
        indices = rng.choice(count, size=point_count, replace=False)
    else:
        base = rng.permutation(count)
        extra = rng.choice(count, size=point_count - count, replace=True)
        indices = np.concatenate((base, extra))
        indices = indices[rng.permutation(point_count)]
    return np.asarray(points[indices], dtype=np.float32)


def convert_sequence(
    source: SequenceSource,
    point_count: int,
    seed: int,
    legacy_offsets: bool,
    hand_focus: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    radar_frames: List[np.ndarray] = []
    keypoint_frames: List[np.ndarray] = []
    for frame_index, path in enumerate(source.files):
        payload = load_frame(path)
        radar = clean_matrix(payload["pc_xyziv_ti"], 5, "pc_xyziv_ti", path)
        keypoints = clean_matrix(payload["pc_xyz_key"], 3, "pc_xyz_key", path)
        if keypoints.shape[0] != 32:
            raise ValueError(
                f"{path}: pc_xyz_key must contain 32 joints, got {keypoints.shape[0]}."
            )

        if legacy_offsets:
            radar[:, 0] -= 0.2
            radar[:, 1] += 0.3
            radar[:, 2] -= 0.9
            keypoints[:, 2] -= 0.1

        candidates = radar
        if hand_focus:
            mask = (
                (radar[:, 0] >= -0.3)
                & (radar[:, 0] <= 0.3)
                & (radar[:, 1] >= 1.1)
                & (radar[:, 1] <= 1.5)
                & (radar[:, 2] >= -0.2)
                & (radar[:, 2] <= 0.3)
            )
            hand_points = radar[mask]
            if len(hand_points) > 8:
                candidates = hand_points

        rng = np.random.default_rng(seed + frame_index)
        radar_frames.append(resample_points(candidates, point_count, rng))
        keypoint_frames.append(np.asarray(keypoints, dtype=np.float32))

    return np.stack(radar_frames), np.stack(keypoint_frames)


def stratified_split(
    labels: Sequence[str], test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must lie strictly between 0 and 1.")
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train: List[int] = []
    test: List[int] = []
    for label in sorted(set(labels.tolist()), key=natural_key):
        indices = np.flatnonzero(labels == label)
        indices = rng.permutation(indices)
        test_count = max(1, int(round(len(indices) * test_ratio)))
        if len(indices) > 1:
            test_count = min(test_count, len(indices) - 1)
        test.extend(indices[:test_count].tolist())
        train.extend(indices[test_count:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(test))


def save_split(
    output_root: Path,
    split: str,
    radar: np.ndarray,
    keypoints: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    np.save(split_root / "list_all_ti.npy", radar[indices])
    np.save(split_root / "list_all_key2.npy", keypoints[indices])
    np.save(split_root / "list_all_label.npy", labels[indices])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat-root",
        action="append",
        required=True,
        help="Read-only matData root. Repeat this option for multiple views/batches.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=25)
    parser.add_argument("--point-count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7588)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--split-mode",
        choices=("test-only", "train-test"),
        default="train-test",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--no-legacy-offsets",
        action="store_true",
        help="Disable the coordinate offsets from VR-PCT/dataprocess.py.",
    )
    parser.add_argument(
        "--no-hand-focus",
        action="store_true",
        help="Disable the legacy hand-region preference before 96-point sampling.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roots = [Path(value).expanduser().resolve() for value in args.mat_root]
    output_root = Path(args.output_dir).expanduser().resolve()
    for root in roots:
        if is_within(output_root, root):
            raise ValueError(
                f"Output directory must be outside every read-only MAT root: {root}"
            )
    sources = list(discover_sequences(roots, args.sequence_length))
    if args.max_sequences > 0:
        sources = sources[: args.max_sequences]
    if not sources:
        raise RuntimeError("No complete MAT sequences were found.")

    radar_samples: List[np.ndarray] = []
    keypoint_samples: List[np.ndarray] = []
    labels: List[str] = []
    provenance: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []
    for index, source in enumerate(sources):
        try:
            radar, keypoints = convert_sequence(
                source,
                point_count=args.point_count,
                seed=args.seed + index * args.sequence_length,
                legacy_offsets=not args.no_legacy_offsets,
                hand_focus=not args.no_hand_focus,
            )
        except (KeyError, ValueError, OSError) as exc:
            skipped.append({"group": str(source.group), "reason": str(exc)})
            continue
        radar_samples.append(radar)
        keypoint_samples.append(keypoints)
        labels.append(source.label)
        provenance.append(
            {
                "mat_root": str(source.root),
                "group": str(source.group),
                "label": source.label,
                "frames": [str(path) for path in source.files],
            }
        )

    if not radar_samples:
        raise RuntimeError("Every discovered sequence failed validation.")
    radar_array = np.asarray(radar_samples, dtype=np.float32)
    keypoint_array = np.asarray(keypoint_samples, dtype=np.float32)
    label_array = np.asarray(labels)

    if args.split_mode == "test-only":
        split_indices = {"test": np.arange(len(label_array))}
    else:
        train_indices, test_indices = stratified_split(
            label_array, args.test_ratio, args.seed
        )
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise RuntimeError(
                "The requested train/test split is empty; add sequences or use --split-mode test-only."
            )
        split_indices = {"train": train_indices, "test": test_indices}

    summary = {
        "source_roots": [str(root) for root in roots],
        "output_root": str(output_root),
        "source_is_read_only": True,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "point_count": args.point_count,
        "legacy_offsets": not args.no_legacy_offsets,
        "hand_focus": not args.no_hand_focus,
        "radar_shape": list(radar_array.shape),
        "keypoint_shape": list(keypoint_array.shape),
        "labels": sorted(set(labels), key=natural_key),
        "splits": {
            split: indices.tolist() for split, indices in split_indices.items()
        },
        "provenance": provenance,
        "skipped": skipped,
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "provenance"}, indent=2))
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    for split, indices in split_indices.items():
        save_split(
            output_root,
            split,
            radar_array,
            keypoint_array,
            label_array,
            indices,
        )
    with (output_root / "preparation_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved deterministic VR-PCT arrays to {output_root}")


if __name__ == "__main__":
    main()
