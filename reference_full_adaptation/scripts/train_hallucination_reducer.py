"""Train the CHR-LMM hallucination reducer without loading the full LMM.

Both the legacy CHR-LMM NPY layout and the real VR-PCT NPY layout are accepted.
VR-PCT does not provide a hallucination-oracle label, so its discrepancy-map
loss is masked out instead of being trained against fabricated targets.
Checkpoint selection uses validation data only. The held-out test partition is
evaluated once after restoring the best validation checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CHRLMM.data.multimodal_adapter import MultimodalSequenceAdapter
from CHRLMM.model.CHRLMM_hallucination import CHRLMMHallucinationReducer


RADAR_FEATURE_ORDERS = {
    "chrlmm": ["x", "y", "z", "doppler", "intensity"],
    "vrpct": ["x", "y", "z", "intensity", "velocity"],
}


class ReducerDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        pointnum=96,
        max_samples=0,
        data_format="auto",
        frame_mode="middle",
        sequence_length=0,
        radar_feature_dim=0,
        label_map=None,
        label_mapping=None,
    ):
        self.adapter = MultimodalSequenceAdapter(
            data_path=data_dir,
            split=split,
            data_format=data_format,
            frame_mode=frame_mode,
            sequence_length=sequence_length,
            pointnum=pointnum,
            radar_feature_dim=radar_feature_dim,
            label_map=label_map,
            label_mapping=label_mapping,
        )
        length = len(self.adapter)
        if max_samples > 0:
            length = min(max_samples, length)
        self.indices = np.arange(length)
        self.info = self.adapter.info

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        points, vision = self.adapter.get_modalities(index)
        hallucinated, hallucination_mask = self.adapter.hallucination_target(index)
        return {
            "points": torch.from_numpy(points.copy()),
            "vision": torch.from_numpy(vision.copy()),
            "label": torch.tensor(
                int(self.adapter.label_ids[index]), dtype=torch.long
            ),
            "hallucinated": torch.tensor(hallucinated, dtype=torch.float32),
            "hallucination_mask": torch.tensor(
                hallucination_mask, dtype=torch.float32
            ),
        }


def compute_accuracy(logits, labels):
    prediction = logits.argmax(dim=-1)
    return (prediction == labels).float().mean().item()


def aggregate_class_logits(outputs, key):
    """Average token predictions, and time when a sequence was supplied."""

    values = outputs.get("sequence_outputs", {}).get(key, outputs[key])
    reduce_dims = tuple(range(1, values.ndim - 1))
    return values.mean(dim=reduce_dims) if reduce_dims else values


def compute_map_loss(outputs, hallucinated, supervision_mask):
    discrepancy = outputs.get("sequence_outputs", {}).get(
        "discrepancy_map", outputs["discrepancy_map"]
    )
    reduce_dims = tuple(range(1, discrepancy.ndim))
    map_score = torch.sigmoid(discrepancy.mean(dim=reduce_dims) - 1.0).clamp(
        1e-5, 1.0 - 1e-5
    )
    per_sample = F.binary_cross_entropy(
        map_score, hallucinated, reduction="none"
    )
    supervised = supervision_mask.sum()
    if supervised.item() == 0:
        return map_score.sum() * 0.0
    return (per_sample * supervision_mask).sum() / supervised


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["label"].to(device)
            hallucinated = batch["hallucinated"].to(device)
            supervision_mask = batch["hallucination_mask"].to(device)
            outputs = model(points, vision)
            logits = aggregate_class_logits(outputs, "corrected_predictions")
            class_loss = F.cross_entropy(logits, labels)
            map_loss = compute_map_loss(
                outputs, hallucinated, supervision_mask
            )
            loss = class_loss + 0.2 * map_loss
            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += batch_size
    denominator = max(total_samples, 1)
    return total_loss / denominator, total_correct / denominator


def _validate_split_contract(train_set, test_set):
    fields = (
        "radar_feature_dim",
        "vision_feature_dim",
        "radar_tokens",
        "vision_tokens",
    )
    differences = {
        field: (getattr(train_set.info, field), getattr(test_set.info, field))
        for field in fields
        if getattr(train_set.info, field) != getattr(test_set.info, field)
    }
    if differences:
        raise ValueError(f"Train/test sensor contracts differ: {differences}")


def stratified_train_validation_indices(labels, validation_ratio, seed):
    """Split training samples deterministically while preserving class coverage."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must lie strictly between 0 and 1.")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size < 2:
        raise ValueError("At least two one-dimensional labels are required.")

    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        if class_indices.size < 2:
            train_indices.extend(class_indices.tolist())
            continue
        validation_count = int(round(class_indices.size * validation_ratio))
        validation_count = min(max(validation_count, 1), class_indices.size - 1)
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())

    if not train_indices or not validation_indices:
        raise ValueError(
            "The requested validation split is empty; provide a pre-partitioned "
            "validation set or more training samples."
        )
    return (
        np.asarray(sorted(train_indices), dtype=np.int64),
        np.asarray(sorted(validation_indices), dtype=np.int64),
    )


def _is_synthetic_dataset(data_dir):
    metadata_path = Path(data_dir) / "synthetic_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return "synthetic" in str(metadata.get("notice", "")).lower()


def _load_validation_data(args, train_set):
    try:
        validation_set = ReducerDataset(
            args.data_dir,
            split=args.validation_split,
            pointnum=args.pointnum,
            max_samples=args.max_validation_samples,
            data_format=train_set.info.data_format,
            frame_mode=args.frame_mode,
            sequence_length=args.sequence_length,
            radar_feature_dim=train_set.info.radar_feature_dim,
            label_mapping=train_set.adapter.label_to_id,
        )
        if validation_set.adapter.split_root == train_set.adapter.split_root:
            raise FileNotFoundError(
                "The validation split resolved to the training data root."
            )
    except FileNotFoundError as error:
        fallback_allowed = (
            args.allow_sample_level_validation
            or _is_synthetic_dataset(args.data_dir)
        )
        if not fallback_allowed:
            raise FileNotFoundError(
                f"No pre-partitioned '{args.validation_split}' split was found. "
                "Real overlapping-window data require a trajectory-level validation "
                "partition. Use --allow-sample-level-validation only for independent "
                "samples or code-path checks."
            ) from error

        labels = train_set.adapter.label_ids[train_set.indices]
        train_indices, validation_indices = stratified_train_validation_indices(
            labels, args.validation_ratio, args.seed
        )
        validation_source = {
            "mode": "sample_stratified",
            "ratio": args.validation_ratio,
            "seed": args.seed,
            "synthetic": _is_synthetic_dataset(args.data_dir),
        }
        return (
            Subset(train_set, train_indices.tolist()),
            Subset(train_set, validation_indices.tolist()),
            validation_source,
        )

    _validate_split_contract(train_set, validation_set)
    validation_source = {
        "mode": "prepartitioned",
        "split": args.validation_split,
    }
    return train_set, validation_set, validation_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="npyData")
    parser.add_argument(
        "--data-format",
        default="auto",
        choices=["auto", "chrlmm", "vrpct"],
    )
    parser.add_argument(
        "--frame-mode",
        default="middle",
        choices=["first", "middle", "last", "all"],
    )
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--radar-feature-dim", type=int, default=0)
    parser.add_argument("--output", default="checkpoints/chrlmm_reducer_laptop.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pointnum", type=int, default=96)
    parser.add_argument(
        "--vision-dim",
        type=int,
        default=0,
        help="Optional assertion; 0 infers the dimension from the input data.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Optional assertion; 0 infers classes from labels or --label-map.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7588)
    parser.add_argument(
        "--allow-sample-level-validation",
        action="store_true",
        help=(
            "Allow a stratified validation subset from training samples. "
            "Do not use this for overlapping windows from real trajectories."
        ),
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"]
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.alpha < 0 or args.beta < 0:
        parser.error("--alpha and --beta must be non-negative.")
    if args.tau <= 0 or args.gamma <= 0:
        parser.error("--tau and --gamma must be positive.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    train_set = ReducerDataset(
        args.data_dir,
        split="train",
        pointnum=args.pointnum,
        max_samples=args.max_train_samples,
        data_format=args.data_format,
        frame_mode=args.frame_mode,
        sequence_length=args.sequence_length,
        radar_feature_dim=args.radar_feature_dim,
        label_map=args.label_map,
    )
    test_set = ReducerDataset(
        args.data_dir,
        split="test",
        pointnum=args.pointnum,
        max_samples=args.max_test_samples,
        data_format=train_set.info.data_format,
        frame_mode=args.frame_mode,
        sequence_length=args.sequence_length,
        radar_feature_dim=train_set.info.radar_feature_dim,
        label_mapping=train_set.adapter.label_to_id,
    )
    _validate_split_contract(train_set, test_set)
    if test_set.adapter.split_root == train_set.adapter.split_root:
        raise ValueError(
            "The test split resolved to the training data root. Provide distinct "
            "train and test partitions."
        )
    train_data, validation_data, validation_source = _load_validation_data(
        args, train_set
    )

    vision_dim = train_set.info.vision_feature_dim
    if args.vision_dim > 0 and args.vision_dim != vision_dim:
        raise ValueError(
            f"--vision-dim={args.vision_dim} conflicts with input dimension {vision_dim}."
        )
    num_classes = train_set.info.num_classes
    if args.num_classes > 0:
        if train_set.adapter.label_ids.max(initial=-1) >= args.num_classes:
            raise ValueError(
                f"--num-classes={args.num_classes} cannot encode the supplied labels."
            )
        num_classes = args.num_classes

    print(
        "Resolved input: "
        f"format={train_set.info.data_format}, "
        f"radar=[{train_set.info.radar_tokens},{train_set.info.radar_feature_dim}], "
        f"vision=[{train_set.info.vision_tokens},{vision_dim}], "
        f"frames={train_set.info.sequence_length}, classes={num_classes}"
    )
    print(
        "Resolved partitions: "
        f"train={len(train_data)}, validation={len(validation_data)}, "
        f"test={len(test_set)}, validation_source={validation_source['mode']}"
    )
    if not train_set.info.has_hallucination_targets:
        print(
            "No detection/recognition oracle found; discrepancy-map loss is disabled."
        )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = CHRLMMHallucinationReducer(
        point_dim=train_set.info.radar_feature_dim,
        vision_dim=vision_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        top_k=args.top_k,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        gamma=args.gamma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    best_validation_accuracy = -1.0
    best_epoch = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    radar_order = list(RADAR_FEATURE_ORDERS[train_set.info.data_format])
    if train_set.info.radar_feature_dim > len(radar_order):
        radar_order.extend(
            f"feature_{index}"
            for index in range(len(radar_order), train_set.info.radar_feature_dim)
        )
    radar_order = radar_order[: train_set.info.radar_feature_dim]
    data_schema = {
        "format": train_set.info.data_format,
        "frame_mode": args.frame_mode,
        "sequence_length": train_set.info.sequence_length,
        "radar_shape": [
            train_set.info.radar_tokens,
            train_set.info.radar_feature_dim,
        ],
        "radar_feature_order": radar_order,
        "radar_normalization": "center and unit-radius normalize XYZ only",
        "vision_shape": [train_set.info.vision_tokens, vision_dim],
        "vision_normalization": "none",
        "label_to_id": train_set.adapter.label_to_id,
        "has_hallucination_targets": train_set.info.has_hallucination_targets,
            "seed": args.seed,
            "discrepancy": {
                "alpha": args.alpha,
                "beta": args.beta,
                "tau": args.tau,
                "gamma": args.gamma,
            },
        "validation": validation_source,
        "partition_sizes": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_set),
        },
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            points = batch["points"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["label"].to(device)
            hallucinated = batch["hallucinated"].to(device)
            supervision_mask = batch["hallucination_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(points, vision)
            corrected_logits = aggregate_class_logits(
                outputs, "corrected_predictions"
            )
            point_logits = aggregate_class_logits(outputs, "point_predictions")
            vision_logits = aggregate_class_logits(outputs, "vision_predictions")
            class_loss = (
                F.cross_entropy(corrected_logits, labels)
                + 0.3 * F.cross_entropy(point_logits, labels)
                + 0.3 * F.cross_entropy(vision_logits, labels)
            )
            map_loss = compute_map_loss(
                outputs, hallucinated, supervision_mask
            )
            loss = class_loss + 0.2 * map_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_correct += (
                corrected_logits.argmax(dim=-1) == labels
            ).sum().item()
            total_samples += batch_size

        train_loss = total_loss / max(total_samples, 1)
        train_accuracy = total_correct / max(total_samples, 1)
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )
        print(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} | "
            f"train acc {train_accuracy * 100:.2f}% | "
            f"validation loss {validation_loss:.4f} | "
            f"validation acc {validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "point_dim": train_set.info.radar_feature_dim,
                        "vision_dim": vision_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_classes": num_classes,
                        "top_k": args.top_k,
                        "alpha": args.alpha,
                        "beta": args.beta,
                        "tau": args.tau,
                        "gamma": args.gamma,
                    },
                    "data_schema": data_schema,
                    "best_validation_accuracy": best_validation_accuracy,
                    "best_epoch": best_epoch,
                },
                output_path,
            )

    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    checkpoint["final_test_loss"] = test_loss
    checkpoint["final_test_accuracy"] = test_accuracy
    torch.save(checkpoint, output_path)

    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_validation_accuracy": best_validation_accuracy,
                "best_epoch": best_epoch,
                "final_test_loss": test_loss,
                "final_test_accuracy": test_accuracy,
                "checkpoint": str(output_path),
                "data_schema": data_schema,
            },
            handle,
            indent=2,
        )
    print(f"Saved best reducer checkpoint: {output_path}")
    print(
        f"Best validation accuracy: {best_validation_accuracy * 100:.2f}% "
        f"at epoch {best_epoch}"
    )
    print(
        f"Final held-out test loss: {test_loss:.4f} | "
        f"test accuracy: {test_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
