"""Strictly restore a recorded native adapter and inspect one real feature cache.

This checks the native engineering adapter, not the author-reported
full-adaptation reference benchmark. It does not run or score the 7B decoder.
No labels are consumed by the sensory forward pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summary(values) -> dict:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not a.size:
        return {"count": 0}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite diagnostic value")
    quantiles = np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {"count": int(a.size), "mean": float(a.mean()), "std_population": float(a.std()),
            "quantiles": dict(zip(["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"], map(float, quantiles)))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--adapter-source", type=Path, required=True,
                   help="Use the frozen adapter.py whose SHA is in checkpoint metadata")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Refusing to overwrite a diagnostic execution")
    args.output_dir.mkdir(parents=True)
    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "rahumpointllm":
        raise ValueError("Checkpoint is not the native token adapter variant")
    meta = checkpoint["metadata"]
    expected = meta["source_sha256"]["adapter.py"]
    actual_source = digest(args.adapter_source)
    if actual_source != expected:
        raise ValueError("Frozen adapter source SHA does not match checkpoint")
    spec = importlib.util.spec_from_file_location("inspected_native_adapter", args.adapter_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.PointLLMTokenAdapter(module.AdapterConfig(**checkpoint["adapter_config"])).eval()
    model.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    with np.load(args.feature_cache, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    radar = torch.from_numpy(arrays["radar_tokens"])
    reference = torch.from_numpy(arrays["reference_tokens"])
    context = {key.removeprefix("context_"): torch.from_numpy(value)
               for key, value in arrays.items() if key.startswith("context_")}
    if any(any(t in key.lower() for t in ("label", "category", "ground_truth", "claim")) for key in context):
        raise ValueError("Unexpected label-like forward metadata")
    with torch.inference_mode():
        output = model(radar, reference, context)
    if not torch.isfinite(output["sequence_corrected_tokens"]).all():
        raise ValueError("Nonfinite reconstructed native tokens")
    config = checkpoint["adapter_config"]
    records = []
    stored_outputs = {}
    for frame_index, frame in enumerate(output["sequence_outputs"]):
        mask = context["radar_mask"][:, frame_index].bool()
        available = frame["reference_available"]
        active = mask & available
        if not torch.allclose(frame["utility_weights"].sum(-1), torch.ones_like(available, dtype=radar.dtype), atol=1e-6):
            raise ValueError("Utility weights do not sum to one")
        if not torch.allclose(frame["attention"].sum(-1), available.to(radar.dtype), atol=1e-6):
            raise ValueError("Attention normalization differs from actual reference availability")
        if not torch.equal(frame["utility_weights"][~available, 1], torch.zeros_like(frame["utility_weights"][~available, 1])):
            raise ValueError("Missing-reference groups receive nonzero reference utility")
        attention = frame["attention"].cpu().numpy()
        utility = frame["utility_weights"].cpu().numpy()
        entropy = -(attention * np.log(np.maximum(attention, 1e-30))).sum(-1)
        concentration = attention.max(-1)
        for batch_index, group_index in np.argwhere(mask.cpu().numpy()):
            i = (int(batch_index), int(group_index))
            records.append({
                "batch": i[0], "frame": frame_index, "radar_group": i[1],
                "reference_available": bool(available[i]),
                "native_radar_utility": float(utility[i + (0,)]),
                "native_reference_utility": float(utility[i + (1,)]),
                "attention_entropy_nats": float(entropy[i]),
                "attention_max": float(concentration[i]),
                "history_count": float(frame["temporal_history_count"][i]),
                "geometric_logprior_std": float(frame["log_geometric_prior"][i].std(unbiased=False)),
            })
        for name in ("attention", "utility_weights", "temporal_history_count", "log_geometric_prior",
                     "reference_available", "selected_reference_indices", "discrepancy"):
            stored_outputs[f"frame{frame_index}_{name}"] = frame[name].cpu().numpy()
        stored_outputs[f"frame{frame_index}_radar_mask"] = mask.cpu().numpy()
        stored_outputs[f"frame{frame_index}_active_mask"] = active.cpu().numpy()
    if not records:
        raise ValueError("No real active radar groups")
    with (args.output_dir / "per_group_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output_dir / "observed_native_diagnostics.npz", **stored_outputs)
    active_records = [r for r in records if r["reference_available"]]
    native_utility = np.asarray([r["native_radar_utility"] for r in active_records])
    report = {
        "status": "completed_strict_restore_real_feature_adapter_forward",
        "scope": "One real TRAIN engineering window and a native adapter checkpoint after two updates; not a held-out benchmark or a measurement of the original reference gate.",
        "no_decoder_or_new_training_run": True,
        "checkpoint_sha256": digest(args.checkpoint),
        "feature_cache_sha256": digest(args.feature_cache),
        "adapter_source_sha256": actual_source,
        "diagnostic_script_sha256": digest(Path(__file__)),
        "checks": {"adapter_strict_state_restore": True, "frozen_source_matches_checkpoint": True,
                   "reconstructed_tokens_finite": True, "utility_normalized": True,
                   "attention_normalized_on_available_only": True,
                   "missing_reference_utility_exact_zero": True, "no_ground_truth_in_forward": True},
        "checkpoint_metadata_completed_steps": meta.get("completed_steps"),
        "checkpoint_dataset_manifest_sha256": meta.get("dataset_manifest_sha256"),
        "checkpoint_engineering_scope": meta.get("engineering_scope"),
        "config_from_strictly_restored_checkpoint": config,
        "observed_reference_tokens": int(reference.shape[2] - int(config["reference_has_cls"])),
        "observed_local_attention_candidates": int(output["attention"].shape[-1]),
        "observed_batch_frames_radar_tokens": list(radar.shape),
        "observed_batch_frames_reference_tokens": list(reference.shape),
        "observed_available_radar_groups": len(active_records),
        "observed_radar_groups": len(records),
        "native_radar_utility_available_only": summary(native_utility),
        "native_fraction_abs_utility_minus_half_lt_001_available_only":
            float(np.mean(np.abs(native_utility - .5) < .01)) if native_utility.size else None,
        "native_attention_entropy_nats_available_only": summary([r["attention_entropy_nats"] for r in active_records]),
        "native_attention_max_available_only": summary([r["attention_max"] for r in active_records]),
        "native_history_count_all_measured_groups": summary([r["history_count"] for r in records]),
        "native_geometric_logprior_std_available_only": summary([r["geometric_logprior_std"] for r in active_records]),
        "original_reference_gate_distribution": None,
        "original_reference_gate_distribution_reason": "No original full-adaptation trained checkpoint or recorded attention/reliability tensors supplied to this diagnostic.",
        "torch": str(torch.__version__),
        "elapsed_s": time.perf_counter() - start,
    }
    for name in ("per_group_diagnostics.csv", "observed_native_diagnostics.npz"):
        report[f"{name}_sha256"] = digest(args.output_dir / name)
    (args.output_dir / "diagnostic_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
