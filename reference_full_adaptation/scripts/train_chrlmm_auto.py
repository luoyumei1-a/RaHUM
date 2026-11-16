"""Interactive CHR-LMM training launcher.

The launcher detects GPU memory, asks the user to confirm a training profile,
and builds a suitable command for either laptop-scale reducer training or
full LMM training on large accelerators.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CHRLMM.data.multimodal_adapter import inspect_multimodal_data


def detect_gpu():
    try:
        import torch
    except Exception:
        return {"cuda": False, "name": "unknown", "vram_gb": 0.0}

    if not torch.cuda.is_available():
        return {"cuda": False, "name": "none", "vram_gb": 0.0}

    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": props.name,
        "vram_gb": props.total_memory / (1024 ** 3),
        "count": torch.cuda.device_count(),
    }


def infer_data_shape(args):
    return inspect_multimodal_data(
        data_path=args.data_dir,
        split="train",
        data_format=args.data_format,
        sequence_length=args.sequence_length,
        pointnum=args.pointnum,
        radar_feature_dim=args.radar_feature_dim,
        label_map=args.label_map,
    )


def recommended_profile(vram_gb, gpu_name):
    name = gpu_name.lower()
    if "h200" in name or "h100" in name or vram_gb >= 80:
        return "strong"
    if vram_gb >= 20:
        return "adapter"
    return "laptop"


def ask(prompt, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def bool_arg(value):
    return "True" if value else "False"


def command_to_text(command):
    if platform.system().lower().startswith("win"):
        return subprocess.list2cmdline(command)
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def common_lmm_args(args, model_path, output_dir, point_ckpt, pointnum, vision_dim):
    cmd = [
        sys.executable,
        "CHRLMM/train/train.py",
        "--model_name_or_path",
        model_path,
        "--data_path",
        args.data_dir,
        "--anno_path",
        str(Path(args.data_dir) / "train" / "conversations.json"),
        "--output_dir",
        output_dir,
        "--pointnum",
        str(pointnum),
        "--data_format",
        args.resolved_data_format,
        "--frame_mode",
        args.frame_mode,
        "--sequence_length",
        str(args.resolved_sequence_length),
        "--radar_feature_dim",
        str(args.resolved_radar_feature_dim),
        "--use_vision",
        "True",
        "--vision_feature_dim",
        str(vision_dim),
        "--num_target_classes",
        str(args.resolved_num_classes),
        "--use_color",
        "False",
        "--remove_unused_columns",
        "False",
        "--dataloader_num_workers",
        str(args.num_workers),
        "--logging_steps",
        "10",
        "--save_strategy",
        "steps",
        "--save_total_limit",
        "2",
        "--report_to",
        "none",
    ]
    if point_ckpt:
        cmd.extend(["--point_backbone_ckpt", point_ckpt])
    if args.label_map:
        cmd.extend(["--label_map", args.label_map])
    return cmd


def build_laptop_command(args, pointnum, vision_dim, vram_gb):
    batch_size = 8 if vram_gb and vram_gb < 8.5 else 16
    hidden_dim = 96 if vram_gb and vram_gb < 8.5 else 128
    command = [
        sys.executable,
        "scripts/train_hallucination_reducer.py",
        "--data-dir",
        args.data_dir,
        "--output",
        "checkpoints/chrlmm_reducer_laptop.pt",
        "--epochs",
        str(args.epochs or 20),
        "--batch-size",
        str(batch_size),
        "--pointnum",
        str(pointnum),
        "--data-format",
        args.resolved_data_format,
        "--frame-mode",
        args.frame_mode,
        "--sequence-length",
        str(args.resolved_sequence_length),
        "--radar-feature-dim",
        str(args.resolved_radar_feature_dim),
        "--vision-dim",
        str(vision_dim),
        "--hidden-dim",
        str(hidden_dim),
        "--top-k",
        "16",
        "--num-classes",
        str(args.resolved_num_classes),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.label_map:
        command.extend(["--label-map", args.label_map])
    return command


def build_adapter_command(args, model_path, point_ckpt, pointnum, vision_dim):
    cmd = common_lmm_args(
        args,
        model_path=model_path,
        output_dir="checkpoints/chrlmm_stage1_adapter",
        point_ckpt=point_ckpt,
        pointnum=pointnum,
        vision_dim=vision_dim,
    )
    cmd.extend(
        [
            "--num_train_epochs",
            str(args.epochs or 20),
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "16",
            "--learning_rate",
            "1e-4",
            "--model_max_length",
            "1024",
            "--gradient_checkpointing",
            "True",
            "--fp16",
            "True",
            "--fix_llm",
            "True",
            "--fix_pointnet",
            "True",
            "--tune_mm_mlp_adapter",
            "True",
            "--tune_hallucination_reducer",
            "True",
        ]
    )
    return [cmd]


def build_strong_commands(args, model_path, point_ckpt, pointnum, vision_dim):
    stage1 = common_lmm_args(
        args,
        model_path=model_path,
        output_dir="checkpoints/chrlmm_stage1",
        point_ckpt=point_ckpt,
        pointnum=pointnum,
        vision_dim=vision_dim,
    )
    stage1.extend(
        [
            "--num_train_epochs",
            str(args.epochs or 20),
            "--per_device_train_batch_size",
            "4",
            "--gradient_accumulation_steps",
            "4",
            "--learning_rate",
            "1e-4",
            "--model_max_length",
            "2048",
            "--gradient_checkpointing",
            "True",
            "--bf16",
            "True",
            "--fix_llm",
            "True",
            "--fix_pointnet",
            "True",
            "--tune_mm_mlp_adapter",
            "True",
            "--tune_hallucination_reducer",
            "True",
        ]
    )

    stage2 = common_lmm_args(
        args,
        model_path="checkpoints/chrlmm_stage1",
        output_dir="checkpoints/chrlmm_stage2",
        point_ckpt=None,
        pointnum=pointnum,
        vision_dim=vision_dim,
    )
    stage2.extend(
        [
            "--num_train_epochs",
            "30",
            "--per_device_train_batch_size",
            "2",
            "--gradient_accumulation_steps",
            "8",
            "--learning_rate",
            "2e-5",
            "--model_max_length",
            "2048",
            "--gradient_checkpointing",
            "True",
            "--bf16",
            "True",
            "--fix_llm",
            "False",
            "--fix_pointnet",
            "False",
            "--tune_mm_mlp_adapter",
            "False",
            "--tune_hallucination_reducer",
            "True",
            "--stage_2",
            "True",
        ]
    )
    return [stage1, stage2]


def maybe_check_path(path, label):
    if path and not Path(path).exists():
        print(f"Warning: {label} does not exist yet: {path}")


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["auto", "laptop", "adapter", "strong"], default="auto")
    parser.add_argument("--data-dir", default="npyData")
    parser.add_argument("--data-format", choices=["auto", "chrlmm", "vrpct"], default="auto")
    parser.add_argument(
        "--frame-mode",
        choices=["first", "middle", "last", "all"],
        default="middle",
    )
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--pointnum", type=int, default=96)
    parser.add_argument("--radar-feature-dim", type=int, default=0)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--point-ckpt", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    args = parser.parse_args()

    gpu = detect_gpu()
    data_info = infer_data_shape(args)
    pointnum = args.pointnum
    vision_dim = data_info.vision_feature_dim
    if args.num_classes > 0 and args.num_classes < data_info.num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} cannot encode {data_info.num_classes} input classes."
        )
    args.resolved_data_format = data_info.data_format
    args.resolved_sequence_length = data_info.sequence_length
    args.resolved_radar_feature_dim = data_info.radar_feature_dim
    args.resolved_num_classes = args.num_classes or data_info.num_classes
    print("Detected training environment")
    print("=" * 36)
    print(f"CUDA available: {gpu['cuda']}")
    print(f"GPU: {gpu['name']}")
    print(f"VRAM: {gpu['vram_gb']:.1f} GB")
    print(
        f"Data: format={data_info.data_format}, frames={data_info.sequence_length}, "
        f"radar=[{pointnum},{data_info.radar_feature_dim}], "
        f"vision=[{data_info.vision_tokens},{vision_dim}], "
        f"classes={args.resolved_num_classes}"
    )
    print()

    if not args.yes:
        gpu["name"] = ask("Confirm GPU name", gpu["name"])
        vram_text = ask("Confirm GPU VRAM in GB", f"{gpu['vram_gb']:.1f}")
        try:
            gpu["vram_gb"] = float(vram_text)
        except (TypeError, ValueError):
            print("Invalid VRAM value; using detected value.")

    rec = recommended_profile(gpu["vram_gb"], gpu["name"])

    print()
    print("Training profile recommendation")
    print("=" * 36)
    print(f"GPU: {gpu['name']}")
    print(f"VRAM: {gpu['vram_gb']:.1f} GB")
    print(f"Recommended profile: {rec}")
    print()
    print("Profiles")
    print("  laptop : <=12 GB VRAM, train hallucination reducer only")
    print("  adapter: 16-48 GB VRAM, frozen LLM with trainable projection/reducer")
    print("  strong : >=80 GB VRAM, full stage1 + stage2 LMM training")
    print()

    profile = args.profile
    if profile == "auto":
        profile = rec
    if not args.yes:
        profile = ask("Choose profile: laptop / adapter / strong", profile)
    if profile not in {"laptop", "adapter", "strong"}:
        raise ValueError(f"Unknown profile: {profile}")

    model_path = args.model_path
    point_ckpt = args.point_ckpt
    if profile in {"adapter", "strong"}:
        if not model_path:
            model_path = ask("LLM checkpoint path", "checkpoints/PointLLM_7B_v1.1_init")
        if not point_ckpt:
            point_ckpt = ask(
                "PointBERT checkpoint path",
                "checkpoints/PointLLM_7B_v1.1_init/point_bert_v1.2.pt",
            )
        maybe_check_path(model_path, "LLM checkpoint")
        maybe_check_path(point_ckpt, "PointBERT checkpoint")

    if profile == "laptop":
        commands = [build_laptop_command(args, pointnum, vision_dim, gpu["vram_gb"])]
    elif profile == "adapter":
        commands = build_adapter_command(args, model_path, point_ckpt, pointnum, vision_dim)
    else:
        commands = build_strong_commands(args, model_path, point_ckpt, pointnum, vision_dim)

    print()
    print("Planned command(s)")
    print("=" * 36)
    for idx, cmd in enumerate(commands, start=1):
        print(f"[{idx}] {command_to_text(cmd)}")
    print()

    run_now = args.run
    if not args.yes:
        run_now = ask("Run now? y/N", "N").lower() == "y"
    if not run_now:
        print("Dry run only. Copy a command above when you are ready.")
        return

    os.makedirs("checkpoints", exist_ok=True)
    for idx, cmd in enumerate(commands, start=1):
        print(f"Running command {idx}/{len(commands)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
