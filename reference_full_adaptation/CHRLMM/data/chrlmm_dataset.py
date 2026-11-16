"""CHR-LMM dataset for synchronized mmWave radar and vision inputs."""

import copy
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .multimodal_adapter import MultimodalSequenceAdapter

try:
    import transformers
except ModuleNotFoundError:
    class _PreTrainedTokenizer:
        pass

    class _TransformersFallback:
        PreTrainedTokenizer = _PreTrainedTokenizer

    transformers = _TransformersFallback()

try:
    from .utils import preprocess_multimodal_point_cloud, preprocess_v1
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise

    def preprocess_multimodal_point_cloud(*args, **kwargs):
        raise ImportError("Install transformers to use CHR-LMM text preprocessing.")

    def preprocess_v1(*args, **kwargs):
        raise ImportError("Install transformers to use CHR-LMM text preprocessing.")


def make_chrlmm_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Create the training and optional validation datasets."""

    data_collator = DataCollatorForCHRLMMDataset(tokenizer=tokenizer)
    common = dict(
        data_path=data_args.data_path,
        anno_path=data_args.anno_path,
        pointnum=data_args.pointnum,
        conversation_types=data_args.conversation_types,
        tokenizer=tokenizer,
        use_color=data_args.use_color,
        use_vision=data_args.use_vision,
        data_args=data_args,
    )

    print("Loading training datasets.")
    train_dataset = CHRLMMDataset(split="train", **common)
    print("Done!")

    if not data_args.split_train_val:
        return dict(
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )

    if data_args.data_debug_num > 0:
        print("Debug mode, using training set as val set.")
        val_dataset = train_dataset
    else:
        print("Loading validation datasets.")
        val_dataset = CHRLMMDataset(split="val", **common)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )


class CHRLMMDataset(Dataset):
    """Model-facing view of CHR-LMM or VR-PCT multimodal sequences."""

    def __init__(
        self,
        data_path=None,
        anno_path=None,
        tokenizer=None,
        pointnum=96,
        split="train",
        conversation_types=None,
        use_color=False,
        use_vision=True,
        data_args=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.split = split
        self.pointnum = pointnum
        self.use_color = use_color
        self.use_vision = use_vision
        self.data_args = data_args
        self.normalize_pc = True
        self.point_indicator = "<point>"
        self.conversation_types = conversation_types or ("simple_description",)
        self.point_backbone_config = (
            getattr(data_args, "point_backbone_config", None)
            if data_args is not None
            else None
        )

        data_format = getattr(data_args, "data_format", "auto")
        frame_mode = getattr(data_args, "frame_mode", "middle")
        sequence_length = getattr(data_args, "sequence_length", 0)
        radar_feature_dim = getattr(data_args, "radar_feature_dim", 0)
        label_map = getattr(data_args, "label_map", None)

        print(f"Loading data from {data_path}")
        self.sensor_data = MultimodalSequenceAdapter(
            data_path=data_path,
            split=split,
            data_format=data_format,
            frame_mode=frame_mode,
            sequence_length=sequence_length,
            pointnum=pointnum,
            radar_feature_dim=radar_feature_dim,
            legacy_xyz_only=not use_color,
            label_map=label_map,
        )
        self.data_format = self.sensor_data.data_format
        self.mmwave_data = self.sensor_data.radar
        self.vision_data = self.sensor_data.vision if use_vision else None
        self.labels = self.sensor_data.labels
        self.label_ids = self.sensor_data.label_ids
        self.weather = self.sensor_data.weather
        self.conversations = self.sensor_data.conversations
        self.radar_feature_dim = self.sensor_data.info.radar_feature_dim
        self.vision_feature_dim = self.sensor_data.info.vision_feature_dim

        self.list_data_dict = [
            {
                "id": index,
                "label": self.labels[index],
                "weather": self.weather[index],
                "conversations": self.conversations[index],
                "conversation_type": "simple_description",
            }
            for index in range(len(self.labels))
        ]
        print(
            f"Loaded {len(self.list_data_dict)} {self.data_format} samples "
            f"for {split} split."
        )

        if self.data_args is not None and self.data_args.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[: self.data_args.data_debug_num]
            print(f"Debug mode, using {len(self.list_data_dict)} samples")
        elif self.data_args is not None and self.data_args.split_train_val:
            boundary = int(self.data_args.split_ratio * len(self.list_data_dict))
            if self.split == "train":
                self.list_data_dict = self.list_data_dict[:boundary]
                print(f"Train set size: {len(self.list_data_dict)}")
            else:
                self.list_data_dict = self.list_data_dict[boundary:]
                print(f"Val set size: {len(self.list_data_dict)}")

    def _load_point_cloud(self, index):
        point_cloud, _ = self.sensor_data.get_modalities(index)
        return point_cloud

    def _load_vision_features(self, index):
        if not self.use_vision or self.vision_data is None:
            return None
        _, vision_features = self.sensor_data.get_modalities(index)
        return vision_features

    def pc_norm(self, point_cloud):
        """Normalize XYZ while preserving any additional radar channels."""

        point_cloud = np.asarray(point_cloud, dtype=np.float32).copy()
        xyz = point_cloud[:, :3]
        other_features = point_cloud[:, 3:]
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.sum(xyz * xyz, axis=1)).max(initial=0.0)
        if scale > 1e-6:
            xyz = xyz / scale
        return np.concatenate((xyz, other_features), axis=1)

    def __getitem__(self, index):
        source = self.list_data_dict[index]
        sources = [source] if isinstance(index, int) else source
        assert len(sources) == 1, "sources should be a list of length 1"

        has_point = self.point_indicator in sources[0]["conversations"][0]["value"]
        if has_point:
            data_id = source["id"]
            point_cloud, vision_features = self.sensor_data.get_modalities(data_id)
            if not self.use_vision:
                vision_features = None

            if self.tokenizer is None:
                return {
                    "point_clouds": torch.from_numpy(point_cloud.astype(np.float32)),
                    "vision_features": (
                        torch.from_numpy(vision_features.astype(np.float32))
                        if vision_features is not None
                        else None
                    ),
                    "labels": source["label"],
                    "class_labels": torch.tensor(
                        self.label_ids[data_id], dtype=torch.long
                    ),
                    "weather": source["weather"],
                }

            sources = preprocess_multimodal_point_cloud(
                copy.deepcopy([entry["conversations"] for entry in sources]),
                self.point_backbone_config,
                point_indicator=self.point_indicator,
            )
        else:
            sources = copy.deepcopy(
                [entry["conversations"] for entry in sources]
            )

        data_dict = preprocess_v1(sources, self.tokenizer)
        if isinstance(index, int):
            data_dict = {
                "input_ids": data_dict["input_ids"][0],
                "labels": data_dict["labels"][0],
            }

        if has_point:
            data_dict["point_clouds"] = torch.from_numpy(
                point_cloud.astype(np.float32)
            )
            data_dict["class_labels"] = torch.tensor(
                self.label_ids[data_id], dtype=torch.long
            )
            data_dict["weather"] = source["weather"]
            if vision_features is not None:
                data_dict["vision_features"] = torch.from_numpy(
                    vision_features.astype(np.float32)
                )
        return data_dict

    def __len__(self):
        return len(self.list_data_dict)


class DataCollatorForCHRLMMDataset:
    """Pad text and stack fixed-shape multimodal tensors."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        if "point_clouds" in instances[0]:
            point_clouds = [instance["point_clouds"] for instance in instances]
            if all(
                value is not None and value.shape == point_clouds[0].shape
                for value in point_clouds
            ):
                batch["point_clouds"] = torch.stack(point_clouds)
            else:
                batch["point_clouds"] = point_clouds

        if "vision_features" in instances[0]:
            vision_features = [
                instance.get("vision_features") for instance in instances
            ]
            vision_features = [value for value in vision_features if value is not None]
            if vision_features:
                if all(
                    value.shape == vision_features[0].shape
                    for value in vision_features
                ):
                    batch["vision_features"] = torch.stack(vision_features)
                else:
                    batch["vision_features"] = vision_features

        if "class_labels" in instances[0]:
            batch["class_labels"] = torch.stack(
                [instance["class_labels"] for instance in instances]
            )
        if "weather" in instances[0]:
            batch["weather"] = [instance["weather"] for instance in instances]
        return batch
