"""Data package exports with optional heavy dependencies loaded lazily."""

__all__ = [
    "load_objaverse_point_cloud",
    "pc_norm",
    "farthest_point_sample",
    "preprocess_v1",
    "preprocess_multimodal_point_cloud",
    "DataCollatorForPointTextDataset",
    "ObjectPointCloudDataset",
    "make_object_point_data_module",
    "ModelNet",
    "CHRLMMDataset",
    "make_chrlmm_data_module",
    "DataCollatorForCHRLMMDataset",
    "MultimodalDataInfo",
    "MultimodalSequenceAdapter",
    "inspect_multimodal_data",
    "load_label_mapping",
]


def __getattr__(name):
    if name in {
        "load_objaverse_point_cloud",
        "pc_norm",
        "farthest_point_sample",
        "preprocess_v1",
        "preprocess_multimodal_point_cloud",
        "DataCollatorForPointTextDataset",
    }:
        from . import utils

        return getattr(utils, name)
    if name in {"ObjectPointCloudDataset", "make_object_point_data_module"}:
        from .object_point_dataset import ObjectPointCloudDataset, make_object_point_data_module

        return {
            "ObjectPointCloudDataset": ObjectPointCloudDataset,
            "make_object_point_data_module": make_object_point_data_module,
        }[name]
    if name == "ModelNet":
        from .modelnet import ModelNet

        return ModelNet
    if name in {"CHRLMMDataset", "make_chrlmm_data_module", "DataCollatorForCHRLMMDataset"}:
        from .chrlmm_dataset import CHRLMMDataset, make_chrlmm_data_module, DataCollatorForCHRLMMDataset

        return {
            "CHRLMMDataset": CHRLMMDataset,
            "make_chrlmm_data_module": make_chrlmm_data_module,
            "DataCollatorForCHRLMMDataset": DataCollatorForCHRLMMDataset,
        }[name]
    if name in {
        "MultimodalDataInfo",
        "MultimodalSequenceAdapter",
        "inspect_multimodal_data",
        "load_label_mapping",
    }:
        from . import multimodal_adapter

        return getattr(multimodal_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
