"""CHR-LMM hallucination reduction modules.

This file implements the design described in the manuscript:
spatial-semantic alignment, discrepancy-based hallucination indication,
top-K reliability selection, and reliability-aware hallucination elimination.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalAlignmentModule(nn.Module):
    """Spatial-semantic alignment between sparse mmWave features and vision tokens."""

    def __init__(self, point_dim: int = 384, vision_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.point_dim = point_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def project_points_to_image(self, point_xyz, camera_intrinsics, radar_to_camera):
        """Project radar points into the image plane with camera intrinsics/extrinsics."""
        batch, num_points, _ = point_xyz.shape
        ones = torch.ones(batch, num_points, 1, device=point_xyz.device, dtype=point_xyz.dtype)
        point_h = torch.cat([point_xyz, ones], dim=-1)
        camera_xyz = torch.matmul(point_h, radar_to_camera.transpose(1, 2))
        depth = camera_xyz[..., 2].clamp_min(1e-4)
        uvw = torch.matmul(camera_xyz, camera_intrinsics.transpose(1, 2))
        return uvw[..., :2] / depth.unsqueeze(-1)

    def build_spatial_prior(self, projected_uv, vision_uv, image_size=None, sigma: float = 32.0):
        """Build a Gaussian correspondence prior for projected radar and vision tokens."""
        if image_size is not None:
            scale = projected_uv.new_tensor([image_size[0], image_size[1]]).view(1, 1, 2)
            projected_uv = projected_uv / scale
            vision_uv = vision_uv / scale
            sigma = sigma / float(max(image_size))
        delta = projected_uv.unsqueeze(2) - vision_uv.unsqueeze(1)
        dist2 = torch.sum(delta * delta, dim=-1)
        return torch.exp(-dist2 / (2.0 * sigma * sigma + 1e-8)).clamp_min(1e-6)

    def forward(
        self,
        point_features,
        vision_features,
        point_xyz: Optional[torch.Tensor] = None,
        vision_uv: Optional[torch.Tensor] = None,
        camera_intrinsics: Optional[torch.Tensor] = None,
        radar_to_camera: Optional[torch.Tensor] = None,
        image_size=None,
    ):
        point_feat = self.point_proj(point_features)
        vision_feat = self.vision_proj(vision_features)

        logits = torch.bmm(point_feat, vision_feat.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        if point_xyz is not None and vision_uv is not None and camera_intrinsics is not None and radar_to_camera is not None:
            projected_uv = self.project_points_to_image(point_xyz, camera_intrinsics, radar_to_camera)
            spatial_prior = self.build_spatial_prior(projected_uv, vision_uv, image_size=image_size)
            logits = logits + torch.log(spatial_prior)

        attention_weights = torch.softmax(logits, dim=-1)
        aligned_features = torch.bmm(attention_weights, vision_feat)
        aligned_features = self.norm(aligned_features + point_feat)
        return aligned_features, vision_feat, attention_weights


class DiscrepancyDetector(nn.Module):
    """Discrepancy-based hallucination indicator."""

    def __init__(self, hidden_dim: int = 512, num_classes: int = 11, alpha: float = 0.3,
                 beta: float = 0.2, tau: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.point_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.vision_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def compute_discrepancy(self, point_pred, vision_pred, temporal_history=None):
        batch, num_points, num_classes = point_pred.shape
        num_vision = vision_pred.shape[1]
        point_exp = point_pred.unsqueeze(2).expand(batch, num_points, num_vision, num_classes)
        vision_exp = vision_pred.unsqueeze(1).expand(batch, num_points, num_vision, num_classes)

        l2_distance = torch.norm(point_exp - vision_exp, p=2, dim=-1)
        point_prob = F.softmax(point_exp, dim=-1)
        vision_prob = F.softmax(vision_exp, dim=-1)
        point_log_prob = F.log_softmax(point_exp, dim=-1)
        vision_log_prob = F.log_softmax(vision_exp, dim=-1)
        symmetric_kl = (
            F.kl_div(point_log_prob, vision_prob, reduction="none").sum(dim=-1)
            + F.kl_div(vision_log_prob, point_prob, reduction="none").sum(dim=-1)
        )

        temporal_inconsistency = torch.zeros_like(l2_distance)
        if temporal_history:
            diffs = []
            for hist_point, hist_vision in temporal_history:
                hist_point = hist_point.to(point_pred.device)
                hist_vision = hist_vision.to(vision_pred.device)
                hist_point_exp = hist_point.unsqueeze(2).expand(batch, num_points, num_vision, num_classes)
                hist_vision_exp = hist_vision.unsqueeze(1).expand(batch, num_points, num_vision, num_classes)
                diffs.append(torch.norm(hist_point_exp - hist_vision_exp, p=2, dim=-1))
            temporal_inconsistency = torch.stack(diffs, dim=0).mean(dim=0)

        discrepancy = l2_distance + self.alpha * symmetric_kl + self.beta * temporal_inconsistency
        hallucination_map = (discrepancy > self.tau).float()
        return discrepancy, hallucination_map

    def forward(self, point_features, vision_features, temporal_history=None):
        point_pred = self.point_predictor(point_features)
        vision_pred = self.vision_predictor(vision_features)
        discrepancy, hallucination_map = self.compute_discrepancy(point_pred, vision_pred, temporal_history)
        return hallucination_map, discrepancy, point_pred, vision_pred


class AdaptiveReliabilityWeighting(nn.Module):
    """Reliability-aware top-K group selection."""

    def __init__(self, feature_dim: int = 512, top_k: int = 96, gamma: float = 2.0):
        super().__init__()
        self.top_k = top_k
        self.gamma = gamma
        self.point_reliability = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.vision_reliability = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, point_features, vision_features, attention_weights, discrepancy_map=None):
        batch, num_points, _ = point_features.shape
        num_vision = vision_features.shape[1]
        point_rel = self.point_reliability(point_features).squeeze(-1)
        vision_rel = self.vision_reliability(vision_features).squeeze(-1)

        point_rel_pairs = point_rel.unsqueeze(2) * attention_weights
        vision_rel_pairs = vision_rel.unsqueeze(1) * attention_weights
        point_weights = torch.exp(self.gamma * point_rel_pairs)
        vision_weights = torch.exp(self.gamma * vision_rel_pairs)
        total = point_weights + vision_weights + 1e-8
        point_weights = point_weights / total
        vision_weights = vision_weights / total

        group_scores = attention_weights * (point_rel_pairs + vision_rel_pairs) / 2.0
        if discrepancy_map is not None:
            norm_discrepancy = discrepancy_map / (discrepancy_map.amax(dim=(1, 2), keepdim=True) + 1e-8)
            group_scores = group_scores * (1.0 - norm_discrepancy).clamp_min(0.0)

        k = min(self.top_k, num_points * num_vision)
        flat_scores = group_scores.reshape(batch, -1)
        top_k_scores, top_k_indices = torch.topk(flat_scores, k=k, dim=1)
        top_k_point_idx = top_k_indices // num_vision
        top_k_vision_idx = top_k_indices % num_vision
        top_k_groups = torch.stack([top_k_point_idx, top_k_vision_idx], dim=-1)

        batch_idx = torch.arange(batch, device=point_features.device).unsqueeze(1).expand(batch, k)
        fusion_point = point_weights[batch_idx, top_k_point_idx, top_k_vision_idx]
        fusion_vision = vision_weights[batch_idx, top_k_point_idx, top_k_vision_idx]
        fusion_weights = torch.stack([fusion_point, fusion_vision], dim=-1)
        return top_k_groups, fusion_weights, top_k_scores


class ContrastiveHallucinationElimination(nn.Module):
    """Reliability-guided evidence reconstruction and auxiliary prediction.

    The historical class name is retained for checkpoint compatibility. Expert
    gate logits are optimized through the corrected-prediction loss, without an
    InfoNCE or other contrastive-learning objective.
    """

    def __init__(self, feature_dim: int = 512, num_classes: int = 11, num_experts: int = 5):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 2,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=3,
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(),
                    nn.Linear(feature_dim // 2, num_classes),
                )
                for _ in range(num_experts)
            ]
        )
        self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        self.output_proj = nn.Linear(feature_dim, num_classes)

    def forward(self, point_features, vision_features, top_k_groups, fusion_weights, point_pred, vision_pred):
        batch, k, _ = top_k_groups.shape
        batch_idx = torch.arange(batch, device=point_features.device).unsqueeze(1).expand(batch, k)
        point_idx = top_k_groups[:, :, 0]
        vision_idx = top_k_groups[:, :, 1]

        selected_point = point_features[batch_idx, point_idx]
        selected_vision = vision_features[batch_idx, vision_idx]
        point_weight = fusion_weights[:, :, 0].unsqueeze(-1)
        vision_weight = fusion_weights[:, :, 1].unsqueeze(-1)
        fused_features = point_weight * selected_point + vision_weight * selected_vision

        decoded_features = self.decoder(tgt=fused_features, memory=selected_vision)
        expert_outputs = torch.stack([expert(decoded_features) for expert in self.experts], dim=2)
        weights = F.softmax(self.expert_weights, dim=0).view(1, 1, -1, 1)
        mixed_output = (expert_outputs * weights).sum(dim=2)
        corrected_predictions = self.output_proj(decoded_features) + mixed_output
        return {
            "corrected_features": decoded_features,
            "corrected_predictions": corrected_predictions,
        }


class CHRLMMHallucinationReducer(nn.Module):
    """Full CHR-LMM detection and elimination module."""

    def __init__(self, point_dim: int = 384, vision_dim: int = 2048, hidden_dim: int = 512,
                 num_classes: int = 11, top_k: int = 96, alpha: float = 0.3,
                 beta: float = 0.2, tau: float = 0.75, gamma: float = 2.0,
                 temporal_history_length: int = 5):
        super().__init__()
        if temporal_history_length < 0:
            raise ValueError("temporal_history_length must be non-negative.")
        self.temporal_history_length = temporal_history_length
        self.alignment = HierarchicalAlignmentModule(point_dim, vision_dim, hidden_dim)
        self.detector = DiscrepancyDetector(hidden_dim, num_classes, alpha=alpha, beta=beta, tau=tau)
        self.weighting = AdaptiveReliabilityWeighting(hidden_dim, top_k=top_k, gamma=gamma)
        self.elimination = ContrastiveHallucinationElimination(hidden_dim, num_classes)

    def _forward_frame(
        self,
        point_features,
        vision_features,
        temporal_history=None,
        point_xyz: Optional[torch.Tensor] = None,
        vision_uv: Optional[torch.Tensor] = None,
        camera_intrinsics: Optional[torch.Tensor] = None,
        radar_to_camera: Optional[torch.Tensor] = None,
        image_size=None,
    ):
        aligned_features, projected_vision_features, attention_weights = self.alignment(
            point_features,
            vision_features,
            point_xyz=point_xyz,
            vision_uv=vision_uv,
            camera_intrinsics=camera_intrinsics,
            radar_to_camera=radar_to_camera,
            image_size=image_size,
        )
        hallucination_map, discrepancy_map, point_pred, vision_pred = self.detector(
            aligned_features, projected_vision_features, temporal_history
        )
        top_k_groups, fusion_weights, reliability_scores = self.weighting(
            aligned_features, projected_vision_features, attention_weights, discrepancy_map
        )
        elimination_outputs = self.elimination(
            aligned_features, projected_vision_features, top_k_groups, fusion_weights, point_pred, vision_pred
        )
        return {
            "hallucination_map": hallucination_map,
            "discrepancy_map": discrepancy_map,
            "corrected_features": elimination_outputs["corrected_features"],
            "corrected_predictions": elimination_outputs["corrected_predictions"],
            "top_k_groups": top_k_groups,
            "reliability_scores": reliability_scores,
            "point_predictions": point_pred,
            "vision_predictions": vision_pred,
            "fusion_weights": fusion_weights,
            "attention_weights": attention_weights,
        }

    @staticmethod
    def _frame_value(value, frame_index: int, frame_count: int):
        if value is None:
            return None
        if torch.is_tensor(value) and value.ndim >= 4 and value.shape[1] == frame_count:
            return value[:, frame_index]
        return value

    def forward(
        self,
        point_features,
        vision_features,
        temporal_history=None,
        point_xyz: Optional[torch.Tensor] = None,
        vision_uv: Optional[torch.Tensor] = None,
        camera_intrinsics: Optional[torch.Tensor] = None,
        radar_to_camera: Optional[torch.Tensor] = None,
        image_size=None,
    ):
        """Reduce one frame [B,N,D] or a synchronized sequence [B,T,N,D]."""

        if point_features.ndim == 3:
            if vision_features.ndim != 3:
                raise ValueError(
                    "Single-frame point and vision features must both be rank 3."
                )
            return self._forward_frame(
                point_features,
                vision_features,
                temporal_history=temporal_history,
                point_xyz=point_xyz,
                vision_uv=vision_uv,
                camera_intrinsics=camera_intrinsics,
                radar_to_camera=radar_to_camera,
                image_size=image_size,
            )

        if point_features.ndim != 4 or vision_features.ndim != 4:
            raise ValueError(
                "Sequence point and vision features must have shapes [B,T,N,D] "
                f"and [B,T,V,D], got {point_features.shape} and {vision_features.shape}."
            )
        if point_features.shape[:2] != vision_features.shape[:2]:
            raise ValueError(
                "Point and vision sequences must share batch and frame dimensions, "
                f"got {point_features.shape[:2]} and {vision_features.shape[:2]}."
            )

        frame_count = point_features.shape[1]
        history = list(temporal_history or [])
        if self.temporal_history_length == 0:
            history = []
        elif len(history) > self.temporal_history_length:
            history = history[-self.temporal_history_length :]

        frame_outputs = []
        for frame_index in range(frame_count):
            current = self._forward_frame(
                point_features[:, frame_index],
                vision_features[:, frame_index],
                temporal_history=history or None,
                point_xyz=self._frame_value(point_xyz, frame_index, frame_count),
                vision_uv=self._frame_value(vision_uv, frame_index, frame_count),
                camera_intrinsics=self._frame_value(
                    camera_intrinsics, frame_index, frame_count
                ),
                radar_to_camera=self._frame_value(
                    radar_to_camera, frame_index, frame_count
                ),
                image_size=image_size,
            )
            frame_outputs.append(current)
            if self.temporal_history_length > 0:
                history.append(
                    (
                        current["point_predictions"].detach(),
                        current["vision_predictions"].detach(),
                    )
                )
                history = history[-self.temporal_history_length :]

        sequence_outputs = {
            key: torch.stack([output[key] for output in frame_outputs], dim=1)
            for key in frame_outputs[0]
        }
        result = dict(frame_outputs[-1])
        result["sequence_outputs"] = sequence_outputs
        return result
