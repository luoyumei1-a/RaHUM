"""PointLLM-native radar/reference token correction, with no point-label readout.

The learned PointBERT/PointLLM backbone is owned by the runtime. This module
consumes its actual CLS+group tokens and returns the same token shape. Ground
truth is accepted only by ``compute_auxiliary_losses``, never by forward.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AdapterConfig:
    token_dim: int = 384
    reference_dim: int = 384
    groups: int = 96
    reference_has_cls: bool = True
    reference_kind: str = 'lidar_geometry'  # alternatively rgb_patch
    local_memory: int = 8
    geometry_dims: int = 2  # published Radar Ghost has no radar elevation
    spatial_sigma: float = 1.0  # metres for lidar_geometry, pixels for rgb_patch
    history_window: int = 5
    history_sigma_m: float = 1.0  # independent of RGB pixel bandwidth
    attribute_dim: int = 2  # dataset-declared measured physical channels, schema recorded separately
    num_experts: int = 5
    num_categories: int = 2  # actual oracle-main ontology: pedestrian / cyclist
    disable_reference: bool = False  # clean radar/temporal adapter control


def _mlp(din: int, hidden: int, dout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dout))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(dim) / w.sum(dim).clamp_min(1.)


def _gather_memory(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # [B,M,D], [B,Q,K] -> [B,Q,K,D], without copying all M candidates per query.
    b = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[b, indices]


class PointLLMTokenAdapter(nn.Module):
    """Residual reconstruction of real PointBERT tokens before the LM projector.

    There is no persistent cross-batch history. Discrepancy affects continuous
    attention, utility and reconstruction; experts output feature residuals.
    """
    def __init__(self, config: AdapterConfig | None = None, **kwargs):
        super().__init__()
        self.config = config or AdapterConfig(**kwargs)
        c = self.config
        if c.reference_kind not in {'lidar_geometry', 'rgb_patch'}:
            raise ValueError('reference_kind must explicitly name lidar_geometry or rgb_patch')
        if c.geometry_dims not in {2, 3} or c.spatial_sigma <= 0 or c.history_sigma_m <= 0 or c.local_memory < 1:
            raise ValueError('Invalid physical geometry configuration')
        d = c.token_dim
        self.radar_norm = nn.LayerNorm(d)
        self.attribute_projection = _mlp(c.attribute_dim, d // 2, d)
        self.reference_projection = nn.Sequential(nn.Linear(c.reference_dim, d), nn.LayerNorm(d))
        self.query_projection = nn.Linear(d, d, bias=False)
        self.key_projection = nn.Linear(d, d, bias=False)
        # Learned latent discrepancy has no invented LiDAR direct/MP semantics.
        self.discrepancy = _mlp(2 * d + 2, d // 2, 1)
        self.discrepancy_penalty_raw = nn.Parameter(torch.tensor(0.54132485))
        self.temporal_projection = _mlp(7, d // 2, d)
        self.history_projection = nn.Linear(d, d, bias=False)
        # Six observed-support channels and seven causal-history descriptors.
        self.utility = _mlp(2 * d + 6 + 7, d // 2, 2)
        self.reconstruction = _mlp(3 * d + 6 + 7, d, d)
        self.experts = nn.ModuleList([_mlp(d, d // 2, d) for _ in range(c.num_experts)])
        self.expert_router = nn.Linear(d, c.num_experts)
        self.output_residual = nn.Linear(d, d)
        self.cls_residual = nn.Linear(d, d, bias=False)
        # Exact identity to the same-budget native PointLLM at initialization.
        nn.init.zeros_(self.output_residual.weight)
        nn.init.zeros_(self.output_residual.bias)
        self.category_head = nn.Linear(d, c.num_categories)
        self.radar_category_head = nn.Linear(d, c.num_categories)
        # Query-conditioned context category, not a class label for a LiDAR point.
        self.reference_query_category_head = nn.Linear(d, c.num_categories)
        self.support_head = nn.Linear(d, 1)
        # Camera-only evidence can reach the CLS without a radar-group match.
        # This is a global visual evidence path, not guaranteed object coverage.
        if c.reference_kind == 'rgb_patch' and not c.disable_reference:
            self.rgb_global_reconstruction = _mlp(2 * d, d, d)
            self.rgb_global_residual = nn.Linear(d, d)
            nn.init.zeros_(self.rgb_global_residual.weight)
            nn.init.zeros_(self.rgb_global_residual.bias)
        if c.disable_reference:
            for module in [self.reference_projection,self.query_projection,self.key_projection,self.discrepancy,self.utility,self.reference_query_category_head]:
                module.requires_grad_(False)
            self.discrepancy_penalty_raw.requires_grad_(False)

    def _validate(self, radar_tokens, reference_tokens, context):
        c = self.config
        if radar_tokens.ndim != 4 or radar_tokens.shape[2:] != (c.groups + 1, c.token_dim):
            raise ValueError(f'radar_tokens must be [B,T,{c.groups+1},{c.token_dim}]')
        b, t = radar_tokens.shape[:2]
        if reference_tokens.ndim != 4 or reference_tokens.shape[:2] != (b, t) or reference_tokens.shape[-1] != c.reference_dim:
            raise ValueError('Reference tokens must share [B,T] and declared reference_dim')
        for name in context:
            if any(s in name.lower() for s in ('label', 'category', 'ground_truth', 'supervision', 'claim', 'instance_id')):
                raise ValueError(f'Ground-truth-like key is forbidden in forward physical_context: {name}')
        required = ['radar_centers', 'radar_attributes', 'radar_mask', 'reference_mask',
                    'frame_timestamps', 'reference_timestamps', 'sequence_ids',
                    'calibration_valid', 'history_valid']
        required += ['reference_centers'] if c.reference_kind == 'lidar_geometry' else ['projected_uv', 'reference_uv', 'projection_valid']
        for name in required:
            if name not in context:
                raise ValueError(f'Missing explicit physical metadata: {name}')
        if context['radar_centers'].shape != (b, t, c.groups, 3):
            raise ValueError('radar_centers must be actual group centers in the common physical frame')
        if context['radar_attributes'].shape != (b, t, c.groups, c.attribute_dim):
            raise ValueError('radar_attributes must use the same actual group memberships')
        m = reference_tokens.shape[2] - int(c.reference_has_cls)
        if m < 1 or context['reference_mask'].shape != (b, t, m):
            raise ValueError('reference_mask must exclude any reference CLS token')
        if context['radar_mask'].shape != (b, t, c.groups):
            raise ValueError('radar_mask must exclude radar CLS')
        if c.reference_kind == 'lidar_geometry' and context['reference_centers'].shape != (b,t,m,3):
            raise ValueError('LiDAR reference_centers must be [B,T,M,3] common-frame metres')
        if c.reference_kind == 'rgb_patch':
            if context['projected_uv'].shape != (b,t,c.groups,2) or context['reference_uv'].shape != (b,t,m,2) or context['projection_valid'].shape != (b,t,c.groups):
                raise ValueError('RGB UV metadata must match actual group/patch counts')
        for name in ['frame_timestamps', 'reference_timestamps', 'sequence_ids', 'calibration_valid', 'history_valid']:
            if context[name].shape != (b, t):
                raise ValueError(f'{name} must have shape [B,T]')
        times = context['frame_timestamps']
        if not torch.isfinite(times).all():
            raise ValueError('Frame timestamps must be finite; use variable-length real windows rather than fake padded time')
        seq = context['sequence_ids']
        same = seq[:, 1:] == seq[:, :-1]
        if not bool(same.all()):
            raise ValueError('Each input window must belong to one recording; reset by starting a separate sample')
        if ((times[:, 1:] <= times[:, :-1]) & same).any():
            raise ValueError('Frame timestamps must increase within each recording')
        reference_present = context['reference_mask'].any(-1)
        if not torch.isfinite(context['reference_timestamps'][reference_present]).all():
            raise ValueError('Available reference observations require finite timestamps')
        if not torch.isfinite(radar_tokens).all():
            raise ValueError('Native radar token tensor must be finite, including explicit missing-modality slots')
        if not torch.isfinite(reference_tokens[:, :, int(c.reference_has_cls):][context['reference_mask'].bool()]).all():
            raise ValueError('Available reference tokens must be finite')
        if ((context['reference_timestamps'] > times) & reference_present).any():
            raise ValueError('A future reference observation cannot enter this causal adapter')
        return b, t, m

    def _history(self, native, centers, attrs, masks, context, j):
        b, _, q, d = native.shape
        temporal = native.new_zeros(b, q, 7)
        previous = native.new_zeros(b, q, d)
        counts = native.new_zeros(b, q)
        if self.config.history_window == 0:
            return temporal, previous, counts
        # Ratios are invariant to timestamps expressed in seconds or raw ticks.
        times = context['frame_timestamps'].to(torch.float64)
        seq = context['sequence_ids']; tiny = torch.finfo(torch.float64).tiny
        history_valid = context['history_valid'].bool()
        gd = self.config.geometry_dims; scale_m = self.config.history_sigma_m
        for batch in range(b):
            if not bool(history_valid[batch, j]):
                continue
            candidates = [p for p in range(j - 1, max(-1, j - self.config.history_window - 1), -1)
                          if bool(history_valid[batch, p]) and bool(seq[batch, p] == seq[batch, j])
                          and bool(times[batch, p] < times[batch, j]) and bool(masks[batch, p].any())]
            if not candidates:
                continue
            observed = [times[batch, n] - times[batch, n - 1] for n in range(1, j + 1)
                        if bool(seq[batch, n] == seq[batch, n - 1]) and bool(seq[batch, n] == seq[batch, j])
                        and bool(times[batch, n] > times[batch, n - 1])]
            typical = torch.stack(observed).median() if observed else (times[batch, j] - times[batch, candidates[0]])
            for p in candidates:
                dist = torch.cdist(centers[batch, j, :, :gd].float(), centers[batch, p, :, :gd].float())
                dist = dist.masked_fill(~masks[batch, p][None, :], float('inf'))
                index = dist.argmin(-1)
                delta = centers[batch, p, index] - centers[batch, j]
                if gd == 2:
                    delta = torch.cat((delta[:, :2], torch.zeros_like(delta[:, 2:])), -1)
                age = ((times[batch, j] - times[batch, p]) / typical.clamp_min(tiny)).to(native.dtype)
                distance = delta[:, :gd].norm(dim=-1) / scale_m
                # A group-mean measured vr difference is an empirical descriptor,
                # not a claimed Cartesian/ego-motion-compensated velocity residual.
                value = torch.cat((delta / scale_m, distance[:, None],
                                   (attrs[batch, p, index, 0] - attrs[batch, j, :, 0])[:, None],
                                   age.expand(q, 1), torch.ones(q, 1, device=native.device, dtype=native.dtype)), -1)
                active = masks[batch, j].to(native.dtype)
                temporal[batch] += value * active[:, None]
                previous[batch] += native[batch, p, index] * active[:, None]
                counts[batch] += active
        temporal = temporal / counts.clamp_min(1.)[..., None]
        previous = previous / counts.clamp_min(1.)[..., None]
        return temporal, previous, counts

    def forward(self, radar_tokens: torch.Tensor, reference_tokens: torch.Tensor,
                physical_context: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        if self.config.disable_reference:
            physical_context=dict(physical_context)
            bb,tt=radar_tokens.shape[:2];dev=radar_tokens.device
            reference_tokens=radar_tokens.new_zeros(bb,tt,1+int(self.config.reference_has_cls),self.config.reference_dim)
            physical_context['reference_mask']=torch.zeros(bb,tt,1,device=dev,dtype=torch.bool)
            physical_context['reference_timestamps']=physical_context['frame_timestamps'].clone()
            physical_context['calibration_valid']=torch.zeros(bb,tt,device=dev,dtype=torch.bool)
            if self.config.reference_kind=='lidar_geometry':physical_context['reference_centers']=radar_tokens.new_zeros(bb,tt,1,3)
            else:
                physical_context['reference_uv']=radar_tokens.new_zeros(bb,tt,1,2)
                physical_context['projected_uv']=radar_tokens.new_zeros(bb,tt,self.config.groups,2)
                physical_context['projection_valid']=torch.zeros(bb,tt,self.config.groups,device=dev,dtype=torch.bool)
        b, t, m = self._validate(radar_tokens, reference_tokens, physical_context)
        c = self.config; d = c.token_dim; device = radar_tokens.device
        dtype = self.output_residual.weight.dtype
        base = radar_tokens.to(dtype)
        context = {k: v.to(device) if torch.is_tensor(v) else v for k, v in physical_context.items()}
        centers = context['radar_centers'].to(dtype)
        attrs = context['radar_attributes'].to(dtype)
        masks = context['radar_mask'].bool()
        refs = reference_tokens[:, :, int(c.reference_has_cls):].to(device=device, dtype=dtype)
        ref_mask = context['reference_mask'].bool()
        present_by_frame=ref_mask.any(-1)
        # Missing-reference padding may be arbitrary, including NaN; never let it
        # enter projections, distance kernels, observation age or calibration.
        refs=torch.where(ref_mask[...,None],refs,torch.zeros_like(refs))
        context['reference_timestamps']=torch.where(present_by_frame,context['reference_timestamps'],context['frame_timestamps'])
        context['calibration_valid']=context['calibration_valid'].bool() & present_by_frame
        if c.reference_kind=='lidar_geometry':
            context['reference_centers']=torch.where((ref_mask & context['calibration_valid'][...,None])[...,None],context['reference_centers'],torch.zeros_like(context['reference_centers']))
        else:
            context['reference_uv']=torch.where((ref_mask & context['calibration_valid'][...,None])[...,None],context['reference_uv'],torch.zeros_like(context['reference_uv']))
            context['projection_valid']=context['projection_valid'].bool() & context['calibration_valid'][...,None]
            context['projected_uv']=torch.where(context['projection_valid'][...,None],context['projected_uv'],torch.zeros_like(context['projected_uv']))
        native = base[:, :, 1:]
        queries = self.radar_norm(native + self.attribute_projection(attrs))
        ref_features = self.reference_projection(refs)
        outputs = []
        for j in range(t):
            q = queries[:, j]; r = ref_features[:, j]
            corr = torch.bmm(self.query_projection(q), self.key_projection(r).transpose(1, 2)) / math.sqrt(d)
            calibrated = context['calibration_valid'][:, j].bool()
            if c.reference_kind == 'lidar_geometry':
                metric_distance = torch.cdist(centers[:, j, :, :c.geometry_dims].float(), context['reference_centers'][:, j, :, :c.geometry_dims].float()).to(dtype)
                projection_valid = torch.ones(b, c.groups, device=device, dtype=torch.bool)
            else:
                metric_distance = torch.cdist(context['projected_uv'][:, j].float(), context['reference_uv'][:, j].float()).to(dtype)
                projection_valid = context['projection_valid'][:, j].bool()
            candidate_valid = ref_mask[:, j, None, :].expand(b, c.groups, m) & masks[:, j, :, None]
            if c.reference_kind == 'rgb_patch':
                candidate_valid = candidate_valid & projection_valid[:, :, None]
            # Geometry selects local memory when available. Otherwise semantic selection
            # is explicit; the geometric prior and its descriptors are then disabled.
            selection = torch.where(calibrated[:, None, None], -metric_distance, corr.detach())
            selection = selection.masked_fill(~candidate_valid, -float('inf'))
            k = min(c.local_memory, m)
            index = selection.topk(k, dim=-1).indices
            valid = candidate_valid.gather(-1, index)
            r_local = _gather_memory(r, index)
            distance = metric_distance.gather(-1, index)
            corr_local = corr.gather(-1, index)
            known_distance = torch.where(calibrated[:, None, None], torch.log1p(distance / c.spatial_sigma), torch.zeros_like(distance))
            time_values = context['frame_timestamps'].to(torch.float64)
            reference_times = context['reference_timestamps'].to(torch.float64)
            tiny_time = torch.finfo(torch.float64).tiny
            time_diff = time_values[:, j] - reference_times[:, j]
            # Dimensionless age. No vr[m/s] * unknown raw-tick displacement is used.
            intervals = time_values[:, 1:j + 1] - time_values[:, :j]
            if j:
                same = (context['sequence_ids'][:, 1:j + 1] == context['sequence_ids'][:, :j]) & (context['sequence_ids'][:, 1:j + 1] == context['sequence_ids'][:, j:j+1])
                typical = torch.stack([intervals[n][same[n]].median() if same[n].any() else time_diff[n].clamp_min(tiny_time) for n in range(b)])
            else:
                typical = time_diff.clamp_min(tiny_time)
            ref_age = (time_diff / typical.clamp_min(tiny_time)).to(dtype)
            difference = q[:, :, None, :] - r_local
            disp_input = torch.cat((difference, difference.abs(), known_distance[..., None], ref_age[:, None, None, None].expand(b, c.groups, k, 1)), -1)
            discrepancy_logits = self.discrepancy(disp_input).squeeze(-1)
            discrepancy = F.softplus(discrepancy_logits)
            log_g = -.5 * (distance / c.spatial_sigma).square().clamp(max=80.)
            log_g = torch.where(calibrated[:, None, None], log_g, torch.zeros_like(log_g))
            scores = corr_local + log_g - F.softplus(self.discrepancy_penalty_raw) * discrepancy
            available = valid.any(-1)
            safe = valid.clone(); safe[..., 0] |= ~available
            attention = scores.masked_fill(~safe, -1e9).softmax(-1) * valid.to(dtype)
            reference_context = (attention[..., None] * r_local).sum(-2)
            weighted_discrepancy = (attention * discrepancy).sum(-1)
            observed_distance = (attention * known_distance).sum(-1)
            nearest_distance = known_distance.masked_fill(~valid, float('inf')).amin(-1)
            nearest_distance = torch.where(available, nearest_distance, torch.zeros_like(nearest_distance))
            observed = torch.stack((nearest_distance, observed_distance, weighted_discrepancy,
                                    valid.to(dtype).mean(-1), ref_age[:, None].expand(b, c.groups),
                                    calibrated[:, None].expand(b, c.groups).to(dtype)), -1)
            observed = observed * (masks[:, j] & available)[..., None]
            temporal, history, history_counts = self._history(native, centers, attrs, masks, context, j)
            temporal_embedding = (self.temporal_projection(temporal) + self.history_projection(history)) * temporal[..., -1:]
            utility_logits = self.utility(torch.cat((q, reference_context, observed, temporal), -1))
            # No multiplication by a normalized candidate attention probability.
            utility_logits = utility_logits.clone()
            utility_logits[..., 0] = torch.where(available, utility_logits[..., 0], torch.zeros_like(utility_logits[..., 0]))
            utility_logits[..., 1] = utility_logits[..., 1].masked_fill(~available, -1e9)
            utility = utility_logits.softmax(-1)
            reconstruction_input = torch.cat((utility[..., :1] * q, utility[..., 1:] * reference_context,
                                               temporal_embedding, observed, temporal), -1)
            feature = self.reconstruction(reconstruction_input)
            experts = torch.stack([expert(feature) for expert in self.experts], dim=-2)
            expert_weights = self.expert_router(feature).softmax(-1)
            expert_features = (experts * expert_weights[..., None]).sum(-2)
            residual = self.output_residual(feature + expert_features) * masks[:, j, :, None]
            patch = base[:, j, 1:] + residual
            cls = base[:, j, :1] + self.cls_residual(_masked_mean(residual, masks[:, j], 1))[:, None, :]
            global_visual_context = None
            global_visual_present = ref_mask[:,j].any(-1) & (not c.disable_reference)
            if c.reference_kind=='rgb_patch' and not c.disable_reference:
                global_visual_context=_masked_mean(r,ref_mask[:,j],1)
                global_feature=self.rgb_global_reconstruction(torch.cat((base[:,j,0],global_visual_context),-1))
                # Mask AFTER trainable projections: biases cannot create missing-RGB evidence.
                global_delta=self.rgb_global_residual(global_feature)*global_visual_present[:,None].to(dtype)
                cls=cls+global_delta[:,None,:]
            corrected = torch.cat((cls, patch), dim=1)
            corrected_pool = _masked_mean(patch, masks[:, j], 1)
            outputs.append({'corrected_tokens': corrected.to(radar_tokens.dtype),
                            'category_logits': self.category_head(corrected_pool),
                            'radar_category_logits': self.radar_category_head(_masked_mean(native[:, j], masks[:, j], 1)),
                            'reference_query_category_logits': self.reference_query_category_head(_masked_mean(reference_context, masks[:, j], 1)),
                            'support_logits': self.support_head(feature).squeeze(-1),
                            'discrepancy_logits': discrepancy_logits, 'discrepancy': discrepancy,
                            'attention': attention, 'utility_weights': utility, 'expert_weights': expert_weights,
                            'reference_available': available, 'selected_reference_indices': index,
                            'global_visual_context': global_visual_context, 'global_visual_reference_present': global_visual_present,
                            'temporal_features': temporal, 'temporal_history_count': history_counts, 'observability': observed, 'log_geometric_prior': log_g})
        result = dict(outputs[-1])
        result['sequence_corrected_tokens'] = torch.stack([x['corrected_tokens'] for x in outputs], dim=1)
        result['sequence_outputs'] = outputs
        result['reference_kind'] = c.reference_kind
        result['reference_enabled'] = not c.disable_reference
        return result


def compute_auxiliary_losses(output: Mapping[str, torch.Tensor], supervision: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Separate optional losses; independent GT cannot enter adapter.forward.

    Class IDs here are contiguous dataset ontology indices, not raw CMTO codes.
    Reference-query supervision is masked unless independently justified. Support
    targets concern annotated physical groups, not language-claim hallucination.
    """
    losses = {}
    for name, key, mask_key in [('category', 'category_logits', 'category_mask'),
                                 ('radar_category', 'radar_category_logits', 'radar_category_mask'),
                                 ('reference_query_category', 'reference_query_category_logits', 'reference_query_category_mask')]:
        logits = output[key]
        target = supervision.get('category_ids')
        mask = supervision.get(mask_key)
        if (name=='reference_query_category' and not output.get('reference_enabled',True)) or target is None or mask is None or not bool(mask.any()):
            losses[name] = logits.sum() * 0.
        else:
            mask = mask.bool().to(logits.device); target = target.to(logits.device).long()
            losses[name] = F.cross_entropy(logits[mask], target[mask])
    support_target = supervision.get('support_targets'); support_mask = supervision.get('support_mask')
    if support_target is None or support_mask is None or not bool(support_mask.any()):
        losses['support'] = output['support_logits'].sum() * 0.
    else:
        mask = support_mask.bool().to(output['support_logits'].device)
        losses['support'] = F.binary_cross_entropy_with_logits(output['support_logits'][mask], support_target.to(output['support_logits'].device)[mask].float())
    return losses


def build_adapter(**kwargs) -> PointLLMTokenAdapter:
    return PointLLMTokenAdapter(AdapterConfig(**kwargs))
