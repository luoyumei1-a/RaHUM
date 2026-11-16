"""Post-run02 contract checks, with frozen-source and real-cache compatibility.
Unit fixtures here are numerical checks, not a model performance experiment.
"""
from pathlib import Path
import importlib.util,sys,json,hashlib
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from adapter import PointLLMTokenAdapter,AdapterConfig,compute_auxiliary_losses
from test_adapter_contract import fixture

def close(a,b):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

def main():
 torch.set_num_threads(1);base=Path(__file__).resolve().parent
 spec=importlib.util.spec_from_file_location('run02_adapter_frozen',base/'run02_source_frozen/adapter.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
 checkpoint=torch.load(base/'probe_run_02/actual_backbone_probe_adapters.pt',map_location='cpu',weights_only=False)
 original=old.PointLLMTokenAdapter(old.AdapterConfig(**checkpoint['adapter_config'])).eval();original.load_state_dict(checkpoint['adapter_state_dict'],strict=True)
 revised=PointLLMTokenAdapter(AdapterConfig(**checkpoint['adapter_config'])).eval();revised.load_state_dict(checkpoint['adapter_state_dict'],strict=True)
 with np.load(base/'probe_run_02/actual_PointBERT_train_cache.npz',allow_pickle=False)as z:data={k:torch.from_numpy(z[k])for k in z.files}
 r=data.pop('radar_tokens');v=data.pop('reference_tokens')
 before=original(r,v,data)['corrected_tokens'];after=revised(r,v,data)['corrected_tokens']
 diff=float((before-after).abs().max().detach());assert torch.equal(before,after),diff
 # With no reference support, arbitrary reference contents/geometry/time/calibration
 # including nonfinite padded values cannot influence correction.
 for kind in ['lidar_geometry','rgb_patch']:
  rr,vv,ctx=fixture();cfg=AdapterConfig(reference_kind=kind,reference_has_cls=True,spatial_sigma=32. if kind=='rgb_patch' else 1.)
  model=PointLLMTokenAdapter(cfg).eval()
  with torch.no_grad():
   model.output_residual.weight.normal_(std=.003)
   if kind=='rgb_patch':model.rgb_global_residual.weight.normal_(std=.003);model.rgb_global_residual.bias.fill_(.7)
  if kind=='rgb_patch':
   ctx.pop('reference_centers');ctx.update(projected_uv=torch.randn(1,3,96,2),reference_uv=torch.randn(1,3,96,2),projection_valid=torch.ones(1,3,96,dtype=torch.bool))
  ctx['reference_mask'].zero_();first=model(rr,vv,ctx)['corrected_tokens']
  changed={k:x.clone()for k,x in ctx.items()};changed['reference_timestamps'].fill_(float('nan'));changed['calibration_valid']=torch.full((1,3),float('nan'))
  if kind=='lidar_geometry':changed['reference_centers'].fill_(float('nan'))
  else:changed['reference_uv'].fill_(float('nan'));changed['projected_uv'].fill_(float('nan'));changed['projection_valid'].zero_()
  second=model(rr,torch.full_like(vv,float('nan')),changed)['corrected_tokens']
  assert torch.isfinite(second).all();assert torch.equal(first,second),kind
 # An explicit clean control ignores every reference and disables reference losses.
 rr,vv,ctx=fixture();clean=PointLLMTokenAdapter(AdapterConfig(disable_reference=True)).eval()
 with torch.no_grad():clean.output_residual.weight.normal_(std=.003)
 a=clean(rr,vv,ctx);alter={k:x.clone()for k,x in ctx.items()};alter['reference_timestamps'].fill_(float('nan'));alter['reference_centers'].fill_(float('nan'));alter['calibration_valid'].zero_()
 b=clean(rr,None,alter);assert torch.equal(a['corrected_tokens'],b['corrected_tokens'])
 losses=compute_auxiliary_losses(a,{'category_ids':torch.tensor([1]),'reference_query_category_mask':torch.tensor([True])})
 assert float(losses['reference_query_category'])==0.
 clean.zero_grad(set_to_none=True);loss_a=a['corrected_tokens'].square().mean();loss_a.backward();grad_a={n:p.grad.clone()for n,p in clean.named_parameters()if p.grad is not None};clean.zero_grad(set_to_none=True)
 b2=clean(rr,None,alter);b2['corrected_tokens'].square().mean().backward()
 for n,p in clean.named_parameters():
  if n in grad_a:assert torch.equal(grad_a[n],p.grad),n
  elif p.grad is not None:raise AssertionError('Reference change created a gradient '+n)
 # Camera-global evidence does not require any radar group correspondence.
 rr,vv,ctx=fixture();rgb=PointLLMTokenAdapter(AdapterConfig(reference_kind='rgb_patch',reference_has_cls=True,spatial_sigma=32.)).eval()
 ctx.pop('reference_centers');ctx.update(projected_uv=torch.zeros(1,3,96,2),reference_uv=torch.randn(1,3,96,2),projection_valid=torch.zeros(1,3,96,dtype=torch.bool));ctx['radar_mask'].zero_()
 with torch.no_grad():rgb.rgb_global_residual.weight.normal_(std=.003)
 out=rgb(rr,vv,ctx);assert out['global_visual_reference_present'].all();assert not out['reference_available'].any()
 decoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(384,6,768,dropout=0.,batch_first=True),1).eval();head=nn.Linear(384,19)
 decoder.requires_grad_(False);head.requires_grad_(False)
 x=torch.cat((out['corrected_tokens'],torch.randn(1,1,384)),1);mask=torch.triu(torch.full((98,98),float('-inf')),diagonal=1)
 loss=F.cross_entropy(head(decoder(x,mask=mask))[:,-1],torch.tensor([3]));loss.backward()
 gradients={name:sum(float(p.grad.norm())for key,p in rgb.named_parameters()if key.startswith(name)and p.grad is not None)for name in ['rgb_global_reconstruction','rgb_global_residual','reference_projection']}
 assert all(x>1e-12 for x in gradients.values()),gradients
 report={'status':'PASS','old_run02_source_sha256':hashlib.sha256((base/'run02_source_frozen/adapter.py').read_bytes()).hexdigest(),'new_adapter_sha256':hashlib.sha256((base/'adapter.py').read_bytes()).hexdigest(),'strict_saved_run02_lidar_checkpoint_load':True,'actual_cached_valid_reference_output_max_abs_difference':diff,'actual_valid_reference_outputs_bitwise_identical':True,'masked_reference_tokens_geometry_time_calibration_nan_invariant':True,'explicit_no_reference_control_and_reference_loss_disabled':True,'clean_reference_disabled_forward_loss_gradient_invariant':True,'camera_global_path_with_no_radar_correspondence_fixture_lm_gradient_norms':gradients,'actual_RGB_7B_execution':False,'scope':'contract and realcachedLiDARcompatibility, not hallucinationperformance; newRGBglobalpathneedsactual7Bprobe'}
 (base/'postprobe_adapter_test_result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
