"""Unit fixtures check the adapter's geometry/causality/gradient contract.
These deterministic numerical fixtures are not experiment or benchmark data.
"""
import copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from adapter import AdapterConfig,PointLLMTokenAdapter,compute_auxiliary_losses

def fixture(batch=1,frames=3,groups=96,dim=384):
 torch.manual_seed(7588);r=torch.randn(batch,frames,groups+1,dim);v=torch.randn(batch,frames,groups+1,dim)
 center=torch.randn(batch,1,groups,3).expand(batch,frames,groups,3).clone()
 center+=torch.arange(frames).view(1,frames,1,1)*.06
 attrs=torch.randn(batch,frames,groups,2);attrs[...,1]+=16
 times=(torch.arange(frames,dtype=torch.float64)+1)[None,:].expand(batch,frames)*100000
 ctx={'radar_centers':center,'reference_centers':center+.15,'radar_attributes':attrs,
      'radar_mask':torch.ones(batch,frames,groups,dtype=torch.bool),'reference_mask':torch.ones(batch,frames,groups,dtype=torch.bool),
      'frame_timestamps':times,'reference_timestamps':times-10000,'sequence_ids':torch.arange(batch)[:,None].expand(batch,frames),
      'calibration_valid':torch.ones(batch,frames,dtype=torch.bool),'history_valid':torch.ones(batch,frames,dtype=torch.bool)}
 return r,v,ctx

def assertclose(a,b):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

def run():
 torch.set_num_threads(1);r,v,ctx=fixture();model=PointLLMTokenAdapter().eval()
 out=model(r,v,ctx)
 assert out['corrected_tokens'].shape==(1,97,384)
 assert torch.equal(out['sequence_corrected_tokens'],r),'Identity residual init changed native tokens'
 assert (out['temporal_features'][...,-1]==1).all()
 assert out['log_geometric_prior'].min()<-.001
 # Nonzero residual probe makes invariance tests sensitive to the full correction path.
 with torch.no_grad():model.output_residual.weight.normal_(std=.002)
 full=model(r,v,ctx)
 prefix={k:x[:,:2].clone()for k,x in ctx.items()};prefix_out=model(r[:,:2],v[:,:2],prefix)
 assertclose(full['sequence_corrected_tokens'][:,1],prefix_out['corrected_tokens'])
 changed={k:x.clone()for k,x in ctx.items()};changed['frame_timestamps'][:,-1]+=10000000;changed['reference_timestamps'][:,-1]+=10000000
 changed['radar_centers'][:,-1]+=999;changed['reference_centers'][:,-1]-=999
 rr=r.clone();vv=v.clone();rr[:,-1]*=100;vv[:,-1]*=-100
 future=model(rr,vv,changed)
 assertclose(future['sequence_corrected_tokens'][:,:2],full['sequence_corrected_tokens'][:,:2])
 # Timestamp-unit invariance (seconds versus original ticks) must hold.
 seconds={k:x.clone()for k,x in ctx.items()};seconds['frame_timestamps']/=1e6;seconds['reference_timestamps']/=1e6
 seconds_out=model(r,v,seconds);assertclose(seconds_out['corrected_tokens'],full['corrected_tokens'])
 assertclose(seconds_out['temporal_features'],full['temporal_features'])
 assert (full['temporal_history_count']==2).all()
 r7,v7,c7=fixture(frames=7);seven=model(r7,v7,c7)
 assert (seven['temporal_history_count']==5).all()
 assertclose(seven['temporal_features'][...,5],torch.full((1,96),3.))
 mixed={k:x.clone()for k,x in ctx.items()};mixed['sequence_ids'][:,1]=9
 try:model(r,v,mixed)
 except ValueError:pass
 else:raise AssertionError('Mixed-recording window accepted')
 # One batch permutation cannot expose another recording as history.
 rb,vb,cb=fixture(batch=2);both=model(rb,vb,cb);idx=torch.tensor([1,0]);permuted=model(rb[idx],vb[idx],{k:x[idx]for k,x in cb.items()})
 assertclose(both['corrected_tokens'][idx],permuted['corrected_tokens'])
 missing={k:x.clone()for k,x in ctx.items()};missing['reference_mask'].zero_();no_ref=model(r,v,missing)
 assert torch.isfinite(no_ref['corrected_tokens']).all()
 assert (no_ref['utility_weights'][...,0]==1).all() and (no_ref['utility_weights'][...,1]==0).all()
 moved={k:x.clone()for k,x in ctx.items()};moved['reference_centers'][...,0]+=3.
 assert not torch.allclose(model(r,v,moved)['corrected_tokens'],full['corrected_tokens'])
 try:model(r,v,{**ctx,'category_labels':torch.tensor([1])})
 except ValueError:pass
 else:raise AssertionError('Ground truth entered forward context')
 try:model(r,v,{**ctx,'reference_timestamps':ctx['frame_timestamps']+1})
 except ValueError:pass
 else:raise AssertionError('Future reference accepted')
 # Fixed utility biases prove no candidate-count attenuation of the gate.
 with torch.no_grad():model.utility[-1].weight.zero_();model.utility[-1].bias[:]=torch.tensor([3.,-3.])
 probe=model(r,v,ctx);assert (probe['utility_weights'][...,0]>.997).all()
 # A frozen causal decoder must backpropagate only through inserted adapter inputs.
 model=PointLLMTokenAdapter().eval()
 with torch.no_grad():model.output_residual.weight.normal_(std=.002)
 decoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(384,6,768,dropout=0.,batch_first=True),1).eval()
 vocab=nn.Linear(384,29);decoder.requires_grad_(False);vocab.requires_grad_(False)
 token=model(r,v,ctx)['corrected_tokens'];word=torch.randn(1,2,384)
 x=torch.cat((token,word),1);mask=torch.triu(torch.full((99,99),float('-inf')),diagonal=1)
 logits=vocab(decoder(x,mask=mask));loss=F.cross_entropy(logits[:,-2:].reshape(-1,29),torch.tensor([3,7]));loss.backward()
 names=['discrepancy','utility','experts','expert_router','output_residual']
 norms={name:sum(float(p.grad.norm())for key,p in model.named_parameters()if key.startswith(name) and p.grad is not None)for name in names}
 assert all(value>1e-10 for value in norms.values()),norms
 assert all(p.grad is None for p in decoder.parameters())
 assert all(p.grad is None for p in model.category_head.parameters()),'Aux classifier was part of LM path'
 # RGB is a separate interface: patch pixels and calibrated projection validity.
 cfg=AdapterConfig(reference_kind='rgb_patch',reference_has_cls=False,reference_dim=64,spatial_sigma=32.)
 rgb=PointLLMTokenAdapter(cfg);uv=torch.rand(1,3,96,2)*224
 rgbctx={k:x for k,x in ctx.items()if k!='reference_centers'}
 rgbctx.update(projected_uv=uv,reference_uv=uv+5,projection_valid=torch.ones(1,3,96,dtype=torch.bool))
 rgbref=torch.randn(1,3,96,64);rgb_out=rgb(r,rgbref,rgbctx)
 assert rgb_out['corrected_tokens'].shape==(1,97,384)
 rgb_other=PointLLMTokenAdapter(AdapterConfig(reference_kind='rgb_patch',reference_has_cls=False,reference_dim=64,spatial_sigma=64.));rgb_other.load_state_dict(rgb.state_dict())
 assertclose(rgb_out['temporal_features'],rgb_other(r,rgbref,rgbctx)['temporal_features'])
 result={'status':'PASS','actual_shape':[1,3,97,384],'identity_init':True,'future_and_cross_batch_invariance':True,'no_reference_safe':True,'ground_truth_and_future_reference_rejected':True,'continuous_frozen_causal_fixture_gradient_norms':norms,'rgb_and_lidar_are_distinct':True,'timestamp_seconds_ticks_invariant':True,'history_aggregates_at_most_five_actual_past_frames':True,'rgb_pixel_bandwidth_does_not_rescale_meter_history':True,'num_feature_experts':len(model.experts),'is_performance_experiment':False,'actual7B_backprop':'separate runtime-owned probe, not established by this unit fixture'}
 Path(__file__).with_name('adapter_contract_test_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':run()
