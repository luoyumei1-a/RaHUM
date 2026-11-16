"""Use actual cached sensors/encoders to verify calibration/provenance isolation.
No language model is loaded and no semantic performance metric is calculated.
"""
from pathlib import Path
import json,tempfile,copy,hashlib
from unittest.mock import patch
import numpy as np
from nuscenes_inputs import assemble_adapter_inputs,load_rgb_cache
from rgb_encoder import preprocessing_geometry,project_camera_points


def main():
 base=Path(__file__).resolve().parent/'nuscenes_real_rgb_probe';raw_path=base/'raw_window_manifest.json';raw=json.loads(raw_path.read_text())
 pc=base/'pointbert_actual/actual_pointbert_tokens.npz';rgb=load_rgb_cache(base/'rgb_actual/actual_rgb_tokens.npz')
 read_text=Path.read_text
 def no_gt(path,*a,**kw):
  if path.name=='supervision_only.json' or '/paired/truth/' in str(path):raise AssertionError('GT opened during forward assembly')
  return read_text(path,*a,**kw)
 with patch.object(Path,'read_text',no_gt):arrays,receipt=assemble_adapter_inputs(raw_path,pc,rgb)
 ctx=arrays['physical_context'];errors=[]
 for j,f in enumerate(raw['frames']):
  centers=ctx['radar_centers'][0,j].numpy().astype(np.float64);T=np.asarray(f['T_camera_from_global']);camera=centers@T[:3,:3].T+T[:3,3]
  geometry=preprocessing_geometry(*f['image_size_wh'],camera_intrinsic=f['camera_intrinsic'])
  independent=project_camera_points(camera,f['camera_intrinsic'],geometry)
  valid=independent['projection_valid']&(camera[:,2]>.1)
  assert np.array_equal(valid,ctx['projection_valid'][0,j].numpy())
  if valid.any():errors.append(float(np.max(abs(independent['native_uv'][valid]-ctx['projected_uv'][0,j].numpy()[valid]))))
  assert np.isfinite(ctx['projected_uv'][0,j].numpy()).all()
  assert np.all(ctx['projected_uv'][0,j].numpy()[~valid]==0.)
 def reject(call):
  try:call()
  except ValueError:return
  raise AssertionError('Stale/wrong-window encoder input was accepted')
 with tempfile.TemporaryDirectory()as tmp:
  wrong=Path(tmp)/'wrong_window.json';changed=copy.deepcopy(raw);changed['selection']='different sourceidentity';wrong.write_text(json.dumps(changed))
  reject(lambda:assemble_adapter_inputs(wrong,pc,rgb))
 wrong=copy.deepcopy(rgb);wrong['source_image_metadata'][0]['image_sha256']='0'*64
 reject(lambda:assemble_adapter_inputs(raw_path,pc,wrong))
 wrong=copy.deepcopy(rgb);wrong['reference_tokens']=np.asarray(wrong['reference_tokens']).copy();wrong['reference_tokens'][0,0,0,0]+=.1
 reject(lambda:assemble_adapter_inputs(raw_path,pc,wrong))
 from adapter import PointLLMTokenAdapter,AdapterConfig
 import torch
 torch.set_num_threads(1)
 adapter=PointLLMTokenAdapter(AdapterConfig(reference_kind='rgb_patch',reference_has_cls=False,spatial_sigma=receipt['spatial_sigma_native_pixels'])).eval()
 with torch.no_grad():out=adapter(arrays['radar_tokens'],arrays['reference_tokens'],ctx)
 assert torch.equal(out['corrected_tokens'],arrays['radar_tokens'][:,-1])
 report={'status':'PASS_actual_radar_RGB_encoder_adapter_assembly_not7B','ground_truth_never_opened_by_assembly':True,'wrong_window_wrong_image_changed_feature_cache_rejected':True,
         'native_shape':list(arrays['radar_tokens'].shape),'RGB_shape':list(arrays['reference_tokens'].shape),'corrected_shape':list(out['corrected_tokens'].shape),
         'independent_camera_projection_max_abs_pixels':max(errors),'crop_valid_counts':[int(x)for x in ctx['projection_valid'].sum(-1)[0]],
         'calibrated_local_reference_group_counts':[int(x['reference_available'].sum())for x in out['sequence_outputs']],
         'global_visual_path_present':out['global_visual_reference_present'].tolist(),'identity_initialization':True,
         'official_sample_time_evaluation':True,'all404_cameras_checked_before_GTsampletime':True,'spatial_sigma_native_pixels':receipt['spatial_sigma_native_pixels'],
         'actual7B_RGB_execution':False,'semantic_performance_claim':False,'adapter_sha256':hashlib.sha256((Path(__file__).with_name('adapter.py')).read_bytes()).hexdigest()}
 (base/'actual_input_assembly_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
