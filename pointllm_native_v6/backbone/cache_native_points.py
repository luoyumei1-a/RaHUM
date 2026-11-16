"""Strict official PointBERT-only cache; never instantiates the 7B language model."""
from pathlib import Path
import argparse,json,gc
import torch
from torch import nn
from easydict import EasyDict
from runtime import PointTransformer,CapturingGroup,PointLLMBackboneRuntime,parameter_fingerprint,file_sha,MODEL_REPO,MODEL_REVISION
from nuscenes_inputs import cache_actual_pointbert

class FrozenPointEncoderOnly:
    encode_measured_points=PointLLMBackboneRuntime.encode_measured_points
    def __init__(self,checkpoint_dir,groups=96,threads=2):
        torch.set_num_threads(threads);self.device=torch.device('cpu');self.groups=groups;self.point_tokens=groups+1
        cfg=EasyDict(trans_dim=384,depth=12,drop_path_rate=.1,cls_dim=40,num_heads=6,group_size=32,num_group=groups,encoder_dims=256,point_dims=6,projection_hidden_layer=2,projection_hidden_dim=[1024,2048],use_max_pool=False)
        self.point_encoder=PointTransformer(cfg,use_max_pool=False);self.point_encoder.group_divider=CapturingGroup(groups,32)
        directory=Path(checkpoint_dir);mapping=json.loads((directory/'pytorch_model.bin.index.json').read_text())['weight_map']
        selected={k:v for k,v in mapping.items()if k.startswith('model.point_backbone.')};weights={}
        for name in sorted(set(selected.values())):
            shard=torch.load(directory/name,map_location='cpu',mmap=True,weights_only=True)
            for k,v in selected.items():
                if v==name:weights[k.removeprefix('model.point_backbone.')]=shard[k]
            del shard
        self.point_encoder.load_state_dict(weights,strict=True);del weights;gc.collect()
        self.point_encoder.requires_grad_(False).eval()
        self.receipt={'official_repo':MODEL_REPO,'official_revision':MODEL_REVISION,'strict_load':True,'native_groups':groups,'native_tokens':groups+1,'point_encoder_parameter_sha256':parameter_fingerprint(self.point_encoder),'checkpoint_index_sha256':file_sha(directory/'pytorch_model.bin.index.json'),'point_encoder_parameter_count':sum(p.numel()for p in self.point_encoder.parameters()),'all_parameters_frozen':True,'language_model_loaded':False,'attributes':'two dataset-declared measured physical channels; sameactualFPS/KNNpooling; nuScenes secondchannel is rawRCS, not logamplitude','runtime_source_sha256':file_sha(Path(__file__).with_name('runtime.py'))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--raw-window',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    enc=FrozenPointEncoderOnly(a.checkpoint_dir);cache=cache_actual_pointbert(a.raw_window,enc.encode_measured_points,a.output)
    receipt=cache.with_name('actual_pointbert_tokens_receipt.json');r=json.loads(receipt.read_text());r['actual_point_encoder']=enc.receipt;receipt.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'status':'PASS_actual_strictPointBERTcache_no7Bloaded','cache':str(cache),'encoder':enc.receipt},indent=2))
if __name__=='__main__':main()
