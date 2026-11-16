"""Inference-compatible stochastic depth; eval mode is exact identity."""
import torch
from torch import nn
class DropPath(nn.Module):
 def __init__(self, drop_prob=0.):
  super().__init__(); self.drop_prob=drop_prob
 def forward(self,x):
  if self.drop_prob==0. or not self.training:return x
  keep=1-self.drop_prob
  mask=torch.empty((x.shape[0],)+(1,)*(x.ndim-1),dtype=x.dtype,device=x.device).bernoulli_(keep)
  return x*mask/keep
