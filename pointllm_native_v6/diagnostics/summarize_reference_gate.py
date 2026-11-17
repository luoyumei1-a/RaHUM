"""Summarize recorded original-reference attention/reliability tensors.

Run only on tensors exported during an actual trained reference-model execution.
Required NPZ arrays: attention[B,Np,Nv], point_reliability[B,Np],
vision_reliability[B,Nv]. Optional selected_groups[B,K,2] and
valid_pairs[B,Np,Nv] identify the actual selected/valid candidate populations.
This tool does not generate model inputs, labels, or performance values.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(a, half_gate=False):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if not a.size:
        return {"count": 0}
    result = {"count": int(a.size), "mean": float(a.mean()), "std_population": float(a.std()),
            "min": float(a.min()), "p05": float(np.quantile(a,.05)), "median": float(np.median(a)),
            "p95": float(np.quantile(a,.95)), "max": float(a.max())}
    if half_gate:
        result["fraction_abs_minus_half_lt_001"] = float(np.mean(np.abs(a-.5)<.01))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recorded-npz", type=Path, required=True)
    p.add_argument("--provenance-json", type=Path, required=True,
                   help="Actual run record with checkpoint_sha256, source_sha256, split, sigma_s and sigma_s_unit")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    meta = json.loads(a.provenance_json.read_text())
    for field in ("checkpoint_sha256","source_sha256","split","sigma_s","sigma_s_unit"):
        if field not in meta:
            raise ValueError("Missing actual-run provenance: "+field)
    if a.gamma <= 0:
        raise ValueError("gamma must be positive")
    with np.load(a.recorded_npz,allow_pickle=False) as f:
        att=f["attention"].astype(np.float64)
        rp=f["point_reliability"].astype(np.float64)
        rv=f["vision_reliability"].astype(np.float64)
        valid=f["valid_pairs"].astype(bool) if "valid_pairs" in f else np.ones_like(att,dtype=bool)
        selected=f["selected_groups"] if "selected_groups" in f else None
    if att.ndim!=3 or rp.shape!=att.shape[:2] or rv.shape!=(att.shape[0],att.shape[2]) or valid.shape!=att.shape:
        raise ValueError("Recorded tensor shape mismatch")
    for name,value in (("attention",att),("point_reliability",rp),("vision_reliability",rv)):
        if not np.isfinite(value).all() or value.min()<0 or value.max()>1:
            raise ValueError(name+" must be finite probabilities")
    if not np.allclose(att.sum(-1),1.,atol=1e-5):
        raise ValueError("Attention must be row-normalized over actual reference candidates")
    logit=a.gamma*att*(rp[:,:,None]-rv[:,None,:])
    omega=1/(1+np.exp(-logit))
    entropy=-(att*np.log(np.maximum(att,1e-300))).sum(-1)
    result={"scope":"Observed original-reference tensor diagnostics; no hallucination benchmark is inferred",
            "run_provenance":meta,"gamma":a.gamma,"N_v":int(att.shape[-1]),
            "all_valid_pairs_omega":summarize(omega[valid], half_gate=True),
            "all_pairs_attention_entropy_nats":summarize(entropy),
            "all_pairs_mean_absolute_half_deviation":float(np.mean(np.abs(omega-.5))),
            "all_pairs_analytic_upper_bound_gamma_over_4Nv":a.gamma/(4*att.shape[-1]),
            "analytic_bound_is_not_a_selected_topK_bound":True}
    if selected is not None:
        if selected.ndim!=3 or selected.shape[0]!=att.shape[0] or selected.shape[-1]!=2 or not np.issubdtype(selected.dtype,np.integer):
            raise ValueError("Invalid actual selected_groups array")
        if (selected<0).any() or (selected[:,:,0]>=att.shape[1]).any() or (selected[:,:,1]>=att.shape[2]).any():
            raise ValueError("Selected pair index out of bounds")
        b=np.arange(att.shape[0])[:,None]
        selected_valid=valid[b,selected[:,:,0],selected[:,:,1]]
        result["actual_selected_pairs_omega"]=summarize(omega[b,selected[:,:,0],selected[:,:,1]][selected_valid], half_gate=True)
        result["actual_selected_pairs_attention"]=summarize(att[b,selected[:,:,0],selected[:,:,1]][selected_valid])
        result["K"]=int(selected.shape[1])
    for path,name in ((a.recorded_npz,"recorded_npz"),(a.provenance_json,"provenance_json")):
        result[name+"_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError("Refusing to overwrite diagnostics")
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
