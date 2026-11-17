"""Real-input sensitivity of local memory size at one fixed native checkpoint.

The checkpoint was trained for two engineering updates with K=8. Other K
values change inference memory only. This is not a retrained-model ablation,
language-generation evaluation, or recognition/hallucination benchmark.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--feature-cache",type=Path,required=True)
    p.add_argument("--adapter-source",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--k-values",nargs="+",type=int,required=True)
    p.add_argument("--timing-repeats",type=int,default=5)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    if len(a.k_values)!=len(set(a.k_values)) or min(a.k_values)<1 or a.timing_repeats<1:
        raise ValueError("Distinct positive local budgets and repeats are required")
    a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads)
    cp=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    if cp.get("variant")!="rahumpointllm":raise ValueError("Not a native adapter checkpoint")
    if sha(a.adapter_source)!=cp["metadata"]["source_sha256"]["adapter.py"]:
        raise ValueError("Frozen source does not match checkpoint")
    spec=importlib.util.spec_from_file_location("swept_native_adapter",a.adapter_source)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    with np.load(a.feature_cache,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    radar=torch.from_numpy(arrays["radar_tokens"]);reference=torch.from_numpy(arrays["reference_tokens"])
    context={k.removeprefix("context_"):torch.from_numpy(v) for k,v in arrays.items() if k.startswith("context_")}
    if any(any(x in k.lower() for x in ("label","category","ground_truth","claim")) for k in context):
        raise ValueError("Label-like metadata must not enter forward")
    rows=[];last_tokens={};raw_timings={}
    nv=int(reference.shape[2]-int(cp["adapter_config"]["reference_has_cls"]))
    if max(a.k_values)>nv:raise ValueError("Budget exceeds actual reference tokens")
    for k in a.k_values:
        cfg=dict(cp["adapter_config"]);cfg["local_memory"]=k
        model=mod.PointLLMTokenAdapter(mod.AdapterConfig(**cfg)).eval()
        model.load_state_dict(cp["adapter_state_dict"],strict=True)
        with torch.inference_mode():
            out=model(radar,reference,context)  # one warm-up; excluded from times
            times=[]
            for repeat in range(a.timing_repeats):
                start=time.perf_counter();out=model(radar,reference,context)
                times.append((time.perf_counter()-start)*1000.)
        weights=[];entropies=[];max_attention=[];active_counts=[]
        for t,frame in enumerate(out["sequence_outputs"]):
            active=context["radar_mask"][:,t].bool() & frame["reference_available"]
            utility=frame["utility_weights"]
            att=frame["attention"]
            if not torch.allclose(utility.sum(-1),torch.ones_like(active,dtype=utility.dtype),atol=1e-6):
                raise ValueError("Invalid utility normalization")
            if not torch.allclose(att.sum(-1),frame["reference_available"].to(att.dtype),atol=1e-6):
                raise ValueError("Invalid masked attention normalization")
            weights.extend(utility[...,0][active].tolist())
            entropies.extend((-(att*att.clamp_min(1e-30).log()).sum(-1))[active].tolist())
            max_attention.extend(att.amax(-1)[active].tolist())
            active_counts.append(int(active.sum()))
        corrected=out["sequence_corrected_tokens"]
        if not torch.isfinite(corrected).all():raise ValueError("Nonfinite corrected tokens")
        weights=np.asarray(weights)
        row={"local_K":k,"checkpoint_trained_local_K":cp["adapter_config"]["local_memory"],
             "available_radar_groups":int(weights.size),"native_radar_utility_mean":float(weights.mean()),
             "native_radar_utility_min":float(weights.min()),"native_radar_utility_max":float(weights.max()),
             "attention_entropy_nats_mean":float(np.mean(entropies)),
             "attention_max_mean":float(np.mean(max_attention)),
             "corrected_sequence_relative_L2_change_from_native":float((corrected-radar).norm()/radar.norm()),
             "forward_ms_median":float(np.median(times)),"forward_ms_min":float(min(times)),
             "forward_ms_max":float(max(times))}
        rows.append(row);last_tokens[k]=corrected.numpy();raw_timings[str(k)]=times
    trained_k=cp["adapter_config"]["local_memory"]
    if trained_k in last_tokens:
        baseline=last_tokens[trained_k]
        denom=np.linalg.norm(baseline)
        for row in rows:
            row["relative_L2_change_from_trained_K_output"]=float(np.linalg.norm(last_tokens[row["local_K"]]-baseline)/denom)
    with (a.output_dir/"local_memory_sweep.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt={"status":"completed_real_input_fixed_checkpoint_local_memory_sensitivity",
             "scope":__doc__.strip(),"checkpoint_sha256":sha(a.checkpoint),"feature_cache_sha256":sha(a.feature_cache),
             "adapter_source_sha256":sha(a.adapter_source),"diagnostic_script_sha256":sha(__file__),
             "checkpoint_completed_steps":cp["metadata"].get("completed_steps"),
             "checkpoint_config":cp["adapter_config"],"observed_reference_tokens":nv,
             "timing": {"device":"cpu","torch_threads":a.threads,"warmup_forwards_per_K":1,
                        "recorded_forwards_per_K":a.timing_repeats,"measured_component":"adapter-only forward, complete three-frame cached-feature window; excludes encoders, projector and 7B decoder",
                        "recorded_forward_ms_by_K":raw_timings},
             "torch":str(torch.__version__),"rows":rows,"csv_sha256":sha(a.output_dir/"local_memory_sweep.csv")}
    (a.output_dir/"memory_sweep_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
