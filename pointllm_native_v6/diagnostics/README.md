# Real-input reliability diagnostics

These scripts inspect the recorded **native PointLLM adapter** and preserve its
checkpoint and real feature-cache identities. This adapter uses an unscaled
learned two-way utility softmax. It is a different implementation from the older
attention-scaled reliability formula in the full-adaptation reference source.

The supplied checkpoint completed two engineering training updates. The example
contains one three-frame nuScenes-mini TRAIN window. The scripts do not retrain
the model, rerun the language decoder, or measure held-out recognition accuracy
or hallucination reduction. Source sensor observations are real recorded data.

## Reproduce the observed utility distribution

Install the project's CPU dependencies, including PyTorch and NumPy, then run
from the repository root:

```bash
python diagnostics/inspect_native_reliability.py \
  --checkpoint backbone/rgb_runner_01/inference.pt \
  --feature-cache backbone/nuscenes_real_rgb_probe/adapter_inputs_v1/actual_radar_rgb_adapter_inputs.npz \
  --adapter-source backbone/rgb_runner_01_source_frozen/adapter.py \
  --output-dir diagnostic_runs/reliability_01
```

The script requires the frozen source SHA recorded in checkpoint metadata and
strictly loads every adapter parameter. It checks finite reconstructed tokens,
utility normalization, availability-masked attention normalization, exact zero
reference utility when reference evidence is absent, and exclusion of label-like
keys from the sensory forward context. `actual_native_reliability/` contains the
executed receipt, per-group CSV, and recorded diagnostic tensors. Its additional
histogram and ECDF CSVs are computed from the 42 available radar groups, with a
fixed 0.05 histogram-bin width. The other 246 groups remain in the output and
have a deterministic missing-reference utility, not a learned reliability
observation. Their radar/temporal residual path can still change their features.

## Reproduce the local-memory sensitivity analysis

```bash
python diagnostics/sweep_native_memory.py \
  --checkpoint backbone/rgb_runner_01/inference.pt \
  --feature-cache backbone/nuscenes_real_rgb_probe/adapter_inputs_v1/actual_radar_rgb_adapter_inputs.npz \
  --adapter-source backbone/rgb_runner_01_source_frozen/adapter.py \
  --k-values 1 2 4 8 12 16 24 32 \
  --output-dir diagnostic_runs/memory_01
```

The same fixed checkpoint was trained with local memory K=8. Every other budget
changes only inference memory at the same learned parameters; it is not a
retrained-model ablation. The script records actual output changes, attention
entropy, utility, and CPU forward times. Each budget receives one untimed warm-up
and five timed forwards. Timing covers the complete three-frame cached-feature
adapter only and excludes the sensor encoders, projector, and PointLLM decoder.
Timing variation is machine-dependent, not uncertainty across dataset samples.

## Original-reference tensors

`summarize_reference_gate.py` is an additional recorder-analysis interface for
real attention and reliability tensors from an independently executed original
reference model. It requires a checkpoint SHA, source SHA, split and actual
spatial-bandwidth provenance. No such original-reference distribution is
substituted with the native adapter observations here.
