# Real-cache training and generation contract

`train_pointllm.py` and `eval_pointllm.py` execute the official PointLLM 7B v1.2 language decoder. They use frozen measured-sensor encoder caches to avoid repeating encoder work. A binary perception head does not produce the response text.

## Backbone and execution

- Official checkpoint: `RunsenXu/PointLLM_7B_v1.2`, revision `37d8c15aaff8e7e05f04729dcb6960d5758e9f86`.
- PointBERT and the 384→1024→2048→4096 projector load every tensor strictly. The language body loads the actual official weights as frozen NF4/BF16, with actual language cross-entropy backpropagating through it into the trainable sensory prefix and projector.
- The execution budget is adapted to 96 groups, or 97 tokens including CLS, versus the official default of 512 groups. All compared variants must use the same sensory cache and 97-token budget.
- Input measured-point row budget is separate from group/token budget. The completed engineering caches sampled 96 measured rows per frame. Repeated rows in a sparse scan are not additional independent measurements. The cache receipt records the original rows, distinct source rows, FPS/KNN groups, normalization and random seed.
- Radar XYZ enters PointBERT with three zero missing-color channels. Radar velocity/RCS are dataset-declared physical attributes in the adapter, not invented RGB values. The attribute schema and physical coordinate/time units remain dataset-specific and are recorded in the encoder/input receipts.
- CPU bitsandbytes inference packing is not a training path. The runtime rejects packed weights and uses gradient-enabled canonical NF4, a frozen language body, non-reentrant checkpointing, and `use_cache=False` during language-objective updates.
- A private, provenance- and fingerprint-verified canonical NF4 language cache is optional. It is not included in a source release, and the runner can load directly from the official checkpoint without it.

## Versioned sensory manifest

A schema-version-1 JSON object contains:

| Field | Meaning |
|---|---|
| `frozen_before_training` | Must be `true`; the scene/source selection is fixed independently of generated outputs. |
| `measurement_data_kind` | Must be `real`. |
| `point_token_budget` | Must be `97` for this matched-budget implementation. |
| `class_vocabulary` | Source annotation ontology used by auxiliary heads, not a claim that all classes occur in the selected split. |
| `adapter_config` | Full `AdapterConfig` dictionary, including reference kind, dimensions, geometry and history units. |
| `records` | All fixed case records, including unavailable input cases. |

Each record specifies `case_id`, `scene_id`, `split` (`train`, `validation`, `test`), `feature_cache`, `feature_cache_sha256`, `input_receipt`, `input_receipt_sha256`, and the generic `prompt`. Relative paths resolve against the manifest directory. Available records also bind `source_sample_tokens`; optional `evaluation_timestamp_us` binds the independently declared query time. Scene IDs and identical encoder caches may not cross splits. An unavailable input uses `input_available=false` and remains in generation output with an explicit execution error; no points, image or future sensor sample are manufactured.

The NPZ contains only numeric measured-feature arrays:

| NPZ key | Shape / contract |
|---|---|
| `radar_tokens` | Float `[1,T,97,384]`, actual frozen PointBERT tokens. |
| `reference_tokens` | Float `[1,T,M,384]`, actual source-declared reference tokens; LiDAR CLS and RGB patches have distinct adapter modes. |
| `context_radar_centers` | Float `[1,T,96,3]`, measured-frame/shared-world centers with documented meters. |
| `context_reference_centers` | Float `[1,T,M,3]` for LiDAR geometry; RGB mode uses its separately documented geometry placeholders and actual patch UV. |
| `context_radar_attributes` | Float `[1,T,96,2]`, pooled with the actual PointBERT group indices. |
| `context_radar_mask`, `context_reference_mask` | Boolean validity masks. |
| `context_frame_timestamps`, `context_reference_timestamps` | Documented causal decision/reference times; do not cast large epoch timestamps to float32 before differencing. |
| `context_sequence_ids` | Numeric recording IDs within the fixed window. |
| `context_calibration_valid`, `context_history_valid` | Boolean validity indicators, not guessed calibration. |
| `context_projected_uv`, `context_reference_uv`, `context_projection_valid` | RGB mode: actual native-image projected point and patch coordinates and projection validity. |

No labels, class IDs, annotation boxes, text captions or instance IDs belong in the feature NPZ. The loader whitelists context keys, rejects nonfinite/text/object arrays, verifies both artifact hashes, checks the input receipt's whole-cache SHA, and binds the source window sample IDs.

## Independent target sidecar

Training records additionally name `target_sidecar` and `target_sidecar_sha256`. This separate JSON supplies `response`, `independent_annotation_source`, `annotation_scope` and evidence references. Where available, `sample_token`, `evaluation_timestamp_us`, measured-window SHA and time-contract SHA must agree with the sensory record/receipt. A partial positive scene response is not an exhaustive caption or a camera visibility label.

Auxiliary category supervision is only enabled with `category_verified=true`, an independently supported `category_id`, and explicit masks. A whole-scene multi-object category is never copied to all radar or reference groups. Support labels need their own independent annotation source. The completed nuScenes RGB engineering run used only a GT-supported partial language response; all auxiliary losses were disabled.

`eval_pointllm.py` never calls `dataset.targets` and never reads target-sidecar content. Generated language is the unedited autoregressive decoder output with token IDs retained. Generated claims are not automatically assigned truth values by the sensory model.

## Matched variants and finite budgets

`plain_projector` trains only the official projector with the same language CE. It does not run the reference adapter or reference/category auxiliary objectives. `rahumpointllm` trains the adapter and optionally the same projector with actual language CE. The runtime also supports `pretrained_native` at evaluation as the unadapted official model under the same token budget.

Training requires an explicit positive `--max-steps`; there is no implicit 50-epoch run. The saved checkpoint is the latest fixed-budget update, not chosen by test results. One atomic checkpoint includes model, optimizer, completed-step metadata, random states and deterministic case schedule. Inference-only exports omit the optimizer and are rejected for optimizer resume. Resume also rejects changed source code, adapter configuration, variant, projector freeze status, dataset, protocol, seed, optimizer settings or invalid completed-step budgets before creating an output directory. All runs, including resume, require a fresh output directory and preserve prior receipts/checkpoints. A new run never silently claims earlier training stages were completed. The recorded guard checks were executed, but an uninterrupted-versus-resumed full 7B numerical trajectory was not separately benchmarked.

Generation retains every selected split record, including execution failures and empty responses. It records model/checkpoint SHA, input/source-case SHA, prompt, split, scene, seed, fixed token budget, protocol SHA, raw response and generated token IDs. An optional `--compare-raw` calls `evaluation.claim_metrics.require_matched_comparison`, which rejects different cases, seeds, prompts, source inputs, splits or generation protocols. Independent supported/contradicted/unverifiable annotations, coverage and response denominators are required for claim scoring.

## Commands

Download the exact official checkpoint into `CHECKPOINT_DIR` under its original model license. CPU dependencies are listed in `requirements-cpu.txt`; install the CPU PyTorch build from the official CPU index rather than a CUDA stack. `LANGUAGE_CACHE_DIR` below is optional and private; omit that argument to load/requantize the official checkpoint directly.

```bash
python train_pointllm.py \
  --checkpoint-dir CHECKPOINT_DIR --language-cache-dir LANGUAGE_CACHE_DIR \
  --dataset DATASET.json --protocol PROTOCOL.json --output-dir RUN_RAHUM \
  --variant rahumpointllm --max-steps 2 --seed 7588 \
  --learning-rate 0.0001 --aux-weight 0 --threads 2 --export-inference

python train_pointllm.py \
  --checkpoint-dir CHECKPOINT_DIR --language-cache-dir LANGUAGE_CACHE_DIR \
  --dataset DATASET.json --protocol PROTOCOL.json --output-dir RUN_PLAIN \
  --variant plain_projector --max-steps 2 --seed 7588 \
  --learning-rate 0.0001 --aux-weight 0 --threads 2 --export-inference

python eval_pointllm.py \
  --checkpoint-dir CHECKPOINT_DIR --language-cache-dir LANGUAGE_CACHE_DIR \
  --adaptation-checkpoint RUN_PLAIN/inference.pt --variant plain_projector \
  --dataset DATASET.json --protocol PROTOCOL.json --split test \
  --output-dir GEN_PLAIN --seed 7588 --max-new-tokens 32 --threads 2

python eval_pointllm.py \
  --checkpoint-dir CHECKPOINT_DIR --language-cache-dir LANGUAGE_CACHE_DIR \
  --adaptation-checkpoint RUN_RAHUM/inference.pt --variant rahumpointllm \
  --dataset DATASET.json --protocol PROTOCOL.json --split test \
  --output-dir GEN_RAHUM --seed 7588 --max-new-tokens 32 --threads 2 \
  --compare-raw GEN_PLAIN/raw_generations.jsonl
```

Use `--split train` only for an explicitly marked TRAIN engineering generation such as the completed connectivity probes. Neither two updates nor a loss decrease establish held-out hallucination mitigation.

The tested environment is CPU PyTorch 2.9.0, Transformers 4.57.6 and bitsandbytes 0.50.2. A GPU path is exposed via `--device cuda:0`, but no GPU execution or full-training performance is claimed by the CPU engineering receipts. Model/data source licenses remain separate; do not include official weights, raw nuScenes samples or derived measurement caches in a blanket-licensed public source archive.
