# RaHUM PointLLM code and real examples — v10

RaHUM provides native PointBERT-token correction before the original
PointLLM projector and complete frozen PointLLM-7B language decoder. This release
contains the real radar–RGB feature example, original recorded two-update
training and generation receipts, strict-restoration checks, and real-input
reliability and local-memory diagnostics.

## Experiment configurations and source identity

The author-provided sensing, response-error and component-ablation summaries are retained under `historical_summary/` with their original source identity. Their exact arrays are not jittered, completed, or reassigned to a different checkpoint. They describe an earlier author configuration; they are distinct from the native PointLLM revision and its recorded executions in `backbone/`. The supplied native records do not independently reproduce those summary tables. Preserved source filenames and source metadata may retain earlier internal method names.

The native record is one three-frame nuScenes-mini TRAIN feature window, two actual optimization updates per matched variant, strict checkpoint restoration and unmodified generated responses. The v10 local-memory diagnostic reuses one fixed checkpoint at K=8,16,24,32,48,64,96,128; it performs no retraining, additional language decoding or held-out claim evaluation. There is no record of 50-epoch full-model fine-tuning or a 6,800-sequence benchmark in this package.

The author-provided 14 TB corpus links and the compact official nuScenes example are different data sources. Dataset access links do not establish the number, split or labels of cases used by any particular experiment.

## Release contents and evidence

| Directory | Role | Evidence scope |
|---|---|---|
| `backbone/` | Native radar/reference correction and complete PointLLM interfaces | Recorded two-update TRAIN executions, matched projector baseline, small adaptation checkpoints and original generated responses |
| `diagnostics/` | Strict checkpoint and feature restoration; utility/memory inspection | Executed real three-frame cached-feature forward, 42 available RGB-supported radar groups; fixed-checkpoint local K sensitivity |
| `reference_full_adaptation/` | Preserved historical CHR-LMM/HiRAC source | Source provenance and configuration history only; distinct model, no benchmark attribution |
| `dataset_tools/` | Official real-data download and manifest preparation | Source hashes, causal sensor pairing and separated annotation sidecars |

No held-out recognition or generated-claim improvement is inferred from the
TRAIN engineering runs or internal diagnostics. No generated sensor fixture or simulated-performance table is presented as
measured experimental evidence in this release. The whole official 7B
pretrained checkpoint is downloaded from its publisher rather than redistributed.

## Quick integrity check

```bash
python verify_real_example.py
```

The portable real feature cache has SHA256
`01f2e9c765fb4efef8f4df8f77e70adfb6bf4c73f4cf5683139d7615520720a0`.
It contains radar tokens `[1,3,97,384]` and real RGB tokens `[1,3,256,384]`.
The published dataset licence and all source hashes remain explicit.

## Actual reliability diagnostics

`diagnostics/README.md` gives exact commands. The strictly restored native
checkpoint uses 256 reference patches, local memory K=8, and spatial bandwidth
49.2247619629 native camera pixels. This bandwidth is the geometric mean of the
actual patch-grid spacings after encoder resizing/cropping. The learned gate is
a two-way utility softmax without multiplication by normalized attention.

The recorded three-frame input has 288 radar groups; 42 have valid RGB
projection support (13, 15 and 14 per frame). The remaining groups remain present
with radar-only utility, and may still receive a radar/temporal residual. The
utility distribution is computed only over the 42 available groups. The active v10 fixed-checkpoint sensitivity scan uses local
K=8,16,24,32,48,64,96,128 without retraining.
Adapter-only CPU timings exclude sensor encoders, projector and language decoder.

# RaHUM with a native PointLLM backbone

This package implements radar/reference correction **inside PointLLM**:
measured points → official PointBERT → 97 corrected point tokens → original
projector → the complete 7B causal decoder → generated language. Detection
and category heads are optional training supervision, not the final output.

The actual CPU implementation has completed real 7B language-loss updates,
strict checkpoint restoration, and autoregressive generation. It is a
reproducible backbone revision and engineering example, not a new claim of
held-out hallucination reduction. The small included adaptation checkpoints
are explicitly two-update engineering checkpoints.

## Setup and pretrained weights

Use Python 3.12 and a PyTorch 2.9 installation suitable for your hardware.
The CPU execution used BF16 compute with canonical NF4 language weights,
FP32 correction/projector weights, and a frozen pretrained point encoder.

```bash
python -m venv .venv
.venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r backbone/requirements-cpu.txt
.venv/bin/python backbone/download_official_model.py --output-dir checkpoints/PointLLM_7B_v1.2
```

The downloader fetches the complete official PointLLM-7B v1.2 files at
revision `37d8c15aaff8e7e05f04729dcb6960d5758e9f86` and checks published-file
sizes and SHA256 values. The 27 GB source weights and the private 4 GB
canonical language cache are not redistributed here. All learned
PointBERT/projector tensors load strictly. The 96-group setting produces
97 tokens including CLS and is used by every matched model. A budget of
96 sampled measurement rows is not 8,192 independent measured points.

## Real radar and RGB example

`backbone/nuscenes_real_rgb_probe/runner_dataset.json` is a portable view of
one real official mini-training window. Its original pre-training manifest
is retained alongside it. Packaging rebases filesystem paths only; sensor
tokens, target sidecars, prompts, splits, and data hashes remain unchanged.

The frozen feature example combines actual PointBERT radar tokens and
actual DINOv2-small RGB patch tokens. The physical side channels are
ego-motion-compensated radial-velocity projection (m/s) and measured RCS
(dBsm); see `diagnostics/PHYSICAL_UNITS.md`. The camera input and radar inputs are
causally paired, ego-pose compensated, calibrated, and bound to image and
measurement hashes. GT stays in separate target sidecars. The partial
training target `The scene contains a car.` is supported by the publisher
annotations; it is not a rich human scene caption or a test answer.

```bash
.venv/bin/python backbone/train_pointllm.py \
  --checkpoint-dir checkpoints/PointLLM_7B_v1.2 \
  --dataset backbone/nuscenes_real_rgb_probe/runner_dataset.json \
  --protocol backbone/nuscenes_real_rgb_probe/runner_protocol.json \
  --output-dir runs/rahum --variant rahumpointllm \
  --max-steps 2 --seed 7588 --learning-rate 0.0001 --aux-weight 0 \
  --export-inference

.venv/bin/python backbone/train_pointllm.py \
  --checkpoint-dir checkpoints/PointLLM_7B_v1.2 \
  --dataset backbone/nuscenes_real_rgb_probe/runner_dataset.json \
  --protocol backbone/nuscenes_real_rgb_probe/runner_protocol.json \
  --output-dir runs/plain --variant plain_projector \
  --max-steps 2 --seed 7588 --learning-rate 0.0001 --aux-weight 0 \
  --export-inference
```

Use `--device cuda` for a compatible GPU environment. CPU execution was
verified here; this package does not claim a measured GPU speedup. Full
training budgets must be chosen before inspecting held-out outputs. The
trainer saves one atomic checkpoint with optimizer/RNG state and validates
the source/configuration contract for resume, alongside an optional inference-only export. A rebased example manifest
does not silently bypass the original checkpoint's training-resume guards.

## Restore and generate unmodified responses

```bash
.venv/bin/python backbone/eval_pointllm.py \
  --checkpoint-dir checkpoints/PointLLM_7B_v1.2 \
  --adaptation-checkpoint runs/plain/inference.pt \
  --dataset backbone/nuscenes_real_rgb_probe/runner_dataset.json \
  --protocol backbone/nuscenes_real_rgb_probe/runner_protocol.json \
  --output-dir runs/plain_generation --variant plain_projector \
  --split train --seed 7588 --max-new-tokens 32

.venv/bin/python backbone/eval_pointllm.py \
  --checkpoint-dir checkpoints/PointLLM_7B_v1.2 \
  --adaptation-checkpoint runs/rahum/inference.pt \
  --dataset backbone/nuscenes_real_rgb_probe/runner_dataset.json \
  --protocol backbone/nuscenes_real_rgb_probe/runner_protocol.json \
  --output-dir runs/rahum_generation --variant rahumpointllm \
  --split train --seed 7588 --max-new-tokens 32 \
  --compare-raw runs/plain_generation/raw_generations.jsonl \
  --compare-model-id plain_projector
```

Generation never opens target-sidecar contents, never rewrites an answer,
and retains missing-input/error cases in the output ledger. The comparison
guard requires matching cases, splits, prompts, source hashes, token budgets,
seeds and generation protocols. TRAIN output verifies execution; it is not
a held-out metric. For a benchmark, provide independent recording-level
training/validation/test manifests and evaluate the fixed held-out split.

`backbone/evaluation/claim_metrics.py` aggregates independently annotated
claims bound to each raw response hash. It reports coverage and supported,
contradicted and unverifiable counts. Four error types can co-occur, so their
sum is not the probability of any error. Unverifiable claims are not silently
converted to incorrect claims or correct answers.
Failed executions remain recorded and prevent a primary hallucination-rate
calculation until a missing-execution protocol is specified; an empty answer
from a successful generation is retained with its coverage information.

## Rebuild measured sensor caches

The optional `dataset_tools` scripts download the official nuScenes mini,
validate its archive and retain radar, front-camera and metadata files.
Run the downloader, selected extraction, paired manifest builder and causal
pair builder in that order. They write relative to their own directory.

```bash
.venv/bin/python dataset_tools/download_official_mini.py
.venv/bin/python dataset_tools/extract_selected.py
.venv/bin/python dataset_tools/build_paired_manifest.py
.venv/bin/python dataset_tools/build_causal_pairs.py
.venv/bin/python backbone/rgb_probe.py --checkpoint-dir checkpoints/dinov2-small --download-only
```

`nuscenes_inputs.py`, `cache_native_points.py` and `rgb_probe.py` prepare
actual windows and frozen features; see their `--help` and
`backbone/ADAPTER_V2_CONTRACT.md`. The official mini has 10 scenes and
404 keyframes; 394 have preceding radar inputs under the fixed causal
policy. It is not the full nuScenes benchmark or the author-collected 14 TB
corpus. The data licence, reference type and exact annotation timestamp
remain explicit.

## Attribution

Retain `backbone/THIRD_PARTY_NOTICES.md`, the official PointLLM attribution and
model terms, the DINOv2 Apache-2.0 notice, and the nuScenes/Radar Ghost data terms.
No blanket licence here overrides third-party restrictions.

## v10 figure and release verification

`matlab/RaHUM_Figures_v10.m` is the single article-ordered MATLAB source. It exports the native diagnostic plots, the exact author-provided five-class matrix and six separately identified recorded-input/annotation panels. `matlab/ONLINE_EXPORT_RECEIPT.json` binds the actual MATLAB Online PDF/FIG exports to the source SHA256. Input/annotation panels are not model predictions.

Run `python verify_release.py` after extraction to verify frozen source identities, the real feature cache, the matched small checkpoints, the K8–128 CSV and the MATLAB Online export source. `VERSION_MIGRATION_V10.json` records changes without rewriting old run receipts. Older K1–32 diagnostics remain under their original directory names as historical execution records; the active v10 K8–128 results are `diagnostics/actual_k8_128/`.
