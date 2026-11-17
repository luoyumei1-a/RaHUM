# RaHUM

Radar Hallucination Understanding and Mitigation.

This repository provides real radar examples, native point-token correction for
PointLLM-7B, source/data provenance, and reproducible internal diagnostics.
Measurement artifacts, independently annotated returns, and generated semantic
assertions have separate evaluation units.

## Native PointLLM implementation

The [`pointllm_native_v6/`](pointllm_native_v6/README.md) directory contains the
versioned native implementation: measured radar → official PointBERT → corrected
97 point tokens → original projector → complete frozen PointLLM-7B decoder.
Its recorded execution used two TRAIN language-objective updates and a matched
projector-only baseline. This engineering example is not a held-out
hallucination-reduction benchmark. Original raw generated responses and receipts
are retained without answer rewriting.

The native diagnostics restore that recorded adapter against its exact frozen source
and real feature-cache hashes. The input contains three frames, 96 radar groups
per frame and 256 RGB patch tokens. Of 288 radar groups, 42 have valid RGB
projection support. The fixed checkpoint uses local memory K=8 and spatial
bandwidth 49.2247619629 native camera pixels. Its learned two-way utility softmax
does not multiply logits by normalized candidate attention. The same weights are
also inspected at local K=1,2,4,8,12,16,24,32; this is inference sensitivity,
not a retrained-model performance comparison.

See [`pointllm_native_v6/diagnostics/`](pointllm_native_v6/diagnostics/README.md)
for actual CSVs, receipts and reproduction scripts. Adapter-only CPU times
exclude the sensor encoders, projector and 7B decoder. The full official
pretrained weights are downloaded from their publisher, not committed here.
Small adaptation checkpoints are available in the code-and-example bundle or
can be regenerated using the documented training commands.

## Real examples and source corpus

The existing [`examples/real_numeric/`](examples/real_numeric/) files preserve
the exact numeric measurements from six original MAT records; no RGB pairing is
inferred from their names. The native radar–RGB feature example has SHA256
`01f2e9c765fb4efef8f4df8f77e70adfb6bf4c73f4cf5683139d7615520720a0`.
It derives from a real official nuScenes-mini TRAIN window and is separate from
the author-provided acquisition corpus linked below. The authors describe that
corpus as 14 TB; this does not identify an evaluated split or annotation count.

| Data Type | Google Links | Baidu Links |
|---|---|---|
| mmWave raw IQ, mmWave Point Cloud, Kinect | [Google Drive](https://drive.google.com/drive/folders/1Hk5WnxPbsS_3Ilbs9x-ut50CIZU6m9Ns?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1lbIBcwvK1LpAyI_cy62UUA?pwd=o3g5) |

## HiRAC predecessor summaries and figure revision

RaHUM evolved from HiRAC. The authors confirmed the prior HiRAC experiment
summaries as their earlier results. `historical_summary/` retains their exact
weather sensing, response-incidence, and ablation summaries, with source hashes
and metric units. These predecessor summaries are not assigned to the new native
adapter checkpoint or presented as new runs of its revised utility gate.

The v9 MATLAB source retains the original five-class HiRAC confusion matrix and
visual style in a separately attributed panel. The other v9 plot extensions use
actual fixed-checkpoint diagnostics: three frame-utility curves and descriptive
entropy quantiles across eight local-memory budgets. No values were jittered or
interpolated into additional training epochs. See `figures/` and
`pointllm_native_v6/diagnostics/actual_frame_memory_sweep/` for their inputs.

`reference_full_adaptation/` preserves the different historical CHR-LMM/HiRAC
source design and original file hashes. It remains separate from the executed
native PointLLM interface.

## Attribution

Retain all PointLLM, PointBERT, DINOv2, nuScenes and Radar Ghost licence and source
notices. Public datasets and pretrained checkpoints retain their own terms.
No blanket repository licence overrides those restrictions.

