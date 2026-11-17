# RaHUM

Radar Hallucination Understanding and Mitigation.

This repository provides measured radar examples, a radar-adapted PointLLM-7B
interface, and reproducible evidence-reconstruction diagnostics. Physical-return
labels, sensing predictions, and generated scene assertions have separate
evaluation units.

## Implementations and evaluation configurations

The [`pointllm_native_v6/`](pointllm_native_v6/README.md) directory contains the
native implementation: measured XYZ → pretrained PointBERT → 97 corrected point
tokens → pretrained projector → complete frozen PointLLM-7B decoder. The radar
front end samples 96 measured rows, appends zero RGB channels, and uses 96 local
groups plus one global token. This adapts the original 8192-point, 513-token input
convention; it does not claim the same input distribution.

The recorded native example uses two language-objective updates on one real
three-frame nuScenes-mini training window. Raw generated responses, frozen
sources, and receipts are retained. This example establishes the executed
interface and supports fixed-checkpoint analysis; it is not a held-out language
hallucination benchmark.

The authors separately supplied local sensing, response-incidence, and component
ablation summaries in [`historical_summary/`](historical_summary/). Their numeric
values are preserved. These sensing/response configurations are separate from
the native checkpoint and its revised utility normalization. Their response
incidence metric counts error types per response, rather than atomic claims.

## Version 10 diagnostics and figures

The fixed native checkpoint is inspected at local reference-memory budgets
K = 8, 16, 24, 32, 48, 64, 96, 128. Every setting retains 97 output tokens and
3,296,719 correction parameters. Of 288 radar groups, 42 have valid RGB
projection support. Inference-memory changes do not retrain the checkpoint.
Adapter-only CPU timing uses two threads, one warm-up, and five recorded repeats;
it excludes encoders, projector, and the decoder.

See [`pointllm_native_v6/diagnostics/`](pointllm_native_v6/diagnostics/README.md)
for exact CSVs, input conventions, physical units, receipts, and rerun commands.
RCS is measured in dBsm; radial velocity is measured in m/s.

[`figures/RaHUM_Figures_v10.m`](figures/RaHUM_Figures_v10.m) is the single
article-ordered plotting source. It was executed in MATLAB Online and exported
11 vector PDFs and 11 editable MATLAB FIG files. Six panels show measured radar
and independent official annotations from three identified scenes; they are
not model predictions. The five-class confusion matrix preserves the separate
author-supplied sensing summary. No values were randomized to change trends.

## Real examples and acquisition resources

The [`examples/real_numeric/`](examples/real_numeric/) directory preserves numeric
measurements from six original MAT records. Pairing is never inferred from
filenames. The native radar–RGB feature cache has SHA256
`01f2e9c765fb4efef8f4df8f77e70adfb6bf4c73f4cf5683139d7615520720a0`.
It comes from a real official nuScenes-mini training window, separately from
the authors' 14 TB radar–Kinect acquisition collection linked below.

| Data Type | Google Links | Baidu Links |
|---|---|---|
| mmWave raw IQ, mmWave Point Cloud, Kinect | [Google Drive](https://drive.google.com/drive/folders/1Hk5WnxPbsS_3Ilbs9x-ut50CIZU6m9Ns?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1lbIBcwvK1LpAyI_cy62UUA?pwd=o3g5) |

## Reproduction and attribution

Run `python verify_release.py` inside the extracted code-and-example bundle to
check frozen source, cache, checkpoint, numeric, and plotting bindings. Full
official pretrained weights are obtained from their publisher and are not
committed here. Small adaptation checkpoints are included in the downloadable
code-and-example bundle.

Retain PointLLM, PointBERT, DINOv2, nuScenes, and Radar Ghost source and license
notices. Source datasets and weights retain their own terms. No repository
license overrides those restrictions, and full public sensor archives are not
redistributed here.
