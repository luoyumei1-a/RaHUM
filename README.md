# RaHUM

Radar Hallucination Understanding and Mitigation.

This repository provides measured radar examples, an official PointLLM-7B inference adapter, and reproducible support-analysis tools. Measurement artifacts, independently annotated return labels, and generated semantic assertions are evaluated separately.

## Real examples

The small numeric examples preserve the exact radar and skeleton values from six original author-provided MAT records. No RGB pairing is inferred. Source hashes, shapes, field names, and round-trip checks are in `examples/real_numeric/provenance.json`.

```bash
python -m pip install numpy
python examples/real_numeric/verify_csv.py
```

Full source MAT examples are included in the downloadable code-and-example bundle. The linked acquisition corpus contains mmWave raw IQ, point clouds, and Kinect records. The authors report a total corpus size of 14 TB; individual released experiments identify the exact files and annotations they use.

| Data Type | Google Links | Baidu Links |
|---|---|---|
| mmWave raw IQ, mmWave Point Cloud, Kinect | [Google Drive](https://drive.google.com/drive/folders/1Hk5WnxPbsS_3Ilbs9x-ut50CIZU6m9Ns?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1lbIBcwvK1LpAyI_cy62UUA?pwd=o3g5) |

## Official PointLLM execution

See `pointllm/README_CPU_REPRODUCTION.md` for the pinned official checkpoint, CPU NF4 configuration, measured-row preprocessing, and execution commands. The complete official 7B weights are obtained from their publisher, with hashes checked by the supplied downloader. Weights are not included in this repository. Official PointLLM provides 7B and 13B variants; a separate model is not renamed PointLLM-3B.

## Data and attribution

Examples are real recorded numeric measurements. They do not include independent per-return artifact labels or full-scene semantic annotations. Coordinate/channel semantics require the relevant acquisition documentation. Geometric support, return classification, and verifiable semantic claims have separate denominators.

Third-party source, checkpoints, and public datasets retain their respective licenses. See `pointllm/THIRD_PARTY_NOTICES.md` and source-specific notices. No blanket license for this repository overrides those terms.
