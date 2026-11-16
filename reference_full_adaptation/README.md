# Original PointLLM reference source

This source snapshot preserves the author-supplied CHR-LMM / HiRAC implementation under its original `CHRLMM` package name. It contains radar–vision alignment, discrepancy, attention-conditioned reliability, top-K grouping, reconstruction, auxiliary expert heads, the PointLLM feature-insertion path and two-stage training source. Original source bytes and names are retained for source and checkpoint compatibility.

## Separate configuration identities

The paper's reference experiment reports 25-frame input, 96 radar points, d=384, K=96, five history frames, five experts, 20 projector/reducer epochs and 30 full-model epochs. `reported_configuration.json` records those reported settings. The unmodified source `config/chrlmm_config.yaml` includes its own d=512 and VR-PCT loader defaults. Those defaults are not silently declared to match a recovered trained experiment checkpoint.

[`../pointllm_native_v6/`](../pointllm_native_v6/README.md) is a separately identified implementation: native point-token correction and the original projector feed a complete frozen PointLLM-7B decoder. Its recorded real radar–RGB execution comprises two TRAIN updates and unmodified generation with a matched projector baseline. That engineering run is not a new full-training reproduction of the reference paper's reported performance. Do not merge the two configurations' numerical evidence.

## Provenance and use

`SOURCE_PROVENANCE.json` lists exact source hashes. Original model, PointBERT and trainer sources came from the nested author-supplied `HiRAC_code.zip`; data readers and launchers came from the enclosing `CHR-LMM code.zip`. Full 7B weights, trained full-model checkpoints, private manuscripts and reviews are not committed here. Dataset generation fixtures, constant-outcome evaluation scripts and bytecode caches are excluded.

The reference tree preserves original dependency and training interfaces, including `CHRLMM/train/train.py` and `scripts/train_chrlmm_auto.py`. Appropriate official pretrained weights, runtime dependencies, recording-level train/validation/test sensor manifests, calibration and independent supervision are needed for a matched experiment. The exact same small real feature example already published under `pointllm_native_v6` remains available.

Retain upstream attribution and licences. See [`../pointllm_native_v6/backbone/THIRD_PARTY_NOTICES.md`](../pointllm_native_v6/backbone/THIRD_PARTY_NOTICES.md); no blanket licence here overrides PointLLM, PointBERT or dataset terms.
