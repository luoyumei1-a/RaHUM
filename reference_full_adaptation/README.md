# Author-supplied PointLLM reference source

This directory preserves the original CHR-LMM / HiRAC PointLLM source under its original `CHRLMM` package name. `model/CHRLMM.py` inserts corrected point features through the PointLLM projector before language generation. `CHRLMM_hallucination.py` contains the alignment, discrepancy, attention-conditioned reliability, top-K grouping, decoder and auxiliary expert heads. Original identifiers and code bytes are retained for source and checkpoint compatibility.

The original model, training and PointBERT files came from the nested `HiRAC_code.zip`; dataset readers and training launchers came from the enclosing author-supplied `CHR-LMM code.zip`. Their exact relative source identities and SHA-256 hashes are recorded in `SOURCE_PROVENANCE.json` at the package root. No full model checkpoint is bundled.

## Configuration identities

`reported_configuration.json` records the supplied paper's 25-frame, 96-point, d=384, K=96, 5-history, 5-expert configuration, with 20 projector/reducer epochs and 30 full-model epochs. `config/chrlmm_config.yaml` is the source archive's unmodified default, including its own d=512 and VR-PCT loader settings. These defaults are not rewritten or silently equated with the reported experiment's checkpoint. The source config's 11 target labels are preserved as config labels, not asserted to be a recovered benchmark annotation manifest.

This is a historical source snapshot, not the active native adapter and not a
record of an executed 20+30-epoch benchmark. Historical reported settings are
preserved as metadata and must not be substituted for actual checkpoint data.
No positive-results table from this snapshot is included in the active release.

## Training source

The reference tree retains the original two-stage PointLLM training entry `CHRLMM/train/train.py`, the reducer-training helper, and the adaptive launcher. Use pre-partitioned real sensor data with explicit calibration, timestamps, supervision and a verified ontology. Training parameters must be matched to the intended reference experiment. The full model requires an appropriate GPU/runtime and official pretrained weights. Original PointLLM/PointBERT dependencies and third-party terms still apply; see `../backbone/THIRD_PARTY_NOTICES.md`.

The old synthetic-data generators, their generated NPY fixtures, constant-outcome evaluation scripts, cached bytecode, nested ZIP archives and all demo outputs are excluded. The code used for the actual CPU prefix-adaptation engineering runs is separately preserved in `../backbone`. Do not merge the two configurations' numerical evidence.
