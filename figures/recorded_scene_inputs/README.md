# Three recorded scene inputs

These CSVs are compact coordinate views derived from independently paired official nuScenes-mini radar measurements. The box-corner CSVs and JSON files are publisher annotations, not SOTA or RaHUM predictions. No recognition-improvement inference is made from these input/annotation panels.

`CASE_SOURCE_MANIFEST.json` preserves original case selection, tokens, measured-file hashes, causal time offsets and source identities. Its absolute `measurement_npz` entries identify the original build location; the MATLAB renderer loads only the relative CSVs. Raw PCD/JPEG sensor files and the complete nuScenes archive are not included. Recover them through the official download tools under the nuScenes terms. The three illustrative scenes are separate from the one three-frame native adapter diagnostic window.
