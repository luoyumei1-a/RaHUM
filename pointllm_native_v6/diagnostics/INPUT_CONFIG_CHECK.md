# PointLLM input configuration check

This check uses the supplied local model/config artifacts and frozen runner code. It does not download a different model or perform a new language evaluation.

## What is preserved and what changes

| Item | Pretrained/configured PointLLM setting | Recorded radar execution |
|---|---|---|
| Checkpoint identity | `RunsenXu/PointLLM_7B_v1.2`, revision `37d8c15aaff8e7e05f04729dcb6960d5758e9f86` | Same pinned learned checkpoint tensors initialize encoder/projector and full language decoder |
| Point input convention | Config names `PointTransformer_8192point_2layer`; `use_color=true` | Radar-derived XYZ with zero-filled three color channels |
| Number of raw measured returns | Object point-cloud pretraining convention | 266, 252, 243 finite radar returns in the three source frames |
| Points actually encoded | Named 8192-point pretrained configuration | **96 sampled distinct radar rows per frame**, not all 266/252/243 rows |
| Point grouping | Local copied default YAML: 512 groups, 32 neighbors/group | 96 groups, 32 neighbors/group |
| Prefix length | 512 local groups + one CLS = 513 tokens | 96 local groups + one CLS = 97 tokens |
| Point encoder channels | Color-enabled checkpoint expects six channels | Six channels, with RGB explicitly absent/zero |
| Physical attributes | Not a substitute for measured RGB | Two physical radar channels are separately pooled by the actual group indices |
| Language model | All official decoder layers | Complete decoder retained, frozen; NF4 storage/BF16 computation |

The current adapter's `local_memory K` is a separate number of candidate RGB reference patches. Extending that `K` to 128 does **not** turn the 97-token PointLLM prefix into 129 or 513 tokens, nor does it restore the original 8192-point input distribution.

## Exact local evidence

All following paths are relative to the extracted package root.

1. `backbone/evidence/config.json:23` names `PointTransformer_8192point_2layer`; line 29 sets `use_color: true`.
2. `reference_full_adaptation/CHRLMM/model/pointbert/PointTransformer_8192point_2layer.yaml:8-17` records 32 neighbors, 512 groups, and `npoints: 8192`. The YAML's base `point_dims: 3` should not be mistaken for the color-enabled checkpoint's executed channel count; the actual radar runner sets six channels explicitly.
3. `backbone/run02_source_frozen/runtime.py:26-27` pins the official model and revision.
4. `backbone/run02_source_frozen/runtime.py:71-84` sets groups=96, 32 neighbors, `point_dims=6`, and projector dimensions `384 -> 1024 -> 2048 -> 4096`.
5. `backbone/run02_source_frozen/runtime.py:85-105` selects official encoder/projector tensors, loads them strictly, and loads all language layers, rejecting missing or unaccounted tensors.
6. `backbone/run02_source_frozen/runtime.py:97-111` freezes PointBERT and the language decoder, exposes the projector as trainable, and records quantization and the altered group budget. Thus “unchanged official PointLLM evaluation protocol” would be false; “official checkpoint initialization with a radar-adapted front end” is accurate.
7. `backbone/run02_source_frozen/runtime.py:146-169` is the actual point preprocessing: random sampling of 96 real rows, centering/radius normalization, appending three zero color channels, running PointBERT, and retaining the original source row IDs/group neighbors. It does not colorize radar using the RGB camera.
8. `backbone/nuscenes_real_rgb_probe/pointbert_actual/actual_pointbert_tokens_receipt.json` independently records each frame's original rows (266/252/243), `model_rows=96`, `distinct_source_rows_sampled=96`, and no replacement sampling. Its encoder receipt records 96 groups and 97 tokens.
9. `backbone/vendor_pointbert/point_encoder.py:185` comments on the original `G+1=513` token convention. `backbone/vendor_pointbert/UPSTREAM_ATTRIBUTION.txt:1-2` identifies the copied official source commit `cb72f4e6ab625ddab92f84931127e12bc326b4be`.

## Wording suitable for the manuscript

> We initialize the point encoder, projector, and language decoder from PointLLM-7B v1.2. For radar input, we sample 96 measured XYZ points per frame, append zero-valued RGB channels, and form 96 local groups of 32 neighbors plus a global token. The resulting 97-token prefix is a radar-adapted input configuration rather than the original 8192-point, 513-token convention. The complete 7B decoder remains frozen with NF4 weight storage and BF16 computation; adaptation optimizes the projector and radar correction module.

This wording describes a genuine model adaptation. It does not establish that the original pretrained projector transfers well to sparse radar, that zeros represent observed color, or that two recorded optimization updates establish semantic reliability. Those questions require held-out evaluation and independently supported scene claims.

## Additional physical-channel precision

The point-token receipt explicitly states that the second nuScenes physical channel is raw RCS, **not log amplitude**. The generic runner's input-validation message says `[vr,log-amplitude]`, but the cached source receipt documents the actual semantics. Reporting should use measured radial velocity/RCS for this input window rather than silently relabeling RCS as log amplitude.

The precise source RCS unit is **dBsm**, not amplitude; see `PHYSICAL_UNITS.md`. The legacy field name `source_rcs_native_unconverted` in frozen receipts is retained unchanged because no numerical conversion was applied.
