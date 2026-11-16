# PointLLM token-adapter contract

This implementation corrects actual PointBERT latent features before the original PointLLM projector. It does not replace the language model with a per-return classifier. Current engineering checks do not establish hallucination-mitigation performance.

## Native backbone and forward boundary

- Official frozen PointBERT, with the same learned tensors and an explicitly adapted group count of 96, produces `[B,T,97,384]`: one CLS plus 96 real FPS/KNN groups. A matched baseline must use the same group count, measured point budget, preprocessing, projector and language checkpoint.
- `PointLLMTokenAdapter(AdapterConfig(...))` receives `radar_tokens`, `reference_tokens`, and a measurement-only `physical_context`; it returns `corrected_tokens` with shape `[B,97,384]` for the current frame, plus explicitly named diagnostics and optional auxiliary logits.
- The original projector maps each corrected 384-dimensional token to the language embedding dimension. All 97 tokens enter the actual language model; response-token cross-entropy can backpropagate to adapter features and the projector while PointBERT and language weights remain frozen.
- An actual two-step official 7B LiDAR-reference connectivity probe has passed. The RGB encoder, real RGB/radar assembly and adapter checks have passed separately; an actual RGB-to-7B probe is a distinct pending execution at this contract revision.

## References and physical channels

`reference_kind` must explicitly be `lidar_geometry` or `rgb_patch`. LiDAR geometry is never described as RGB vision. Both modes use real reference tokens. RGB uses 256 frozen DINOv2-small patch embeddings without a CLS. The three-frame RGB cache is bound to the pinned encoder, actual image hashes, original pixel centers, resize/crop affine and source-window hash.

Required context includes actual group centers, same-membership pooled physical attributes, group/reference masks, per-frame timestamps, reference timestamps, sequence IDs, history validity and calibration availability. LiDAR adds common-frame reference centers. RGB adds original-pixel projected radar centers, original-pixel patch centers and an explicit crop/projection-valid mask. Native groups without a camera correspondence remain in the input and output.

The two physical channels are dataset-declared. The Radar Ghost probe uses measured radial speed and transformed amplitude. The nuScenes probe uses the radial component of compensated sensor-plane velocity along that sensor's original line of sight, and unconverted source RCS. RCS is not called amplitude. No physical attribute is encoded as a fabricated RGB color.

## Continuous correction and causal history

Geometry selects a local reference memory and supplies a Gaussian prior, with a bandwidth declared in metres or original-image pixels as appropriate. Learned latent discrepancy enters the selected continuous attention scores, observability summary and reconstruction. Two utility logits are normalized independently of candidate-attention probabilities. Five feature experts produce 384-dimensional residuals routed into corrected point tokens. These are latent feature corrections; optional category/support logits are not substituted for language outputs.

History uses only explicitly supplied past frames in one recording. For each current group, nearest real groups from at most five earlier eligible frames are averaged. Descriptor distances have a separate metre scale; age uses timestamp ratios, with no invented conversion from unknown source ticks to seconds and no radial-speed-times-ticks displacement. There is no state carried across minibatches.

RGB additionally supplies a masked mean of all available patch features to a zero-initialized CLS residual. This global visual evidence path permits reference information without a radar-group correspondence. Its projected output is multiplied by reference presence after all trainable layers so biases cannot inject missing-reference evidence. It does not guarantee that every object is represented.

## Missing reference and clean control

Masked reference contents, coordinates, timestamps and calibration values cannot affect correction, including arbitrary NaN padding. `disable_reference=True` provides a clean radar/history adapter control: reference inputs are ignored, reference-only parameters are frozen and reference auxiliary loss is disabled. Tests verify identical forward values, losses and gradients under reference perturbations. This mode is distinct from merely hiding reference tokens at evaluation after training with reference supervision.

## Supervision isolation and evaluation time

Forward rejects ground-truth-like context keys. Targets enter only separate loss functions or response-token labels. Whole-scene multi-object nuScenes windows are not assigned an unsupported single-class auxiliary label; the current engineering RGB response is a separately documented positive partial response, with all category/support masks disabled.

For all 404 official mini anchors, camera time was checked to be no later than the official annotated sample time; retained radar measurements precede the camera anchor. Model frame timestamps and GT evaluation use official sample time. Projection uses the actual camera pose and calibration at camera time. Actor motion between the camera and sample times is not fabricated or claimed corrected. Projection is not an annotation of camera visibility, absence or a radar ghost.

## Input provenance

`nuscenes_inputs.py` separates raw-window export, supervision export, evaluation-time binding and feature assembly. It rejects mismatched raw windows, altered image hashes, changed feature bytes and stale encoder-cache receipts. The assembled NPZ receipt includes its whole-file SHA-256, which every consuming dataset/generation loader must verify before tensor loading. Public code should acquire licensed nuScenes data locally rather than redistribute these raw RGB/radar caches.

The source and input hashes are recorded in `adapter_source_freeze_v2.json`. Old run02 sources remain under `run02_source_frozen/`; the old checkpoint loads strictly into the revised LiDAR adapter and produces bitwise-identical actual valid-reference outputs.
