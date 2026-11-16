# Attribution and terms for the real feature example

The files in `backbone/nuscenes_real_rgb_probe/` contain a small derived
feature/context example and annotation/provenance metadata from **nuScenes
v1.0-mini**, by the nuScenes/Motional dataset creators.

- Dataset source and terms: https://www.nuscenes.org/terms-of-use
- Official source archive: https://www.nuscenes.org/data/v1.0-mini.tgz
- Dataset license: **Creative Commons Attribution-NonCommercial-ShareAlike
  4.0 International**, https://creativecommons.org/licenses/by-nc-sa/4.0/,
  together with the applicable official dataset terms.
- Dataset paper: Caesar et al., *nuScenes: A multimodal dataset for
  autonomous driving*, CVPR 2020.

Retain this attribution, the license link, the noncommercial restriction
and the applicable share-alike conditions for redistribution/adaptations of
these data-derived files. These files are not relicensed under a blanket
software license. No endorsement by the dataset creators is implied.

The example contains three real training-window measurements transformed
with source calibration and ego poses, actual frozen PointBERT radar
features, and actual frozen DINOv2-small image features. It does not include
the original camera pixels or model weights. Transformations, sampling,
missing-color channels and the adapted 97-token budget are documented in
the accompanying receipts. The partial language target is independently
supported by the listed official annotation tokens; it is not a complete
scene caption, camera visibility annotation or test response.

The input feature NPZ and independent target sidecar remain separate. The
public manifest changes filesystem paths only. Actual recorded training
and generation artifacts retain their original source and artifact hashes.
No result in this example establishes held-out hallucination mitigation.

PointLLM source/model and DINOv2 retain their separate publisher terms;
see `backbone/THIRD_PARTY_NOTICES.md` and the original publisher notices.
