# Third-party notices

`vendor_pointbert/` is adapted from the official PointLLM source at commit `cb72f4e6ab625ddab92f84931127e12bc326b4be`:
https://github.com/InternRobotics/PointLLM/tree/cb72f4e6ab625ddab92f84931127e12bc326b4be/pointllm/model/pointbert

The official project README specifies Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. Preserve upstream attribution and its applicable share-alike and noncommercial conditions. A blanket license for newly written RaHUM files does not replace these conditions. CPU compatibility changes replace the `timm` DropPath dependency with a pure-Torch equivalent and record the same official FPS/KNN group indices without changing learned tensors.

Official model: https://huggingface.co/RunsenXu/PointLLM_7B_v1.2/tree/37d8c15aaff8e7e05f04729dcb6960d5758e9f86 . Its model card separately specifies CC BY-NC 4.0. The full model, private quantized language cache and tokenizer binary are not bundled. Use the pinned publisher downloader and observe the original license. There is no claim that an official PointLLM 3B checkpoint was executed or identified.

Cite Xu et al., *PointLLM: Empowering Large Language Models to Understand Point Clouds*, ECCV 2024.

The optional real RGB encoder uses https://huggingface.co/facebook/dinov2-small at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, retaining its original publisher Apache 2.0 terms and notices. It is distinct from the PointLLM point encoder and is kept frozen in the recorded RGB engineering execution.

Radar Ghost and nuScenes data retain their source data licenses. Sensor samples, derived measurement/token caches, source annotations and official model weights must not be silently relicensed or redistributed as unrestricted RaHUM software. The source-only release can include download/processing instructions and provenance receipts while users obtain datasets from their publishers under the applicable terms.


DINOv2-small is downloaded from the official facebook/dinov2-small repository at revision ed25f3a31f01632728cabb09d1542f84ab7b0056. Model metadata declares Apache-2.0. Its weights are not bundled.

nuScenes data is subject to CC BY-NC-SA 4.0 and the publisher dataset terms. This bundle contains a tiny, attributed frozen-feature engineering example and provenance, not the full sensor archive. The dataset download remains official. Radar Ghost data and labels retain their publisher CC BY-NC-SA 4.0 terms and attribution.
