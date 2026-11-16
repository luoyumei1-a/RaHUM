# Predeclared semantic-claim annotation protocol

## Scope

This pilot uses a manually annotated **oracle main-target radar-return crop** from the official RadarGhost test split. CMTO's main-target category is the only semantic truth supplied. The category is not included in the model prompt. The experiment does not supply a full-scene object inventory or establish that unannotated objects do not exist.

The question, fixed before any grounded pilot generation, is:

> Describe the object represented by this point cloud in one sentence. State uncertainty if its category is unclear.

All predefined raw and mitigated outputs use the same official PointLLM-7B v1.2 revision, CPU NF4 quantization, 8192 sampled measured rows, zero missing-color channels, seed 7588, greedy decoding, and maximum 32 generated tokens. Every output is retained regardless of quality. A token-limit truncation is marked explicitly.

## Annotation

1. Copy the exact generated text and record source-file and request hashes.
2. Extract the target-category assertion literally, if the text makes one. Do not treat a confidence caveat as erasing an explicit category assertion (e.g., “probably a chair” still asserts chair). A pure abstention makes no category assertion.
3. A pedestrian/person/human description is compatible with the manually annotated pedestrian category. A cyclist/bicyclist/person riding a bicycle is compatible with cyclist. “Bicycle” is recorded as compatible **coarse constituent category**, not exact cyclist identification; report this distinction if it occurs. A generic human claim for a cyclist is likewise coarse compatible, not an exact-class answer.
4. A clearly disjoint object class (e.g., chair, airplane, table) is **Contradicted** by the annotated oracle target identity. A compatible category is **Supported** at the corresponding exact/coarse granularity. A vague geometric shape, incomprehensible sentence, ambiguous body-part description, or category whose relationship is not established is **Unverifiable**, rather than automatically wrong.
5. Gender, color, material, detailed anatomy, count, articulation, height, and motion are **Unverifiable** unless a separately identified independent annotation supports them. Planar z=0 is a representation convention, not measured height. A point being labeled multipath is not evidence that its associated semantic object does not exist.
6. List additional atomic attributes separately; retain Unverifiable attributes in the record. Do not fold unverified details into confirmed false claims.
7. Report exact-category accuracy, compatible coarse-category coverage, number of category assertions, Supported/Contradicted/Unverifiable counts, and abstentions separately. The **category-claim contradiction rate** is Contradicted / (Supported + Contradicted), with its denominator shown. The **verifiable category coverage** is (Supported + Contradicted) / all predefined examples. Do not label this limited quantity as complete-scene hallucination rate or as validation across all perception-model families.

The final annotation table must expose every sample, both variants, the untouched response, and the ground-truth source so the author can independently verify the adjudication. A small pilot cannot establish broad superiority or a statistically secure mitigation effect.
