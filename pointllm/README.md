# Official PointLLM-7B v1.2 on a CPU-only machine

This adapter executes the official complete fine-tuned 7B checkpoint; it does not train a stand-in language model. Source and checkpoint revisions, quantization, measured-row preprocessing, and generated text are recorded per example.

## Environment

Tested on Python 3.12, an AVX512/BF16-capable Intel CPU, and 9.7 GiB RAM. The first measured-cloud execution used about 6.94 GiB peak RSS and roughly 129 s for 48 generated tokens. Concurrent CPU work can increase latency. Keep more than 32 GB free for the original FP32 checkpoint shards and enough RAM for the quantized model.

```bash
python3.12 -m venv runtime_v5
runtime_v5/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
runtime_v5/bin/python -m pip install -r requirements_cpu.txt
runtime_v5/bin/python download_official.py
runtime_v5/bin/python pointllm_cpu_infer.py \
  --cloud /absolute/path/to/measured_points.npz --key points \
  --geometry-only --threads 2 --max-new-tokens 32 \
  --output /absolute/path/to/actual_generation.json
```

The pinned downloader checks the publisher's checkpoint SHA256 values. The `evidence` directory contains the pinned small checkpoint metadata required by this release. Complete model weights are downloaded from the publisher rather than redistributed in the code bundle.

The Llama module uses NF4 4-bit weights and BF16 CPU compute; the exact official PointBERT/projector weights remain FP32. These precision conditions are part of the experimental configuration. On the tested runtime, the packaged native CPU kernels ran successfully even though the optional community kernel extension was unavailable.

## Inputs and evaluation

The adapter accepts NPY, NPZ, or MATLAB arrays with XYZ. Only a true Nx6 XYZRGB input may supply color; geometry-only and Nx5 radar inputs use zero RGB. Doppler/intensity are not RGB. Sparse clouds repeat measured rows to the official 8192-point budget with a fixed seed; no new returns are interpolated. Physical units and calibration must be supplied by the dataset documentation; normalization does not establish them.

`run_claim_pilot.py` executes an immutable preselected manifest. `pointllm_worker.py` keeps one actual model in memory and executes bounded JSON-file requests. It exits when a `STOP` file appears in its queue directory or after the specified idle timeout.

```bash
runtime_v5/bin/python pointllm_worker.py \
  --queue-dir /absolute/path/to/claim_queue --threads 2 --idle-seconds 1800
```

A request has `cloud_path`, `key`, `expected_source_sha256`, `geometry_only`, `question`, `seed`, `max_new_tokens`, and optional provenance `metadata`. Only the cloud and question enter the model. The worker writes every untouched response and token sequence under `results/` and writes explicit input/execution errors without fabricating outputs.

The official-test pilot evaluates conditional semantic assertions about **oracle main-target crops**. Its manual CMTO labels support only the independently annotated categories. They do not supply a full-scene absent-object ground truth. See `claim_annotation_protocol.md` and the frozen selection/protocol files. Raw/mitigated outputs are paired under the same model, point budget, seed, prompt, and decoding settings. Empty or geometrically degenerate filtered inputs count as coverage failures rather than being silently replaced with oracle-selected returns.

## Attribution and model sizes

Official code: `InternRobotics/PointLLM` at `cb72f4e6ab625ddab92f84931127e12bc326b4be`.

Official checkpoint: `RunsenXu/PointLLM_7B_v1.2` at `37d8c15aaff8e7e05f04729dcb6960d5758e9f86`.

The official project provides 7B and 13B variants. No official PointLLM 3B checkpoint was identified, and no unrelated 3B model is substituted under that name. MiniGPT-3D/Phi-2 is a distinct point-cloud-language model and would require a separate implementation and execution receipt.

The vendored PointBERT components retain their upstream attribution and license requirements (`vendor_pointbert/UPSTREAM_ATTRIBUTION.txt`). The official project README specifies CC BY-NC-SA 4.0, and the HF model card separately specifies CC BY-NC 4.0. Do not relicense or redistribute the weights under a blanket license for a new repository.
