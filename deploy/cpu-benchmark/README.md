# CPU Pod cold-start comparison

Both images prepare exactly the same minimal native T2VA quality model set: FL2VA INT8, official NVFP4 encoder, video VAE and audio VAE. No optional LoRAs, Ref2VA, enhancer or post-processing models are included. The smaller encoder corresponds to the RTX 5090 model option; this CPU benchmark does not test GPU compatibility or generation.

The runtime is identical Python 3.12 amd64 with the same pinned Hugging Face client. The HF image starts with no weights and downloads pinned upstream revisions. The image variant ships original files and does not download weights on startup. Both read and SHA-256 verify every complete file before emitting a flushed `models_ready` JSON event. A timeout prevents an unbounded download attempt. There is no model network volume.

Use identical CPU Pod size, data center and container disk for both tests. Record submission time, platform download and extraction events, first `benchmark_start` and `models_ready`. Report deployment-to-ready separately from container-to-ready. Include cold versus cached layers, scheduler changes, host identity when visible and any overlapping download/extraction; do not add overlapping phase durations. CPU model-file readiness excludes CUDA, ComfyUI startup and inference.

Build the two Dockerfiles from the repository root and deploy fresh Pods. Save container and system logs before stopping the temporary Pods. The result is also written to `/tmp/cpu-startup-result.json` inside each Pod. Images hold after completion so a platform restart policy does not silently rerun a warm test.
