# Pinned ComfyUI scheduling sources

`graph.py` and `caching.py` are unmodified upstream files from ComfyUI v0.38.0,
commit `6b747c0428c343e1417219641db93a4fb7cb69ae`. `SOURCE.json` records the
repository paths and SHA256 values. The upstream GPL license is in `LICENSE`.

`tests/scheduler_harness.py` loads the original `ExecutionList`, topological
sort, dynamic prompt and `NullCache` definitions. Imports requiring torch are
omitted, and annotations are deferred for Python 3.9 test hosts. Scheduling
method bodies are unchanged. GPU kernels are CPU stand-ins; production MMH3
gates and release barriers execute with a simulated model manager.

These tests verify scheduling, output lifetimes, release ordering and duplicate
sampling. They do not establish CUDA kernel compatibility or measured VRAM use.
When changing the image's ComfyUI version, update the source snapshots, source
metadata and regression expectations together.
