# Fetch model weights directly from Hugging Face during the Runpod build.
FROM --platform=linux/amd64 python:3.12-slim-bookworm@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0
RUN python -m pip install --no-cache-dir "huggingface_hub[hf_xet]==2.2.0"
COPY deploy/cpu-benchmark/benchmark.py deploy/cpu-benchmark/models.json /opt/cpu-benchmark/
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 HF_XET_HIGH_PERFORMANCE=0 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/tmp/hf-home
# One verified native file per layer; no model image is pulled during packaging.
RUN python -u /opt/cpu-benchmark/benchmark.py --source hf --hold-seconds 0 --build-file 0
RUN python -u /opt/cpu-benchmark/benchmark.py --source hf --hold-seconds 0 --build-file 1
RUN python -u /opt/cpu-benchmark/benchmark.py --source hf --hold-seconds 0 --build-file 2
RUN python -u /opt/cpu-benchmark/benchmark.py --source hf --hold-seconds 0 --build-file 3
RUN python -u /opt/cpu-benchmark/benchmark.py --source image --hold-seconds 0
ENTRYPOINT ["python", "-u", "/opt/cpu-benchmark/benchmark.py"]
CMD ["--source", "image"]
