# Preserve the verified runtime filesystem and environment in one uncached layer.
FROM ghcr.io/luoziyong/mmh3-rtx5090-cpu-benchmark@sha256:05c48cec4c075342a03764a44fb332a4b07da3bf1505a99b879dcb81172dd977 AS runtime_files
COPY deploy/pod-benchmark/ /opt/pod-benchmark/
FROM scratch
COPY --from=runtime_files / /
ENV PATH="/opt/comfyui-baked/.venv-cu130/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONUNBUFFERED="1"
ENV IMAGEIO_FFMPEG_EXE="/usr/bin/ffmpeg"
ENV FILEBROWSER_CONFIG="/workspace/runpod-slim/.filebrowser.json"
ENV VENV_SUFFIX="cu130"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"
ENV NVIDIA_REQUIRE_CUDA=""
ENV NVIDIA_DISABLE_REQUIRE="true"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="all"
ENV PIP_CONSTRAINT="/opt/serverless/constraints-cu130.txt"
ENV GIT_TERMINAL_PROMPT="0"
ENV COMFYUI_DIR="/workspace/runpod-slim/ComfyUI"
ENV WORKFLOWS_DIR="/opt/workflows"
ENV MMH3_GPU_PROFILE="rtx5090_32"
ENV MMH3_PRESET="quality"
ENV MMH3_ENCODER_FAMILY="official"
ENV WORKFLOW_SYNC_ENABLED="false"
ENV DISABLE_TENSORRT_AUTO_INSTALL="true"
ENV JOB_TIMEOUT_S="1800"
ENV MMH3_MODEL_SOURCE="volume"
ENV VOL_ROOT="/workspace"
ENV HF_HUB_DISABLE_XET="1"
ENV HF_HUB_ENABLE_HF_TRANSFER="0"
ENV HF_HUB_DISABLE_PROGRESS_BARS="1"
ENV HF_HOME="/tmp/mmh3-hf"
ENV BENCH_VOLUME="/workspace"
ENV BENCH_TIMEOUT_SECONDS="1800"
WORKDIR "/workspace/runpod-slim"
EXPOSE 22/tcp
EXPOSE 8000/tcp
EXPOSE 8080/tcp
EXPOSE 8188/tcp
EXPOSE 8888/tcp
ENTRYPOINT ["python3", "-u", "/opt/pod-benchmark/benchmark.py"]
