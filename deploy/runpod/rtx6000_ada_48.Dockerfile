# Generated from Dockerfile by scripts/render_runpod_dockerfiles.py; do not edit.
# runpod-minimax-h3 — MiniMax H3 INT8 serverless worker (CUDA 13.0)
#
# - Single active Pruned INT8 DiT, profile-selected INT8/NVFP4 encoder, INT8 video VAE.
# - RTX 6000 Ada / L40S / PRO 6000 MIG 48GB / RTX 5090 32GB, staged GPU loading.
# - Missing volume files are auto-downloaded at worker boot from
#   models-manifest.txt (populate-volume.sh, VOLUME_AUTOPOPULATE=false to skip)
# - For Runpod's managed registry, use the generated deploy/runpod Dockerfiles;
#   native image weights are linked directly, while DiTs remain on a volume.
# - model_bundle.py also supports GHCR-sized layers for external registries.
# - runpod serverless handler drives ComfyUI per-job (handler.py)
#

# Verified linux/amd64 image manifest; avoid mutable-tag/index resolution drift.
ARG MMH3_PLATFORM=linux/amd64
FROM --platform=${MMH3_PLATFORM} huchukato/comfyui-base:cu130@sha256:9a913ebdc8eba4ad304efcd206871365e85236231404279591d968093432c151
ARG COMFYUI_REF=v0.38.0
ENV PATH="/opt/comfyui-baked/.venv-cu130/bin:${PATH}"

# The INT8 video VAE and stage gates require a known ComfyUI core. Keep
# custom_nodes/models supplied by the base, but replace the core sources.
RUN git clone --depth 1 --branch "$COMFYUI_REF" https://github.com/Comfy-Org/ComfyUI.git /tmp/comfyui-core && \
    cp -a /tmp/comfyui-core/. /opt/comfyui-baked/ && \
    rm -rf /tmp/comfyui-core && \
    python3 -m pip install --no-cache-dir -r /opt/comfyui-baked/requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Custom nodes — same stack as the pod image
# ──────────────────────────────────────────────────────────────────────────────
ENV GIT_TERMINAL_PROMPT=0
RUN dl() { \
    local user="$1" repo="$2" branch="${3:-main}" dir="$4"; \
    echo "Cloning $user/$repo ($branch) -> $dir"; \
    git clone --depth 1 --branch "$branch" "https://github.com/$user/$repo.git" "/opt/comfyui-baked/custom_nodes/$dir"; \
  } && \
    cd /opt/comfyui-baked/custom_nodes && \
    dl Kosinkadink ComfyUI-VideoHelperSuite main ComfyUI-VideoHelperSuite && \
    dl yolain ComfyUI-Easy-Use main ComfyUI-Easy-Use && \
    dl huchukato ComfyUI-QwenVL-Mod main ComfyUI-QwenVL-Mod && \
    dl huchukato ComfyUI-RIFE-TensorRT-Auto master ComfyUI-RIFE-TensorRT-Auto && \
    dl huchukato ComfyUI-Upscaler-TensorRT-Auto master ComfyUI-Upscaler-TensorRT-Auto && \
    dl huchukato ComfyUI-TagForge v2.3.0 ComfyUI-TagForge && \
    dl huchukato ComfyUI-PerfectVideoResolution master ComfyUI-PerfectVideoResolution && \
    dl huchukato ComfyUI-MiniMaxH3-Director main ComfyUI-MiniMaxH3-Director && \
    dl pixaroma ComfyUI-Pixaroma main ComfyUI-Pixaroma && \
    dl kijai ComfyUI-KJNodes main ComfyUI-KJNodes && \
    dl ltdrdata ComfyUI-Impact-Pack Main ComfyUI-Impact-Pack && \
    dl ltdrdata ComfyUI-Impact-Subpack main ComfyUI-Impact-Subpack


# ──────────────────────────────────────────────────────────────────────────────
# Requirements + runpod serverless SDK
# ──────────────────────────────────────────────────────────────────────────────
# libgl1/libglib2.0-0: opencv-python (VideoHelperSuite) fails to import without
# them → the whole pack never registers → missing_node_type on VHS_* nodes.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN bash -c 'cd /opt/comfyui-baked/custom_nodes && for node_dir in */; do \
        if [ -f "$node_dir/requirements.txt" ]; then \
            echo "Installing requirements for $node_dir..." && \
            pip install --no-cache-dir -r "$node_dir/requirements.txt" || echo "Failed to install requirements for $node_dir"; \
        fi; \
    done' && \
    pip install --no-cache-dir --no-deps "transformers>=5.2.0" && \
    pip install --no-cache-dir 'runpod>=1.12' requests boto3 "huggingface_hub[cli]" hf_transfer && \
    pip install --no-cache-dir tensorrt_cu13==10.15.1.29 && \
    pip cache purge

# ──────────────────────────────────────────────────────────────────────────────
# Model dirs (empty — real models come from the network volume via
# extra_model_paths.yaml written by the entrypoint)
# ──────────────────────────────────────────────────────────────────────────────
RUN for dir in vae diffusion_models unet text_encoders loras LLM onnx tensorrt; do \
        mkdir -p "/opt/comfyui-baked/models/$dir"; \
    done

# ──────────────────────────────────────────────────────────────────────────────
# Serverless entrypoint + handler + workflow templates (API format)
# ──────────────────────────────────────────────────────────────────────────────
COPY handler.py /opt/serverless/handler.py
COPY worker_config.py workflow_runtime.py http_api.py runtime_workspace.py model_bundle.py /opt/serverless/
COPY custom_nodes/mmh3_memory/ /opt/comfyui-baked/custom_nodes/mmh3_memory/
COPY serverless-entrypoint.sh /opt/serverless/entrypoint.sh
COPY populate-volume.sh /opt/serverless/populate-volume.sh
COPY models-manifest.txt /opt/serverless/models-manifest.txt
COPY workflows-api/mmh3/ /opt/workflows/
RUN chmod +x /opt/serverless/entrypoint.sh /opt/serverless/populate-volume.sh
RUN python3 /opt/serverless/runtime_workspace.py stamp \
    --baked /opt/comfyui-baked --comfy-ref "$COMFYUI_REF"

ENV COMFYUI_DIR="/workspace/runpod-slim/ComfyUI" \
    WORKFLOWS_DIR="/opt/workflows" \
    MMH3_GPU_PROFILE="auto" \
    MMH3_PRESET="quality" \
    MMH3_ENCODER_FAMILY="official" \
    WORKFLOW_SYNC_ENABLED="false" \
    DISABLE_TENSORRT_AUTO_INSTALL="true" \
    JOB_TIMEOUT_S="1800" \
    VENV_SUFFIX="cu130"

EXPOSE 8000

# Base image carries ENTRYPOINT ["/start.sh"] (pod mode) — a bare CMD would be
# passed to it as args and the runpod handler would never start. Override the
# entrypoint so the container runs the serverless handler directly.
ENTRYPOINT ["/opt/serverless/entrypoint.sh"]

# Native image weights; DiTs are downloaded onto the session network volume.
ENV MMH3_GPU_PROFILE="rtx6000_ada_48"
ENV MMH3_ENCODER_FAMILY="official"
ENV MMH3_MODEL_SOURCE="hybrid"
ENV MMH3_BAKED_MODELS_DIR="/opt/mmh3-models"
ENV MMH3_LOCAL_ROOT="/tmp/mmh3-data"
ENV VOL_ROOT="/runpod-volume"
ENV VOLUME_AUTOPOPULATE="true"
ENV MMH3_DOWNLOAD_ENHANCER="false"
ENV MMH3_DOWNLOAD_SINGULARITY="false"
ENV MMH3_DOWNLOAD_STYLE_LORAS="false"
RUN python3 /opt/serverless/model_bundle.py bake --profile rtx6000_ada_48
