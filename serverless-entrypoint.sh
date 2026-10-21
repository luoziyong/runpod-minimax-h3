#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_COMFYUI_DIR="${COMFYUI_DIR:-/workspace/runpod-slim/ComfyUI}"
BAKED="${COMFYUI_BAKED_DIR:-/opt/comfyui-baked}"
export MMH3_MODEL_SOURCE="${MMH3_MODEL_SOURCE:-volume}"
case "$MMH3_MODEL_SOURCE" in
    volume|hybrid)
        VOL_ROOT="${VOL_ROOT:-/runpod-volume}"
        if [ ! -d "$VOL_ROOT" ]; then
            echo "Mount a persistent model volume at $VOL_ROOT before starting this worker" >&2
            exit 1
        fi
        ;;
    image)
        VOL_ROOT="${VOL_ROOT:-/tmp/mmh3-data}"
        mkdir -p "$VOL_ROOT"
        ;;
    *) echo "MMH3_MODEL_SOURCE must be volume, image or hybrid" >&2; exit 1 ;;
esac
VENV_DIR="$BAKED/.venv-${VENV_SUFFIX:-cu130}"
export VOL_ROOT
export MMH3_RUNTIME_CONFIG="${MMH3_RUNTIME_CONFIG:-/tmp/mmh3/runtime.json}"
RUNTIME_DIR="$(dirname "$MMH3_RUNTIME_CONFIG")"

# Use the image's Python environment, never an old workspace virtualenv.
if [ -x "$VENV_DIR/bin/python3" ]; then
    export PATH="$VENV_DIR/bin:$PATH"
fi
python3 "$SCRIPT_DIR/worker_config.py" prepare --output-dir "$RUNTIME_DIR"
if [ "$MMH3_MODEL_SOURCE" != "volume" ]; then
    python3 "$SCRIPT_DIR/model_bundle.py" restore \
        --bundle-dir "${MMH3_BAKED_MODELS_DIR:-/opt/mmh3-models}" \
        --runtime-config "$MMH3_RUNTIME_CONFIG"
fi
COMFYUI_DIR="$(python3 "$SCRIPT_DIR/runtime_workspace.py" prepare \
    --baked "$BAKED" --workspace "$WORKSPACE_COMFYUI_DIR" --runtime-config "$MMH3_RUNTIME_CONFIG")"
export COMFYUI_DIR
export MANIFEST="$RUNTIME_DIR/models-manifest.txt"
if [ "$MMH3_MODEL_SOURCE" = "hybrid" ]; then
    export MANIFEST="$RUNTIME_DIR/models-volume-manifest.txt"
fi
if [ "$MMH3_MODEL_SOURCE" != "image" ] && [ "${VOLUME_AUTOPOPULATE:-true}" = "true" ]; then
    "$SCRIPT_DIR/populate-volume.sh"
fi
python3 - "$MMH3_RUNTIME_CONFIG" <<'CHECK'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
manifest = Path(sys.argv[1]).with_name('models-manifest.txt')
missing = []
for line in manifest.read_text().splitlines():
    sub, name, _ = line.split('\t')
    path = Path(config['models_dir']) / sub / name
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(str(path))
if missing:
    raise SystemExit('Selected models missing: ' + ', '.join(missing))
CHECK

MODELS_DIR="$(python3 -c 'import json, os; print(json.load(open(os.environ["MMH3_RUNTIME_CONFIG"]))["models_dir"])')"
cat > "$COMFYUI_DIR/extra_model_paths.yaml" <<YAML
runpod_volume:
  base_path: $MODELS_DIR
  diffusion_models: diffusion_models
  unet: unet
  loras: loras
  vae: vae
  text_encoders: text_encoders
  LLM: LLM
YAML

# Image and hybrid modes keep assembled weights and TRT caches on the local disk.
# Engines are not portable between Ada, 5090 and a 48GB MIG slice. Original
# prebuilt RTX PRO 6000 engines are never copied into these cache directories.
CACHE_KEY="$(python3 -c 'import json, os; print(json.load(open(os.environ["MMH3_RUNTIME_CONFIG"]))["trt_cache_key"])')"
mkdir -p "$COMFYUI_DIR/models" "$MODELS_DIR/tensorrt/$CACHE_KEY" "$MODELS_DIR/onnx"
for kind in tensorrt onnx; do
    target="$MODELS_DIR/$kind"
    [ "$kind" != tensorrt ] || target="$target/$CACHE_KEY"
    dest="$COMFYUI_DIR/models/$kind"
    if [ -d "$dest" ] && [ ! -L "$dest" ]; then
        mv "$dest" "$dest.legacy.$(date +%s)"
    fi
    ln -sfn "$target" "$dest"
done
case "${WORKER_MODE:-serverless}" in
    serverless) exec python3 "$SCRIPT_DIR/handler.py" ;;
    api) exec python3 "$SCRIPT_DIR/http_api.py" ;;
    *) echo "WORKER_MODE must be serverless or api" >&2; exit 1 ;;
esac
