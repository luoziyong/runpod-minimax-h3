"""CPU-only lifecycle probe using the production cu130 environment and 5090 files."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

START = time.monotonic()
RUN_ID = uuid.uuid4().hex[:12]
EVENTS = []
EVENT_FILE = None


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def emit(event, **values):
    row = dict(event=event, utc=utc(), elapsed_s=round(time.monotonic() - START, 3),
               run_id=RUN_ID, **values)
    EVENTS.append(row)
    line = json.dumps(row, separators=(",", ":"))
    print("MMH3_POD_BENCH " + line, flush=True)
    if EVENT_FILE is not None:
        with EVENT_FILE.open("a") as stream:
            stream.write(line + "\n")
    return row


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=90)
    return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def disk_snapshot(stage, volume, models):
    files = list(models.rglob("*.safetensors"))
    emit("disk_snapshot", stage=stage, model_count=len(files),
         model_logical_bytes=sum(p.stat().st_size for p in files),
         df=command("df", "-B1", "-T", "/", str(volume), "/tmp"),
         du=command("du", "-B1", "-s", str(models), str(volume / "mmh3-benchmark")),
         declared_volume_gb=100)


def download(model, root):
    from huggingface_hub import hf_hub_download
    result = {}
    expected_source = f"{model['subdir']}/{model['filename']}"
    if model["source_path"] not in (model["filename"], expected_source):
        raise ValueError("Unexpected upstream model layout")
    # LightX2V stores its LoRA at the repository root; Comfy-Org uses subfolders.
    local_dir = root / model["subdir"] if model["source_path"] == model["filename"] else root

    def fetch():
        try:
            result["path"] = hf_hub_download(repo_id=model["repo"], revision=model["revision"],
                                              filename=model["source_path"], local_dir=local_dir)
        except BaseException as exc:
            result["error"] = exc

    start = time.monotonic()
    thread = threading.Thread(target=fetch, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(15)
        if thread.is_alive():
            emit("download_progress", filename=model["filename"], seconds=round(time.monotonic() - start, 3))
    if "error" in result:
        raise result["error"]
    expected = root / model["subdir"] / model["filename"]
    if Path(result["path"]) != expected or expected.stat().st_size != model["size"]:
        raise RuntimeError(f"Downloaded file path or size mismatch: {expected}")
    return round(time.monotonic() - start, 3)


def reset_test_models_once(volume, models, state_dir, manifest):
    token = os.environ.get("BENCH_RESET_MODELS_ONCE", "")
    if not token:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", token):
        raise ValueError("Invalid reset token")
    receipt = state_dir / f"reset-{token}.json"
    if receipt.exists():
        return False
    if models.is_symlink() or models.resolve() != volume.resolve() / "models" or not (state_dir / "volume-marker.json").is_file():
        raise RuntimeError("Refusing to reset an unmarked or redirected model directory")
    known = {m["filename"] for m in manifest["models"]}
    for path in models.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.name not in known
                                 and ".cache" not in path.relative_to(models).parts):
            raise RuntimeError(f"Refusing to delete unexpected test-volume data: {path}")
    emit("cold_cache_reset_start", token=token)
    shutil.rmtree(models)
    models.mkdir()
    receipt.write_text(json.dumps(dict(run_id=RUN_ID, utc=utc(), token=token)))
    emit("cold_cache_reset_complete", token=token, model_count=0)
    return True


def get_json(path):
    with urllib.request.urlopen("http://127.0.0.1:8188" + path, timeout=10) as response:
        return json.load(response)


def main():
    global EVENT_FILE
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Benchmark deadline")))
    signal.alarm(int(os.environ.get("BENCH_TIMEOUT_SECONDS", "1800")))
    volume = Path(os.environ.get("BENCH_VOLUME", "/workspace")).resolve()
    if not volume.is_dir() or volume.stat().st_dev == Path("/").stat().st_dev:
        raise RuntimeError("Benchmark requires a separately mounted network volume; refusing container-disk fallback")
    models = volume / "models"
    models.mkdir(exist_ok=True)
    state_dir = volume / "mmh3-benchmark"
    state_dir.mkdir(exist_ok=True)
    marker = state_dir / "volume-marker.json"
    previous_marker = json.loads(marker.read_text()) if marker.exists() else None
    if previous_marker is None:
        marker.write_text(json.dumps(dict(id=uuid.uuid4().hex, created_utc=utc())))
    run_dir = state_dir / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    EVENT_FILE = run_dir / "events.jsonl"
    temporary_marker = Path("/tmp/mmh3-container-marker.json")
    previous_temporary_marker = temporary_marker.read_text() if temporary_marker.exists() else None
    temporary_marker.write_text(json.dumps(dict(run_id=RUN_ID, utc=utc())))
    packages = {}
    for name in ("torch", "huggingface_hub", "transformers", "runpod", "tensorrt_cu13"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    emit("container_start", pod_id=os.environ.get("RUNPOD_POD_ID"), hostname=platform.node(),
         cpu=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name")), None), packages=packages,
         volume_marker=json.loads(marker.read_text()), volume_marker_preexisting=previous_marker is not None,
         container_marker_preexisting=previous_temporary_marker is not None,
         scope="CPU mode; no CUDA initialization, model tensor loading or video generation")
    disk_snapshot("before_models", volume, models)
    manifest = json.loads(Path(__file__).with_name("models.json").read_text())
    if reset_test_models_once(volume, models, state_dir, manifest):
        disk_snapshot("after_cold_cache_reset", volume, models)
    sys.path.insert(0, "/opt/serverless")
    import worker_config
    import runtime_workspace
    os.environ.update(VOL_ROOT=str(volume), MMH3_MODEL_SOURCE="volume", MMH3_GPU_PROFILE="rtx5090_32")
    runtime = worker_config.build_runtime("rtx5090_32")  # Explicitly no fabricated GPU properties.
    selected = {(m["subdir"], m["filename"]) for m in worker_config.selected_models(runtime)}
    if selected != {(m["subdir"], m["filename"]) for m in manifest["models"]}:
        raise RuntimeError("Pinned benchmark manifest differs from production 5090 model selection")
    downloaded, reused = [], []
    for model in manifest["models"]:
        path = models / model["subdir"] / model["filename"]
        if path.is_file() and path.stat().st_size == model["size"]:
            reused.append(model["filename"])
            emit("model_reused", filename=model["filename"], bytes=model["size"])
            continue
        emit("download_start", filename=model["filename"], bytes=model["size"])
        seconds = download(model, models)
        downloaded.append(dict(filename=model["filename"], seconds=seconds, bytes=model["size"]))
        emit("download_complete", **downloaded[-1])
    inventory = emit("model_inventory_ready", model_bytes=manifest["model_bytes"],
                     downloaded_count=len(downloaded), reused_count=len(reused))
    emit("runtime_prepare_start")
    comfy = runtime_workspace.prepare_workspace("/opt/comfyui-baked", "/tmp/mmh3-workspace/ComfyUI",
                                                "cpu-benchmark-rtx5090")
    runtime["trt_cache_key"] = "cpu-benchmark-no-cuda"
    runtime["measurement_mode"] = "cpu_only"
    config = Path("/tmp/mmh3/runtime.json")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(runtime))
    os.environ["MMH3_RUNTIME_CONFIG"] = str(config)
    (comfy / "extra_model_paths.yaml").write_text(
        f"runpod_volume:\n  base_path: {models}\n  diffusion_models: diffusion_models\n"
        "  unet: unet\n  loras: loras\n  vae: vae\n  text_encoders: text_encoders\n")
    emit("runtime_prepare_complete", comfy_dir=str(comfy), image_build=runtime_workspace.read_stamp(Path("/opt/comfyui-baked"))["build_id"])
    args = [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", "8188",
            "--disable-auto-launch", "--use-pytorch-cross-attention", "--disable-xformers"] + runtime["extra_args"]
    emit("comfy_start", command=args)
    log = (run_dir / "comfy.log").open("ab", buffering=0)
    process = subprocess.Popen(args, cwd=comfy, stdout=log, stderr=subprocess.STDOUT)
    until = time.monotonic() + 600
    while time.monotonic() < until:
        if process.poll() is not None:
            print((run_dir / "comfy.log").read_text(errors="replace")[-24000:], flush=True)
            raise RuntimeError(f"ComfyUI exited with {process.returncode}")
        try:
            stats = get_json("/system_stats")
            info = get_json("/object_info")
        except Exception:
            time.sleep(1)
            continue
        required = {"MMH3StageBarrier", "MMH3StageGate", "MMH3TensorRTLoader", "VAEDecodeTiled"}
        missing = sorted(required - info.keys())
        if missing:
            raise RuntimeError(f"Required staged-loading nodes missing: {missing}")
        service = emit("service_ready", devices=stats.get("devices"), node_count=len(info), required_nodes=sorted(required))
        break
    else:
        raise TimeoutError("ComfyUI did not become ready")
    disk_snapshot("service_ready_before_full_hash", volume, models)
    emit("integrity_check_start", note="Separate full network-volume read; not included in service-ready timing")
    verified = []
    for model in manifest["models"]:
        path = models / model["subdir"] / model["filename"]
        start = time.monotonic()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != model["sha256"]:
            raise RuntimeError(f"Model SHA-256 mismatch: {path}")
        row = dict(filename=model["filename"], sha256=actual, seconds=round(time.monotonic() - start, 3))
        verified.append(row)
        emit("model_verified", **row)
    disk_snapshot("all_verified", volume, models)
    report = dict(status="completed", run_id=RUN_ID, pod_id=os.environ.get("RUNPOD_POD_ID"),
                  profile="rtx5090_32", inventory_ready=inventory, service_ready=service,
                  downloaded=downloaded, reused=reused, verified=verified,
                  total_model_bytes=manifest["model_bytes"], events=EVENTS.copy(),
                  scope="CPU model files and ComfyUI health only; excludes CUDA/model loading/inference")
    (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    emit("benchmark_complete", result=report)
    signal.alarm(0)
    # The controller stops the Pod after collecting logs; keep it stable meanwhile.
    while process.poll() is None:
        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        emit("benchmark_failed", error=str(error), error_type=type(error).__name__)
        raise
