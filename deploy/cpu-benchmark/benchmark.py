"""Measure identical model readiness on CPU Pods; no GPU inference is performed."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def emit(event, **values):
    # Runpod treats bare JSON as structured logs and drops fields without "message".
    print("MMH3_BENCHMARK " + json.dumps({"event": event, "utc": utc_now(), **values}), flush=True)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model, root):
    # Use a new local_dir on each fresh Pod, without reusing a persistent HF cache.
    from huggingface_hub import hf_hub_download

    result = {}

    def run():
        try:
            result["path"] = hf_hub_download(
                repo_id=model["repo"], filename=model["source_path"],
                revision=model["revision"], local_dir=root,
            )
        except BaseException as error:
            result["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    start = time.monotonic()
    thread.start()
    while thread.is_alive():
        thread.join(timeout=10)
        if thread.is_alive():
            emit("download_wait", filename=model["filename"], elapsed_s=round(time.monotonic() - start, 3))
    if "error" in result:
        raise result["error"]
    path = Path(result["path"])
    expected = root / model["subdir"] / model["filename"]
    if path != expected:
        raise RuntimeError(f"Unexpected download path: {path}")
    return round(time.monotonic() - start, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("hf", "image"), required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models.json"))
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--hold-seconds", type=int, default=3600)
    parser.add_argument("--output", type=Path, default=Path("/tmp/cpu-startup-result.json"))
    parser.add_argument("--build-file", type=int, help="Download one manifest index into its own build layer")
    args = parser.parse_args()
    start, start_utc = time.monotonic(), utc_now()
    signal.alarm(args.timeout_seconds)
    manifest = json.loads(args.manifest.read_text())
    root = args.model_root
    root.mkdir(parents=True, exist_ok=True)
    if args.build_file is not None:
        if args.source != "hf" or args.hold_seconds != 0 or not 0 <= args.build_file < len(manifest["models"]):
            parser.error("--build-file requires --source hf, --hold-seconds 0 and a valid model index")
        model = manifest["models"][args.build_file]
        path = root / model["subdir"] / model["filename"]
        if path.exists():
            raise RuntimeError(f"Build download target already exists: {path}")
        emit("build_download_start", filename=model["filename"], size=model["size"])
        elapsed = download(model, root)
        if path.stat().st_size != model["size"] or hash_file(path) != model["sha256"]:
            raise RuntimeError(f"Build model size or SHA-256 mismatch: {path}")
        emit("build_model_verified", filename=model["filename"], download_seconds=elapsed)
        return
    versions = {}
    for package in ("huggingface_hub", "hf_xet"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    emit("benchmark_start", source=args.source, pid=os.getpid(), python=sys.version,
         platform=platform.platform(), cpu_count=os.cpu_count(), packages=versions,
         pod_id=os.environ.get("RUNPOD_POD_ID"), data_center=os.environ.get("RUNPOD_DC_ID"),
         model_count=len(manifest["models"]), model_bytes=manifest["model_bytes"])
    if args.source == "hf" and any(root.rglob("*.safetensors")):
        raise RuntimeError("HF benchmark requires an empty model directory")
    results = []
    for model in manifest["models"]:
        item = {"filename": model["filename"], "size": model["size"], "sha256": model["sha256"]}
        path = root / model["subdir"] / model["filename"]
        if args.source == "hf":
            emit("download_start", filename=model["filename"], size=model["size"])
            item["download_seconds"] = download(model, root)
            emit("download_complete", filename=model["filename"], elapsed_s=item["download_seconds"])
        if not path.is_file() or path.stat().st_size != model["size"]:
            raise RuntimeError(f"Missing model or size mismatch: {path}")
        hash_start = time.monotonic()
        emit("verify_start", filename=model["filename"])
        actual = hash_file(path)
        item["verify_seconds"] = round(time.monotonic() - hash_start, 3)
        if actual != model["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        emit("verify_complete", filename=model["filename"], elapsed_s=item["verify_seconds"], sha256=actual)
        results.append(item)
    expected = {root / m["subdir"] / m["filename"] for m in manifest["models"]}
    if set(root.rglob("*.safetensors")) != expected:
        raise RuntimeError("Unexpected extra model files")
    report = {"status": "models_ready", "source": args.source, "started_utc": start_utc,
              "ready_utc": utc_now(), "container_to_models_ready_seconds": round(time.monotonic() - start, 3),
              "model_bytes": manifest["model_bytes"], "model_count": len(results),
              "pod_id": os.environ.get("RUNPOD_POD_ID"), "packages": versions, "models": results,
              "measurement_scope": "Files present and full SHA-256 verified; excludes GPU/ComfyUI initialization"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    emit("models_ready", **report)
    signal.alarm(0)
    # Keep the completed result available for collection; avoid automatic restarts rerunning the test.
    until = time.monotonic() + args.hold_seconds
    while time.monotonic() < until:
        time.sleep(min(30, until - time.monotonic()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("benchmark_failed", error=str(error), error_type=type(error).__name__)
        raise
