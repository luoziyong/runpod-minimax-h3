"""Atomic ONNX downloads shared by TensorRT workers on different GPUs."""

import fcntl
import functools
import hashlib
import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

_INSTALL_LOCK = threading.Lock()


@contextmanager
def model_lock(path):
    path = Path(path).resolve()
    directory = path.parent / ".download-locks"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(path.name.encode()).hexdigest()
    with (directory / key).open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def valid_onnx(path):
    # Transport receipt and integrity, not a substitute for the ONNX parser.
    path = Path(path)
    try:
        with path.open("rb") as stream:
            if path.stat().st_size < 1024 or stream.read(1) != b"\x08":
                return False
        receipt = json.loads(receipt_path(path).read_text())
        return receipt == payload_receipt(path)
    except (FileNotFoundError, ValueError):
        return False


def receipt_path(path):
    path = Path(path)
    return path.with_name(f".{path.name}.mmh3-complete.json")


def payload_receipt(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def quarantine_partial(path):
    """Run under model_lock, including for caches from older worker versions."""
    path = Path(path)
    if path.exists() and not valid_onnx(path):
        backup = path.with_name(f".{path.name}.invalid-{uuid.uuid4().hex}")
        os.replace(path, backup)
        receipt = receipt_path(path)
        if receipt.exists():
            os.replace(receipt, backup.with_name(backup.name + ".receipt.json"))
        logging.warning("[mmh3-cache] Unverified ONNX moved to %s", backup)


def atomic_downloader(download):
    @functools.wraps(download)
    def wrapped(url, save_path):
        target = Path(save_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with model_lock(target):
            if valid_onnx(target):
                return
            quarantine_partial(target)
            # Retain .onnx so upstream downloaders also run their format checks.
            temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.part.onnx")
            receipt_tmp = temporary.with_suffix(".receipt.json")
            try:
                download(url=url, save_path=str(temporary))
                with temporary.open("rb") as stream:
                    if temporary.stat().st_size < 1024 or stream.read(1) != b"\x08":
                        raise RuntimeError(f"Incomplete or invalid ONNX download: {target.name}")
                    os.fsync(stream.fileno())
                with receipt_tmp.open("w") as stream:
                    json.dump(payload_receipt(temporary), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                os.replace(receipt_tmp, receipt_path(target))
            finally:
                temporary.unlink(missing_ok=True)
                receipt_tmp.unlink(missing_ok=True)

    wrapped._mmh3_atomic_download = True
    return wrapped


def prepare_loader(loader, models_dir, settings):
    model = settings.get("model")
    if not isinstance(model, str) or not model or Path(model).name != model or model in (".", ".."):
        raise ValueError("TensorRT model must be a plain model name")
    directory = Path(models_dir) / "onnx"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{model}.onnx"
    with model_lock(target):
        quarantine_partial(target)

    function = getattr(loader, "__func__", loader)
    namespace = function.__globals__
    # The upstream loader resolves download_file in its module globals. Install
    # once rather than stacking temporary patches across concurrent threads.
    with _INSTALL_LOCK:
        download = namespace.get("download_file")
        if not callable(download):
            raise RuntimeError("Unsupported TensorRT node: download_file helper is unavailable")
        if not getattr(download, "_mmh3_atomic_download", False):
            namespace["download_file"] = atomic_downloader(download)
