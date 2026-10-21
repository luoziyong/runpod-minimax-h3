"""Run image-owned ComfyUI code while preserving an existing workspace's data."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import uuid
from pathlib import Path

STAMP = ".mmh3-image.json"
DATA_DIRS = {"input", "output", "user", "models", "temp", "logs"}
MUTABLE_FILES = {STAMP, "extra_model_paths.yaml", "comfy.log"}
REQUIRED = {"main.py", "comfy/model_management.py", "comfy_execution/graph.py",
            "custom_nodes/mmh3_memory/__init__.py", "custom_nodes/mmh3_memory/trt_cache.py"}


def ignored(directory, names, root):
    return {name for name in names if name in {".git", "__pycache__", ".DS_Store"}
            or name.startswith(".venv") or name.endswith((".pyc", ".pyo"))
            or (Path(directory) == root and name in DATA_DIRS | MUTABLE_FILES)}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root):
    files = {}
    for directory, dirs, names in os.walk(root):
        excluded = ignored(directory, dirs + names, root)
        dirs[:] = sorted(name for name in dirs if name not in excluded)
        for name in sorted(set(names) - excluded):
            path = Path(directory) / name
            files[path.relative_to(root).as_posix()] = file_hash(path)
    missing = REQUIRED - files.keys()
    if missing:
        raise ValueError(f"Baked ComfyUI is missing required files: {sorted(missing)}")
    return files


def stamp_image(baked, comfy_ref, environment=None):
    baked = Path(baked).resolve()
    if environment is None:
        environment = {"python": platform.python_version(), "packages": sorted(
            (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())}
    stamp = {"schema": 1, "comfy_ref": comfy_ref, "environment": environment,
             "files": image_files(baked)}
    stamp["build_id"] = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    (baked / STAMP).write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    return stamp


def read_stamp(baked):
    path = baked / STAMP
    if not path.is_file():
        raise RuntimeError(f"Image build manifest missing: {path}; rebuild this worker image")
    stamp = json.loads(path.read_text())
    payload = {key: value for key, value in stamp.items() if key != "build_id"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if stamp.get("schema") != 1 or stamp.get("build_id") != digest or not REQUIRED <= stamp.get("files", {}).keys():
        raise ValueError("Invalid image build manifest; rebuild this worker image")
    for name in stamp["files"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid image manifest path")
    return stamp


def matches_image(directory, stamp):
    if directory.is_symlink() or not (directory / STAMP).is_file():
        return False
    try:
        if json.loads((directory / STAMP).read_text()) != stamp:
            return False
        return all(file_hash(directory / name) == expected for name, expected in stamp["files"].items())
    except (OSError, ValueError):
        return False


def shared_data_links(directory, workspace):
    return (all((directory / name).is_symlink()
                and (directory / name).resolve() == (workspace / name).resolve()
                for name in ("input", "output", "user"))
            and (directory / "models").is_dir() and not (directory / "models").is_symlink())


def prepare_workspace(baked, workspace, runtime_key):
    baked, workspace = Path(baked).resolve(), Path(workspace).resolve()
    if baked == workspace or baked in workspace.parents or workspace in baked.parents:
        raise ValueError("Baked image and workspace must be separate directories")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", runtime_key) or runtime_key in (".", ".."):
        raise ValueError("Invalid GPU runtime key")
    stamp = read_stamp(baked)
    workspace.mkdir(parents=True, exist_ok=True)
    managed = workspace / ".mmh3"
    if managed.is_symlink():
        raise ValueError("Workspace .mmh3 must be a managed directory, not a symlink")
    managed.mkdir(exist_ok=True)
    with (managed / ".workspace.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for name in ("input", "output", "user"):
            (workspace / name).mkdir(exist_ok=True)
        parent = managed / "releases" / stamp["build_id"]
        parent.mkdir(parents=True, exist_ok=True)
        runtime = parent / runtime_key
        if matches_image(runtime, stamp) and shared_data_links(runtime, workspace):
            return runtime
        staging = parent / f".{runtime_key}.tmp-{uuid.uuid4().hex}"
        try:
            shutil.copytree(baked, staging, symlinks=True,
                            ignore=lambda directory, names: ignored(directory, names, baked))
            (staging / STAMP).write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
            for name in ("input", "output", "user"):
                (staging / name).symlink_to(workspace / name, target_is_directory=True)
            (staging / "models").mkdir()
            if not matches_image(staging, stamp):
                raise RuntimeError("Image files do not match their build manifest")
            backup = None
            if runtime.exists() or runtime.is_symlink():
                backup = runtime.with_name(f"{runtime_key}.previous-{uuid.uuid4().hex}")
                os.replace(runtime, backup)
            try:
                os.replace(staging, runtime)
            except BaseException:
                if backup is not None:
                    os.replace(backup, runtime)
                raise
            logging.warning("[mmh3-workspace] Using image %s at %s; legacy code/data preserved at %s",
                            stamp["build_id"][:12], runtime, workspace)
            return runtime
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    stamp_parser = sub.add_parser("stamp")
    stamp_parser.add_argument("--baked", required=True)
    stamp_parser.add_argument("--comfy-ref", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--baked", required=True)
    prepare_parser.add_argument("--workspace", required=True)
    prepare_parser.add_argument("--runtime-config", required=True)
    args = parser.parse_args()
    if args.action == "stamp":
        stamp = stamp_image(args.baked, args.comfy_ref)
        print(stamp["build_id"])
    else:
        path = Path(args.runtime_config)
        runtime = json.loads(path.read_text())
        directory = prepare_workspace(args.baked, args.workspace, runtime["trt_cache_key"])
        runtime.update(comfyui_dir=str(directory), workspace_dir=str(Path(args.workspace).resolve()),
                       image_build=read_stamp(Path(args.baked).resolve())["build_id"])
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(runtime, indent=2) + "\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(directory)


if __name__ == "__main__":
    main()
