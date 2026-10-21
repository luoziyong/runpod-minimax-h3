"""Bake one GPU profile for Runpod, or package it into GHCR-sized layers."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from worker_config import PROFILES, build_runtime, selected_models

PART_BYTES = 4 * 1024 ** 3  # GHCR allows 10 GB per layer; leave room for tar headers.
BUFFER_BYTES = 4 * 1024 ** 2
BUNDLE_DIR = "/opt/mmh3-models"
LOCAL_ROOT = "/tmp/mmh3-data"
RESTORE_RESERVE_BYTES = 8 * 1024 ** 3
ROOT = Path(__file__).resolve().parent


def model_selection(profile, encoder_family="official", volume_diffusion_models=False):
    # Do not inherit auto detection or optional downloads from the build host.
    runtime = build_runtime(profile, environ={"MMH3_ENCODER_FAMILY": encoder_family})
    rows = selected_models(runtime)
    volume = [row for row in rows if volume_diffusion_models and row["subdir"] == "diffusion_models"]
    image = [row for row in rows if row not in volume]
    return runtime, image, volume


def bake_models(bundle_dir, profile, encoder_family="official"):
    """Download native image files on a CPU builder; leave only the DiTs on the volume."""
    runtime, rows, volume = model_selection(profile, encoder_family, volume_diffusion_models=True)
    bundle_dir = Path(bundle_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = bundle_dir / "image-models.txt"
    manifest.write_text("".join(f"{row['subdir']}\t{row['filename']}\t{row['url']}\n" for row in rows))
    subprocess.run(["bash", str(ROOT / "populate-volume.sh")],
                   env=dict(os.environ, VOL_ROOT=str(bundle_dir), MANIFEST=str(manifest)), check=True)
    bundle = {"schema": 1, "layout": "files", "profile": runtime["profile"],
              "encoder_family": encoder_family, "model_groups": runtime["model_groups"],
              "models": [], "volume_models": volume}
    for row in rows:
        path = bundle_dir / "models" / row["subdir"] / row["filename"]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Selected image model missing or empty: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(BUFFER_BYTES), b""):
                digest.update(block)
        bundle["models"].append({"subdir": row["subdir"], "filename": row["filename"],
                                 "size": path.stat().st_size, "sha256": digest.hexdigest()})
        # A failed HF attempt followed by a successful wget can leave a second
        # partial copy. Keep that for failed downloads, but never bake it into an image.
        hf_temporary = path.parent / f".tmp_hf_{path.name}"
        if hf_temporary.is_dir():
            shutil.rmtree(hf_temporary)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)
    expected_paths = {Path(row["subdir"]) / row["filename"] for row in rows}
    extras = [path for path in (bundle_dir / "models").rglob("*")
              if path.is_file() and path.suffix in (".safetensors", ".gguf")
              and path.relative_to(bundle_dir / "models") not in expected_paths]
    if extras:
        raise ValueError("Unexpected weights in the image model directory; use a clean bundle directory")
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def pack_models(models_dir, output, profile, base_image, encoder_family="official", part_bytes=PART_BYTES,
                volume_diffusion_models=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", base_image):
        raise ValueError("Invalid base image reference")
    if not 0 < part_bytes <= PART_BYTES:
        raise ValueError("Model parts must be at most 4 GiB")
    runtime, rows, volume = model_selection(profile, encoder_family, volume_diffusion_models)
    models_dir, output = Path(models_dir).resolve(), Path(output).resolve()
    for row in rows:
        source = models_dir / row["subdir"] / row["filename"]
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Selected model missing or empty: {source}")
    # A failed pack never replaces a previous usable build context.
    output.mkdir(parents=True, exist_ok=False)
    bundle = {"schema": 1, "profile": runtime["profile"], "encoder_family": encoder_family,
              "model_groups": runtime["model_groups"], "models": [], "volume_models": volume}
    copies = []
    for row in rows:
        source = models_dir / row["subdir"] / row["filename"]
        expected = source.stat().st_size
        model = {"subdir": row["subdir"], "filename": row["filename"], "size": expected, "parts": []}
        total = 0
        with source.open("rb") as stream:
            while total < expected:
                index = len(copies)
                layer = f"layers/{index:05d}"
                name = f"part-{index:05d}.bin"
                directory = output / layer
                directory.mkdir(parents=True)
                digest, size = hashlib.sha256(), 0
                with (directory / name).open("wb") as target:
                    while size < part_bytes:
                        block = stream.read(min(BUFFER_BYTES, part_bytes - size))
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                if size == 0:
                    raise ValueError(f"Model changed while packaging: {source}")
                model["parts"].append({"name": name, "size": size, "sha256": digest.hexdigest()})
                copies.append(f"COPY {json.dumps([layer + '/', BUNDLE_DIR + '/parts/'])}")
                total += size
        if total != expected or source.stat().st_size != expected:
            raise ValueError(f"Model changed while packaging: {source}")
        bundle["models"].append(model)
        print(f"Packaged {row['filename']}: {total} bytes, {len(model['parts'])} layer(s)", flush=True)
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    settings = {"MMH3_MODEL_SOURCE": "hybrid" if volume else "image", "MMH3_BAKED_MODELS_DIR": BUNDLE_DIR,
                "MMH3_GPU_PROFILE": runtime["profile"], "MMH3_ENCODER_FAMILY": encoder_family,
                "MMH3_DOWNLOAD_ENHANCER": "false", "MMH3_DOWNLOAD_SINGULARITY": "false",
                "MMH3_DOWNLOAD_STYLE_LORAS": "false", "VOLUME_AUTOPOPULATE": "true" if volume else "false",
                "MMH3_LOCAL_ROOT": LOCAL_ROOT, "VOL_ROOT": "/runpod-volume" if volume else LOCAL_ROOT}
    lines = [f"FROM {base_image}",
             f"COPY {json.dumps(['bundle.json', BUNDLE_DIR + '/bundle.json'])}", *copies]
    lines += [f"ENV {key}={json.dumps(value)}" for key, value in settings.items()]
    (output / "Dockerfile").write_text("\n".join(lines) + "\n")
    return bundle


def validate_bundle(bundle, runtime):
    if bundle.get("schema") != 1:
        raise ValueError("Unsupported image model bundle schema")
    if (bundle.get("profile") != runtime["profile"]
            or bundle.get("encoder_family") != runtime["encoder_family"]
            or sorted(bundle.get("model_groups", [])) != sorted(runtime["model_groups"])):
        raise ValueError("Image model configuration differs from this worker; rebuild the model image")
    expected = {(row["subdir"], row["filename"]) for row in selected_models(runtime)}
    models = bundle.get("models", [])
    volume = bundle.get("volume_models", [])
    actual = {(model["subdir"], model["filename"]) for model in models}
    on_volume = {(model["subdir"], model["filename"]) for model in volume}
    if (actual | on_volume != expected or actual & on_volume
            or len(models) + len(volume) != len(expected)
            or any(model["subdir"] != "diffusion_models" for model in volume)):
        raise ValueError("Image model bundle does not contain exactly the selected models")
    source = "hybrid" if volume else "image"
    if runtime.get("model_source") != source:
        raise ValueError(f"This model bundle requires MMH3_MODEL_SOURCE={source}")
    layout = bundle.get("layout", "parts")
    if layout not in ("parts", "files"):
        raise ValueError("Unsupported image model layout")
    names = set()
    for model in models:
        parts = model.get("parts", [])
        if type(model.get("size")) is not int or model["size"] <= 0:
            raise ValueError("Invalid image model size")
        if layout == "files":
            if not re.fullmatch(r"[0-9a-f]{64}", model.get("sha256", "")) or parts:
                raise ValueError("Invalid native image model checksum or unexpected parts")
            continue
        if not parts:
            raise ValueError("Missing image model parts")
        for part in parts:
            if (not re.fullmatch(r"part-[0-9]{5,}\.bin", part.get("name", ""))
                    or part["name"] in names
                    or type(part.get("size")) is not int or not 0 < part["size"] <= PART_BYTES
                    or not re.fullmatch(r"[0-9a-f]{64}", part.get("sha256", ""))):
                raise ValueError("Invalid image model part")
            names.add(part["name"])
        if sum(part["size"] for part in parts) != model["size"]:
            raise ValueError("Image model part sizes do not match the model size")


def copy_part(path, part, target=None):
    digest, size = hashlib.sha256(), 0
    if not path.is_file() or path.stat().st_size != part["size"]:
        raise ValueError(f"Missing or incomplete image model part: {path}")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(block)
            size += len(block)
            if target is not None:
                target.write(block)
    if size != part["size"] or digest.hexdigest() != part["sha256"]:
        raise ValueError(f"Image model part checksum mismatch: {path}")


def restore_space_needed(bundle, destination):
    allocated, peak = 0, 0
    for model in bundle["models"]:
        target = destination / model["subdir"] / model["filename"]
        required = model["size"] if bundle.get("layout", "parts") == "parts" and len(model["parts"]) > 1 else 0
        previous = target.stat().st_size if target.is_file() and not target.is_symlink() else 0
        # Atomic replacement needs room for the new file before releasing the old one.
        peak = max(peak, allocated + required)
        allocated += required - previous
    return peak


def restore_models(bundle_dir, runtime):
    bundle_dir = Path(bundle_dir).resolve()
    encoded = (bundle_dir / "bundle.json").read_bytes()
    bundle = json.loads(encoded)
    validate_bundle(bundle, runtime)
    destination = Path(runtime["models_dir"]).resolve()
    if destination == bundle_dir or bundle_dir in destination.parents or destination in bundle_dir.parents:
        raise ValueError("Image model bundle and writable model directory must be separate")
    destination.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(encoded).hexdigest()
    ready = destination / ".bundle-ready"
    with (destination / ".bundle.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if bundle.get("volume_models"):
            volume_root = Path(runtime["volume_root"])
            if not volume_root.is_dir():
                raise ValueError(f"Hybrid model volume is missing: {volume_root}")
            for model in bundle["volume_models"]:
                source = volume_root / "models" / model["subdir"] / model["filename"]
                target = destination / model["subdir"] / model["filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() and target.resolve() == source.resolve():
                    continue
                if target.exists() or target.is_symlink():
                    raise ValueError(f"Hybrid volume link conflicts with an existing model: {target}")
                target.symlink_to(source)
        if ready.is_file() and ready.read_text() == identity:
            if all((destination / model["subdir"] / model["filename"]).is_file()
                   and (destination / model["subdir"] / model["filename"]).stat().st_size == model["size"]
                   for model in bundle["models"]):
                return bundle
        required = restore_space_needed(bundle, destination)
        free = shutil.disk_usage(destination).free
        if free < required + RESTORE_RESERVE_BYTES:
            raise RuntimeError(f"Insufficient container disk: preparing image models needs "
                               f"{required / 1e9:.2f} GB plus 8 GiB headroom; "
                               f"only {free / 1e9:.2f} GB is free. Increase the container disk "
                               "or deploy the volume-based runtime image instead.")
        ready.unlink(missing_ok=True)
        for model in bundle["models"]:
            target = destination / model["subdir"] / model["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.bundle.tmp")
            temporary.unlink(missing_ok=True)
            try:
                parts = model.get("parts", [])
                if bundle.get("layout") == "files":
                    source = bundle_dir / "models" / model["subdir"] / model["filename"]
                    copy_part(source, model)
                    temporary.symlink_to(source)
                elif len(parts) == 1:
                    source = bundle_dir / "parts" / parts[0]["name"]
                    copy_part(source, parts[0])
                    temporary.symlink_to(source)
                else:
                    with temporary.open("wb") as stream:
                        for part in parts:
                            copy_part(bundle_dir / "parts" / part["name"], part, stream)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        ready.write_text(identity)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    bake = commands.add_parser("bake", help="Bake native model files for Runpod's managed image builder")
    bake.add_argument("--profile", required=True, choices=PROFILES)
    bake.add_argument("--encoder-family", default="official", choices=("official", "heretic"))
    bake.add_argument("--bundle-dir", default=BUNDLE_DIR)
    pack = commands.add_parser("pack", help="Create a Docker build context from downloaded model files")
    pack.add_argument("--models-dir", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--profile", required=True, choices=PROFILES)
    pack.add_argument("--encoder-family", default="official", choices=("official", "heretic"))
    pack.add_argument("--base-image", required=True, help="Runtime image built from this repository's Dockerfile")
    pack.add_argument("--volume-diffusion-models", action="store_true",
                      help="Keep both DiTs on a network volume; bake only the encoder, VAEs and LoRAs")
    manifest = commands.add_parser("manifest", help="Print only the models to download into the image")
    manifest.add_argument("--profile", required=True, choices=PROFILES)
    manifest.add_argument("--encoder-family", default="official", choices=("official", "heretic"))
    manifest.add_argument("--volume-diffusion-models", action="store_true")
    restore = commands.add_parser("restore", help="Restore image models on the worker's temporary disk")
    restore.add_argument("--bundle-dir", default=BUNDLE_DIR)
    restore.add_argument("--runtime-config", required=True)
    args = parser.parse_args()
    if args.action == "bake":
        bundle = bake_models(args.bundle_dir, args.profile, args.encoder_family)
        print(f"Native image models ready: {bundle['profile']}; "
              f"{sum(model['size'] for model in bundle['models'])} bytes; no assembly needed at startup")
    elif args.action == "pack":
        bundle = pack_models(args.models_dir, args.output, args.profile, args.base_image, args.encoder_family,
                             volume_diffusion_models=args.volume_diffusion_models)
        total = sum(model["size"] for model in bundle["models"])
        writable = sum(model["size"] for model in bundle["models"] if len(model["parts"]) > 1)
        print(f"Build context: {args.output}; model bytes: {total}; restore bytes: {writable}")
    elif args.action == "manifest":
        _, rows, _ = model_selection(args.profile, args.encoder_family, args.volume_diffusion_models)
        for row in rows:
            print(f"{row['subdir']}\t{row['filename']}\t{row['url']}")
    else:
        runtime = json.loads(Path(args.runtime_config).read_text())
        bundle = restore_models(args.bundle_dir, runtime)
        if bundle.get("volume_models"):
            on_volume = {(model["subdir"], model["filename"]) for model in bundle["volume_models"]}
            rows = [row for row in selected_models(runtime) if (row["subdir"], row["filename"]) in on_volume]
            Path(args.runtime_config).with_name("models-volume-manifest.txt").write_text(
                "".join(f"{row['subdir']}\t{row['filename']}\t{row['url']}\n" for row in rows))
        print(f"Image models ready: {bundle['profile']} ({len(bundle['models'])} files)")


if __name__ == "__main__":
    main()
