import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_bundle
import runtime_workspace
from worker_config import build_runtime, selected_models

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "rtx6000_ada_48"


def make_bundle(root, profile=PROFILE, hybrid=False):
    models = root / "downloaded"
    runtime = build_runtime(profile, environ={"MMH3_MODEL_SOURCE": "hybrid" if hybrid else "image",
                            "VOL_ROOT": str(root / "volume" if hybrid else root / "local"),
                            "MMH3_LOCAL_ROOT": str(root / "local")})
    expected = {}
    for index, row in enumerate(selected_models(runtime)):
        path = models / row["subdir"] / row["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytes([index + 1]) * (13 if row["subdir"] == "vae" else 79)
        path.write_bytes(data)
        expected[(row["subdir"], row["filename"])] = data
        if hybrid and row["subdir"] == "diffusion_models":
            target = root / "volume/models" / row["subdir"] / row["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    context, image = root / "context", root / "image-models"
    with contextlib.redirect_stdout(io.StringIO()):
        bundle = model_bundle.pack_models(models, context, profile, "mmh3:runtime", part_bytes=32,
                                         volume_diffusion_models=hybrid)
    (image / "parts").mkdir(parents=True)
    shutil.copyfile(context / "bundle.json", image / "bundle.json")
    for part in (context / "layers").glob("*/*.bin"):
        shutil.copyfile(part, image / "parts" / part.name)
    return runtime, bundle, context, image, expected


def make_native_bundle(root, profile=PROFILE, volume_encoder=False):
    runtime, _, context, _, expected = make_bundle(root, profile, hybrid=True)
    if volume_encoder:
        for (subdir, filename), data in expected.items():
            if subdir == "text_encoders":
                target = root / "volume/models" / subdir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    tools = root / "download-tools"
    tools.mkdir()
    (tools / "populate-volume.sh").write_text('''#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import os,shutil
from pathlib import Path
rows=Path(os.environ['MANIFEST']).read_text().splitlines()
assert len(rows)==int(os.environ['FIXTURE_IMAGE_COUNT'])
for row in rows:
    sub,name,url=row.split('\\t')
    assert sub!='diffusion_models'
    target=Path(os.environ['VOL_ROOT'])/'models'/sub/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(os.environ['FIXTURE_MODELS'])/sub/name,target)
    partial=target.parent/('.tmp_hf_'+name)
    partial.mkdir()
    (partial/'unused.safetensors').write_bytes(b'failed HF partial download')
    target.with_name(name+'.tmp').write_bytes(b'old wget partial download')
PY
''')
    image = root / "native-image-models"
    with patch.object(model_bundle, "ROOT", tools), \
            patch.dict(os.environ, FIXTURE_MODELS=str(root / "downloaded"),
                       FIXTURE_IMAGE_COUNT="4" if volume_encoder else "5"):
        bundle = model_bundle.bake_models(image, profile, volume_encoder=volume_encoder)
    return runtime, bundle, context, image, expected


class ModelBundleTests(unittest.TestCase):
    def test_native_images_link_all_weights_without_assembling_encoder(self):
        for profile in (PROFILE, "rtx5090_32"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                runtime, bundle, context, image, expected = make_native_bundle(Path(directory), profile)
                self.assertEqual(bundle["layout"], "files")
                self.assertEqual(len(bundle["models"]), 5)
                self.assertEqual(len(bundle["volume_models"]), 2)
                self.assertFalse(list((image / "models").rglob("*.tmp")))
                self.assertFalse(list((image / "models").rglob(".tmp_hf_*")))
                root = Path(runtime["models_dir"])
                self.assertEqual(model_bundle.restore_space_needed(bundle, root), 0)
                model_bundle.restore_models(image, runtime)
                for model in bundle["models"]:
                    target = root / model["subdir"] / model["filename"]
                    self.assertTrue(target.is_symlink())
                    self.assertEqual(target.resolve(), (image / "models" / model["subdir"] / model["filename"]).resolve())
                    self.assertEqual(target.read_bytes(), expected[(model["subdir"], model["filename"])])

    def test_native_image_rejects_corrupt_files_before_publishing_links(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_native_bundle(Path(directory))
            model = bundle["models"][0]
            source = image / "models" / model["subdir"] / model["filename"]
            source.write_bytes(b"x" * source.stat().st_size)
            with self.assertRaisesRegex(ValueError, "checksum"):
                model_bundle.restore_models(image, runtime)
            self.assertFalse((Path(runtime["models_dir"]) / ".bundle-ready").exists())

    def test_native_bake_failure_does_not_publish_a_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "populate-volume.sh").write_text("#!/bin/bash\nexit 1\n")
            with patch.object(model_bundle, "ROOT", root), self.assertRaises(subprocess.CalledProcessError):
                model_bundle.bake_models(root / "image", PROFILE)
            self.assertFalse((root / "image/bundle.json").exists())

    def test_pack_selects_one_encoder_and_separate_small_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_bundle(Path(directory))
            self.assertEqual(bundle["profile"], PROFILE)
            self.assertEqual(len(bundle["models"]), 7)
            self.assertEqual(len([key for key in expected if key[0] == "text_encoders"]), 1)
            self.assertTrue(all("nvfp4" not in key[1] for key in expected))
            dockerfile = (context / "Dockerfile").read_text()
            copies = [json.loads(line[5:]) for line in dockerfile.splitlines() if line.startswith("COPY ")]
            self.assertIn('ENV MMH3_GPU_PROFILE="rtx6000_ada_48"', dockerfile)
            self.assertIn('ENV MMH3_MODEL_SOURCE="image"', dockerfile)
            self.assertIn('ENV VOLUME_AUTOPOPULATE="false"', dockerfile)
            self.assertEqual(len(copies), 1 + sum(len(model["parts"]) for model in bundle["models"]))
            for source, destination in copies[1:]:
                files = list((context / source).iterdir())
                self.assertEqual(len(files), 1)
                self.assertLessEqual(files[0].stat().st_size, 32)
                self.assertEqual(destination, model_bundle.BUNDLE_DIR + "/parts/")

    def test_restore_is_offline_and_does_not_copy_small_models(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_bundle(Path(directory))
            model_bundle.restore_models(image, runtime)
            for model in bundle["models"]:
                target = Path(runtime["models_dir"]) / model["subdir"] / model["filename"]
                self.assertEqual(target.read_bytes(), expected[(model["subdir"], model["filename"])])
                self.assertEqual(target.is_symlink(), len(model["parts"]) == 1)
            with patch.object(model_bundle, "copy_part", side_effect=AssertionError("Restored files should be reused")):
                model_bundle.restore_models(image, runtime)

    def test_corrupt_or_missing_parts_never_publish_partial_model(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                runtime, bundle, context, image, expected = make_bundle(Path(directory))
                model = bundle["models"][0]
                part = image / "parts" / model["parts"][0]["name"]
                if missing:
                    part.unlink()
                else:
                    part.write_bytes(b"x" * part.stat().st_size)
                with self.assertRaises(ValueError):
                    model_bundle.restore_models(image, runtime)
                target = Path(runtime["models_dir"]) / model["subdir"] / model["filename"]
                self.assertFalse(target.exists())
                self.assertFalse(list(target.parent.glob("*.tmp")))
                self.assertFalse((Path(runtime["models_dir"]) / ".bundle-ready").exists())

    def test_removed_restored_model_is_recreated(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_bundle(Path(directory))
            model_bundle.restore_models(image, runtime)
            model = bundle["models"][0]
            path = Path(runtime["models_dir"]) / model["subdir"] / model["filename"]
            path.write_bytes(b"partial")
            model_bundle.restore_models(image, runtime)
            self.assertEqual(path.read_bytes(), expected[(model["subdir"], model["filename"])])

    def test_low_disk_is_rejected_before_model_data_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_bundle(Path(directory))
            usage = shutil._ntuple_diskusage(total=50_000_000_000, used=49_000_000_000, free=1_000_000_000)
            with patch.object(model_bundle.shutil, "disk_usage", return_value=usage), \
                    patch.object(model_bundle, "copy_part", side_effect=AssertionError("No writes on low disk")):
                with self.assertRaisesRegex(RuntimeError, "Insufficient container disk"):
                    model_bundle.restore_models(image, runtime)
            self.assertFalse((Path(runtime["models_dir"]) / ".bundle-ready").exists())

    def test_restore_capacity_accounts_for_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_bundle(Path(directory))
            root = Path(runtime["models_dir"])
            sizes = [model["size"] for model in bundle["models"] if len(model["parts"]) > 1]
            self.assertEqual(model_bundle.restore_space_needed(bundle, root), sum(sizes))
            model_bundle.restore_models(image, runtime)
            self.assertEqual(model_bundle.restore_space_needed(bundle, root), max(sizes))

    def test_different_profile_encoder_or_optional_weights_require_rebuild(self):
        overrides = ({"MMH3_ENCODER_FAMILY": "heretic"}, {"MMH3_DOWNLOAD_ENHANCER": "true"},
                     {"MMH3_DOWNLOAD_SINGULARITY": "true"})
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_bundle(Path(directory))
            variants = [build_runtime(PROFILE, environ=env) for env in overrides]
            variants.append(build_runtime("rtx5090_32", environ={}))
            for variant in variants:
                with self.subTest(groups=variant["model_groups"]), self.assertRaisesRegex(ValueError, "rebuild"):
                    model_bundle.restore_models(image, variant)

    def test_manifest_paths_duplicate_models_and_oversized_parts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle, context, image, expected = make_bundle(Path(directory))
            variants = []
            unsafe = copy.deepcopy(bundle)
            unsafe["models"][0]["parts"][0]["name"] = "../../escape"
            variants.append(unsafe)
            duplicate = copy.deepcopy(bundle)
            duplicate["models"].append(duplicate["models"][0])
            variants.append(duplicate)
            oversized = copy.deepcopy(bundle)
            oversized["models"][0]["parts"][0]["size"] = 10_000_000_000
            variants.append(oversized)
            for variant in variants:
                with self.assertRaises(ValueError):
                    model_bundle.validate_bundle(variant, runtime)

    def test_pack_requires_complete_models_and_preserves_existing_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                model_bundle.pack_models(root / "missing", root / "output", PROFILE, "mmh3:runtime")
            self.assertFalse((root / "output").exists())
            runtime, bundle, context, image, expected = make_bundle(root)
            with self.assertRaises(FileExistsError):
                model_bundle.pack_models(root / "downloaded", context, PROFILE, "mmh3:runtime")
            self.assertEqual(json.loads((context / "bundle.json").read_text()), bundle)

    def test_image_and_volume_model_roots_and_invalid_source(self):
        image = build_runtime(PROFILE, environ={"MMH3_MODEL_SOURCE": "image"})
        volume = build_runtime(PROFILE, environ={})
        self.assertEqual(image["models_dir"], "/tmp/mmh3-data/models")
        self.assertEqual(volume["models_dir"], "/runpod-volume/models")
        hybrid = build_runtime(PROFILE, environ={"MMH3_MODEL_SOURCE": "hybrid"})
        self.assertEqual(hybrid["models_dir"], "/tmp/mmh3-data/models")
        self.assertEqual(hybrid["volume_root"], "/runpod-volume")
        with self.assertRaisesRegex(ValueError, "MMH3_MODEL_SOURCE"):
            build_runtime(PROFILE, environ={"MMH3_MODEL_SOURCE": "unknown"})

    def test_hybrid_profiles_bake_separate_encoders_and_share_only_dits(self):
        for profile, encoder in ((PROFILE, "int8"), ("rtx5090_32", "nvfp4")):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                runtime, bundle, context, image, expected = make_bundle(Path(directory), profile, True)
                self.assertEqual(len(bundle["models"]), 5)
                self.assertEqual(len(bundle["volume_models"]), 2)
                self.assertTrue(all(model["subdir"] == "diffusion_models" for model in bundle["volume_models"]))
                encoders = [model["filename"] for model in bundle["models"] if model["subdir"] == "text_encoders"]
                self.assertEqual(len(encoders), 1)
                self.assertIn(encoder, encoders[0])
                self.assertIn('ENV MMH3_MODEL_SOURCE="hybrid"', (context / "Dockerfile").read_text())
                model_bundle.restore_models(image, runtime)
                for key, data in expected.items():
                    path = Path(runtime["models_dir"]) / key[0] / key[1]
                    self.assertEqual(path.read_bytes(), data)
                    if key[0] == "diffusion_models":
                        self.assertTrue(path.is_symlink())
                        self.assertIn(str(Path(directory) / "volume"), str(path.resolve()))

    def test_hybrid_pack_does_not_require_downloaded_dits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, bundle, context, image, expected = make_bundle(root, hybrid=True)
            shutil.rmtree(root / "downloaded/diffusion_models")
            with contextlib.redirect_stdout(io.StringIO()):
                result = model_bundle.pack_models(root / "downloaded", root / "second", PROFILE,
                                                 "mmh3:runtime", volume_diffusion_models=True)
            self.assertEqual(len(result["models"]), 5)

    def test_entrypoint_starts_with_image_models_and_no_existing_model_volume(self):
        self.check_entrypoint()

    def test_hybrid_entrypoint_downloads_only_dits_for_each_gpu(self):
        for profile in (PROFILE, "rtx5090_32"):
            with self.subTest(profile=profile):
                self.check_entrypoint(profile, hybrid=True)

    def test_runpod_native_entrypoint_starts_both_fixed_profiles(self):
        for profile in (PROFILE, "rtx5090_32"):
            with self.subTest(profile=profile):
                self.check_entrypoint(profile, hybrid=True, native=True)

    def test_compact_native_entrypoint_downloads_encoder_and_dits_only(self):
        self.check_entrypoint(hybrid=True, native=True, volume_encoder=True)

    def check_entrypoint(self, profile=PROFILE, hybrid=False, native=False, volume_encoder=False):
        with tempfile.TemporaryDirectory(prefix="mmh3 image boot ") as directory:
            root = Path(directory)
            if native:
                runtime, bundle, context, image, expected = make_native_bundle(root, profile, volume_encoder)
                if volume_encoder:
                    self.assertEqual(len(bundle["models"]), 4)
                    self.assertEqual(len(bundle["volume_models"]), 3)
                    self.assertEqual({model["subdir"] for model in bundle["models"]}, {"vae", "loras"})
            else:
                runtime, bundle, context, image, expected = make_bundle(root, profile, hybrid)
            baked, scripts = root / "baked", root / "scripts"
            scripts.mkdir()
            for name in runtime_workspace.REQUIRED:
                path = baked / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# fixture: {name}\n")
            runtime_workspace.stamp_image(baked, "v0.38.0", environment={"fixture": True})
            for name in ("serverless-entrypoint.sh", "worker_config.py", "runtime_workspace.py",
                         "model_bundle.py", "models-manifest.txt"):
                shutil.copyfile(ROOT / name, scripts / name)
            torch_fixture = '''from types import SimpleNamespace
__version__ = 'fixture'
version = SimpleNamespace(cuda='13.0')
cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
    get_device_properties=lambda index: SimpleNamespace(name='NVIDIA RTX 6000 Ada',
        total_memory=48*1024**3, multi_processor_count=142, major=8, minor=9))
'''
            if profile == "rtx5090_32":
                torch_fixture = torch_fixture.replace("NVIDIA RTX 6000 Ada", "NVIDIA GeForce RTX 5090").replace(
                    "total_memory=48", "total_memory=32").replace("major=8, minor=9", "major=12, minor=0")
            (scripts / "torch.py").write_text(torch_fixture)
            (scripts / "populate-volume.sh").write_text("#!/bin/sh\necho UNEXPECTED_DOWNLOAD >&2\nexit 88\n")
            if hybrid:
                for model in bundle["volume_models"]:
                    (root / "volume/models" / model["subdir"] / model["filename"]).unlink()
                (scripts / "populate-volume.sh").write_text('''#!/usr/bin/env python3
import os,shutil
from pathlib import Path
rows=Path(os.environ['MANIFEST']).read_text().splitlines()
assert len(rows)==int(os.environ['FIXTURE_VOLUME_COUNT'])
for row in rows:
    sub,name,url=row.split('\\t')
    assert sub in ('diffusion_models','text_encoders')
    target=Path(os.environ['VOL_ROOT'])/'models'/sub/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(os.environ['FIXTURE_MODELS'])/sub/name,target)
''')
                (scripts / "populate-volume.sh").chmod(0o755)
            (scripts / "handler.py").write_text('''import json,os
from pathlib import Path
runtime=json.loads(Path(os.environ['MMH3_RUNTIME_CONFIG']).read_text())
print(json.dumps({'models_dir':runtime['models_dir'], 'model_source':runtime['model_source'],
                 'comfyui_dir':runtime['comfyui_dir']}))
''')
            env = dict(os.environ, MMH3_MODEL_SOURCE="hybrid" if hybrid else "image", MMH3_GPU_PROFILE=profile,
                       MMH3_ENCODER_FAMILY="official", MMH3_DOWNLOAD_ENHANCER="false",
                       MMH3_DOWNLOAD_SINGULARITY="false", MMH3_DOWNLOAD_STYLE_LORAS="false",
                       MMH3_BAKED_MODELS_DIR=str(image), VOL_ROOT=runtime["volume_root"],
                       MMH3_LOCAL_ROOT=str(root / "local"), FIXTURE_MODELS=str(root / "downloaded"),
                       FIXTURE_VOLUME_COUNT=str(len(bundle.get("volume_models", []))),
                       COMFYUI_BAKED_DIR=str(baked), COMFYUI_DIR=str(root / "workspace"),
                       MMH3_RUNTIME_CONFIG=str(root / "runtime/runtime.json"),
                       VOLUME_AUTOPOPULATE="true", WORKER_MODE="serverless", PYTHONDONTWRITEBYTECODE="1")
            self.assertFalse((root / "local").exists())
            result = subprocess.run(["bash", str(scripts / "serverless-entrypoint.sh")],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(state["model_source"], "hybrid" if hybrid else "image")
            self.assertEqual(state["models_dir"], str(root / "local/models"))
            for key, data in expected.items():
                self.assertEqual((Path(state["models_dir"]) / key[0] / key[1]).read_bytes(), data)
            for kind in ("onnx", "tensorrt"):
                cache = Path(state["comfyui_dir"]) / "models" / kind
                self.assertTrue(cache.is_symlink())
                self.assertIn(str(root / "local"), str(cache.resolve()))


if __name__ == "__main__":
    unittest.main()
