import concurrent.futures
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runtime_workspace as workspace

ROOT = Path(__file__).resolve().parents[1]


def make_image(root, version="new"):
    root.mkdir(parents=True, exist_ok=True)
    for name in workspace.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# image {version}: {name}\n")
    stamp = workspace.stamp_image(root, "v0.38.0", environment={"fixture": version})
    return stamp


class WorkspaceTests(unittest.TestCase):
    def test_legacy_code_models_virtualenv_and_user_data_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix="mmh3 workspace ") as directory:
            root = Path(directory)
            baked, legacy = root / "baked", root / "existing"
            stamp = make_image(baked)
            original = {"main.py": b"old core", "custom_nodes/user_node/node.py": b"old user node",
                        ".venv-cu130/bin/python3": b"old python", "models/vae/local.safetensors": b"local weights",
                        "input/ref.png": b"reference", "output/old.mp4": b"old output",
                        "user/default/workflows/mine.json": b"user workflow"}
            for name, data in original.items():
                path = legacy / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            runtime = workspace.prepare_workspace(baked, legacy, "ada-test")
            self.assertNotEqual(runtime, legacy)
            self.assertIn(stamp["build_id"], str(runtime))
            self.assertTrue(workspace.matches_image(runtime, stamp))
            self.assertFalse((runtime / ".venv-cu130").exists())
            self.assertFalse((runtime / "custom_nodes/user_node").exists())
            for name, data in original.items():
                self.assertEqual((legacy / name).read_bytes(), data)
            self.assertEqual((runtime / "input/ref.png").read_bytes(), b"reference")
            (runtime / "output/new.mp4").write_bytes(b"new output")
            self.assertEqual((legacy / "output/new.mp4").read_bytes(), b"new output")

    def test_same_image_is_reused_and_modified_runtime_is_backed_up_then_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stamp = make_image(root / "baked")
            runtime = workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            timestamp = (runtime / "main.py").stat().st_mtime_ns
            self.assertEqual(workspace.prepare_workspace(root / "baked", root / "workspace", "gpu"), runtime)
            self.assertEqual((runtime / "main.py").stat().st_mtime_ns, timestamp)
            self.assertFalse(list(runtime.parent.glob("*.previous-*")))
            (runtime / "main.py").write_text("locally changed")
            workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            self.assertTrue(workspace.matches_image(runtime, stamp))
            backups = list(runtime.parent.glob("gpu.previous-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "main.py").read_text(), "locally changed")

    def test_image_upgrade_and_four_gpu_keys_keep_separate_runtime_and_engine_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "baked", "first")
            keys = ["rtx6000_ada_48-a", "l40s_48-b", "pro6000_mig_48-c", "rtx5090_32-d"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                runtimes = list(pool.map(lambda key: workspace.prepare_workspace(
                    root / "baked", root / "workspace", key), keys))
            self.assertEqual(len(set(runtimes)), 4)
            self.assertTrue(all((runtime / "models").is_dir() for runtime in runtimes))
            make_image(root / "baked", "second")
            upgraded = workspace.prepare_workspace(root / "baked", root / "workspace", keys[0])
            self.assertNotEqual(upgraded, runtimes[0])
            self.assertIn("first", (runtimes[0] / "main.py").read_text())
            self.assertIn("second", (upgraded / "main.py").read_text())

    def test_failed_copy_does_not_replace_existing_runtime_or_leave_a_partial_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "baked")
            runtime = workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            (runtime / "main.py").write_text("preserve me")
            def partial_copy(source, destination, **kwargs):
                destination.mkdir()
                (destination / "main.py").write_text("incomplete copy")
                raise OSError("disk full")

            with patch.object(workspace.shutil, "copytree", side_effect=partial_copy):
                with self.assertRaisesRegex(OSError, "disk full"):
                    workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            self.assertEqual((runtime / "main.py").read_text(), "preserve me")
            self.assertFalse(list(runtime.parent.glob(".gpu.tmp-*")))

    def test_modified_data_link_is_repaired_without_discarding_its_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "baked")
            runtime = workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            (runtime / "input").unlink()
            (runtime / "input").mkdir()
            (runtime / "input/local-ref.png").write_bytes(b"preserve this reference")
            workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            self.assertTrue(workspace.shared_data_links(runtime, (root / "workspace").resolve()))
            backup = next(runtime.parent.glob("gpu.previous-*"))
            self.assertEqual((backup / "input/local-ref.png").read_bytes(), b"preserve this reference")

    def test_failed_atomic_publication_restores_previous_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "baked")
            runtime = workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            (runtime / "main.py").write_text("preserve on rollback")
            replace = os.replace

            def fail_publish(source, target):
                if Path(source).name.startswith(".gpu.tmp-"):
                    raise OSError("publish failed")
                return replace(source, target)

            with patch.object(workspace.os, "replace", side_effect=fail_publish):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    workspace.prepare_workspace(root / "baked", root / "workspace", "gpu")
            self.assertEqual((runtime / "main.py").read_text(), "preserve on rollback")
            self.assertFalse(list(runtime.parent.glob(".gpu.tmp-*")))

    def test_missing_invalid_or_mismatched_image_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baked = root / "baked"
            baked.mkdir()
            with self.assertRaisesRegex(RuntimeError, "manifest missing"):
                workspace.prepare_workspace(baked, root / "workspace", "gpu")
            self.assertFalse((root / "workspace").exists())
            stamp = make_image(baked)
            (baked / "main.py").write_text("modified after build")
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                workspace.prepare_workspace(baked, root / "workspace", "gpu")
            self.assertFalse(list((root / "workspace/.mmh3/releases").rglob(".gpu.tmp-*")))
            stamp["files"]["../outside"] = "invalid"
            (baked / workspace.STAMP).write_text(json.dumps(stamp))
            with self.assertRaises(ValueError):
                workspace.prepare_workspace(baked, root / "workspace", "gpu")

    def test_unsafe_workspace_layouts_and_runtime_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "baked")
            for key in ("..", "../../escape", "/absolute"):
                with self.assertRaises(ValueError):
                    workspace.prepare_workspace(root / "baked", root / "workspace", key)
            with self.assertRaisesRegex(ValueError, "separate"):
                workspace.prepare_workspace(root / "baked", root / "baked/nested", "gpu")

    def test_entrypoint_uses_image_python_and_new_core_with_existing_main_py(self):
        with tempfile.TemporaryDirectory(prefix="mmh3 entrypoint ") as directory:
            root = Path(directory)
            baked, legacy, scripts, volume = (root / name for name in ("baked", "legacy", "scripts", "volume"))
            legacy.mkdir(); scripts.mkdir(); volume.mkdir()
            (legacy / "main.py").write_text("old core")
            old_python = legacy / ".venv-cu130/bin/python3"
            old_python.parent.mkdir(parents=True)
            old_python.write_text("#!/bin/sh\necho OLD_PYTHON_SELECTED >&2\nexit 75\n")
            old_python.chmod(0o755)
            image_python = baked / ".venv-cu130/bin/python3"
            image_python.parent.mkdir(parents=True)
            image_python.write_text("#!/bin/sh\nprintf 'image\\n' >> \"$PYTHON_RECORD\"\nexec "
                                    + shlex.quote(sys.executable) + " \"$@\"\n")
            image_python.chmod(0o755)
            make_image(baked)
            for name in ("serverless-entrypoint.sh", "runtime_workspace.py"):
                shutil.copyfile(ROOT / name, scripts / name)
            (scripts / "worker_config.py").write_text('''import json,os,sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--output-dir')+1]);root.mkdir(parents=True,exist_ok=True)
(root/'runtime.json').write_text(json.dumps({'profile':'rtx6000_ada_48','trt_cache_key':'fixture',
    'models_dir':str(Path(os.environ['VOL_ROOT'])/'models')}))
(root/'models-manifest.txt').write_text('')
''')
            (scripts / "handler.py").write_text('''import json,os
from pathlib import Path
root=Path(os.environ['COMFYUI_DIR'])
print(json.dumps({'comfyui_dir':str(root),'core':(root/'main.py').read_text(),
                 'stage_nodes':(root/'custom_nodes/mmh3_memory/__init__.py').is_file()}))
''')
            env = dict(os.environ, COMFYUI_DIR=str(legacy), COMFYUI_BAKED_DIR=str(baked),
                       VOL_ROOT=str(volume), MMH3_RUNTIME_CONFIG=str(root / "runtime/runtime.json"),
                       VOLUME_AUTOPOPULATE="false", WORKER_MODE="serverless", VENV_SUFFIX="cu130",
                       PYTHON_RECORD=str(root / "python-record"), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(["bash", str(scripts / "serverless-entrypoint.sh")],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertTrue(output["stage_nodes"])
            self.assertIn("image new", output["core"])
            self.assertNotEqual(output["comfyui_dir"], str(legacy))
            self.assertEqual((legacy / "main.py").read_text(), "old core")
            self.assertNotIn("OLD_PYTHON_SELECTED", result.stderr)
            self.assertTrue((root / "python-record").read_text().splitlines())
            config = json.loads((root / "runtime/runtime.json").read_text())
            self.assertEqual(config["comfyui_dir"], output["comfyui_dir"])
            self.assertEqual(config["workspace_dir"], str(legacy.resolve()))


if __name__ == "__main__":
    unittest.main()
