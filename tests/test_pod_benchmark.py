import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pod_probe", ROOT / "deploy/pod-benchmark/benchmark.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PodBenchmarkTests(unittest.TestCase):
    def model(self, source):
        return dict(repo="fixture/repo", revision="0" * 40, source_path=source,
                    subdir="loras", filename="model.safetensors", size=7)

    def test_both_upstream_layouts_land_in_comfy_model_subfolder(self):
        def fake_download(*, filename, local_dir, **kwargs):
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")
            return str(target)

        fake_hub = types.SimpleNamespace(hf_hub_download=fake_download)
        for source in ("model.safetensors", "loras/model.safetensors"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                    probe.download(self.model(source), root)
                self.assertEqual((root / "loras/model.safetensors").read_bytes(), b"fixture")
                self.assertFalse((root / "model.safetensors").exists())

    def test_reset_is_opt_in_once_only_and_rejects_unknown_data(self):
        with tempfile.TemporaryDirectory() as folder:
            volume = Path(folder)
            models, state = volume / "models", volume / "mmh3-benchmark"
            models.mkdir(); state.mkdir()
            (state / "volume-marker.json").write_text(json.dumps({"id": "fixture"}))
            model = models / "model.safetensors"
            model.write_bytes(b"fixture")
            manifest = {"models": [self.model("model.safetensors")]}
            with patch.dict(os.environ, {"BENCH_RESET_MODELS_ONCE": ""}):
                self.assertFalse(probe.reset_test_models_once(volume, models, state, manifest))
            unknown = models / "keep.txt"
            unknown.write_text("user data")
            with patch.dict(os.environ, {"BENCH_RESET_MODELS_ONCE": "regression"}), patch.object(probe, "emit"):
                with self.assertRaisesRegex(RuntimeError, "unexpected test-volume data"):
                    probe.reset_test_models_once(volume, models, state, manifest)
                self.assertTrue(model.exists())
                unknown.unlink()
                self.assertTrue(probe.reset_test_models_once(volume, models, state, manifest))
                self.assertFalse(model.exists())
                model.write_bytes(b"fixture")
                self.assertFalse(probe.reset_test_models_once(volume, models, state, manifest))
                self.assertTrue(model.exists())


if __name__ == "__main__":
    unittest.main()
