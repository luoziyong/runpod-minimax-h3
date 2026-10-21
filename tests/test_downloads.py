import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DownloadTests(unittest.TestCase):
    def run_download(self, directory, fail=False):
        directory = Path(directory)
        tools = directory / "tools"
        tools.mkdir(exist_ok=True)
        for name, script in {
            "hf": "#!/bin/sh\nexit 1\n",
            "flock": "#!/bin/sh\nexit 0\n",
            "wget": """#!/usr/bin/env python3
import os,sys
from pathlib import Path
path=Path(sys.argv[sys.argv.index('-O')+1])
path.write_bytes(b'partial' if os.environ.get('MOCK_FAIL')=='1' else b'fixture weights')
raise SystemExit(1 if os.environ.get('MOCK_FAIL')=='1' else 0)
""",
        }.items():
            path = tools / name
            path.write_text(script)
            path.chmod(0o755)
        manifest = directory / "manifest.txt"
        manifest.write_text("vae\tmodel.safetensors\thttps://huggingface.co/test/model/resolve/main/model.safetensors\n")
        env = dict(os.environ, VOL_ROOT=str(directory), MANIFEST=str(manifest),
                   PATH=str(tools) + os.pathsep + os.environ["PATH"], MOCK_FAIL="1" if fail else "0")
        return subprocess.run(["bash", str(ROOT / "populate-volume.sh")], env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_failed_download_is_fatal_and_does_not_publish_partial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_download(directory, fail=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("1 download(s) failed", result.stderr)
            self.assertFalse((Path(directory) / "models/vae/model.safetensors").exists())
            self.assertTrue((Path(directory) / "models/vae/model.safetensors.tmp").exists())

    def test_valid_existing_file_is_reused_but_empty_file_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_download(directory).returncode, 0)
            model = Path(directory) / "models/vae/model.safetensors"
            before = model.stat().st_mtime_ns
            self.assertEqual(self.run_download(directory, fail=True).returncode, 0)
            self.assertEqual(model.stat().st_mtime_ns, before)
            model.write_bytes(b"")
            self.assertEqual(self.run_download(directory).returncode, 0)
            self.assertGreater(model.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
