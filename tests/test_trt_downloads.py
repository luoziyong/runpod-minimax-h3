import concurrent.futures
import hashlib
import multiprocessing
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_memory_nodes import load_nodes

PAYLOAD = b"\x08" + b"complete ONNX transport fixture" * 256


def profile_worker(models_dir, loader_type, barrier, started, finish, downloads, results):
    """Separate processes model workers with different hardware engine caches."""
    module, _ = load_nodes()

    def download(url, save_path):
        with downloads.get_lock():
            downloads.value += 1
        with Path(save_path).open("wb") as stream:
            stream.write(PAYLOAD[:2048])
            stream.flush()
            started.set()
            if not finish.wait(10):
                raise TimeoutError("Fixture download was not released")
            stream.write(PAYLOAD[2048:])

    namespace = {"Path": Path, "download_file": download, "models_dir": models_dir}
    # Match the upstream protocol: existence is checked before calling the
    # module-level helper; the engine builder immediately reads the ONNX.
    exec('''class Loader:
    FUNCTION = "load"
    def load(self, model, batch_size):
        path = Path(models_dir) / "onnx" / (model + ".onnx")
        if not path.exists():
            download_file(url="fixture://model", save_path=str(path))
        return (path.read_bytes(),)
''', namespace)
    dependencies = {"folder_paths": types.SimpleNamespace(models_dir=models_dir),
                    "nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={loader_type: namespace["Loader"]})}
    try:
        with patch.dict(sys.modules, dependencies):
            barrier.wait(timeout=10)
            output = module.MMH3TensorRTLoader().load(
                object(), loader_type, '{"model":"shared-model","batch_size":1}')[0]
        results.put((len(output), hashlib.sha256(output).hexdigest()))
    except BaseException as exc:
        results.put(("error", repr(exc)))


class TensorRTDownloadTests(unittest.TestCase):
    def test_four_gpu_processes_share_only_complete_onnx_and_download_once(self):
        context = multiprocessing.get_context("spawn")
        profiles = ("rtx6000_ada_48", "l40s_48", "pro6000_mig_48", "rtx5090_32")
        for loader_type in ("LoadUpscalerTensorrtModel", "AutoLoadRifeTensorrtModel"):
            with self.subTest(loader=loader_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shared = root / "shared-onnx"
                shared.mkdir()
                models = []
                for profile in profiles:
                    path = root / profile / "models"
                    path.mkdir(parents=True)
                    (path / "onnx").symlink_to(shared, target_is_directory=True)
                    models.append(path)
                barrier = context.Barrier(4)
                started, finish = context.Event(), context.Event()
                downloads = context.Value("i", 0)
                results = context.Queue()
                processes = [context.Process(target=profile_worker, args=(
                    str(path), loader_type, barrier, started, finish, downloads, results)) for path in models]
                try:
                    for process in processes:
                        process.start()
                    self.assertTrue(started.wait(10), "Workers never started their download")
                    self.assertFalse((shared / "shared-model.onnx").exists(), "Partial ONNX was published")
                    finish.set()
                    for process in processes:
                        process.join(10)
                        self.assertFalse(process.is_alive(), "Cross-profile download deadlocked")
                        self.assertEqual(process.exitcode, 0)
                    expected = (len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
                    self.assertEqual([results.get(timeout=3) for _ in processes], [expected] * 4)
                    self.assertEqual(downloads.value, 1)
                    self.assertEqual((shared / "shared-model.onnx").read_bytes(), PAYLOAD)
                    self.assertFalse(list(shared.glob("*.part.onnx")))
                    self.assertEqual(len({(path / "tensorrt").resolve() for path in models}), 4)
                finally:
                    finish.set()
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                        if process.pid is not None:
                            process.join(2)
                    results.close()

    def test_failed_download_is_never_published_and_next_attempt_recovers(self):
        module, _ = load_nodes()
        cache = module.trt_cache
        attempts = []

        def flaky(url, save_path):
            attempts.append(save_path)
            Path(save_path).write_bytes(PAYLOAD[:2048])
            if len(attempts) == 1:
                raise OSError("connection interrupted")
            Path(save_path).write_bytes(PAYLOAD)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.onnx"
            download = cache.atomic_downloader(flaky)
            with self.assertRaisesRegex(OSError, "interrupted"):
                download("fixture://model", target)
            self.assertFalse(target.exists())
            self.assertFalse(list(target.parent.glob("*.part.onnx")))
            download("fixture://model", target)
            self.assertEqual(target.read_bytes(), PAYLOAD)
            download("fixture://model", target)
            self.assertEqual(len(attempts), 2)

    def test_incomplete_legacy_cache_is_quarantined_before_upstream_exists_check(self):
        module, _ = load_nodes()
        calls = []

        def download(url, save_path):
            calls.append(save_path)
            Path(save_path).write_bytes(PAYLOAD)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "onnx/shared-model.onnx"
            target.parent.mkdir()
            # A large partial download can have a valid ONNX prefix. Without
            # a completion receipt it still must not pass the existence check.
            legacy = PAYLOAD[:2048]
            target.write_bytes(legacy)
            namespace = {"Path": Path, "download_file": download, "models_dir": directory}
            exec('''class Loader:
    FUNCTION = "load"
    def load(self, model, batch_size):
        path = Path(models_dir) / "onnx" / (model + ".onnx")
        if not path.exists():
            download_file(url="fixture://model", save_path=str(path))
        return (path.read_bytes(),)
''', namespace)
            dependencies = {"folder_paths": types.SimpleNamespace(models_dir=directory),
                            "nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={"LoadUpscalerTensorrtModel": namespace["Loader"]})}
            with patch.dict(sys.modules, dependencies):
                result = module.MMH3TensorRTLoader().load(
                    object(), "LoadUpscalerTensorrtModel", '{"model":"shared-model","batch_size":1}')
            self.assertEqual(result, (PAYLOAD,))
            self.assertEqual(len(calls), 1)
            backups = list(target.parent.glob(".shared-model.onnx.invalid-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), legacy)

    def test_same_size_corruption_and_interrupted_publication_are_recovered(self):
        module, _ = load_nodes()
        cache = module.trt_cache
        calls = []

        def download(url, save_path):
            calls.append(save_path)
            Path(save_path).write_bytes(PAYLOAD)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.onnx"
            wrapped = cache.atomic_downloader(download)
            wrapped("fixture://model", target)
            self.assertTrue(cache.valid_onnx(target))
            target.write_bytes(b"\x08" + b"x" * (len(PAYLOAD) - 1))
            self.assertFalse(cache.valid_onnx(target))
            wrapped("fixture://model", target)
            self.assertEqual(target.read_bytes(), PAYLOAD)
            cache.receipt_path(target).unlink()  # Crash between data/receipt publication.
            self.assertFalse(cache.valid_onnx(target))
            wrapped("fixture://model", target)
            self.assertTrue(cache.valid_onnx(target))
            self.assertEqual(len(calls), 3)

    def test_different_models_can_download_in_parallel(self):
        module, _ = load_nodes()
        cache = module.trt_cache
        together = threading.Barrier(2)

        def download(url, save_path):
            together.wait(timeout=3)
            Path(save_path).write_bytes(PAYLOAD)

        wrapped = cache.atomic_downloader(download)
        with tempfile.TemporaryDirectory() as directory, concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(wrapped, "fixture://model", Path(directory) / name)
                       for name in ("rife.onnx", "upscale.onnx")]
            for future in futures:
                future.result(timeout=5)

    def test_error_payload_is_rejected_and_temporary_file_is_removed(self):
        module, _ = load_nodes()
        cache = module.trt_cache

        def html_error(url, save_path):
            Path(save_path).write_bytes(b"<html>not a model</html>" * 256)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.onnx"
            with self.assertRaisesRegex(RuntimeError, "invalid ONNX"):
                cache.atomic_downloader(html_error)("fixture://model", target)
            self.assertFalse(target.exists())
            self.assertFalse(list(target.parent.glob("*.part.onnx")))


if __name__ == "__main__":
    unittest.main()
