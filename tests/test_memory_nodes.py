import importlib.util
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def download_file(url, save_path):
    Path(save_path).write_bytes(b"\x08" + b"fixture" * 1024)


def load_nodes():
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = Mock()
    mm.soft_empty_cache = Mock()
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    routes = types.SimpleNamespace(post=lambda path: lambda function: function)
    server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes)))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    modules = {"comfy": comfy, "comfy.model_management": mm, "torch": torch,
               "server": server, "aiohttp": types.SimpleNamespace(web=types.SimpleNamespace())}
    path = Path(__file__).resolve().parents[1] / "custom_nodes/mmh3_memory/__init__.py"
    spec = importlib.util.spec_from_file_location("memory_nodes_test", path)
    module = importlib.util.module_from_spec(spec)
    modules[spec.name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, mm


class MemoryNodeTests(unittest.TestCase):
    def test_barrier_unloads_inside_comfy_process_and_passes_data_unchanged(self):
        module, mm = load_nodes()
        value, dependency = object(), object()
        result = module.MMH3StageBarrier().run(value, "after_sampling", dependency)
        self.assertIs(result["result"][0], value)
        self.assertEqual(result["ui"]["mmh3_memory"][0]["stage"], "after_sampling")
        mm.unload_all_models.assert_called_once()
        mm.soft_empty_cache.assert_called_once()

    def test_tensor_rt_resources_are_explicitly_released(self):
        module, _ = load_nodes()
        engine = types.SimpleNamespace(reset=Mock(), engine=object(), cuda_graph_instance=object())
        module.release_models("after_upscale", engine)
        engine.reset.assert_called_once()
        self.assertIsNone(engine.engine)
        self.assertIsNone(engine.cuda_graph_instance)

    def test_release_synchronizes_and_records_device_memory_before_and_after_unload(self):
        module, manager = load_nodes()
        events = []
        state = {"allocated": 24000, "reserved": 26000, "used": 27000}
        manager.current_loaded_models = ["encoder"]

        def unload():
            events.append("unload")
            manager.current_loaded_models.clear()
            state["allocated"] = 0

        def empty():
            events.append("empty_cache")
            state.update(reserved=0, used=600)

        manager.unload_all_models.side_effect = unload
        manager.soft_empty_cache.side_effect = empty
        module.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True, synchronize=lambda: events.append("synchronize"),
            mem_get_info=lambda: ((48000 - state["used"]) * 2 ** 20, 48000 * 2 ** 20),
            memory_allocated=lambda: state["allocated"] * 2 ** 20,
            memory_reserved=lambda: state["reserved"] * 2 ** 20,
            max_memory_allocated=lambda: 25000 * 2 ** 20,
            reset_peak_memory_stats=lambda: events.append("reset_peak")))
        with patch.object(module.gc, "collect", side_effect=lambda: events.append("gc")):
            record = module.release_models("after_conditioning")
        self.assertEqual(events, ["synchronize", "unload", "gc", "empty_cache", "reset_peak"])
        self.assertEqual((record["models_before"], record["models_after"]), (1, 0))
        self.assertEqual((record["before_allocated_mib"], record["allocated_mib"]), (24000, 0))
        self.assertEqual((record["before_device_used_mib"], record["device_used_mib"]), (27000, 600))
        self.assertEqual(record["peak_allocated_mib"], 25000)
        self.assertIsInstance(record["sequence"], int)

    def test_gate_retains_its_data_dependency_until_the_other_stage_finishes(self):
        module, _ = load_nodes()
        gate = module.MMH3StageGate()
        dependency, value = object(), object()
        self.assertFalse(gate.INPUT_TYPES()["required"]["value"][1:])
        self.assertIs(gate.run(dependency, value)[0], value)

    def test_workers_do_not_build_shared_tensor_rt_engine_concurrently(self):
        module, _ = load_nodes()
        state = {"active": 0, "peak": 0}
        guard = threading.Lock()
        errors = []

        class Loader:
            FUNCTION = "load"

            def load(self, model, batch_size):
                with guard:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.02)
                with guard:
                    state["active"] -= 1
                return (model,)

        with tempfile.TemporaryDirectory() as directory:
            modules = {"folder_paths": types.SimpleNamespace(models_dir=directory),
                       "nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={"LoadUpscalerTensorrtModel": Loader})}

            def run():
                try:
                    result = module.MMH3TensorRTLoader().load(
                        object(), "LoadUpscalerTensorrtModel", '{"model":"fixture","batch_size":1}')
                    self.assertEqual(result, ("fixture",))
                except Exception as exc:
                    errors.append(exc)

            with patch.dict(sys.modules, modules):
                workers = [threading.Thread(target=run) for _ in range(3)]
                for thread in workers:
                    thread.start()
                for thread in workers:
                    thread.join()
            self.assertFalse(errors)
            self.assertEqual(state["peak"], 1)


if __name__ == "__main__":
    unittest.main()
