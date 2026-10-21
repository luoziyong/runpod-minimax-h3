"""Run the pinned ComfyUI scheduler/NullCache with CPU stand-ins for GPU work.

The graph traversal, lazy promotion and lifetime of consumer-held outputs are
upstream code. MMH3 gates/barriers are production code. CUDA kernels and model
allocations are stand-ins: this checks scheduling/release order, not VRAM size.
"""

import ast
import asyncio
import collections
import hashlib
import inspect
import json
import types
from pathlib import Path

from test_memory_nodes import load_nodes
from workflow_runtime import CONDITIONERS, MEDIA_OUTPUTS, is_link

FIXTURE = Path(__file__).parent / "fixtures/comfyui_v0_38_0"


def load_scheduler():
    source = json.loads((FIXTURE / "SOURCE.json").read_text())
    namespace = {"asyncio": asyncio, "inspect": inspect,
                 "ExecutionBlocker": object, "is_link": is_link}
    selected = {"graph.py": {"DependencyCycleError", "NodeInputError", "NodeNotFoundError",
                            "DynamicPrompt", "get_input_info", "TopologicalSort", "ExecutionList"},
                "caching.py": {"NullCache"}}
    for name, names in selected.items():
        data = (FIXTURE / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != source["files"][name]["sha256"]:
            raise AssertionError(f"Pinned ComfyUI source changed: {name}")
        tree = ast.parse(data)
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        body += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in names]
        # Keep the upstream implementations unchanged; defer annotations for
        # Python 3.9 hosts and omit imports that would load torch/CUDA.
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                     str(FIXTURE / name), "exec"), namespace)
    return namespace


class Engine:
    def __init__(self, kind):
        self.kind = kind
        self.engine = object()
        self.context = object()
        self.cuda_graph_instance = object()

    def reset(self):
        self.context = None


class SchedulerHarness:
    def __init__(self, legacy_lazy_gate=False):
        self.scheduler = load_scheduler()
        self.memory, self.manager = load_nodes()
        self.manager.current_loaded_models = []
        self.manager.unload_all_models.side_effect = self.manager.current_loaded_models.clear
        self.legacy_lazy_gate = legacy_lazy_gate

    async def run(self, graph):
        memory = self.memory
        mapping = dict(memory.NODE_CLASS_MAPPINGS)
        metadata = {}
        for node in graph.values():
            ct = node["class_type"]
            if ct in mapping:
                continue
            fields = metadata.setdefault(ct, {})
            for key in node["inputs"]:
                lazy = ct == "MiniMaxH3DirectorCS" and key in ("model", "model_ref2va")
                fields[key] = ("*", {"lazy": True}) if lazy else ("*",)

        def definition(ct, fields):
            return type(ct, (), {"INPUT_TYPES": classmethod(lambda cls: {"required": fields}),
                                "FUNCTION": "run", "run": lambda self, **kwargs: None,
                                "OUTPUT_NODE": ct in MEDIA_OUTPUTS or ct in {"AutoRifeTensorrt", "PixaromaShowText"}})

        mapping.update({ct: definition(ct, fields) for ct, fields in metadata.items()})
        if self.legacy_lazy_gate:
            class LegacyGate(memory.MMH3StageGate):
                @classmethod
                def INPUT_TYPES(cls):
                    return {"required": {"dependency": ("*",), "value": ("*", {"lazy": True})}}

                def check_lazy_status(self, dependency, value=None):
                    return ["value"] if value is None else []

            mapping["MMH3StageGate"] = LegacyGate
        self.scheduler["nodes"] = types.SimpleNamespace(NODE_CLASS_MAPPINGS=mapping)
        execution = self.scheduler["ExecutionList"](
            self.scheduler["DynamicPrompt"](graph), self.scheduler["NullCache"]())
        for nid, node in graph.items():
            if getattr(mapping[node["class_type"]], "OUTPUT_NODE", False):
                execution.add_node(nid)
        events, records, engines, latents = [], [], [], {}
        calls = collections.Counter()
        for _ in range(1000):
            if execution.is_empty():
                return {"calls": calls, "events": events, "stages": records,
                        "engines": engines, "latents": latents}
            nid, error, exc = await execution.stage_node_execution()
            if error is not None:
                raise exc
            node, args, missing = graph[nid], {}, []
            ct = node["class_type"]
            cls = mapping[ct]
            for key, value in node["inputs"].items():
                if not is_link(value):
                    args[key] = value
                    continue
                cached = execution.get_cache(value[0], nid)
                if cached is None:
                    _, _, options = self.scheduler["get_input_info"](cls, key)
                    if not options.get("lazy", False):
                        raise AssertionError(f"Missing strong dependency: {nid}.{key}")
                    missing.append(key)
                    args[key] = None
                else:
                    args[key] = cached.outputs[value[1]][0]
            if missing:
                if ct == "MMH3StageGate":
                    requested = cls().check_lazy_status(**args)
                elif ct == "MiniMaxH3DirectorCS":
                    requested = missing  # The compiler retained only the selected model socket.
                else:
                    raise AssertionError(f"Unhandled lazy node {ct}")
                for key in requested:
                    execution.make_input_strong_link(nid, key)
                execution.unstage_node_execution()
                continue
            calls[ct] += 1
            events.append(ct)
            if ct == "MMH3StageBarrier":
                result = cls().run(**args)
                records.extend(result["ui"]["mmh3_memory"])
                events.append(args["stage"])
                outputs = result["result"]
            elif ct == "MMH3StageGate":
                outputs = cls().run(**args)
            elif ct == "MMH3TensorRTLoader":
                kind = "rife" if "Rife" in args["loader_type"] else "upscale"
                engine = Engine(kind)
                engines.append(engine)
                events.append("load_" + kind)
                outputs = (engine,)
            else:
                outputs = tuple(object() for _ in range(12))
                resident = ("encoder" if ct in CONDITIONERS else "dit" if ct == "SamplerCustomAdvanced"
                            else "video_vae" if ct == "VAEDecodeTiled" else "audio_vae" if ct == "VAEDecodeAudio" else None)
                if resident:
                    self.manager.current_loaded_models.append(resident)
                if ct in {"VAEDecodeTiled", "VAEDecodeAudio"}:
                    latents[ct] = args["samples"]
            execution.cache_update(nid, types.SimpleNamespace(outputs=[[value] for value in outputs]))
            execution.complete_node_execution()
        raise AssertionError("Pinned scheduler did not finish")
