"""Explicit stage boundaries inside ComfyUI's execution process."""

import gc
import fcntl
import itertools
import json
import logging
import time
from pathlib import Path

import torch
import comfy.model_management as mm

from .trt_cache import prepare_loader

_RELEASE_SEQUENCE = itertools.count(1)


def gpu_memory():
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"allocated_mib": round(torch.cuda.memory_allocated() / 2 ** 20),
            "reserved_mib": round(torch.cuda.memory_reserved() / 2 ** 20),
            "device_used_mib": round((total - free) / 2 ** 20)}


def release_models(stage, engine=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    before = gpu_memory()
    models_before = len(getattr(mm, "current_loaded_models", []))
    # Optional TensorRT engines are not registered with ComfyUI's model manager.
    if engine is not None:
        engine.reset()
        engine.cuda_graph_instance = None
        engine.engine = None
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    stats = {"stage": stage, "time": time.time(), "sequence": next(_RELEASE_SEQUENCE),
             "models_before": models_before,
             "models_after": len(getattr(mm, "current_loaded_models", [])),
             **gpu_memory(), **{f"before_{key}": value for key, value in before.items()}}
    if torch.cuda.is_available():
        stats["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        torch.cuda.reset_peak_memory_stats()
    logging.info("[mmh3-memory] %s", stats)
    return stats


class MMH3StageBarrier:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",), "stage": ("STRING",)},
                "optional": {"dependency": ("*",), "release": ("*",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/memory"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, value, stage, dependency=None, release=None):
        stats = release_models(stage, release)
        return {"ui": {"mmh3_memory": [stats]}, "result": (value,)}


class MMH3StageGate:
    @classmethod
    def INPUT_TYPES(cls):
        # Both inputs must be strong links. With --cache-none a lazy value
        # would be requested after its first consumer discarded the result,
        # re-executing the sampler. Only this gate's execution is delayed.
        return {"required": {"dependency": ("*",), "value": ("*",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/memory"

    def run(self, dependency, value):
        return (value,)


class MMH3TensorRTLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dependency": ("*",),
            "loader_type": (["AutoLoadRifeTensorrtModel", "LoadUpscalerTensorrtModel"],),
            "settings": ("STRING",),
        }}

    RETURN_TYPES = ("*",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/memory"

    def load(self, dependency, loader_type, settings):
        import folder_paths
        import nodes

        if loader_type not in ("AutoLoadRifeTensorrtModel", "LoadUpscalerTensorrtModel"):
            raise ValueError("Unsupported TensorRT loader")
        cls = nodes.NODE_CLASS_MAPPINGS.get(loader_type)
        if cls is None:
            raise RuntimeError(f"{loader_type} is unavailable; check TensorRT custom-node imports")
        directory = Path(folder_paths.models_dir) / "tensorrt"
        directory.mkdir(parents=True, exist_ok=True)
        loader = getattr(cls(), cls.FUNCTION)
        settings = json.loads(settings)
        prepare_loader(loader, folder_paths.models_dir, settings)
        # Engine builds remain isolated by GPU. ONNX downloads use a separate
        # per-file lock in the shared ONNX directory and atomic publication.
        with (directory / ".build.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return loader(**settings)


NODE_CLASS_MAPPINGS = {"MMH3StageBarrier": MMH3StageBarrier,
                       "MMH3StageGate": MMH3StageGate,
                       "MMH3TensorRTLoader": MMH3TensorRTLoader}


def register_release_route():
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/mmh3/release")
    async def release(request):
        running, pending = PromptServer.instance.prompt_queue.get_current_queue()
        if running or pending:
            return web.json_response({"error": "ComfyUI is still busy"}, status=409)
        return web.json_response(release_models("between_jobs"))


register_release_route()
