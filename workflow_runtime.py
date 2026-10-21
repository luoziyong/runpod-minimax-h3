"""Compile exported H3 graphs into a single-DiT, explicitly staged workflow."""

import copy
import json
from pathlib import Path

from worker_config import (AUDIO_VAE, FL2VA, FL2VA_LORA, PRESETS, REF2VA,
                           REF2VA_LORA, SINGULARITY, VIDEO_VAE, encoder_filename)

MEDIA_OUTPUTS = {"SaveVideo", "VHS_VideoCombine", "SaveImage", "SaveAudio",
                 "SaveAudioMP3", "SaveAudioOpus", "SaveAudioAdvanced"}
CONDITIONERS = {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "MiniMaxH3DirectorCS"}
CORE_MEMORY_STAGES = ("after_conditioning", "after_sampling", "after_video_decode", "after_decode")


def validate_memory_stages(records, execution):
    expected = list(CORE_MEMORY_STAGES)
    expected += [f"after_{name}" for name in ("upscale", "rife") if execution.get(name)]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise RuntimeError("Missing memory-stage execution evidence")
    if [row.get("stage") for row in records] != expected:
        raise RuntimeError(f"Unexpected memory stages: expected {expected}")
    sequences = [row.get("sequence") for row in records]
    if (not all(isinstance(value, int) and not isinstance(value, bool) for value in sequences)
            or sequences != list(range(sequences[0], sequences[0] + len(expected)))):
        # ComfyUI history overwrites UI for a re-executed node. Sequence gaps
        # or inversions still reveal a repeated barrier instead of hiding it.
        raise RuntimeError("Non-consecutive memory-stage sequence; a stage may have re-executed")
    if any(row.get("models_after") != 0 for row in records):
        raise RuntimeError("ComfyUI models remain loaded after a memory-stage release")


def is_link(value):
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and isinstance(value[1], int))


def replace_link(graph, old, new, exclude=()):
    for nid, node in graph.items():
        if nid not in exclude:
            for key, value in node.get("inputs", {}).items():
                if value == old:
                    node["inputs"][key] = copy.deepcopy(new)


def bypass(graph, nid, input_key="model"):
    source = graph[nid]["inputs"][input_key]
    replace_link(graph, [nid, 0], source, exclude=(nid,))
    del graph[nid]


def prune(graph):
    roots = [nid for nid, node in graph.items() if node["class_type"] in MEDIA_OUTPUTS]
    if not roots:
        raise ValueError("H3 workflow needs a media output node (SaveVideo or VHS_VideoCombine)")
    seen = set()

    def visit(nid):
        if nid in seen:
            return
        if nid not in graph:
            raise ValueError(f"Workflow contains a dangling link to {nid}")
        seen.add(nid)
        for value in graph[nid].get("inputs", {}).values():
            if is_link(value):
                visit(value[0])

    for nid in roots:
        visit(nid)
    return {nid: node for nid, node in graph.items() if nid in seen}


def compile_workflow(prompt, job, runtime):
    graph = copy.deepcopy(prompt)
    if any(nid.startswith("mmh3_") for nid in graph):
        raise ValueError("mmh3_ node IDs are reserved for runtime staging")
    conditioners = [(nid, n) for nid, n in graph.items() if n["class_type"] in CONDITIONERS]
    if len(conditioners) != 1:
        raise ValueError("A staged job must contain exactly one H3 conditioning/Director node")
    cond_id, conditioner = conditioners[0]
    director = conditioner["class_type"] == "MiniMaxH3DirectorCS"
    inp = conditioner["inputs"]
    reference = conditioner["class_type"] == "MiniMaxH3ReferenceToVideo"
    if director:
        timeline = json.loads(inp.get("timeline_data") or "{}")
        mode = job.get("reference_mode", timeline.get("reference_mode", "REF2VA"))
        reference = not str(mode).lower().startswith("off")
        timeline["reference_mode"] = "REF2VA" if reference else "OFF"
        inp["timeline_data"] = json.dumps(timeline)
        # Removing the socket also protects old Director builds without lazy inputs.
        inp.pop("model" if reference else "model_ref2va", None)
    variant = job["model_variant"]
    if variant == "singularity" and not reference:
        raise ValueError("The optional Singularity recipe is supported for reference generation only")
    schedule = dict(PRESETS[job["preset"]])
    if job["preset"] == "speed" and reference:
        schedule.update(steps=4, shift_video=12.0)
        if variant == "singularity":
            schedule.update(steps=6, scheduler="beta")
    turbo = job["preset"] == "speed"

    # These optional patches can change precision/attention behavior. Keep the
    # presets on core PyTorch attention and change only the sampling schedule.
    for nid, node in list(graph.items()):
        if node["class_type"] in ("BlockSparseAttention", "ModelAttentionBackend", "MiniMaxH3SigmaShift"):
            bypass(graph, nid)
    for nid, node in list(graph.items()):
        args = node.get("inputs", {})
        if "lora_name" in args:
            bypass(graph, nid)
        elif node["class_type"].startswith("QwenVL"):
            args["keep_model_loaded"] = False
            if not job["enhance"]:
                bypass(graph, nid, "prompt_text" if "prompt_text" in args else "prompt")
        elif "keep_model_loaded" in args:
            args["keep_model_loaded"] = False

    for nid, node in list(graph.items()):
        args = node.setdefault("inputs", {})
        ct = node["class_type"]
        if ct == "CLIPLoader":
            args.update(clip_name=encoder_filename(runtime), type="minimax", device="default")
        elif ct == "VAELoader":
            args["vae_name"] = AUDIO_VAE if "audio" in args["vae_name"].lower() else VIDEO_VAE
        elif ct == "UNETLoader":
            is_ref = reference if not director else any(
                token in args["unet_name"].lower() for token in ("ref2va", "singularity"))
            args["unet_name"] = (SINGULARITY if variant == "singularity" else REF2VA) if is_ref else FL2VA
            args["weight_dtype"] = "default"
            source = [nid, 0]
            added = []
            if turbo:
                lora_id = f"mmh3_turbo_{nid}"
                graph[lora_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                    "model": source, "lora_name": REF2VA_LORA if is_ref else FL2VA_LORA,
                    "strength_model": 1.0}}
                source = [lora_id, 0]
                added.append(lora_id)
            if job.get("lora", "none") != "none":
                style_id = f"mmh3_style_{nid}"
                filename = {"realism": "h3-realism-people-t2v-i2v-r2v.safetensors",
                            "char_swap": "h3_character_swap_pro4500_1000.safetensors"}[job["lora"]]
                graph[style_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                    "model": source, "lora_name": filename, "strength_model": 1.0}}
                source = [style_id, 0]
                added.append(style_id)
            if not director and variant != "singularity":
                shift_id = f"mmh3_shift_{nid}"
                graph[shift_id] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
                    "model": source, "shift_video": schedule["shift_video"],
                    "shift_audio": schedule["shift_audio"]}}
                source = [shift_id, 0]
                added.append(shift_id)
            replace_link(graph, [nid, 0], source, exclude=(nid, *added))
        elif ct == "KSamplerSelect":
            args["sampler_name"] = schedule["sampler"]
        elif ct == "BasicScheduler":
            args.update(steps=schedule["steps"], scheduler=schedule["scheduler"])
        if ct in CONDITIONERS:
            if "ref_image_size" in args:
                args["ref_image_size"] = "match"
            if director:
                args.update(shift_video=schedule["shift_video"], shift_audio=schedule["shift_audio"])

    def barrier(name, value, dependency=None, release=None):
        nid = f"mmh3_{name}"
        args = {"value": value, "stage": name}
        if dependency is not None:
            args["dependency"] = dependency
        if release is not None:
            args["release"] = release
        graph[nid] = {"class_type": "MMH3StageBarrier", "inputs": args}
        return [nid, 0]

    def gate(name, dependency, value):
        nid = f"mmh3_{name}"
        graph[nid] = {"class_type": "MMH3StageGate", "inputs": {
            "dependency": dependency, "value": value}}
        return [nid, 0]

    cond_slot, latent_slot = (1, 2) if director else (0, 1)
    encoded = barrier("after_conditioning", [cond_id, cond_slot], [cond_id, latent_slot])
    replace_link(graph, [cond_id, cond_slot], encoded, exclude=(encoded[0],))
    samplers = [nid for nid, n in graph.items() if n["class_type"] == "SamplerCustomAdvanced"]
    if len(samplers) != 1:
        raise ValueError("A staged job must contain exactly one SamplerCustomAdvanced")
    sampled = barrier("after_sampling", [samplers[0], 0])
    replace_link(graph, [samplers[0], 0], sampled, exclude=(sampled[0],))
    video = [nid for nid, n in graph.items() if n["class_type"] in ("VAEDecode", "VAEDecodeTiled")]
    audio = [nid for nid, n in graph.items() if n["class_type"] == "VAEDecodeAudio"]
    if len(video) != 1 or len(audio) != 1:
        raise ValueError("A staged H3 workflow needs one video decoder and one audio decoder")
    video_id, audio_id = video[0], audio[0]
    graph[video_id]["class_type"] = "VAEDecodeTiled"
    graph[video_id]["inputs"].update(tile_size=runtime["tile_size"], overlap=64,
                                     temporal_size=32 if runtime["profile"] == "rtx5090_32" else 64,
                                     temporal_overlap=8)
    decoded_video = barrier("after_video_decode", [video_id, 0])
    audio_samples = graph[audio_id]["inputs"]["samples"]
    graph[audio_id]["inputs"]["samples"] = gate("audio_after_video", decoded_video, audio_samples)
    current_images = barrier("after_decode", decoded_video, [audio_id, 0])

    for name, ct, image_key, model_key in (
            ("upscale", "UpscalerTensorrt", "images", "upscaler_trt_model"),
            ("rife", "AutoRifeTensorrt", "frames", "rife_trt_model")):
        candidates = [(nid, n) for nid, n in graph.items() if n["class_type"] == ct]
        if not job[name]:
            continue
        if len(candidates) != 1:
            raise ValueError(f"{name}=true needs exactly one {ct} node")
        nid, node = candidates[0]
        args = node["inputs"]
        original_loader = args[model_key]
        loader_node = graph[original_loader[0]]
        loader_settings = dict(loader_node["inputs"], batch_size=1)
        loader_id = f"mmh3_{name}_loader"
        # Defer both engine compilation and deserialization until decode ends.
        graph[loader_id] = {"class_type": "MMH3TensorRTLoader", "inputs": {
            "dependency": current_images, "loader_type": loader_node["class_type"],
            "settings": json.dumps(loader_settings)}}
        model = [loader_id, 0]
        args[image_key], args[model_key], args["batch_size"] = current_images, model, 1
        if name == "rife":
            args["keep_model_loaded"] = False
        current_images = barrier(f"after_{name}", [nid, 0], release=model)

    for node in graph.values():
        ct, args = node["class_type"], node.get("inputs", {})
        if ct in ("CreateVideo", "VHS_VideoCombine"):
            args["images"] = current_images
            fps = 24
            if job["rife"]:
                rife = next(n for n in graph.values() if n["class_type"] == "AutoRifeTensorrt")
                fps *= int(rife["inputs"].get("multiplier", 2))
            args["fps" if ct == "CreateVideo" else "frame_rate"] = fps

    graph = prune(graph)
    if any(f"mmh3_{stage}" not in graph for stage in CORE_MEMORY_STAGES):
        raise ValueError("The output must depend on every H3 memory stage")
    for node in graph.values():
        for value in node.get("inputs", {}).values():
            if is_link(value) and value[0] == samplers[0] and value[1] != 0:
                raise ValueError("Staged H3 jobs support sampler output 0 only")
    unets = [n["inputs"]["unet_name"] for n in graph.values() if n["class_type"] == "UNETLoader"]
    if len(unets) != 1:
        raise ValueError(f"Expected one reachable DiT after compilation, found {len(unets)}")
    return graph, {"profile": runtime["profile"], "preset": job["preset"],
                   "task": "ref2va" if reference else "fl2va", "dit": unets[0],
                   "encoder": encoder_filename(runtime), "steps": schedule["steps"],
                   "video_size": job["video_size"], "seconds": job["seconds"],
                   "staged_loading": True, "tiled_decode": True,
                   "enhance": job["enhance"], "upscale": job["upscale"], "rife": job["rife"]}


def validate_model_files(graph, runtime, comfy_dir):
    missing = []
    for node in graph.values():
        args, ct = node.get("inputs", {}), node["class_type"]
        lookup = {"UNETLoader": ("diffusion_models", "unet_name"),
                  "CLIPLoader": ("text_encoders", "clip_name"),
                  "VAELoader": ("vae", "vae_name"),
                  "LoraLoaderModelOnly": ("loras", "lora_name")}
        if ct in lookup:
            subdir, key = lookup[ct]
            name = args[key]
            roots = (Path(runtime["models_dir"]), Path(comfy_dir) / "models")
            if not any((root / subdir / name).is_file() and (root / subdir / name).stat().st_size > 0
                       for root in roots):
                missing.append(f"{subdir}/{name}")
    if missing:
        raise FileNotFoundError("Missing selected model files: " + ", ".join(sorted(set(missing))))
