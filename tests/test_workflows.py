import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import handler
import worker_config as config
from workflow_runtime import compile_workflow, is_link, validate_model_files

ROOT = Path(__file__).resolve().parents[1]


def ancestors(graph, nid, visiting=None):
    visiting = set() if visiting is None else visiting
    if nid in visiting:
        raise AssertionError(f"Cycle at {nid}")
    found = set()
    for value in graph[nid].get("inputs", {}).values():
        if is_link(value):
            found.add(value[0])
            found.update(ancestors(graph, value[0], visiting | {nid}))
    return found


def build(name, profile="rtx6000_ada_48", preset="quality", **overrides):
    runtime = config.build_runtime(profile, environ={"MMH3_DOWNLOAD_SINGULARITY": "true",
                                                    "MMH3_DOWNLOAD_ENHANCER": "true"})
    job = {"workflow": name, "prompt": "A cinematic shot.", "preset": preset,
           "images": ["aW1hZ2U="] * 3, "seed": 123, **overrides}
    job = config.normalize_job(job, runtime)
    graph = json.loads((ROOT / "workflows-api/mmh3" / name).read_text())
    with patch.object(handler, "_upload", side_effect=lambda name, data: name):
        graph = handler._patch(graph, job)
    graph = handler._apply_node_params(graph, job.get("params"))
    compiled, execution = compile_workflow(graph, job, runtime)
    return compiled, execution, runtime


class WorkflowTests(unittest.TestCase):
    def test_every_shipped_graph_on_every_profile_and_preset(self):
        for path in (ROOT / "workflows-api/mmh3").glob("*.json"):
            for profile in config.PROFILES:
                for preset in config.PRESETS:
                    with self.subTest(workflow=path.name, profile=profile, preset=preset):
                        graph, execution, runtime = build(path.name, profile, preset)
                        unets = [n for n in graph.values() if n["class_type"] == "UNETLoader"]
                        loras = [n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
                        self.assertEqual(len(unets), 1)
                        self.assertEqual(len(loras), int(preset == "speed"))
                        self.assertFalse(any(n["class_type"].startswith("QwenVL") for n in graph.values()))
                        self.assertFalse(any("Tensorrt" in n["class_type"] for n in graph.values()))
                        self.assertFalse(any(n["class_type"] in ("BlockSparseAttention", "ModelAttentionBackend")
                                             for n in graph.values()))
                        available = {r["filename"] for r in config.selected_models(runtime)}
                        for nid, node in graph.items():
                            ancestors(graph, nid)  # Includes a dangling-link and cycle check.
                            for key in ("unet_name", "clip_name", "vae_name", "lora_name"):
                                if key in node["inputs"]:
                                    self.assertIn(node["inputs"][key], available)
                        schedulers = [n["inputs"] for n in graph.values() if n["class_type"] == "BasicScheduler"]
                        expected_steps = (20 if preset == "quality" else
                                          6 if "Singularity" in path.name else
                                          4 if execution["task"] == "ref2va" else 8)
                        self.assertEqual(execution["steps"], expected_steps)
                        self.assertTrue(all(s["steps"] == expected_steps for s in schedulers))
                        self.assertEqual(execution["encoder"], config.encoder_filename(runtime))

    def test_default_reference_path_is_official_not_singularity(self):
        graph, execution, _ = build("MiniMaxH3-Turbo-R2VA.json")
        self.assertEqual(execution["dit"], config.REF2VA)
        conditioner = next(n for n in graph.values() if n["class_type"] == "MiniMaxH3ReferenceToVideo")
        self.assertEqual(conditioner["inputs"]["ref_image_size"], "match")

    def test_director_off_selects_fl2va_and_its_own_turbo_schedule(self):
        graph, execution, _ = build("MiniMaxH3-Director-Native.json", preset="speed", reference_mode="OFF")
        self.assertEqual(execution["dit"], config.FL2VA)
        self.assertEqual(execution["steps"], 8)
        director = next(n["inputs"] for n in graph.values() if n["class_type"] == "MiniMaxH3DirectorCS")
        self.assertNotIn("model_ref2va", director)
        self.assertEqual(director["shift_video"], 6)

    def test_stages_are_dependencies_not_just_unload_options(self):
        for post in ({}, {"upscale": True}, {"rife": True}, {"upscale": True, "rife": True}):
            for name in ("MiniMaxH3-Turbo-T2VA.json", "MiniMaxH3-Director-Storyboard.json"):
                with self.subTest(post=post, name=name):
                    graph, _, _ = build(name, **post)
                    sampler = next(nid for nid, n in graph.items() if n["class_type"] == "SamplerCustomAdvanced")
                    video = next(nid for nid, n in graph.items() if n["class_type"] == "VAEDecodeTiled")
                    audio = next(nid for nid, n in graph.items() if n["class_type"] == "VAEDecodeAudio")
                    self.assertIn("mmh3_after_conditioning", ancestors(graph, sampler))
                    self.assertIn("mmh3_after_sampling", ancestors(graph, video))
                    self.assertIn("mmh3_after_video_decode", ancestors(graph, audio))
                    for nid, node in graph.items():
                        ancestors(graph, nid)
                        if node["class_type"] == "MMH3TensorRTLoader":
                            self.assertIn("mmh3_after_decode", ancestors(graph, nid))
                            self.assertEqual(json.loads(node["inputs"]["settings"])["batch_size"], 1)
                        if node["class_type"] == "CreateVideo":
                            self.assertEqual(node["inputs"]["fps"], 48 if post.get("rife") else 24)
                        if node["class_type"] == "VHS_VideoCombine":
                            self.assertEqual(node["inputs"]["frame_rate"], 48 if post.get("rife") else 24)
                    if post.get("upscale") and post.get("rife"):
                        self.assertIn("mmh3_after_upscale", ancestors(graph, "mmh3_rife_loader"))
                    self.assertIn("mmh3_after_decode", graph)

    def test_enhancer_is_unloaded_and_generic_params_cannot_enable_residency(self):
        graph, _, _ = build("MiniMaxH3-Turbo-T2VA.json", enhance=True,
                            params={"105:256:keep_model_loaded": True})
        enhancer = next(n for n in graph.values() if n["class_type"].startswith("QwenVL"))
        self.assertFalse(enhancer["inputs"]["keep_model_loaded"])

    def test_5090_uses_smaller_decode_tiles(self):
        graph, execution, _ = build("MiniMaxH3-Turbo-T2VA.json", "rtx5090_32")
        self.assertEqual(execution["video_size"], "864x480")
        decode = next(n["inputs"] for n in graph.values() if n["class_type"] == "VAEDecodeTiled")
        self.assertEqual(decode["tile_size"], 384)
        self.assertEqual(decode["temporal_size"], 32)

    def test_model_preflight_checks_only_reachable_selected_files(self):
        graph, _, runtime = build("MiniMaxH3-Turbo-T2VA.json")
        with tempfile.TemporaryDirectory() as directory:
            runtime = dict(runtime, models_dir=directory)
            with self.assertRaisesRegex(FileNotFoundError, config.FL2VA):
                validate_model_files(graph, runtime, directory)
            for row in config.selected_models(runtime):
                if row["filename"] == config.REF2VA:
                    continue  # This job does not need the alternative DiT.
                file = Path(directory) / row["subdir"] / row["filename"]
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"fixture")
            validate_model_files(graph, runtime, directory)

    def test_director_timeline_exact_duration_and_no_implicit_llm(self):
        job = {"prompt": "shot one\nshot two\nshot three", "seconds": 5,
               "images": ["aW1hZ2U="], "enhance": False, "analyze": False}
        with patch.object(handler, "_comfy_route") as route:
            timeline, duration = handler._director_timeline(job)
        self.assertEqual(duration, 120)
        segments = json.loads(timeline)["segments"]
        self.assertEqual(sum(s["length"] for s in segments), 120)
        self.assertEqual(segments[-1]["start"] + segments[-1]["length"], 120)
        route.assert_not_called()

    def test_singularity_style_is_inserted_even_when_template_has_no_style_slot(self):
        runtime = config.build_runtime("rtx6000_ada_48", environ={
            "MMH3_DOWNLOAD_SINGULARITY": "true", "MMH3_DOWNLOAD_STYLE_LORAS": "true"})
        job = config.normalize_job({"prompt": "A cinematic shot.", "preset": "speed",
                                    "model_variant": "singularity", "lora": "realism"}, runtime)
        raw = json.loads((ROOT / "workflows-api/mmh3/MiniMaxH3-Turbo-R2VA.json").read_text())
        graph = handler._patch(raw, job)
        graph, execution = compile_workflow(graph, job, runtime)
        self.assertEqual(execution["dit"], config.SINGULARITY)
        self.assertEqual(execution["steps"], 6)
        loras = [n["inputs"]["lora_name"] for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
        self.assertEqual(set(loras), {config.REF2VA_LORA, "h3-realism-people-t2v-i2v-r2v.safetensors"})

    def test_unstaged_sampler_outputs_are_rejected_before_queueing(self):
        runtime = config.build_runtime("rtx6000_ada_48", environ={})
        job = config.normalize_job({"prompt": "Example"}, runtime)
        raw = json.loads((ROOT / "workflows-api/mmh3/MiniMaxH3-Turbo-T2VA.json").read_text())
        raw = handler._patch(raw, job)
        sampler = next(nid for nid, node in raw.items() if node["class_type"] == "SamplerCustomAdvanced")
        audio = next(node for node in raw.values() if node["class_type"] == "VAEDecodeAudio")
        audio["inputs"]["samples"] = [sampler, 1]
        with self.assertRaisesRegex(ValueError, "sampler output 0"):
            compile_workflow(raw, job, runtime)


if __name__ == "__main__":
    unittest.main()
