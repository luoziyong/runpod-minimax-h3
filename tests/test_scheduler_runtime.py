import asyncio
import json
import re
import unittest
from pathlib import Path

import worker_config
from workflow_runtime import validate_memory_stages
from scheduler_harness import FIXTURE, SchedulerHarness
from test_workflows import build


class RuntimeSchedulingTests(unittest.TestCase):
    def test_offline_scheduler_matches_the_default_image_core_version(self):
        source = json.loads((FIXTURE / "SOURCE.json").read_text())
        dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
        self.assertEqual(re.search(r"ARG COMFYUI_REF=(\S+)", dockerfile).group(1), source["version"])

    def assert_pipeline(self, result, post):
        calls, events, records = result["calls"], result["events"], result["stages"]
        for node in ("SamplerCustomAdvanced", "VAEDecodeTiled", "VAEDecodeAudio"):
            self.assertEqual(calls[node], 1, (node, events))
        self.assertEqual(sum(calls[node] for node in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "MiniMaxH3DirectorCS")), 1)
        expected = ["after_conditioning", "after_sampling", "after_video_decode", "after_decode"]
        expected += ["after_" + name for name in ("upscale", "rife") if post.get(name)]
        self.assertEqual([row["stage"] for row in records], expected)
        validate_memory_stages(records, post)
        self.assertTrue(all(row["models_after"] == 0 for row in records))
        self.assertTrue(all(row["models_before"] == 1 for row in records[:4]))
        for release, load in (("after_conditioning", "SamplerCustomAdvanced"),
                              ("after_sampling", "VAEDecodeTiled"),
                              ("after_video_decode", "VAEDecodeAudio")):
            self.assertLess(events.index(release), events.index(load))
        for name in ("upscale", "rife"):
            if post.get(name):
                self.assertLess(events.index("after_decode"), events.index("load_" + name))
        if post.get("upscale") and post.get("rife"):
            self.assertLess(events.index("after_upscale"), events.index("load_rife"))
        self.assertIs(result["latents"]["VAEDecodeTiled"], result["latents"]["VAEDecodeAudio"])
        for engine in result["engines"]:
            self.assertIsNone(engine.engine)
            self.assertIsNone(engine.context)
            self.assertIsNone(engine.cuda_graph_instance)

    def test_all_workflows_profiles_presets_and_postprocessing_execute_once(self):
        harness = SchedulerHarness()
        for path in (Path(__file__).resolve().parents[1] / "workflows-api/mmh3").glob("*.json"):
            for profile in worker_config.PROFILES:
                for preset in worker_config.PRESETS:
                    for post in ({}, {"upscale": True}, {"rife": True}, {"upscale": True, "rife": True}):
                        with self.subTest(workflow=path.name, profile=profile, preset=preset, post=post):
                            graph, _, _ = build(path.name, profile, preset, **post)
                            self.assert_pipeline(asyncio.run(harness.run(graph)), post)

    def test_consecutive_director_jobs_and_enhancement_do_not_reuse_old_latents(self):
        harness = SchedulerHarness()
        prior = None
        for enhance in (False, True):
            for mode in ("OFF", "REF2VA"):
                graph, _, _ = build("MiniMaxH3-Director-Storyboard.json", preset="speed",
                                    reference_mode=mode, enhance=enhance, upscale=True, rife=True)
                for _ in range(2):
                    result = asyncio.run(harness.run(graph))
                    self.assert_pipeline(result, {"upscale": True, "rife": True})
                    current = result["latents"]["VAEDecodeAudio"]
                    self.assertIsNot(current, prior)
                    prior = current

    def test_regression_fixture_detects_the_previous_double_sampling_bug(self):
        graph, _, _ = build("MiniMaxH3-Turbo-T2VA.json")
        result = asyncio.run(SchedulerHarness(legacy_lazy_gate=True).run(graph))
        self.assertEqual(result["calls"]["SamplerCustomAdvanced"], 2)
        self.assertIsNot(result["latents"]["VAEDecodeTiled"], result["latents"]["VAEDecodeAudio"])
        with self.assertRaises(RuntimeError):
            validate_memory_stages(result["stages"], {})

    def test_overwritten_reexecuted_ui_and_missing_releases_fail_runtime_validation(self):
        names = ["after_conditioning", "after_sampling", "after_video_decode", "after_decode"]
        valid = [{"stage": name, "sequence": index + 10, "models_after": 0}
                 for index, name in enumerate(names)]
        validate_memory_stages(valid, {})
        invalid = [None, valid[:-1], valid + [valid[-1]],
                   [dict(row, sequence=seq) for row, seq in zip(valid, (14, 15, 12, 16))],
                   [dict(row, sequence=seq) for row, seq in zip(valid, (10, 11, 12, 14))],
                   [dict(row, models_after=1) for row in valid]]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(RuntimeError):
                validate_memory_stages(records, {})


if __name__ == "__main__":
    unittest.main()
