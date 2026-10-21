import copy
import unittest

import worker_config as config


class ProfileTests(unittest.TestCase):
    def test_gpu_detection_checks_visible_mig_capacity(self):
        cases = [("NVIDIA RTX 6000 Ada Generation", 48, [8, 9], "rtx6000_ada_48"),
                 ("NVIDIA L40S", 48, [8, 9], "l40s_48"),
                 ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 47, [12, 0], "pro6000_mig_48"),
                 ("NVIDIA GeForce RTX 5090", 32, [12, 0], "rtx5090_32")]
        for name, vram, cc, profile in cases:
            gpu = {"name": name, "vram_gib": vram, "capability": cc}
            with self.subTest(profile=profile):
                self.assertEqual(config.build_runtime("auto", gpu, {})["profile"], profile)
        with self.assertRaises(ValueError):
            config.build_runtime("auto", {"name": cases[2][0], "vram_gib": 96, "capability": [12, 0]}, {})
        with self.assertRaisesRegex(ValueError, "visible VRAM"):
            config.build_runtime("pro6000_mig_48", {"name": cases[2][0], "vram_gib": 24,
                                                    "capability": [12, 0]}, {})

    def test_wrong_architecture_is_rejected_before_model_download(self):
        with self.assertRaisesRegex(ValueError, "compute capability"):
            config.build_runtime("rtx6000_ada_48", {"name": "5090", "vram_gib": 48,
                                                   "capability": [12, 0]}, {})

    def test_each_manifest_has_one_encoder_and_no_optional_models_or_engines(self):
        for profile, spec in config.PROFILES.items():
            for family in ("official", "heretic"):
                runtime = config.build_runtime(profile, environ={"MMH3_ENCODER_FAMILY": family})
                rows = config.selected_models(runtime)
                encoders = [r for r in rows if r["subdir"] == "text_encoders"]
                self.assertEqual(len(encoders), 1)
                self.assertIn(spec["encoder"], encoders[0]["filename"])
                self.assertEqual(len(rows), 7)
                self.assertFalse(any(r["subdir"].startswith(("LLM", "tensorrt", "sams")) for r in rows))
                self.assertFalse(any("Singularity" in r["filename"] for r in rows))

    def test_presets_and_input_validation(self):
        runtime = config.build_runtime("rtx6000_ada_48", environ={})
        self.assertEqual(config.normalize_job({}, runtime)["video_size"], "1344x768")
        self.assertEqual(config.normalize_job({"preset": "speed"}, runtime)["video_size"], "864x480")
        for payload in ({"preset": "fast"}, {"seconds": True}, {"seconds": 16},
                        {"seconds": 1.2}, {"images": "url"}, {"video_size": "auto"},
                        {"video_size": "1376x768"}, {"video_size": "513x512"},
                        {"enhance": "false"}, {"enhance": True}, {"model_variant": "singularity"},
                        {"preset": "quality", "config": "native_turbo"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                config.normalize_job(payload, runtime)
        result = config.normalize_job({"config": "native_turbo"}, runtime)
        self.assertEqual(result["preset"], "speed")

    def test_trt_cache_isolated_by_gpu_slice_software_and_revision(self):
        gpu = {"name": "RTX PRO 6000", "vram_gib": 48, "capability": [12, 0], "torch": "test", "cuda": "13.0"}
        runtime = config.build_runtime("pro6000_mig_48", gpu, {})
        key = config.trt_cache_key(runtime)
        for field, replacement in (("name", "RTX 5090"), ("vram_gib", 24), ("torch", "new")):
            other = copy.deepcopy(runtime)
            other["gpu"][field] = replacement
            self.assertNotEqual(config.trt_cache_key(other), key)


if __name__ == "__main__":
    unittest.main()
