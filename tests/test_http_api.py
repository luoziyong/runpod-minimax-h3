import asyncio
import importlib.util
import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import http_api

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts/smoke_test.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class HttpApiTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        async def execute(job):
            self.calls.append(job)
            if job["input"].get("action") == "health":
                return {"status": "ok", "comfyui": "ready"}
            if job["input"].get("fail"):
                raise ValueError("invalid fixture request")
            await asyncio.sleep(job["input"].get("delay", 0.01))
            return {"outputs": [{"filename": "fixture.mp4", "b64": "Zml4dHVyZQ=="}],
                    "execution": {"profile": "rtx6000_ada_48", "memory_stages": [
                        {"stage": name, "sequence": index + 1, "models_after": 0}
                        for index, name in enumerate(("after_conditioning", "after_sampling",
                                                     "after_video_decode", "after_decode"))]}}

        self.mock_worker = patch.object(http_api.worker, "handler", side_effect=execute)
        self.mock_worker.start()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.server = http_api.create_server(self.loop, "fixture-key", host="127.0.0.1", port=0)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
        self.loop.close()
        self.mock_worker.stop()

    def call(self, path, payload=None, authorized=True):
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer fixture-key"
        body = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(self.base + path, data=body, headers=headers), timeout=5) as response:
            return json.load(response)

    def test_health_and_sync_protocol(self):
        self.assertEqual(self.call("/health")["comfyui"], "ready")
        result = self.call("/runsync", {"input": {"preset": "quality"}})
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(self.call("/status/" + result["id"])["status"], "COMPLETED")

    def test_unauthorized_request_never_reaches_worker(self):
        with self.assertRaises(HTTPError) as caught:
            self.call("/runsync", {"input": {}}, authorized=False)
        self.assertEqual(caught.exception.code, 401)
        self.assertFalse(self.calls)

    def test_async_job_can_be_cancelled(self):
        result = self.call("/run", {"input": {"delay": 20}})
        self.assertEqual(result["status"], "IN_QUEUE")
        result = self.call("/cancel/" + result["id"], {})
        self.assertEqual(result["status"], "CANCELLED")
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.02), self.loop).result()

    def test_failures_are_reported_without_losing_job_id(self):
        result = self.call("/run", {"input": {"fail": True}})
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.02), self.loop).result()
        result = self.call("/status/" + result["id"])
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("fixture", result["error"])

    def test_sync_failures_keep_job_id_and_remain_queryable(self):
        result = self.call("/runsync", {"input": {"fail": True}})
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(self.call("/status/" + result["id"]), result)

    def test_gpu_smoke_client_uses_async_protocol_and_requires_stage_evidence(self):
        with patch.object(smoke.time, "sleep"):
            output = smoke.run_job(self.base, "fixture-key", {"preset": "quality"}, timeout=5)
        self.assertTrue(output["outputs"])
        self.assertEqual(len(output["execution"]["memory_stages"]), 4)

    def test_api_key_is_required(self):
        with self.assertRaises(ValueError):
            http_api.create_server(self.loop, "", host="127.0.0.1", port=0)


if __name__ == "__main__":
    unittest.main()
