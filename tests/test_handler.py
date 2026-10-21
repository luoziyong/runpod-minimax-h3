import asyncio
import json
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import handler
from worker_config import build_runtime


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handler._JOB_LOCK = None

    async def test_gpu_jobs_are_serial_but_health_does_not_wait_for_the_lock(self):
        state = {"active": 0, "peak": 0}

        async def execute(job):
            if job["input"].get("action") == "health":
                return "healthy"
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.03)
            state["active"] -= 1
            return job["id"]

        with patch.object(handler, "_handle_job", side_effect=execute), \
                patch.object(handler, "_BOOT_DONE", threading.Event()):
            first = asyncio.create_task(handler.handler({"id": "one", "input": {}}))
            await asyncio.sleep(0.005)
            self.assertEqual(await handler.handler({"input": {"action": "health"}}), "healthy")
            second = asyncio.create_task(handler.handler({"id": "two", "input": {}}))
            self.assertEqual(await asyncio.gather(first, second), ["one", "two"])
        self.assertEqual(state["peak"], 1)

    async def test_timeout_aborts_comfy_work_before_returning(self):
        with patch.object(handler, "_http", return_value=b'{"prompt_id":"p"}'), \
                patch.object(handler, "JOB_TIMEOUT_S", -1), \
                patch.object(handler, "_abort_and_wait") as abort:
            with self.assertRaises(TimeoutError):
                await handler._queue_and_wait({})
        abort.assert_called_once_with("p")

    async def test_cancel_during_submission_waits_for_prompt_id_then_aborts(self):
        started, submitted = threading.Event(), threading.Event()

        def http(path, *args):
            self.assertEqual(path, "/prompt")
            started.set()
            time.sleep(0.04)
            submitted.set()
            return b'{"prompt_id":"submitted-after-cancel"}'

        with patch.object(handler, "_http", side_effect=http), \
                patch.object(handler, "_abort_and_wait") as abort:
            task = asyncio.create_task(handler._queue_and_wait({}))
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(submitted.is_set())
        abort.assert_called_once_with("submitted-after-cancel")

    async def test_lost_submission_response_clears_unknown_work(self):
        with patch.object(handler, "_http", side_effect=TimeoutError("response lost")), \
                patch.object(handler, "_abort_and_wait") as abort:
            with self.assertRaisesRegex(TimeoutError, "response lost"):
                await handler._queue_and_wait({})
        abort.assert_called_once_with(None)

    async def test_repeated_cancellation_waits_for_gpu_abort_cleanup(self):
        started, finished = threading.Event(), threading.Event()

        def http(path, *args):
            if path == "/prompt":
                return b'{"prompt_id":"p"}'
            started.set()
            return b'{}'

        def abort(prompt_id):
            time.sleep(0.04)
            finished.set()

        with patch.object(handler, "_http", side_effect=http), \
                patch.object(handler, "_abort_and_wait", side_effect=abort):
            task = asyncio.create_task(handler._queue_and_wait({}))
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(finished.is_set())

    async def test_cancelled_director_preprocessing_holds_lock_until_thread_finishes(self):
        started, finished = threading.Event(), threading.Event()

        def prepare(graph, inputs):
            started.set()
            time.sleep(0.04)
            finished.set()
            return graph

        with patch.object(handler, "_patch", side_effect=prepare):
            task = asyncio.create_task(handler._prepare_prompt({}, {}))
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(finished.is_set())

    async def test_prompt_validation_error_surfaces_node_diagnostics(self):
        with patch.object(handler, "_http", return_value=b'{"error":"invalid graph","node_errors":{"x":{}}}'):
            with self.assertRaisesRegex(RuntimeError, "node_errors"):
                await handler._queue_and_wait({})

    async def test_successful_job_requires_ordered_release_evidence_before_returning_media(self):
        runtime = build_runtime("rtx6000_ada_48", environ={})
        graph = json.loads((Path(__file__).resolve().parents[1] /
                           "workflows-api/mmh3/MiniMaxH3-Turbo-T2VA.json").read_text())
        records = [{"stage": stage, "sequence": index + 20, "models_after": 0}
                   for index, stage in enumerate(("after_conditioning", "after_sampling", "after_video_decode", "after_decode"))]
        history = {"outputs": {str(index): {"mmh3_memory": [row]} for index, row in enumerate(records)}}
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                selected = json.loads(json.dumps(history))
                if invalid:
                    selected["outputs"]["1"]["mmh3_memory"][0]["sequence"] = 99
                with patch.object(handler, "_ensure_comfy", new_callable=AsyncMock), \
                        patch.object(handler, "load_runtime", return_value=runtime), \
                        patch.object(handler, "_load_workflow", return_value=graph), \
                        patch.object(handler, "_prune_volume_outputs"), \
                        patch.object(handler, "validate_model_files"), \
                        patch.object(handler, "_queue_and_wait", new_callable=AsyncMock, return_value=selected), \
                        patch.object(handler, "_collect_outputs", return_value=([{"filename": "fixture.mp4"}], {})) as collect:
                    if invalid:
                        with self.assertRaisesRegex(RuntimeError, "re-executed"):
                            await handler._handle_job({"id": "fixture", "input": {"prompt": "Example"}})
                        collect.assert_not_called()
                    else:
                        result = await handler._handle_job({"id": "fixture", "input": {"prompt": "Example"}})
                        self.assertEqual(result["execution"]["memory_stages"], records)

    def test_boot_rejects_gpu_only_override(self):
        runtime = build_runtime("rtx6000_ada_48", environ={})
        with patch.object(handler, "load_runtime", return_value=runtime), \
                patch.dict("os.environ", {"COMFYUI_EXTRA_ARGS": "--gpu-only"}):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                handler._boot_comfy()

    def test_missing_stage_nodes_fail_boot_without_ten_minute_retry(self):
        runtime = build_runtime("rtx6000_ada_48", environ={})
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(handler, "load_runtime", return_value=runtime), \
                patch.dict("os.environ", {"COMFYUI_EXTRA_ARGS": ""}), \
                patch("builtins.open", mock_open()), \
                patch.object(handler.subprocess, "Popen", return_value=proc), \
                patch.object(handler, "_http", return_value=b'{}'), \
                patch.object(handler.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "MMH3StageBarrier"):
                handler._boot_comfy()
        sleep.assert_not_called()

    def test_workflow_sync_is_opt_in_and_paths_are_rejected(self):
        with patch.dict("os.environ", {"WORKFLOW_SYNC_ENABLED": "false"}), \
                patch.object(handler.urllib.request, "urlopen") as request:
            handler._sync_workflows()
            request.assert_not_called()
        with self.assertRaises(ValueError):
            handler._load_workflow("../unrelated.json")


if __name__ == "__main__":
    unittest.main()
