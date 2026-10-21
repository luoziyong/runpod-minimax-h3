"""Authenticated Runpod-shaped API for Pods, including the 48GB MIG SKU."""

import asyncio
import hmac
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import handler as worker


def create_server(loop, api_key, host="0.0.0.0", port=8000):
    if not api_key:
        raise ValueError("Set WORKER_API_KEY before exposing the Pod HTTP API")
    jobs = {}
    jobs_lock = threading.Lock()

    def submit(job):
        job = dict(job, id=uuid.uuid4().hex)
        with jobs_lock:
            now = time.monotonic()
            for jid, row in list(jobs.items()):
                if row["future"].done() and now - row["created"] > 3600:
                    del jobs[jid]
            if len(jobs) >= 32:
                oldest = next((jid for jid, row in jobs.items() if row["future"].done()), None)
                if oldest is not None:
                    del jobs[oldest]
            if len(jobs) >= 32:
                raise OverflowError("Pod queue is full; retry after an active job finishes")
            future = asyncio.run_coroutine_threadsafe(worker.handler(job), loop)
            row = {"id": job["id"], "future": future, "created": now}
            jobs[job["id"]] = row
        return row

    def snapshot(row):
        future = row["future"]
        result = {"id": row["id"], "status": "IN_PROGRESS"}
        if future.cancelled():
            result["status"] = "CANCELLED"
        elif future.done():
            try:
                result.update(status="COMPLETED", output=future.result())
            except Exception as exc:
                result.update(status="FAILED", error=str(exc))
        return result

    class RequestHandler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            expected = f"Bearer {api_key}"
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected.encode()):
                self.reply(401, {"error": "Invalid bearer token"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path.startswith("/status/"):
                with jobs_lock:
                    row = jobs.get(self.path.removeprefix("/status/"))
                self.reply(200 if row else 404, snapshot(row) if row else {"error": "Unknown job"})
                return
            if self.path != "/health":
                self.reply(404, {"error": "Use GET /health, GET /status/id or POST /run"})
                return
            try:
                future = asyncio.run_coroutine_threadsafe(worker.handler({"input": {"action": "health"}}), loop)
                self.reply(200, future.result(timeout=10))
            except Exception as exc:
                self.reply(503, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path.startswith("/cancel/"):
                with jobs_lock:
                    row = jobs.get(self.path.removeprefix("/cancel/"))
                if row is None:
                    self.reply(404, {"error": "Unknown job"})
                else:
                    row["future"].cancel()
                    self.reply(200, snapshot(row))
                return
            if self.path not in ("/run", "/runsync"):
                self.reply(404, {"error": "Use POST /run or /runsync"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 64 * 1024 ** 2:
                    self.reply(413, {"error": "JSON request must be between 1 byte and 64 MiB"})
                    return
                job = json.loads(self.rfile.read(size))
                if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
                    raise ValueError("Expected an object with an input object")
                row = submit(job)
                if self.path == "/run":
                    self.reply(200, {"id": row["id"], "status": "IN_QUEUE"})
                else:
                    try:
                        row["future"].result()
                    except Exception:
                        # Keep the task ID and status for failed/cancelled jobs,
                        # matching the asynchronous status endpoint.
                        pass
                    self.reply(200, snapshot(row))
            except OverflowError as exc:
                self.reply(429, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"status": "FAILED", "error": str(exc)})
            except Exception as exc:
                self.reply(500, {"status": "FAILED", "error": str(exc)})

        def log_message(self, fmt, *args):
            # The bearer token and request body are never logged.
            print(f"[http-api] {fmt % args}", flush=True)

    return ThreadingHTTPServer((host, port), RequestHandler)


def main():
    loop = asyncio.new_event_loop()
    api_key = os.environ.get("WORKER_API_KEY", "")
    port = int(os.environ.get("WORKER_API_PORT", "8000"))
    server = create_server(loop, api_key, port=port)
    threading.Thread(target=loop.run_forever, daemon=True).start()
    threading.Thread(target=worker._boot_in_bg, daemon=True).start()
    print(f"[http-api] listening on :{port}; GPU jobs execute serially", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
