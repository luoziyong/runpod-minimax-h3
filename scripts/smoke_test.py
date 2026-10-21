#!/usr/bin/env python3
"""Run real quality/speed T2V jobs on a deployed worker and save acceptance data."""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow_runtime import validate_memory_stages


def request(url, token, payload=None, timeout=60):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
        return json.load(response)


def run_job(base_url, token, job, timeout):
    result = request(base_url + "/run", token, {"input": job})
    deadline = time.monotonic() + timeout
    while result.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        if time.monotonic() >= deadline:
            request(base_url + "/cancel/" + result["id"], token, {})
            raise TimeoutError(f"Job {result['id']} exceeded {timeout}s; cancellation requested")
        time.sleep(2)
        result = request(base_url + "/status/" + result["id"], token)
    if result.get("status") != "COMPLETED":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    output = result["output"]
    execution = output.get("execution", {})
    validate_memory_stages(execution.get("memory_stages"), execution)
    if not output.get("outputs"):
        raise RuntimeError("Generation returned no media outputs")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Endpoint base URL or its /runsync URL")
    parser.add_argument("--api-key-env", default="WORKER_API_KEY")
    parser.add_argument("--preset", choices=("quality", "speed", "both"), default="both")
    parser.add_argument("--size", default="864x480")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", default="work/smoke")
    args = parser.parse_args()
    token = os.environ.get(args.api_key_env)
    if not token:
        raise SystemExit(f"Set {args.api_key_env} before running")
    url = args.url.rstrip("/")
    for suffix in ("/runsync", "/run"):
        if url.endswith(suffix):
            url = url[:-len(suffix)]
            break
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    presets = ("quality", "speed") if args.preset == "both" else (args.preset,)
    for preset in presets:
        job = {"task": "t2v", "preset": preset, "seed": 42,
               "video_size": args.size, "seconds": args.seconds,
               "prompt": "A cinematic tracking shot through a quiet mountain village at sunrise, with natural ambient sound."}
        started = time.monotonic()
        output = run_job(url, token, job, args.timeout)
        media = []
        for index, item in enumerate(output["outputs"]):
            filename = Path(item.get("filename") or f"output-{index}.mp4").name
            if item.get("b64"):
                path = directory / f"{preset}-{filename}"
                path.write_bytes(base64.b64decode(item["b64"], validate=True))
                media.append({"file": str(path)})
            else:
                media.append({"filename": filename, "url": item.get("url")})
        record = {"request": job, "wall_seconds": round(time.monotonic() - started, 3),
                  "execution": output["execution"], "media": media}
        path = directory / f"{preset}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"{preset}: {record['execution']['profile']}, {record['wall_seconds']}s, {path}")


if __name__ == "__main__":
    main()
