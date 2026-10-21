"""Hardware profiles, model selection and boot-time configuration (stdlib only)."""

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
SINGULARITY = "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
ENCODER_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
ENCODER_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
FL2VA_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
REF2VA_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

# Starting points, not measured VRAM limits. Never silently resize after OOM.
PROFILES = {
    "rtx6000_ada_48": {"encoder": "int8", "min_vram_gib": 44, "capability": [8, 9],
                       "reserve_vram": 2, "quality_size": "1344x768", "tile_size": 512},
    "l40s_48": {"encoder": "int8", "min_vram_gib": 44, "capability": [8, 9],
                "reserve_vram": 2, "quality_size": "1344x768", "tile_size": 512},
    "pro6000_mig_48": {"encoder": "nvfp4", "min_vram_gib": 44, "capability": [12, 0],
                       "reserve_vram": 2, "quality_size": "1344x768", "tile_size": 512},
    "rtx5090_32": {"encoder": "nvfp4", "min_vram_gib": 29, "capability": [12, 0],
                   "reserve_vram": 3, "quality_size": "864x480", "tile_size": 384},
}
PRESETS = {
    "quality": {"steps": 20, "sampler": "res_multistep", "scheduler": "simple",
                "shift_video": 12.0, "shift_audio": 3.0},
    "speed": {"steps": 8, "sampler": "euler", "scheduler": "simple",
              "shift_video": 6.0, "shift_audio": 3.0},
}
OPTIONAL_GROUPS = {"enhancer": "MMH3_DOWNLOAD_ENHANCER",
                   "singularity": "MMH3_DOWNLOAD_SINGULARITY",
                   "style": "MMH3_DOWNLOAD_STYLE_LORAS"}


def env_bool(name, default=False, environ=None):
    env = os.environ if environ is None else environ
    value = str(env.get(name, str(default))).strip().lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"{name} must be true or false")
    return value in ("true", "1")


def probe_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA GPU is required; check container GPU access")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU or MIG device per worker")
    cuda = torch.version.cuda or "0"
    if int(cuda.split(".")[0]) < 13:
        raise RuntimeError(f"PyTorch cu130+ is required for INT8 ConvRot; found CUDA {cuda}")
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "vram_gib": props.total_memory / 1024 ** 3,
            "sm_count": props.multi_processor_count,
            "capability": [props.major, props.minor], "torch": torch.__version__, "cuda": cuda}


def detect_profile(gpu):
    name = gpu["name"].lower()
    if "5090" in name:
        return "rtx5090_32"
    if "pro 6000" in name and 40 <= gpu["vram_gib"] < 64:
        return "pro6000_mig_48"
    if "l40" in name:  # Runpod's ADA_48_PRO pool can also schedule an L40.
        return "l40s_48"
    if "6000" in name and "ada" in name:
        return "rtx6000_ada_48"
    raise ValueError(f"Unsupported GPU: {gpu['name']} ({gpu['vram_gib']:.1f} GiB). "
                     f"Supported profiles: {', '.join(PROFILES)}")


def validate_gpu(profile, gpu):
    spec = PROFILES[profile]
    if gpu["vram_gib"] < spec["min_vram_gib"]:
        raise ValueError(f"{profile} needs at least {spec['min_vram_gib']} GiB visible VRAM; "
                         f"found {gpu['vram_gib']:.1f} GiB")
    if list(gpu["capability"]) != spec["capability"]:
        raise ValueError(f"{profile} expects compute capability {spec['capability']}; "
                         f"found {gpu['capability']}")
    if profile == "pro6000_mig_48" and gpu["vram_gib"] >= 64:
        raise ValueError("pro6000_mig_48 needs a visible 48GB MIG slice, not the 96GB parent GPU")


def build_runtime(profile, gpu=None, environ=None):
    env = os.environ if environ is None else environ
    if profile == "auto":
        if gpu is None:
            raise ValueError("auto requires GPU detection")
        profile = detect_profile(gpu)
    if profile not in PROFILES:
        raise ValueError(f"Unknown GPU profile {profile!r}; choose auto or {', '.join(PROFILES)}")
    if gpu is not None:
        validate_gpu(profile, gpu)
    preset = env.get("MMH3_PRESET", "quality")
    if preset not in PRESETS:
        raise ValueError("MMH3_PRESET must be quality or speed")
    spec = dict(PROFILES[profile])
    family = env.get("MMH3_ENCODER_FAMILY", "official")
    if family not in ("official", "heretic"):
        raise ValueError("MMH3_ENCODER_FAMILY must be official or heretic")
    model_source = env.get("MMH3_MODEL_SOURCE", "volume")
    if model_source not in ("volume", "image", "hybrid"):
        raise ValueError("MMH3_MODEL_SOURCE must be volume, image or hybrid")
    default_root = "/tmp/mmh3-data" if model_source == "image" else "/runpod-volume"
    volume_root = env.get("VOL_ROOT", default_root)
    model_root = env.get("MMH3_LOCAL_ROOT", "/tmp/mmh3-data") if model_source == "hybrid" else volume_root
    groups = ["core", "turbo", f"encoder_{family}_{spec['encoder']}"]
    groups += [group for group, key in OPTIONAL_GROUPS.items() if env_bool(key, environ=env)]
    return {"profile": profile, "preset": preset, "gpu": gpu, **spec,
            "encoder_family": family, "model_groups": groups,
            "model_source": model_source,
            "volume_root": volume_root,
            "models_dir": str(Path(model_root) / "models"),
            "extra_args": ["--disable-smart-memory", "--disable-async-offload",
                           "--reserve-vram", str(spec["reserve_vram"]),
                           "--preview-method", "none", "--cache-none"]}


def read_catalog(path=None):
    rows = []
    with open(path or ROOT / "models-manifest.txt", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 4:
                raise ValueError(f"Model catalog line {line}: expected subdir, filename, URL, group")
            subdir, filename, url, group = row
            if (Path(subdir).is_absolute() or ".." in Path(subdir).parts
                    or Path(filename).name != filename or not url.startswith("https://huggingface.co/")):
                raise ValueError(f"Invalid model catalog line {line}")
            rows.append(dict(subdir=subdir, filename=filename, url=url, group=group))
    if len({(r["subdir"], r["filename"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate model catalog entries")
    return rows


def selected_models(runtime):
    groups = set(runtime["model_groups"])
    rows = [r for r in read_catalog() if r["group"] in groups]
    missing = groups - {r["group"] for r in rows}
    if missing:
        raise ValueError(f"Model groups missing from catalog: {sorted(missing)}")
    return rows


def encoder_filename(runtime):
    return next(r["filename"] for r in selected_models(runtime)
                if r["group"].startswith("encoder_"))


def trt_cache_key(runtime):
    # TRT version is also embedded in upstream node filenames.
    identity = {"profile": runtime["profile"], "gpu": runtime.get("gpu"),
                "revision": os.environ.get("MMH3_ENGINE_CACHE_REV", "v1")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return f"{runtime['profile']}-{digest}"


def load_runtime():
    path = Path(os.environ.get("MMH3_RUNTIME_CONFIG", "/tmp/mmh3/runtime.json"))
    if not path.is_file():
        raise RuntimeError(f"Missing runtime profile {path}; run serverless-entrypoint.sh first")
    runtime = json.loads(path.read_text())
    if runtime.get("profile") not in PROFILES:
        raise ValueError("Invalid runtime GPU profile")
    return runtime


def normalize_job(job, runtime):
    job = dict(job)
    preset = job.get("preset") or runtime["preset"]
    if preset not in PRESETS:
        raise ValueError("preset must be quality or speed")
    config = job.get("config")
    if config:
        aliases = {"native": "quality", "native_turbo": "speed",
                   "r2va_native": "quality", "r2va_native_turbo": "speed",
                   "r2va_singularity": "quality", "r2va_singularity_turbo": "speed"}
        if config not in aliases:
            raise ValueError(f"Unsupported legacy config: {config}")
        if job.get("preset") and preset != aliases[config]:
            raise ValueError("config and preset select different sampling schedules")
        preset = aliases[config]
        if "singularity" in config:
            job["model_variant"] = "singularity"
    job["preset"] = preset
    if "model_variant" not in job and "Singularity" in str(job.get("workflow", "")):
        job["model_variant"] = "singularity"
    variant = job.get("model_variant", "official")
    if variant not in ("official", "singularity"):
        raise ValueError("model_variant must be official or singularity")
    if variant == "singularity" and "singularity" not in runtime["model_groups"]:
        raise ValueError("Enable MMH3_DOWNLOAD_SINGULARITY=true and restart before selecting Singularity")
    if job.get("lora", "none") != "none" and "style" not in runtime["model_groups"]:
        raise ValueError("Enable MMH3_DOWNLOAD_STYLE_LORAS=true before selecting a style LoRA")
    if job.get("lora", "none") not in ("none", "realism", "char_swap"):
        raise ValueError("lora must be none, realism or char_swap")
    if job.get("lora", "none") != "none" and variant != "singularity":
        raise ValueError("Style LoRAs require model_variant=singularity")
    job["model_variant"] = variant
    images = job.get("images", [])
    if not isinstance(images, list) or len(images) > 3 or not all(isinstance(v, str) for v in images):
        raise ValueError("images must be a list of up to three URLs or base64 strings")
    job.setdefault("seconds", 5)
    if isinstance(job["seconds"], bool) or not isinstance(job["seconds"], int) or not 1 <= job["seconds"] <= 15:
        raise ValueError("seconds must be an integer from 1 to 15; chain clips for longer videos")
    job.setdefault("video_size", runtime["quality_size"] if preset == "quality" else "864x480")
    size = re.fullmatch(r"(\d+)\s*x\s*(\d+)(?:\s+\([^\n]*\))?", str(job["video_size"]))
    if size is None:
        raise ValueError("video_size must be explicit, e.g. 1344x768 or 864x480; auto is not memory-bounded")
    width, height = map(int, size.groups())
    if min(width, height) < 256 or width % 32 or height % 32 or width * height > 1344 * 768:
        raise ValueError("video_size must use multiples of 32, minimum 256px, and at most 1344*768 pixels")
    job["video_size"] = f"{width}x{height}"
    for key in ("enhance", "analyze", "upscale", "rife"):
        job.setdefault(key, False)
        if not isinstance(job[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if (job["enhance"] or job["analyze"]) and "enhancer" not in runtime["model_groups"]:
        raise ValueError("Enable MMH3_DOWNLOAD_ENHANCER=true and restart before using enhance/analyze")
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifest", "prepare"))
    parser.add_argument("--profile", default=os.environ.get("MMH3_GPU_PROFILE", "auto"))
    parser.add_argument("--output-dir", default="/tmp/mmh3")
    args = parser.parse_args()
    gpu = probe_gpu() if args.action == "prepare" or args.profile == "auto" else None
    runtime = build_runtime(args.profile, gpu)
    rows = selected_models(runtime)
    manifest = "".join(f"{r['subdir']}\t{r['filename']}\t{r['url']}\n" for r in rows)
    if args.action == "manifest":
        print(manifest, end="")
        return
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "models-manifest.txt").write_text(manifest)
    runtime["trt_cache_key"] = trt_cache_key(runtime)
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
