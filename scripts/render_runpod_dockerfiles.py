"""Generate fixed-profile Dockerfiles for Runpod's GitHub image builder."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("rtx6000_ada_48", "rtx5090_32")


def render(source, profile, volume_encoder=False):
    settings = {"MMH3_GPU_PROFILE": profile, "MMH3_ENCODER_FAMILY": "official",
                "MMH3_MODEL_SOURCE": "hybrid", "MMH3_BAKED_MODELS_DIR": "/opt/mmh3-models",
                "MMH3_LOCAL_ROOT": "/tmp/mmh3-data", "VOL_ROOT": "/runpod-volume",
                "VOLUME_AUTOPOPULATE": "true", "MMH3_DOWNLOAD_ENHANCER": "false",
                "MMH3_DOWNLOAD_SINGULARITY": "false", "MMH3_DOWNLOAD_STYLE_LORAS": "false"}
    header = "# Generated from Dockerfile by scripts/render_runpod_dockerfiles.py; do not edit.\n"
    volume_label = "the encoder and DiTs" if volume_encoder else "DiTs"
    suffix = f"\n# Native image weights; {volume_label} are downloaded onto the session network volume.\n"
    suffix += "\n".join(f'ENV {key}="{value}"' for key, value in settings.items()) + "\n"
    encoder_arg = " --volume-encoder" if volume_encoder else ""
    suffix += f"RUN python3 /opt/serverless/model_bundle.py bake --profile {profile}{encoder_arg}\n"
    return header + source.rstrip() + "\n" + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when generated Dockerfiles need updating")
    args = parser.parse_args()
    source = (ROOT / "Dockerfile").read_text()
    stale = []
    variants = [(profile, profile, False) for profile in PROFILES]
    variants.append(("rtx6000_ada_48_compact", "rtx6000_ada_48", True))
    for name, profile, volume_encoder in variants:
        path = ROOT / "deploy/runpod" / f"{name}.Dockerfile"
        expected = render(source, profile, volume_encoder)
        if args.check:
            if not path.is_file() or path.read_text() != expected:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
            print(path.relative_to(ROOT))
    if stale:
        print("Regenerate Runpod Dockerfiles: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
