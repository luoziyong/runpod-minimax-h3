#!/bin/bash
# Download a GPU-selected three-column manifest. Catalog selection is in worker_config.py.
set -euo pipefail

VOL_ROOT="${VOL_ROOT:-/runpod-volume}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$VOL_ROOT/models"
if [ ! -d "$VOL_ROOT" ]; then
    echo "populate: missing volume $VOL_ROOT" >&2
    exit 1
fi
if [ -z "${MANIFEST:-}" ]; then
    MANIFEST="$(mktemp)"
    trap 'rm -f "$MANIFEST"' EXIT
    python3 "$SCRIPT_DIR/worker_config.py" manifest > "$MANIFEST"
fi
if [ ! -s "$MANIFEST" ]; then
    echo "populate: missing or empty selected manifest $MANIFEST" >&2
    exit 1
fi
exec 9>"$VOL_ROOT/.populate.lock"
flock 9
mkdir -p "$M"
failed=0
while IFS=$'\t' read -r sub name url extra; do
    case "$sub" in ''|\#*) continue;; esac
    if [ -n "${extra:-}" ] || [ -z "$url" ]; then
        echo "populate: expected a selected three-column manifest; use worker_config.py manifest" >&2
        exit 1
    fi
    case "$sub/$name" in /*|*../*) echo "populate: invalid destination" >&2; exit 1;; esac
    dest="$M/$sub/$name"
    if [ -s "$dest" ]; then
        echo "populate: present $sub/$name"
        continue
    fi
    mkdir -p "$M/$sub"
    echo "populate: downloading $sub/$name"
    ok=0
    if command -v hf >/dev/null 2>&1 && [[ "$url" =~ ^https://huggingface\.co/ ]]; then
        repo_id=$(echo "$url" | awk -F/ '{print $4"/"$5}')
        repo_path=$(echo "$url" | sed -E 's#https?://[^/]+/[^/]+/[^/]+/resolve/[^/]+/(.+)#\1#')
        tmp_dir="$M/$sub/.tmp_hf_$name"
        mkdir -p "$tmp_dir"
        export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
        if hf download "$repo_id" "$repo_path" --local-dir "$tmp_dir" && [ -s "$tmp_dir/$repo_path" ]; then
            mv -f "$tmp_dir/$repo_path" "$dest"
            rm -rf "$tmp_dir"
            ok=1
        fi
    fi
    if [ "$ok" = 0 ]; then
        wget_args=(-q --continue --tries=5 --timeout=60 -O "$dest.tmp")
        [ -z "${HF_TOKEN:-}" ] || wget_args+=(--header="Authorization: Bearer $HF_TOKEN")
        if wget "${wget_args[@]}" "$url" && [ -s "$dest.tmp" ]; then
            mv -f "$dest.tmp" "$dest"
        else
            echo "populate: FAILED $sub/$name (partial download kept for retry)" >&2
            failed=$((failed + 1))
        fi
    fi
done < "$MANIFEST"
if [ "$failed" -gt 0 ]; then
    echo "populate: $failed download(s) failed" >&2
    exit 1
fi
echo "populate: selected models ready in $M"
