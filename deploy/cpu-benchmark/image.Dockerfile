# Keep the already-published largest model layer, avoiding another compression pass.
FROM --platform=linux/amd64 python:3.12-slim-bookworm@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0 AS runtime
RUN python -m pip install --no-cache-dir "huggingface_hub[hf_xet]==2.2.0"
FROM --platform=linux/amd64 luoziyong/mmh3-dit@sha256:c574e396ccb8c1a8c75aec1a508325f688e44dca575b93f0980e972a214b2271
COPY --link --from=runtime / /
COPY --link --from=luoziyong/mmh3-encoder@sha256:385a45cba3c55d2e8da7e78b9d4e057f618a968131e26398ace2c7718e41cde5 /models/ /models/
COPY --link --from=luoziyong/mmh3-common@sha256:e8c64bec249235ff47272937b105398ec57f29009562e733c5c21c7ccfeb752c /models/vae/ /models/vae/
COPY deploy/cpu-benchmark/benchmark.py deploy/cpu-benchmark/models.json /opt/cpu-benchmark/
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/tmp/hf-home
ENTRYPOINT ["python", "-u", "/opt/cpu-benchmark/benchmark.py"]
CMD ["--source", "image"]
