# GHCR 备选部署

当前首选 Runpod 的 GitHub 构建与自带镜像仓库，见[部署说明](deployment.md)。本文保留自行构建并推送 GHCR 的方案；其分层还原空间预算不适用于 Runpod 原生文件镜像。命令均在仓库根目录执行。

### 3.1 GHCR 混合部署：Ada 主镜像与 5090 备选镜像

对低频使用，推荐 **编码器、VAE、Turbo LoRA 随镜像发布，两个 DiT 放在按需创建的网络卷**。每个镜像固定一种 GPU 配置，均保留 quality/speed 和 FL2VA/Ref2VA；不混入另一种编码器或可选模型。

2026-10-22 查询的模型文件大小如下，单位为十进制 GB，均不含基础运行环境：

| 镜像 | GPU profile / Runpod 池 | 编码器 | 镜像内权重 | 容器临时盘还原 | 网络卷内 DiT |
| --- | --- | --- | --- | --- | --- |
| Ada 主镜像 | `rtx6000_ada_48` / `ADA_48_PRO` | 官方 INT8 ConvRot | 34.47GB | 27.14GB | 41.94GB |
| 5090 备选镜像 | `rtx5090_32` / `ADA_32_PRO` | 官方 NVFP4 AWQ | 23.02GB | 15.69GB | 41.94GB |

5090 保留默认 480p，Ada 画质档默认 768p。两套编码器独立打包，两个 DiT 文件相同，可顺序复用同一个网络卷；GPU 切换仍需先保证数据中心有对应资源。

[GHCR 每个镜像层限制 10GB，上传超时为 10 分钟](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#troubleshooting)。打包器将模型拆成每层最多 4GiB 的块，大编码器在启动时校验 SHA-256 并原子还原，小文件直接链接镜像中的块。**50GB 可写盘可以容纳上述编码器还原预算**，代码另要求至少 8GiB 余量供运行使用；这不是对最终平台容量和 GPU 推理成功的保证，镜像拉取、基础环境和生成缓存仍需实机核验。

若把所有模型都装入 Ada 镜像，启动还原约需 69.08GB，50GB 可写盘无法运行。混合方案将两个 DiT 留在卷中，避免这部分临时盘占用。建议网络卷先分配 **80GB**，容纳约 42GB DiT 和下载、运行余量；容器临时盘先用 50GB 验收，遇到镜像拉取或可用空间不足再提高容量。代码检查的是文件系统实际报告的可用空间。

在有 Docker 的构建机执行。模型原文件、分块构建目录和镜像会同时占空间，单独构建一套混合镜像建议至少 **200GB 可用构建空间**，连续构建两套应额外预留。Mac 使用 `linux/amd64`。

先构建共享运行环境，不需要发布这个中间镜像：

```bash
docker build --platform linux/amd64 -t mmh3-runtime:profiles .
```

选择本次构建的固定配置；每次只选择一组：

```bash
# Ada 主镜像
export MMH3_PROFILE=rtx6000_ada_48
export MMH3_TAG=ada-hybrid-v1

# 构建 5090 备选镜像时，改用下面两项，再执行后面的下载、打包和发布步骤：
# export MMH3_PROFILE=rtx5090_32
# export MMH3_TAG=5090-hybrid-v1

# 换成自己的小写 GitHub 用户名或组织名。
export MMH3_IMAGE='ghcr.io/YOUR_NAMESPACE/runpod-minimax-h3'
```

只下载需要进入镜像的五个权重，构建机不需要 GPU：

```bash
mkdir -p "work/model-download-${MMH3_PROFILE}"
docker run --rm --platform linux/amd64 \
  --entrypoint bash \
  -e MMH3_GPU_PROFILE="${MMH3_PROFILE}" \
  -e VOL_ROOT=/runpod-volume \
  -v "$PWD/work/model-download-${MMH3_PROFILE}:/runpod-volume" \
  mmh3-runtime:profiles -c '
set -euo pipefail
python3 /opt/serverless/model_bundle.py manifest \
  --profile "$MMH3_GPU_PROFILE" --volume-diffusion-models > /tmp/image-models.txt
MANIFEST=/tmp/image-models.txt /opt/serverless/populate-volume.sh
'

python3 model_bundle.py pack \
  --profile "${MMH3_PROFILE}" \
  --encoder-family official \
  --volume-diffusion-models \
  --models-dir "work/model-download-${MMH3_PROFILE}/models" \
  --output "work/baked-${MMH3_TAG}" \
  --base-image mmh3-runtime:profiles

docker build --platform linux/amd64 \
  -t "${MMH3_IMAGE}:${MMH3_TAG}" "work/baked-${MMH3_TAG}"
```

`--output` 必须是尚不存在的目录；失败的打包目录不能用于构建，重试时换一个新目录。已有所需权重时可直接提供 `--models-dir`，混合打包不要求构建机上存在两个 DiT。新增模型层最多 4GiB；更换基础镜像或依赖后，还需检查它们自己的层是否符合 GHCR 限制。

在构建机设置 `GHCR_USER` 和具有 `write:packages` 权限的 classic PAT（`GHCR_TOKEN`），登录并推送：

```bash
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker push "${MMH3_IMAGE}:${MMH3_TAG}"
```

GHCR 首次发布默认私有。Runpod 使用 **Import from Docker Registry** 填写对应镜像；私有镜像配置 GitHub 用户名和具有 `read:packages` 权限的 classic PAT，公开镜像可匿名拉取。生产使用推送后的 digest。

部署时创建并挂载网络卷到 `/runpod-volume`，选择表中的对应 GPU 池，Active workers=0、Max workers=1。镜像已内置固定 profile、官方编码器及以下混合存储设置：

```dotenv
MMH3_MODEL_SOURCE=hybrid
VOLUME_AUTOPOPULATE=true
VOL_ROOT=/runpod-volume
MMH3_LOCAL_ROOT=/tmp/mmh3-data
```

首次启动只向卷下载两个 DiT；编码器、VAE 和 LoRA 从镜像读取。`/tmp/mmh3-data/models` 汇合镜像模型和卷内 DiT 的链接；ONNX/TensorRT 缓存也放临时盘。不要把 GPU profile 改为 `auto` 混用两类编码器，改变 profile、编码器或可选模型组须重新打包，切换 quality/speed 无需重建。`upscale` / `rife` 默认关闭，其可选 ONNX 首次调用时仍需下载。

用完后按顺序处理：保存生成结果，等待队列和 GPU 任务结束，停止相关 worker 并解除 Endpoint 的卷绑定（Pod 则终止 Pod），最后删除网络卷。若两套镜像共用一个卷，应先确认另一套也不再使用它。**仅缩容到 0 或终止 Pod 不会删除网络卷，也不会停止卷的存储计费。** 下次重新创建卷、挂载并下载两个 DiT 即可。

[创建网络卷](https://docs.runpod.io/api-reference-v2/network-volumes/create-a-network-volume)可用 `POST https://api.runpod.io/v2/network-volumes`；[删除卷](https://docs.runpod.io/api-reference-v2/network-volumes/delete-a-network-volume)可用下面的接口，删除会永久清空卷内数据：

```bash
curl -fsS -X DELETE \
  "https://api.runpod.io/v2/network-volumes/${VOLUME_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

首次验收依次检查镜像拉取、空间检查、编码器还原、DiT 下载、`comfyui=ready` 和真实视频生成。CPU 测试覆盖两套配置的分块还原、混合模型读取与启动路径，不代表镜像已经完成 GHCR 发布或 Runpod GPU 验收。

如有更大的临时盘，也保留全模型随镜像部署的选项：下载清单和打包命令均去掉 `--volume-diffusion-models`，生成的镜像自动使用 `MMH3_MODEL_SOURCE=image`，不挂网络卷。Ada 全模型还原约需 69.08GB 再加运行余量，不能套用混合方案的 50GB 预算。
