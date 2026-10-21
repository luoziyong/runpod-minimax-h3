# 部署与 GPU 验收

## 1. 确定 GPU 配置

| GPU | 环境文件 | 默认编码器 | 画质 / 速度默认参数 |
| --- | --- | --- | --- |
| RTX 6000 Ada 48GB | `deploy/rtx6000_ada_48.env` | 官方 INT8 ConvRot | 768p 原生20步 / 480p Turbo |
| L40S 48GB | `deploy/l40s_48.env` | 官方 INT8 ConvRot | 768p 原生20步 / 480p Turbo |
| PRO 6000 MIG 48GB | `deploy/pro6000_mig_48.env` | 官方 NVFP4 AWQ | 768p 原生20步 / 480p Turbo |
| RTX 5090 32GB | `deploy/rtx5090_32.env` | 官方 NVFP4 AWQ | 480p 原生20步 / 480p Turbo |

Turbo：T2V/I2V/FL2V 为 8 步，官方 Ref2VA 为 4 步。均默认 5 秒。MIG 的 48GB 实例只有半卡计算资源，不能套用完整 96GB PRO 6000 的耗时。上述默认尺寸是保守起点，仍需真实任务验收；5090 可显式测试 `video_size=1344x768`，失败时降低尺寸/时长，不会自动偷换输出规格。

生产建议主机内存 128GB 和快速本地盘。卷部署建议至少 160GB 持久模型卷；多个显卡配置/编码器家族共用卷或全开可选权重时建议 200GB。低频使用首选下面的 Runpod 单配置混合镜像，网络卷存放两个 DiT，紧凑 Ada 还包含编码器，用完释放。权重下载与 CPU 卸载会消耗系统内存及磁盘空间，不能只看 GPU 显存。

## 2. 首选：Runpod 构建并托管镜像

通过 Runpod **Import Git Repository** 导入 GitHub 仓库，由 Runpod 根据 Dockerfile 构建、存入其自带镜像仓库并部署 Serverless Endpoint。无需先配置 GHCR，也无需在本机下载模型、构建或推送镜像。[官方 GitHub 部署说明](https://docs.runpod.io/serverless/workers/github-integration)

每个 Endpoint 使用一个固定配置，不把两套编码器装进同一个镜像：

| 用途 | Dockerfile Path | GPU pool | 固定编码器 | 镜像内权重 |
| --- | --- | --- | --- | --- |
| 主配置（紧凑 Ada） | `deploy/runpod/rtx6000_ada_48_compact.Dockerfile` | `ADA_48_PRO` | 官方 INT8 ConvRot，放网络卷 | 两个 VAE、两种 Turbo LoRA |
| Ada 编码器入镜像版 | `deploy/runpod/rtx6000_ada_48.Dockerfile` | `ADA_48_PRO` | 官方 INT8 ConvRot | 约 34.47GB |
| 备选配置 | `deploy/runpod/rtx5090_32.Dockerfile` | `ADA_32_PRO` | 官方 NVFP4 AWQ | 约 23.02GB |

以上仅为权重大小，不是最终镜像大小。紧凑 Ada 只包含视频/音频 VAE 和两种 Turbo LoRA；其余版本还包含本配置的编码器。所有版本的两个 DiT 均在网络卷。Ada 画质档默认 768p，5090 默认 480p，均保留 quality/speed。

### 存储安排

- 镜像：`/opt/mmh3-models/models` 中保存四个完整权重文件（紧凑 Ada）或五个完整权重文件（其余版本），构建时生成大小和 SHA-256 清单。
- 网络卷：`/runpod-volume/models/diffusion_models` 中保存 FL2VA 和 Ref2VA，合计约 **41.94GB**。紧凑 Ada 还在 `text_encoders` 保存本配置的 INT8 编码器，建议 **120GB** 卷；五权重版本建议 **80GB**。
- 容器临时盘：模型路径和镜像/卷文件通过符号链接汇合在 `/tmp/mmh3-data/models`。只存工作区、缓存、输出和少量管理文件，**不再把 27GB/16GB 编码器重建到可写盘**。可先用 **50GB** 做部署验收；实际镜像拉取与平台空间分配仍需核对。

这与 GHCR 的分块镜像不同：Runpod 入口使用原始权重文件，启动先校验再建立链接，没有大文件组装。仍要求临时盘至少有 8GiB 可用余量。新网络卷按镜像清单下载三个模型（紧凑 Ada）或两个 DiT（其余版本），复用已有卷时跳过现有非空文件。第一次下载和启动可能较慢。

### 控制台操作

1. 将整理后的代码放到你可访问的 GitHub 仓库。在 Runpod Settings → Connections 连接该仓库。
2. 在目标 GPU 可用的数据中心创建网络卷，紧凑 Ada 分配 120GB，其余版本分配 80GB。两套配置可顺序复用同一卷，但需确认该数据中心有目标 GPU。
3. Serverless → New Endpoint → **Import Git Repository**，选择仓库、分支和上表对应的 **Dockerfile Path**。各 Dockerfile 的构建上下文是仓库根目录；本地调试时也使用根目录作为 context。
4. 选择表中对应 GPU pool、每个 worker 一张卡、Active workers=0、Max workers=1、CUDA 13.0 或更新的兼容版本。绑定网络卷到 `/runpod-volume`，容器盘先配置 50GB，执行超时建议至少 1800 秒。
5. 保留镜像内置的固定 profile、官方编码器和混合存储设置；可在 Endpoint 中导入 `deploy/rtx6000_ada_48.env` 或 `deploy/rtx5090_32.env` 的相应参数。不要把 profile 改成 `auto`，也不要在同一 Endpoint 中加入另一种编码器架构的 GPU 池。
6. 部署后查看 Builds 日志，再确认镜像模型校验、卷中模型下载、`comfyui=ready`，最后提交真实生成任务。

如果只构建镜像，可先设置 Active workers=0、Max workers=0，暂不创建或绑定卷；推理前再挂载卷并把 Max workers 改为 1。

镜像已设置：

```dotenv
MMH3_MODEL_SOURCE=hybrid
MMH3_ENCODER_FAMILY=official
VOLUME_AUTOPOPULATE=true
VOL_ROOT=/runpod-volume
MMH3_LOCAL_ROOT=/tmp/mmh3-data
```

`MMH3_GPU_PROFILE` 分别固定为 `rtx6000_ada_48` 和 `rtx5090_32`。改变编码器或可选模型组需要重建镜像；切换 quality/speed 无需重建。`upscale` / `rife` 默认关闭，首次调用时仍下载 ONNX 并为实际 GPU 构建引擎。

### 构建限制与回退

Runpod GitHub 构建要求 **Docker build 步骤在 30 分钟内完成，最终镜像不超过 80GB**。原生混合镜像的模型大小虽低于 80GB，但还需加基础环境和依赖，不能据此保证最终大小或耗时达标。完整构建、上传和测试须看 Builds 结果。2026-10-22 的 Ada 五权重构建已完成模型下载、SHA-256 校验和 OCI 镜像导出，但构建缓存导出触及 30 分钟超时，平台终止构建，镜像没有上传成功。因此 Ada 首选紧凑版，把编码器移到网络卷；它仍只选择同一套 Ada 配置。GPU 推理须另做实测。

若紧凑版仍超过时间或镜像大小限制，可继续使用 **Runpod 自带镜像仓库**：把 Dockerfile Path 改为根目录的 `Dockerfile`，使用完整模型卷（建议 160GB），设置 `MMH3_MODEL_SOURCE=volume`、`VOLUME_AUTOPOPULATE=true`，并固定对应 GPU profile。此时镜像只含运行环境，全部所选权重在首次启动下载到卷。外部 GHCR 是后续备选，见 [GHCR 部署](ghcr.md)。

根目录的 Core 和基础镜像 digest 已固定；第三方节点仍继承上游分支/标签。基础镜像为公开镜像，构建不调用 GPU 探测、TensorRT 编译或推理。镜像代码与镜像内权重在构建时准备，卷中模型由启动脚本补齐。冷启动不更新源代码。

### 代码维护

固定配置 Dockerfile 由根目录 `Dockerfile` 生成，公共运行环境只维护一份。修改根目录 Dockerfile 后执行：

```bash
python3 scripts/render_runpod_dockerfiles.py
python3 scripts/render_runpod_dockerfiles.py --check
```

如需本地调试，使用同一个根目录 context：

```bash
docker build --platform linux/amd64 \
  -f deploy/runpod/rtx6000_ada_48_compact.Dockerfile \
  -t mmh3-runpod:ada .

# 5090 对应改用 deploy/runpod/rtx5090_32.Dockerfile。
```

### 使用完释放

先保存视频和需要的结果，等待队列与任务结束，停止相关 worker 并解除 Endpoint 的卷绑定，再删除网络卷；Pod 使用者需终止 Pod。共享卷时先确认另一套配置也不再使用它。**仅缩容到 0、停止 GPU 或终止 Pod 不会删除网络卷，卷仍计费。** 镜像内权重不受删卷影响。紧凑 Ada 下次新建卷需重新下载编码器和两个 DiT；五权重版本只需重新下载两个 DiT。

[创建卷](https://docs.runpod.io/api-reference-v2/network-volumes/create-a-network-volume)可调用 `POST https://api.runpod.io/v2/network-volumes`；[删除卷](https://docs.runpod.io/api-reference-v2/network-volumes/delete-a-network-volume)示例如下。删除会永久清空卷中数据：

```bash
curl -fsS -X DELETE \
  "https://api.runpod.io/v2/network-volumes/${VOLUME_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

### 旧工作目录迁移

升级前停止旧 worker，再使用新镜像启动。`COMFYUI_DIR` 仍指定工作区根目录，默认 `/workspace/runpod-slim/ComfyUI`；即使根目录已有旧 `main.py`，新 worker 也会运行镜像中的固定代码：

```text
<COMFYUI_DIR>/.mmh3/releases/<镜像代码和依赖指纹>/<GPU缓存键>/
```

镜像构建会生成代码哈希清单；启动逐文件校验运行目录，缺失/变更时先准备完整副本，再原子发布。已有旧 Core、自定义节点、模型和虚拟环境保留在原位置；`input/output/user` 通过符号链接共享原数据。运行只使用镜像 Python，不加载旧虚拟环境或旧自定义节点。Runpod 原生镜像中的 VAE、LoRA 通过链接读取，两个 DiT 从 `/runpod-volume/models` 获取；紧凑 Ada 的编码器也从卷中读取，其余原生版本从镜像读取，不会向临时盘复制大权重。完整模型卷模式仍读取卷中的全部所选权重。

镜像更新会使用新指纹目录；同一版本重复启动复用已校验目录。修复被修改的运行副本时，原副本保留为 `*.previous-*`。复制/发布失败会退出并保留原目录，下一次启动可重试。旧文件和历史版本不会自动删除。

健康返回的 `runtime.image_build`、`runtime.comfyui_dir` 和 `runtime.workspace_dir` 分别标识镜像指纹、实际代码目录和数据根目录。回退时先停止新 worker，再启动之前保存 digest 的镜像，模型卷和原数据仍可复用。每个 worker 使用自己的工作区，多个 worker 只共享模型卷。

## 3. Serverless 调用

部署完成后，使用自己的 Runpod API Key 请求健康检查和生成，不要把 Key 提交到仓库。

```bash
export ENDPOINT_ID='your-endpoint-id'
export RUNPOD_API_KEY='your-api-key'

curl -fsS "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"action":"health"}}'

curl -fsS "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/quality.json
```

`/run` 返回任务 ID 后调用 `/status/任务ID`。切换速度档使用 `examples/speed.json`。也可以保持环境默认，在单个请求中传 `preset=quality/speed`。较长的 `/runsync` 可能先返回进行中状态，验收脚本会继续轮询。

## 4. PRO 6000 MIG 48GB：Runpod Pod / 独立 API

公开文档将 `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb` 列为 Pod GPU 型号，但未列出对应独立 Serverless pool ID。本仓库不猜造 `BLACKWELL_48` 等池名。

1. 创建 Pod，选择精确的 **PRO 6000 MIG 48GB**，使用从根目录 `Dockerfile` 自行构建的运行环境镜像，步骤见 [GHCR 备选部署](ghcr.md)。
2. 绑定模型卷到 `/runpod-volume`，将 8000 配为 HTTP 端口。
3. 导入 `deploy/pro6000_mig_48.env` 的变量，再增加：

```dotenv
WORKER_MODE=api
WORKER_API_KEY=replace-with-a-long-random-token
WORKER_API_PORT=8000
```

4. 确认容器只看到一个 48GB MIG 设备。profile 校验使用 CUDA 实际可见显存与计算能力，并记录 SM 数，不取 `nvidia-smi` 的父设备 96GB 数据。
5. 使用 Runpod 为 8000 端口生成的 HTTPS 代理地址；把以下 `POD_API_URL` 替换为控制台实际地址。

```bash
export POD_API_URL='https://your-pod-8000.proxy.runpod.net'
export WORKER_API_KEY='the-token-set-on-the-pod'

curl -fsS "${POD_API_URL}/health" \
  -H "Authorization: Bearer ${WORKER_API_KEY}"

curl -fsS "${POD_API_URL}/run" \
  -H "Authorization: Bearer ${WORKER_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/quality.json

# 将返回的 id 填入 JOB_ID，轮询至 COMPLETED / FAILED / CANCELLED。
export JOB_ID='id-returned-by-run'
curl -fsS "${POD_API_URL}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${WORKER_API_KEY}"

# 需要取消时执行：
curl -fsS -X POST "${POD_API_URL}/cancel/${JOB_ID}" \
  -H "Authorization: Bearer ${WORKER_API_KEY}"
```

Pod API 使用相同的任务结构、图编译器和 GPU 串行锁。优先使用 `/run` + `/status/id`，避免代理因等待生成而超时；`/runsync` 也可用。取消会触发 GPU 清理，后续任务等待清理完成。

Pod 队列与结果保存在进程内，重启后丢失；最多保留 32 项任务，满时先淘汰已完成结果，仍满则返回 HTTP 429。已完成结果会在后续提交时清理过期记录（从提交起超过 1 小时）。及时保存返回的视频；需要持久任务队列时使用外部队列或 Runpod Serverless。

## 5. 自有 NVIDIA 主机 / Docker

主机需安装 NVIDIA Container Toolkit，且兼容 CUDA 13。以 RTX 6000 Ada 为例：

```bash
mkdir -p /srv/mmh3-models

docker run --rm --gpus device=0 --shm-size=8g \
  --env-file deploy/rtx6000_ada_48.env \
  -e WORKER_MODE=api \
  -e WORKER_API_KEY="${WORKER_API_KEY}" \
  -p 127.0.0.1:8000:8000 \
  -v /srv/mmh3-models:/runpod-volume \
  YOUR_REGISTRY/runpod-minimax-h3:profiles
```

L40S/5090 更换 `--env-file`。MIG 使用已有实例的 UUID，例如 `--gpus "device=${MIG_UUID}"` 和 `deploy/pro6000_mig_48.env`；不要把父设备和它的 MIG 实例同时暴露给 worker。本仓库不修改主机 MIG 划分。

本地访问 `http://127.0.0.1:8000/health`、`/run`、`/status/id`，也需要 Bearer Token。首次启动会校验全部所选权重；下载失败直接退出，下一次启动继续下载，不会把半个文件当完成。

## 6. 可选组件

新增环境变量后重启 worker，先下载再调用：

| 环境变量 | 请求参数 | 用途 |
| --- | --- | --- |
| `MMH3_DOWNLOAD_ENHANCER=true` | `enhance=true` / `analyze=true` | 提示词扩写 / Native Director 人物分析 |
| `MMH3_DOWNLOAD_SINGULARITY=true` | `model_variant=singularity` | 参考生成的可选融合模型 |
| `MMH3_DOWNLOAD_STYLE_LORAS=true` | `model_variant=singularity` 且 `lora=realism/char_swap` | 两种可选风格 LoRA |
| `MMH3_ENCODER_FAMILY=heretic` | 无需逐请求指定 | 保留原仓库的 Heretic 编码器特性，精度仍按 GPU 选择 |

`upscale=true` / `rife=true` 不需要预下载完整 PRO 6000 引擎。第一次实际用到该阶段才下载 ONNX、为当前 GPU 编译，并保存到：

```text
/runpod-volume/models/tensorrt/<profile-and-hardware-software-hash>/
```

两种后处理顺序执行、batch=1，完成后释放 context、buffers、CUDA graph 与 engine。共享相同缓存的 worker 使用文件锁串行编译，避免读取未写完的引擎。改变依赖且需要重建时设置新的 `MMH3_ENGINE_CACHE_REV`；已有旧缓存不会自动删除。插帧后同时修正输出 FPS，避免把 48fps 帧序列按 24fps 播放。

不同 GPU 共用 `/runpod-volume/models/onnx`，下载按文件加共享锁；完整下载通过临时文件原子替换，并记录大小/SHA256 完成凭据。异常中断不发布半文件，之后可重试；不同模型仍可并行下载。旧 ONNX 没有完成凭据或校验不符时，会保留为隐藏的 `*.invalid-*` 备份并重新下载一次。升级期间不要同时运行仍直接写 ONNX 的旧镜像。

## 7. 实机验收

先用固定 864×480、5 秒、同一 seed 比较两档，再提高到目标分辨率/时长。以下脚本会执行真实生成，并把返回的视频和执行数据保存在忽略的 `work/smoke/`：

```bash
# Pod / 自有 Docker
python3 scripts/smoke_test.py \
  --url "${POD_API_URL}/runsync" \
  --api-key-env WORKER_API_KEY \
  --size 864x480 --seconds 5

# Serverless
python3 scripts/smoke_test.py \
  --url "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  --api-key-env RUNPOD_API_KEY \
  --size 864x480 --seconds 5
```

验收内容：

1. `execution.profile/encoder/dit/steps` 与目标硬件和预设一致；质量档没有 Turbo LoRA，速度档按任务使用正确步数。
2. 日志及 `execution.memory_stages` 按顺序出现 `after_conditioning`、`after_sampling`、`after_video_decode`、`after_decode`，开启后处理时再依次出现 `after_upscale`、`after_rife`。每阶段只执行一次，`sequence` 在单次任务内连续，`models_after=0`。worker 和验收脚本会拒绝缺失、重复、乱序或仍有受管理模型驻留的结果；序号也能发现 ComfyUI 覆盖重复节点 UI 的情况。
3. 比较 `before_device_used_mib` 与 `device_used_mib`：两者来自 CUDA 的设备内存查询，包含当前设备上的 PyTorch 外分配。`before_allocated_mib/allocated_mib`、`before_reserved_mib/reserved_mib` 只描述 PyTorch allocator；`peak_allocated_mib` 同样只统计 PyTorch 峰值，不包含全部 AIMDO/TensorRT/驱动分配。另用监控系统采样真正的阶段峰值。上下文和仍在使用的张量会占内存，不要求释放后设备占用归零。
4. 连续跑不同提示词至少 3 次，检查显存无持续增长；分别测冷启动、首次引擎编译和稳定生成耗时。
5. 播放视频检查运动、细节、参考一致性和音频。CPU 图测试和样片成功生成都不能替代画质比较。

5090 优先保持 480p；要升到 768p，逐步提高时长和参考数量。某些请求仍会 CPU 卸载，其性能受 PCIe、系统 RAM、磁盘及 CPU 编码影响。超时不会自动改变 seed、尺寸或采样步数重试。

## 8. 本地检查与故障定位

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
bash -n serverless-entrypoint.sh populate-volume.sh

# 无 GPU 也能核对某个配置的下载清单，不会下载权重：
python3 worker_config.py manifest --profile rtx5090_32
```

健康请求可在 ComfyUI 启动中返回。请检查 `comfyui=ready`，而不仅是 `status=ok`。缺少 staging 节点、权重或不匹配硬件会给出明确错误。`ComfyUI did not become idle after abort` 表示 worker 已被隔离，重启后再处理新任务。

`Non-consecutive memory-stage sequence` 或 `Unexpected memory stages` 表示执行路径未满足单次分阶段协议，任务会失败并清理 GPU；检查是否使用了旧节点/镜像或不支持的自定义图。导入图只支持采样器的 output 0（sampled latent），其他采样输出会在入队前拒绝。`Image build manifest missing` 表示镜像未包含本次构建步骤，需要重建镜像。

默认不启用 Sage/Comfy Kitchen attention 或 Sol 稀疏 patch；原仓库也记录了某些快速启动参数与其 INT8 栈的输出问题。`COMFYUI_EXTRA_ARGS` 中覆盖显存、缓存或 attention 策略的选项会被拒绝，避免绕过预设。

模型/采样选择依据：[Comfy-Org H3](https://huggingface.co/Comfy-Org/MiniMax-H3)、[ComfyUI H3 文档](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)、[Turbo 配套参数](https://github.com/ModelTC/Minimax-H3-Turbo)、[Runpod GPU 型号和池](https://github.com/runpod/docs/blob/main/references/gpu-types.mdx)。
