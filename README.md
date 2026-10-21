# MiniMax H3：32–48GB GPU 分阶段推理 Worker

基于 [huchukato/runpod-minimax-h3](https://github.com/huchukato/runpod-minimax-h3) 的 Runpod/容器部署版本，针对 **RTX 6000 Ada 48GB、L40S 48GB、RTX PRO 6000 MIG 48GB、RTX 5090 32GB** 提供硬件配置、画质/速度预设和显式显存释放。

默认使用官方 Pruned INT8 ConvRot FL2VA / Ref2VA、按显卡选择的 32B 编码器、INT8 视频 VAE 和 FP32 音频 VAE。Singularity、Heretic 编码器、提示词增强和 TensorRT 后处理均可选。

## 硬件与两套预设

| `MMH3_GPU_PROFILE` | GPU | 编码器 | 画质档默认尺寸 | 速度档默认尺寸 |
| --- | --- | --- | --- | --- |
| `rtx6000_ada_48` | RTX 6000 Ada 48GB | INT8 ConvRot | 1344×768 | 864×480 |
| `l40s_48` | L40S 48GB | INT8 ConvRot | 1344×768 | 864×480 |
| `pro6000_mig_48` | PRO 6000 Blackwell 的 48GB MIG 实例 | NVFP4 AWQ | 1344×768 | 864×480 |
| `rtx5090_32` | RTX 5090 32GB | NVFP4 AWQ | 864×480 | 864×480 |

`MMH3_GPU_PROFILE=auto` 根据**容器可见的设备与显存**选择，检查 CUDA、计算能力及容量。MIG 配置要求只暴露一个 48GB 实例，不能用 96GB 父设备冒充。Runpod 的 `ADA_48_PRO` 池也包含 L40；自动配置会使用相同的 Ada 模型组合，并在 health 中报告实际型号。

- **`quality`**：原生 20 步，`res_multistep` + `simple`，移除 Turbo LoRA。
- **`speed`**：FL2VA 使用公开配套参数的 768p 8 步 Turbo，`euler` + `simple`；官方 Ref2VA 使用 4 步 Turbo，`euler` + `simple`。参考一致性、细节和音频需与画质档比较。
- 两档默认均为 5 秒、24fps，不启用提示词增强、插帧或超分。请求可显式覆盖尺寸/时长；5090 不会被悄悄改成与 48GB 卡相同的默认分辨率。

这些是待目标 GPU 验收的起始配置，并非四款显卡的实测性能保证。高分辨率、长片段和复杂参考输入仍可能需要 CPU 卸载或降低尺寸。

## 分阶段加载

`可选增强（用完释放） → H3 条件编码 → 释放编码器/参考 VAE → 单 DiT 采样 → 释放 DiT → 视频分块解码 → 释放视频 VAE → 音频解码 → 释放音频 VAE → 可选超分 → 释放引擎 → 可选插帧 → 释放引擎`

- `custom_nodes/mmh3_memory` 在 **ComfyUI 进程内**释放模型，不依赖只清 CUDA allocator cache。
- 图依赖强制视频/音频解码和后处理的执行次序；TensorRT 的编译和加载也在解码后才执行。
- 音频保留对同一采样 latent 的普通依赖，避免 `--cache-none` 下 lazy 输入重跑采样。返回结果前校验阶段顺序、连续释放序号和受管理模型数量；异常任务不会按成功返回。
- Director 会移除未选中的模型输入；不会为了使用 Ref2VA 同时加载 FL2VA。
- 每个 worker 一次只执行一个 GPU 任务。取消/超时后先中断并等待 ComfyUI 空闲；未能清空的 worker 拒绝后续任务。
- 核心使用 PyTorch attention，关闭异步卸载和跨请求节点缓存；视频采用空间/时间分块解码。权重可能保留在系统内存或通过磁盘映射访问，分阶段释放指 GPU 驻留。
- TensorRT 缓存按 GPU 型号、可见 MIG 容量/SM 数、Torch/CUDA 与缓存版本隔离；相同缓存下的引擎构建使用文件锁。共享 ONNX 使用跨 GPU 文件锁、原子下载及 SHA256 完成凭据。不下载上游针对完整 PRO 6000 的预编译引擎。

## 部署

首选 **Runpod 的 GitHub 构建与自带镜像仓库**：导入仓库后选择下面的 Dockerfile，由 Runpod 构建、托管和部署，无需先使用 GHCR。[完整步骤](docs/deployment.md)

| 配置 | Dockerfile Path | GPU pool | 默认编码器 |
| --- | --- | --- | --- |
| Ada 主配置 | [`deploy/runpod/rtx6000_ada_48.Dockerfile`](deploy/runpod/rtx6000_ada_48.Dockerfile) | `ADA_48_PRO` | 官方 INT8 ConvRot |
| 5090 备选配置 | [`deploy/runpod/rtx5090_32.Dockerfile`](deploy/runpod/rtx5090_32.Dockerfile) | `ADA_32_PRO` | 官方 NVFP4 AWQ |

每个镜像只包含本配置的编码器、VAE 和 Turbo LoRA，权重约 34.47GB / 23.02GB；两个 DiT 共约 **41.94GB** 放按需创建的网络卷（建议 80GB），用完保存结果并删除卷。镜像中的权重直接校验、链接读取，**不需要在容器临时盘还原大编码器**，可从 50GB 容器盘开始实机验收。缓存和输出应及时另存。

Runpod GitHub 构建限制为 **Docker build 30 分钟、最终镜像 80GB**，基础环境和依赖也计入最终镜像，需用实际云端构建确认。若超限，可改用根目录 `Dockerfile`，仍由 Runpod 构建托管，全部模型放到按需释放的完整模型卷（建议 160GB）。[GHCR 方案](docs/ghcr.md)保留为后续备选。

Core 固定为 **ComfyUI v0.38.0**，基于固定 digest 的 cu130 镜像，需要 CUDA 13 兼容驱动、单个可见 GPU 和充足系统内存（建议 128GB）。两个 Runpod Dockerfile 从根目录 Dockerfile 生成，修改公共构建逻辑后执行：

```bash
python3 scripts/render_runpod_dockerfiles.py
python3 scripts/render_runpod_dockerfiles.py --check
```

启动时按镜像代码/依赖指纹和 GPU 缓存键准备独立运行目录，校验代码文件，并始终使用镜像 Python。已有 `main.py`、旧虚拟环境和自定义节点保留在原位置；`input/output/user` 共享原数据目录。具体迁移与回退步骤见部署文档。

两个固定镜像分别使用 `ADA_48_PRO` 和 `ADA_32_PRO`。通用完整模型卷 / Hub 入口支持这两个池和 `auto` 检测，不再锁定 `BLACKWELL_96`。官方公开池列表未给 48GB MIG 单独的 Serverless pool ID；MIG 使用精确 GPU 型号的 Pod/API 部署，勿猜造 pool ID。

## 调用

```json
{
  "input": {
    "task": "t2v",
    "preset": "quality",
    "prompt": "A cinematic tracking shot through a quiet mountain village at sunrise, with natural ambient sound.",
    "seconds": 5,
    "seed": 42
  }
}
```

- `task`：`t2v`、`i2v`、`fl2v`、`r2v`、`director`、`storyboard`。
- `preset`：`quality` / `speed`；请求覆盖环境变量 `MMH3_PRESET`。
- `video_size`：显式尺寸，例如 `1344x768`、`768x1344`、`864x480`。尺寸为 32 的倍数，边长至少 256，总像素不超过 1344×768；为保证可预期的内存预算，不再接受 `auto`。
- `seconds`：1–15 的整数；实际帧数按 H3 帧网格对齐。更长视频拆分为多个请求。
- `images`：最多 3 个 URL/base64；`video` 可提供参考视频。`r2v`、Director 和 Storyboard 默认使用 `ref_image_size=match`，避免把参考图无条件扩大到 2048。
- `enhance`、`analyze`、`upscale`、`rife`：布尔值，默认 `false`。`analyze` 是 Native Director 的人物分析开关。
- `model_variant=singularity`：需事先启用下载。此可选配方的速度档保留原仓库的 6 步 `euler/beta`，不与官方 Ref2VA 4 步参数混用。
- 返回 `outputs`、`texts` 和 `execution`；后者含最终模型、预设、尺寸、步数、执行耗时和阶段显存记录。

可继续传旧 `workflow` 文件名或 `prompt_graph`。`config=native/native_turbo/r2va_native/r2va_native_turbo/r2va_singularity/r2va_singularity_turbo` 映射到相应新预设；与显式 `preset` 冲突时会报错。未提供的 10Eros 配方不再静默回退。通用 `params` 仍可修改普通控件，模型、驻留策略和采样配置由编译器接管。

7 个原工作流文件名保留为 API 兼容入口，名字里的 `Turbo` 不决定最终模式；请求的 `preset` 才决定。运行时图只支持一次 H3 条件编码、一次采样及一对视频/音频解码；多次 refine 的复杂导出图需拆分任务。

## 模型来源和可选功能

模型目录清单是 [models-manifest.txt](models-manifest.txt)；启动时从中生成硬件专用下载清单，仅选一个编码器。两套 DiT 在磁盘上共存，每次任务只保留一个可达生成分支。

- [Comfy-Org H3](https://huggingface.co/Comfy-Org/MiniMax-H3)：默认 FL2VA/Ref2VA、官方编码器和 VAEs。
- [LightX2V Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)：保留有公开配套采样参数的版本；未把新文件名直接等同于更高质量。
- `MMH3_ENCODER_FAMILY=heretic`：Ada 下载 [Heretic INT8](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot)，Blackwell 下载 [Heretic NVFP4](https://huggingface.co/Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4)。
- `MMH3_DOWNLOAD_ENHANCER=true`：允许 `enhance=true` / `analyze=true`。
- `MMH3_DOWNLOAD_SINGULARITY=true`：允许选择 Singularity v1.3 Pruned INT8。
- `MMH3_DOWNLOAD_STYLE_LORAS=true`：允许 Singularity 的 `lora=realism/char_swap`。

可选权重不会默认下载。卷模式下 `VOLUME_AUTOPOPULATE=false` 只关闭自动下载，启动仍检查所选权重存在；全镜像模式 `MMH3_MODEL_SOURCE=image` 不执行卷下载脚本，混合模式 `hybrid` 只下载卷中的两个 DiT；修改 GPU 配置、编码器或开启额外权重须重新打包。旧卷中的已有权重不会自动删除。ForgeHub 远程工作流同步默认关闭，显式设置 `WORKFLOW_SYNC_ENABLED=true` 才会同步。

## 检查

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
bash -n serverless-entrypoint.sh populate-volume.sh
python3 scripts/render_runpod_dockerfiles.py --check
python3 worker_config.py manifest --profile rtx6000_ada_48
```

这些检查覆盖配置、全部原工作流的图依赖、阶段释放接口、HTTP 协议、并发/取消和下载失败处理。固定版本的真实 `ExecutionList/NullCache` 会执行 224 种工作流/GPU/预设/后处理组合，断言每任务采样一次、音视频使用同一 latent、各阶段释放后才开始下一阶段；另覆盖跨 GPU 多进程下载及旧目录迁移/失败回滚。

调度测试使用 CPU 代替 CUDA 计算，不测实际显存容量。部署后用 [smoke_test.py](scripts/smoke_test.py) 在真实 GPU 上生成两档样片，检查耗时、释放前后设备占用、音视频及画质。测试夹具来自 ComfyUI v0.38.0 固定提交，来源/哈希和 GPL 许可证见 `tests/fixtures/comfyui_v0_38_0/`。
