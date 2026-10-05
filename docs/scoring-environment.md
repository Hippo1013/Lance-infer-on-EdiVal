# MICE 评分环境

本文记录模型、独立运行环境和最初 GPU 冒烟验证。后续评分适配及历史小样本验证见文末链接；当前Qwen单票v3全量评分已启动，见 [单次评分协议](scoring-qwen-once.md)。

2026-10-04 验收通过：5 项模型资源完整下载，两个环境 `pip check` 通过，GroundingDINO / DINOv3 和两位裁判的 GPU 冒烟通过。证据为 `outputs/setup/scoring_20261004/summary.json`；原全量推理的 16 个源码哈希保持不变。

## 模型资源

权重位于服务器 `/home/chs/model/`，固定 revision 见 `configs/scoring_models.json`。

| 目录 | 来源与用途 |
| --- | --- |
| `GroundingDINO-SwinT-OGC` | IDEA-Research 官方 Swin-T OGC，检测对象；包含原始配置与仅修改 BERT 本地路径的 `.local.py` 配置 |
| `DINOv3-ViT-L-16` | `timm/vit_large_patch16_dinov3.lvd1689m`，非目标区域特征 |
| `bert-base-uncased` | `google-bert/bert-base-uncased`，GroundingDINO 文本编码器 |
| `Qwen3.6-27B` | `Qwen/Qwen3.6-27B`，第一位裁判 |
| `gemma-4-31B-it` | `google/gemma-4-31B-it`，第二位裁判，指令版本 |

下载器检查文件大小、HF LFS 上游 SHA-256、safetensors 头与分片索引；普通文件记录本地 SHA-256。GroundingDINO release asset 未假定存在上游哈希，记录本地 SHA-256 并读取其 state dictionary。单纯下载完成不表示模型已经能推理。

## 运行环境

| Conda 前缀 | 主要依赖 |
| --- | --- |
| `/home/chs/conda/envs/mice-metrics` | Python 3.12、Torch 2.13.0、Torchvision 0.28.0、Transformers 4.53.3、timm 1.0.25、GroundingDINO 原生 CUDA 扩展 |
| `/home/chs/conda/envs/mice-judge` | Python 3.12、vLLM 0.30.0、Transformers 5.14.1；Qwen/Gemma 共用 |

Lance 和共享环境不作修改。CUDA 13 沿用服务器 `/usr/local/cuda-13.0/compat`，不更换宿主驱动。OpenCV 所需 `libxcb`、`libgl`、`libglib` 安装在 metrics Conda 前缀中；模型进程将该前缀的 `lib` 加入动态库路径，调用系统 tmux 前恢复原路径，避免 Conda 的 `libtinfo` 覆盖系统版本。

GroundingDINO 固定源码 `856dde20aee659246248e20734ef9ba5214f5e44`；安装器将旧 Tensor API 的 `.type().is_cuda()`、dispatch `.type()`、`.data<T>()` 分别替换为 `.is_cuda()`、`.scalar_type()`、`.data_ptr<T>()`，不改检测算法或阈值。GPU 冒烟将原生扩展与上游纯 PyTorch 参考实现比较。

DINOv3 使用 timm 加载所下载的 timm 权重。其特征有 CLS 和 4 个 register tokens；评分适配已区分这些前缀与空间 patch，并验证对应遮罩几何；不能直接将所有 `features[:, 1:]` 当作 patch 网格。CC 与上游的差异见评分协议。

## 安装与检查

在服务器项目根目录执行。网络代理、GPU burn 启停遵循本地工作区 `AGENT.md`；每个新下载 shell 单独设置代理。

```bash
source /home/chs/net_proxy/proxy-off.sh
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

/home/chs/conda/envs/lance/bin/python scripts/download_scoring_models.py \
  --output outputs/setup/scoring_20261004
bash scripts/install_scoring.sh
```

安装证据统一写入 `outputs/setup/scoring_20261004/`：`downloads.json`、各模型 `.manifest.json`、`environment.status`、安装日志、两个 `*.freeze.txt` 与 `*.conda-explicit.txt`。安装器临时目录位于系统 `/tmp`，退出时清理。下载和安装在后台 tmux `mice-model-download`、`mice-env-setup` 中运行时，避免重复启动。

按工作区规程释放指定 GPU 后运行：

```bash
bash scripts/run_scoring_smoke.sh 0 metrics \
  --output outputs/setup/scoring_20261004/metrics-smoke.json
bash scripts/run_scoring_smoke.sh 0 judge --model /home/chs/model/Qwen3.6-27B \
  --output outputs/setup/scoring_20261004/qwen-smoke.json
bash scripts/run_scoring_smoke.sh 1 judge --model /home/chs/model/gemma-4-31B-it \
  --output outputs/setup/scoring_20261004/gemma-smoke.json
```

包装脚本拒绝在已有显存占用的卡上启动；正常或失败退出后检查本次使用的 GPU 与原 pane，仅在显存已释放且 pane 无子进程时恢复该卡 burn。所有测试缓存进入唯一 `/tmp` 目录，退出时清理。monitor/watchdog 不作操作。运行中的脚本不得改写。

Metrics 冒烟验证检测权重键、原生 CUDA 扩展数值、真实源图的对象检测、DINOv3 patch 网格及同图/异图特征。Judge 冒烟采用 BF16、单卡、8192 上下文、一次四图输入，检查模型返回的颜色顺序。它们是环境检查，不是裁判准确性校准、正式评分或完整评分 HTTP 服务验收。

本次原生扩展对参考实现最大绝对误差 `5.96e-8`；DINOv3 输出为 `[3,261,1024]`，其中前缀 5 个、空间 patch 256 个；同图与异图的 CLS 余弦相似度分别约 `1.0`、`0.01934`。两位裁判均正确返回四图颜色顺序。最终 metrics 证据为 `metrics-smoke-final.json`；早期缺库/收尾问题记录保留在 setup 目录中，不混作评分结果。

后续已实现现有生成结果适配、分裁判保存与平均、异常/缺项处理，并通过 12 轮覆盖测试和 120 轮双裁判试评工程验收。见 [评分协议](scoring-protocol.md) 与 [评分验收](scoring-validation.md)。独立人工裁判校准尚未完成；当前全量采用Qwen单票v3，不自动启用Gemma或第二票。
