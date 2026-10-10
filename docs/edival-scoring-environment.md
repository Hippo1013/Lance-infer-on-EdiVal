# EdiVal 官方评分环境

本文维护 EdiVal 官方评分组件的权重、独立环境和工程检查。用户已要求使用官方模型与评分逻辑，对既有 Lance EdiVal 512 bare 结果评分。2026-10-06 八项模型资源、三个独立环境与 GPU1 组件检查均通过；全量接入与双卡运行状态见 [官方评分接口](edival-scoring.md)。完整准备证据在服务器 `outputs/setup/edival_official_20261006/`，终态为其中 `summary.json`。

## 官方来源与模型身份

EdiVal 固定版本为 [96d34b0](https://github.com/TianyuCodings/EdiVal/tree/96d34b00d7ea2dc3de90f2bb01f292f6dec6294a)。服务器原样保存 85 个源码、配置与说明文件于 `/home/chs/tools/EdiVal/96d34b00d7ea2dc3de90f2bb01f292f6dec6294a/`；`source.json` 记录原始归档与逐文件 SHA256。评分函数未修改。

所有模型位于 `/home/chs/model/`，HF 固定 revision 见 [模型清单](../configs/edival_scoring_models.json)。

| 目录 | 官方模型与用途 |
| --- | --- |
| `Qwen2-VL-7B-Instruct` | `Qwen/Qwen2-VL-7B-Instruct`；IF 图片问答，保持官方模型 |
| `GroundingDINO-SwinT-OGC` | 官方 Swin-T OGC；对象检测与定位，复用既有权重 |
| `bert-base-uncased` | `google-bert/bert-base-uncased`；GroundingDINO 文本编码器，复用既有文件 |
| `DINOv3-ViT-B-16` | `facebook/dinov3-vitb16-pretrain-lvd1689m`；CC，使用官方 B/16 而非既有 L/16 |
| `RAHF` | `C-Tianyu/RAHF` 的 `rahf_model.pt`；质量评分 |
| `vit-large-patch16-384` | `google/vit-large-patch16-384`；RAHF 图像基础模型 |
| `t5-base` | `google-t5/t5-base`；RAHF 文本基础模型与 tokenizer |
| `HPSv3` | `MizzenAI/HPSv3` 的 `HPSv3.safetensors`；人类偏好分数 |

DINOv3-B 从共享目录 `/media/vlm_model/asset_llm_ckpt/dinov3-vitb16-pretrain-lvd1689m/` 实际复制到自己的模型目录，六个文件逐一校验，模型目录的 `source.json` 留存来源；不是软链接。342662192 字节权重 SHA256 为 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`，与固定官方 revision 发布的 LFS SHA256 一致。因此这份本地权重的使用不再依赖新下载授权。

下载器核对大小、上游 LFS SHA256、safetensors 头及 Qwen 的全部分片和 tensor index。GroundingDINO/BERT 另对照既有下载 manifest 重新计算哈希。GroundingDINO 的 34 个 Python 文件与 EdiVal 附带源码逐字节一致；原生 CUDA 扩展沿用先前的 Tensor API 兼容迁移，不修改检测算法或阈值。

## 独立运行环境

| Conda 前缀 | 组件与主要依赖 |
| --- | --- |
| `/home/chs/conda/envs/EdiVal` | GroundingDINO、DINOv3-B、RAHF；Python3.12、Torch2.13.0、Torchvision0.28.0、Transformers4.57.6 |
| `/home/chs/conda/envs/EdiVal-judge` | Qwen2-VL 官方权重；vLLM0.30.0、Transformers5.14.1 |
| `/home/chs/conda/envs/EdiVal-hps` | 官方 PyPI `hpsv3==1.0.0`；Transformers4.45.2、TRL0.11.4、PEFT0.13.2、Accelerate0.34.2 |

三个环境由现有环境克隆后独立安装，原 Lance/MICE 环境不作修改。这里的冲突是运行库的版本约束：现有 vLLM0.30.0 要求 Transformers≥5.10.4，官方主环境最后使用4.57.6，而 HPSv3 包固定4.45.2。单独的 judge 环境是复用已验证 vLLM 栈的工程选择，不是 Qwen 权重本身存在冲突，也不是官方要求三个环境。官方同样将 HPS 放在另一环境，并用 `update_hps_scores.py` 补齐。

沿用服务器 CUDA13 与 `/usr/local/cuda-13.0/compat`，不修改宿主驱动。Judge 环境另安装 Conda `libxcb/libgl/libglib` 以支持 OpenCV；运行进程将自己的 `lib` 加入动态库路径，调用系统 tmux 前恢复原路径。HPS 使用官方无 FlashAttention 时的 SDPA 路径。

Qwen 检查固定 `VLLM_USE_V2_MODEL_RUNNER=0`，使用 vLLM 自带旧执行后端。默认 V2 后端在本配置的预热阶段触发 CUDA illegal memory access，尚未产生任何问答；切换后单图/双图 helper 均通过。开关来自 [vLLM 官方环境变量](https://docs.vllm.ai/en/stable/configuration/env_vars/)。早期缺 OpenCV 动态库和 V2 后端失败均保留原日志；不是评分样本失败或有效裁判答案。

## 安装入口与证据

以下命令在服务器项目 `/home/chs/exp0_attention/Lance-infer-on-EdiVal` 执行；代理、GPU1 burn 切换遵循本地工作区 `AGENT.md`。安装器拒绝直接改写已有同名环境，重新安装前须检查现场。

```bash
source /home/chs/net_proxy/proxy-off.sh
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
/home/chs/conda/envs/mice-judge/bin/python scripts/download_edival_scoring_models.py \
  --output outputs/setup/edival_official_20261006
bash scripts/install_edival_scoring.sh
```

`downloads.json` 与逐模型 `*.manifest.json` 保存下载/校验身份；`reused-models-validation.json` 保存复用模型与 GroundingDINO 源码核对；`import-checks-final.json`、各环境 `*.pip-check.txt`、`*.freeze.txt`、`*.conda-explicit.txt` 保存依赖身份。早期失败日志单独保留，最终验收引用通过的检查文件。

## GPU 组件检查与评分边界

在按工作区规程释放 GPU1 后执行：

```bash
bash scripts/run_edival_scoring_smoke.sh 1 \
  /home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/setup/edival_official_20261006/new_validation
```

包装脚本仅允许物理 GPU1，检查显存空闲、使用唯一 `/tmp` 缓存目录，结束后清理临时文件并在确认 GPU/pane 均空闲后恢复同卡 burn。只检查指定组件时可设 `EDIVAL_SMOKE_COMPONENTS="metrics judge"` 或 `"hps"`；检查产物使用独立目录，旧结果保留。

组件检查使用既有真实会话0的原图与首轮结果，全部离线加载本地模型。Metrics 调用官方检测、DINOv3 对象/遮罩背景特征及 RAHF 质量函数；RAHF checkpoint 严格匹配全部模型键。Judge 执行原样上游单图/双图 VLM helper，记录实际响应与 token 数，避免官方捕获异常后默认返回 `no` 被误判为检查成功。HPS 使用官方 inferencer 严格加载 checkpoint，以空文本计算两张图的两维原始输出，评分沿用第一维；只替换基础模型本地路径与临时输出路径。

检查中的 RAHF 结果不含 HPS；HPS 在自己的环境单独验证。所有组件均通过后才称环境就绪。这些是加载、运行和来源检查，不是 IF/CC/VQ 的全量结果、裁判准确性校准或 Lance 能力结论。

通过的 metrics 证据为 `validation_metrics_judge/metrics-smoke.json`；最终 Qwen/HPS 证据为 `validation_judge_hps_legacy/judge-smoke.json` 与 `hps-smoke.json`，该包装任务退出0。检测在真实原图找到目标对象；DINOv3 特征为 `[2,201,768]`、同图对象与背景相似度在 `1e-5` 内接近1；RAHF 所有质量计算有限；Qwen 两次实际调用返回非空回答；HPS 严格加载并返回有限的 `[2,2]` 输出。HPS 峰值 PyTorch 分配约16.2GiB；Qwen 使用子进程，父进程的峰值0不表示模型未占显存。

实际联用评分接口及两卡调度见 [官方评分接口](edival-scoring.md)。GroundingDINO须在主环境运行：Transformers5中的BERT缺少其依赖接口，judge环境的独立Qwen验收不代表可以直接联用检测器；IF阶段以同卡主环境子进程运行原样检测函数。

完整评分将既有 `run/edival/<index>/turn_n.png` 显式映射为官方 multipass 输入，并将不同环境的组件接入评分入口。模型、prompt、阈值、CC token 处理与聚合规则沿用固定官方版本；环境检查没有改写推理图像、attention 或已有评分产物。
