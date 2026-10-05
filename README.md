# Lance-infer-on-EdiVal

UMM 课题的 exp0 预实验：使用 Lance 在 MICE、ImgEdit Multi-Turn 和 EdiVal 上进行完整交错历史推理，测量历史约束、指代、版本回溯和误差累积，再分析并干预 attention。

工程验收保证输入与计算正确、公平；编辑失败作为研究数据保留。不改写 benchmark 指令、不挑样本或种子、不替换失败历史、不重试择优，不将评测答案或对象标注输入编辑模型。

## 当前阶段

2026-10-05：MICE 和 ImgEdit 的 bare-v2 全量推理已完成并通过复核；MICE 正在 GPU0 上进行 **Qwen 单票 v3 评分**，EdiVal 正在 GPU1 上进行 **512×512 全量推理与 attention 保存**。ImgEdit bare 结果的 88 轮人工表已全部填写。attention 的系统分析和干预尚未开展。

任务路径、带时间的进度快照、终态条件与部署差异集中在 [exp0 状态与交接](docs/exp0-status.md)。模型、数据、评分结果和 attention 留在服务器，不纳入 Git。

## 文档入口

| 内容 | 文档 |
| --- | --- |
| 全部文档与证据索引 | [文档目录](docs/README.md) |
| 当前 MICE 单票 v3、人工未决及不完整回答 | [Qwen 单次评分](docs/scoring-qwen-once.md) |
| 历史双裁判规则及官方差异 | [评分协议](docs/scoring-protocol.md) |
| 精确 IF/GA 正文、字段和图像顺序 | [裁判提示词](docs/scoring-prompts.md) |
| 评分环境与历史工程验收 | [环境](docs/scoring-environment.md)、[验收](docs/scoring-validation.md) |
| EdiVal 数据、512 尺寸及全量入口 | [EdiVal 推理](docs/edival-inference.md) |
| attention 格式、计算与解释边界 | [Attention 观测](docs/attention-observation.md) |
| 提示封装诊断与原生速度参考 | [伪文字诊断](docs/text-artifact-diagnosis.md)、[速度试跑](docs/native-speed-probe.md) |

## 结果查看

服务由服务器提供，通过 SSH 转发到本机回环地址；中文仅供浏览。

| 数据与用途 | 本机地址 | 说明 |
| --- | --- | --- |
| MICE bare，CM/CU 各 15 会话抽查 | <http://127.0.0.1:8768> | [MICE 页面](docs/mice-review.md) |
| ImgEdit bare，30 会话/88 轮人工评分 | <http://127.0.0.1:8769> | [ImgEdit 页面](docs/imgedit-human-review.md) |
| EdiVal 全部 572 会话，动态进度 | <http://127.0.0.1:8770> | [EdiVal 页面](docs/edival-review.md) |

旧有标签结果及 8765/8767 页面保留独立历史身份，不能与 bare-v2 的清单或评分库混用。人工统计使用项目协议，不能称为官方榜单分数。

## 环境与资源

服务器项目：`/home/chs/exp0_attention/Lance-infer-on-EdiVal`。

| 资源 | 路径 / 版本 |
| --- | --- |
| Lance 环境 | `/home/chs/conda/envs/lance`，Python 3.12.14 |
| 推理依赖 | Torch 2.13.0+cu130、vLLM 0.30.0、Transformers 5.14.1、Diffusers 0.40.0 |
| vLLM-Omni | `b742f86136d28423903a1213adde2a62a11067a2` |
| Lance 权重 | `/home/chs/model/Lance`，revision `7395315758865e6f56ab87ad06a88c7ac172f056` |
| MICE | `/home/chs/dataset/MICE-Bench`，CM/CU 各 360 个三轮会话 |
| ImgEdit | `/home/chs/dataset/ImgEdit-Bench-Multi-Turn/multiturn`，30 会话/88 轮 |
| EdiVal | `/home/chs/dataset/EdiVal`，572 会话/1716 轮，CSV＋ZIP |

```bash
source /media/damoxing/tangzecong/miniconda3/etc/profile.d/conda.sh
conda activate /home/chs/conda/envs/lance
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
```

重建环境入口为 `scripts/install_lance.sh`；可设置 `CONDA_EXE`、`CONDA_ENVS_PATH`。安装前检查现有环境，沿用已有 CUDA 13 兼容库，不修改宿主驱动。评分另用 `mice-metrics`、`mice-judge` 环境。

项目为 editable 安装，部署须明确文件并核对 SHA-256。运行中的冻结源码不能覆盖；GitHub 版本同步不等于活动服务器目录已统一部署。GPU 使用只处理本次所用卡的 burn，保持 monitor/watchdog 原状；设施操作在本地工作区 `AGENT.md` 维护，不上传实验仓库。

## 输入与推理协议

- 第 n 轮提供 `I0,T1,I1,T2,...,Tn`；历史图像均为本模型保存后重新读入的真实 RGB 输出。
- 当前 `lance-history-bare-v2` 保留官方 image-edit 系统提示，user 仅使用原始指令和交错图像，不加历史说明、HISTORY/CURRENT EDIT 或轮次标签。空 label 仅作零 token 的 CFG/缓存边界。
- 每张历史图提供 ViT/VAE 条件；CFG-negative 仅移除当前指令正文，完整历史保留。
- 30 步、shift 3.5、text/image CFG 4/1、区间 `(0.4,1]`、global/min=0、BF16、base seed42；种子按会话、轮次及用途派生，沿用 v1 RNG 命名空间。
- MICE/ImgEdit 使用 768 面积桶；EdiVal 直接以 512×512 生成、保存并回填。ViT 沿用 Lance 的条件编码设置。
- `none` 重算历史，`images` 缓存编码，`prefix` 再缓存历史 K/V。会话结束清空缓存；DP2 按完整会话分片，每卡一个完整模型。

输入协议、配置与数据身份进入运行指纹；恢复拒绝历史缺口、篡改或不同协议，不用当前源码冒充历史验收版本。EdiVal 完整历史输入是项目选择，区别于官方 multipass 的当前指令＋上一轮图。

## 检查与运行入口

以下 CPU 命令不加载模型；EdiVal 在服务器上使用其独立 runtime 执行。

```bash
python -B -m lance_mice.runner --selection all --profile dp2 --dry-run
python -B -m lance_mice.imgedit --selection all --dry-run
python -B -m lance_mice.edival --selection all --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

全量入口会启动真实任务，仅用于另一次已授权的新运行，输出须为新目录：MICE 为 `scripts/full_mice_job.py`，ImgEdit 为 `scripts/full_imgedit_job.py`，EdiVal 为 `scripts/run_edival_job.sh`。MICE/ImgEdit 要求 bare 验收，EdiVal 要求与512验收一致的源码/数据。活动任务不能重复启动；当前评分命令与恢复边界见 [Qwen 协议](docs/scoring-qwen-once.md)。

推理产物为 `run/<split>/<sample_id>/session.json`、`turn_0_input.png` 和逐轮 `turn_n.png/json/attention.npz`；EdiVal 的 split 为 `edival`。状态每20秒更新，完成复核源图/指令、真实历史、CFG/seed、覆盖、attention及冻结源码；失败保留现场，不择优重跑。

## 证据与解释边界

MICE bare 全量为720会话2160轮，ImgEdit bare 全量为30会话88轮，两项 `validation.json` 均 passed。工程复核证明覆盖和来源正确，不能替代能力评分。

attention 为 `target-group-mass-v1`：保存全部30步36层16head的上下文组概率质量，对目标图像query空间取均值；图像ViT/VAE分组可相加。只记录正向条件分支，不是完整query×key矩阵，不能声称得到token级熵或因果结论。

旧temperature=0双裁判的12/120轮验收保留原配置身份；不能验证当前单票v3裁判准确率。人工解决与自动遍历完成独立报告。替代裁判、CC适配、缺项分母和累计GA均有明确项目协议边界。

上游来源与许可范围见 [第三方说明](THIRD_PARTY_NOTICES.md)。本仓库不包含模型权重、数据集或运行结果。
