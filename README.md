# Lance-infer-on-EdiVal

资源准备见 [实验资源清单](docs/experiment-resources.md)，历史实验实现见 [源码版本索引](versions/README.md)。

UMM课题的exp0：用Lance在MICE、ImgEdit Multi-Turn和EdiVal上进行完整交错历史推理，测量历史约束、指代、版本回溯和误差累积，再分析并干预attention。

工程验收保证输入、来源和计算正确；编辑失败保留。不改原指令、不挑样本/种子、不替换失败历史、不重试择优，不把评测答案或对象标注输入编辑模型。模型、数据、图像、attention和评分产物不入Git。

## 状态与文档入口

2026-10-07双协议六组推理与四组评分全部完成，共2644会话7928轮、184.09 GB attention；MICE仍有16个首次回答组件待人工复核。[六组结果](docs/sixrun-results-20261007.md)提供分数、分母和验收结论，[实验说明](docs/sixrun-20261007.md)提供协议与恢复边界，[存储布局](docs/storage-layout-20261007.md)提供迁移后的产物位置。[exp0状态与交接](docs/exp0-status.md)区分当前与历史运行；[文档索引](docs/README.md)提供完整入口。

2026-10-10 EdiVal七项attention描述分析已完成：v2观测修复真实VAE缓存标签，保留首次历史并逐像素复现原输出。分析源码、CPU验证及验收入口见[Attention分析](docs/attention-analysis.md)，观测适用范围见[语义验收](docs/attention-semantic-validation.md)。干预和训练尚未开展。

以下旧版实验与查看页独立保留：[EdiVal 512推理](docs/edival-inference.md)于2026-10-06完成572会话1716轮/attention及全部页面图片复核；[官方能力评分](docs/edival-scoring-results.md) 已完成并通过终态及独立复核；MICE [第一版评分结果](docs/mice-first-edition-results.md) 已完成Qwen单票及31项人工补评，原自动协议见 [Qwen单票v3](docs/scoring-qwen-once.md)。

结果页经SSH转发到本机回环地址，中文仅供浏览；旧有标签页面/评分库保持独立身份。

| 查看范围 | 本机地址 | 说明 |
| --- | --- | --- |
| 最新六组 bare/chat，三 bench 全量对照 | <http://127.0.0.1:8775/?bench=edival> | [六组查看器](docs/sixrun-review.md) |
| MICE单票输出失败31项，已完成补评 | <http://127.0.0.1:8772> | [人工窗口](docs/mice-pending-human-review.md) |
| MICE bare，CM/CU各15会话 | <http://127.0.0.1:8768> | [抽查页面](docs/mice-review.md) |
| ImgEdit bare，30会话88轮人工表 | <http://127.0.0.1:8769> | [人工评分](docs/imgedit-human-review.md) |
| EdiVal全部572会话1716轮，复核完成 | <http://127.0.0.1:8770> | [只读页面](docs/edival-review.md) |

## 环境与资源

两机均已部署六组实验环境与冻结runtime，各自工作根目录为 `/home/chs`；连接使用 `ssh a800_0`、`ssh a800_1`。下表沿用0号机资源身份，1号机部署和六个本机环境基础导入已验收，具体见存储布局。两机同名目录不是共享目录。服务器项目：`/home/chs/exp0_attention/Lance-infer-on-EdiVal`。

| 资源 | 路径 / 版本 |
| --- | --- |
| Lance环境 | `/home/chs/conda/envs/lance`，Python3.12.14 |
| 推理依赖 | Torch2.13.0+cu130、vLLM0.30.0、Transformers5.14.1、Diffusers0.40.0 |
| vLLM-Omni | `b742f86136d28423903a1213adde2a62a11067a2` |
| Lance权重 | `/home/chs/model/Lance`，revision `7395315758865e6f56ab87ad06a88c7ac172f056` |
| MICE / ImgEdit | `/home/chs/dataset/MICE-Bench`、`/home/chs/dataset/ImgEdit-Bench-Multi-Turn/multiturn` |
| EdiVal | `/home/chs/dataset/EdiVal`，CSV＋ZIP |

```bash
source /media/damoxing/tangzecong/miniconda3/etc/profile.d/conda.sh
conda activate /home/chs/conda/envs/lance
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
```

重建入口 `scripts/install_lance.sh`，可设 `CONDA_EXE/CONDA_ENVS_PATH`。安装前检查已有环境，沿用CUDA13兼容库，不改宿主驱动。评分使用独立mice-metrics/mice-judge，见 [环境说明](docs/scoring-environment.md)。

EdiVal 官方评分组件另使用 EdiVal/EdiVal-judge/EdiVal-hps；八项模型资源和 GPU1 组件检查已通过，完整版本见 [EdiVal 评分环境](docs/edival-scoring-environment.md)，官方 multipass 接入与双卡分片入口见 [EdiVal 评分接口](docs/edival-scoring.md)。评分已完成；累计IF、官方CC/O与VQ Δ见 [论文格式评分表](docs/edival-paper-tables.md)。开发工作树、GitHub提交、服务器部署与冻结runtime分开核对；同步方法见[源码同步](docs/source-sync.md)。

项目为editable安装，部署须明确文件并核对SHA-256；活动冻结源码不得覆盖。GitHub同步不代表服务器统一部署。GPU任务只处理本卡burn，monitor/watchdog保持原状；设施命令由本地工作区AGENT维护。

## 输入与推理协议

- 第n轮输入 `I0,T1,I1,...,Tn`；历史图像是Lance保存后重新读入的真实RGB输出。
- `lance-history-bare-v2`保留官方image-edit系统提示，user仅原始指令/交错图像，不加HISTORY/CURRENT EDIT或轮次标签。空label是零token的CFG/缓存边界。
- `lance-history-chat-v1`把第一轮原图与指令放入user、历史真实输出放入assistant，后续指令与输出按角色交错；末尾保留assistant生成前缀，详见六组实验说明。
- 每张历史图提供ViT/VAE条件；CFG-negative仅移除当前指令正文，历史保留。
- 30步、shift3.5、text/image CFG4/1、区间 `(0.4,1]`、global/min0、BF16、base seed42；会话/轮次/用途派生种子沿用v1 RNG。
- MICE/ImgEdit为768面积桶；EdiVal直接生成、保存、回填512×512，ViT条件编码沿用Lance。EdiVal完整历史区别于官方multipass的当前指令＋上一轮图。
- none重算历史、images缓存编码、prefix再缓存历史K/V；结束会话清空。DP2按完整会话分片，每卡一个完整模型。

协议、配置和数据进入指纹，恢复拒绝历史缺口/篡改/身份变化，不用当前源码冒充旧验收。六组实验使用正向 `target-token-region-stats-v1`：FP32逐指令key token均值、历史图ViT/VAE 8×8区域mass、分组mean/std/P10/P50/P90，保留30步36层16head。旧实验使用 `target-group-mass-v1`，两种格式不可混读；字段与限制见 [观测说明](docs/attention-observation.md)。

## 检查与运行入口

以下CPU命令不加载模型；服务器EdiVal使用独立runtime。

```bash
python -B -m lance_mice.runner --selection all --profile dp2 --dry-run
python -B -m lance_mice.imgedit --selection all --dry-run
python -B -m lance_mice.edival --selection all --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

全量入口会执行真实GPU任务，只用于另一次已授权的新运行/新目录，不重复启动活动任务：MICE `scripts/full_mice_job.py`，ImgEdit `scripts/full_imgedit_job.py`，EdiVal `scripts/run_edival_job.sh`。前两项要求bare验收，EdiVal要求512验收的源码/数据身份；评分恢复见单票协议。

推理产物在 `run/<split>/<sample_id>/`：session.json、turn_0_input.png及逐轮turn_n.png/json/attention.npz（EdiVal split为edival）。状态每20秒更新；结束复核源图/指令、真实历史、CFG/seed、覆盖、attention与冻结源码，失败保留现场。

工程通过不替代能力得分；历史双裁判验收不证明当前v3准确率，替代裁判/人工协议、缺项分母和累计GA须披露。上游来源与许可见 [第三方说明](THIRD_PARTY_NOTICES.md)。
