# 实验资源清单

本表记录当前exp0的资源需求与固定来源，供本地理解实验和在服务器准备资源。权重、原始数据与下载缓存保存在服务器；本地只维护清单、配置、方法和结论。下列固定来源沿用0号机记录；2026-10-07六组实验所需权重、三项benchmark、六个环境和EdiVal工具已在1号机部署验收。候选资源仍主要属于0号机，不推定1号机拥有全部候选副本。两机共用权重和原始数据集可长期放公共资源目录；实验环境、工具及产物留各自/home/chs，见 [存储布局](storage-layout-20261007.md)。核查日期：2026-10-07。

## 资源范围

当前实验使用Lance进行完整历史图文交错推理，在MICE、ImgEdit、EdiVal上测量能力，记录attention供后续分析。推理输入沿用模型前轮生成图，不用GT替换历史。六组实验中MICE bare/chat各采用Qwen3.6-27B单票，16个首答组件待人工复核；ImgEdit两版本次未新增评分。历史ImgEdit人工表与MICE31项补评独立保留；EdiVal两版沿用官方评分组件。EdiVal七项attention描述分析已完成，源码与验收见[分析入口](attention-analysis.md)；干预、VINCIE训练适配及候选数据集实验尚未开展。

## 推理模型

来源：[bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)，固定revision `7395315758865e6f56ab87ad06a88c7ac172f056`。服务器目录 `/home/chs/model/Lance`。

| 必要内容 | 用途 |
| --- | --- |
| `Lance_3B/`，含模型、tokenizer及配置 | 图像编辑主体 |
| `Qwen2.5-VL-ViT/` | 图像视觉编码 |
| `Wan2.2_VAE.pth` | 图像潜变量编解码 |
| 根目录 `config.json` | 组件配置 |

下载图像编辑组件即可；当前实验没有视频生成权重需求。使用HF固定revision下载上述目录/文件，保留下载清单与校验记录。下载来源记录在服务器 `/home/chs/download_jobs/lance-edival-20261002/manifest.json`。运行环境安装入口为 `scripts/install_lance.sh`，固定vLLM-Omni源码commit `b742f86136d28423903a1213adde2a62a11067a2`；环境检查和最小验收按 [项目README](../README.md) 执行，下载完成不能替代模型加载及实际生成验收。

## 正式实验数据

| 数据集 | 固定来源与所需内容 | 服务器目录与规模 |
| --- | --- | --- |
| MICE-Bench | [Edit-R2](https://github.com/yuxiaooye/Edit-R2) commit `26b55829246e1a67fc3c8d522324fee3d49cd954` 的测试标注；[官方图像包](https://drive.google.com/file/d/1ztxZtg4VYiZiBr8Pbs9uBXYf9NKOnt3a/view) 中被测试标注引用的源图 | `/home/chs/dataset/MICE-Bench`；CM/CU各360会话，共720会话2160轮 |
| ImgEdit-Bench-Multi-Turn | HF数据仓库 `sysuyy/ImgEdit`，revision `f8de753484a2b6bd37f135fd20d308b66b09a523` 的 `Benchmark.tar`，只解出 `multiturn` | `/home/chs/dataset/ImgEdit-Bench-Multi-Turn/multiturn`；3类30会话88轮；`annotation.json` 实为JSON Lines |
| EdiVal | HF数据仓库 `C-Tianyu/EdiVal`，revision `583d3efe2daf81afb0a13ee2c4df06cdaad1b554` 的 `input_images_resize_512.zip`；官方代码commit `96d34b00d7ea2dc3de90f2bb01f292f6dec6294a` 的 `oai_instruction_generation_output.csv` | `/home/chs/dataset/EdiVal`；606原图中572纳入、34排除，572会话1716轮；CSV前缀须还原为完整会话 |

下载固定版本后先验压缩包完整性、标注解析、所有被引用图像和会话轮次，再按推理协议验收。MICE和ImgEdit的解析入口在 `src/lance_mice/dataset.py`、`imgedit.py`；EdiVal入口为 `scripts/edival_data_check.py`，细节见 [EdiVal推理](edival-inference.md)。

| 文件 | 字节数 | 已下载文件SHA256 |
| --- | ---: | --- |
| MICE图像压缩包 | 1633500031 | `60b465583a7bb7f2bf056555fbbcc872ee30f75dfc5b523c4da897bb733bc0a6` |
| ImgEdit `Benchmark.tar` | 50319360 | `d59ed1d4f68bfd068667ae76c98619e59ade30a5434fe9237cf10bb79450786e` |
| EdiVal 512 ZIP | 51709607 | `c15d9010ab50dd6ddaed81c3ef0f573591149ee31ca266124e39f4e8518d0c79` |
| EdiVal CSV | 872227 | `1c17f9f579c5337d3846f45714153f87f27cc0f85119d897dcb62e5301351555` |

这些SHA256标识已使用的下载副本；未发布官方校验值的文件不称为“官方SHA256”。

## MICE评分资源

权重根目录 `/home/chs/model/`。固定版本以 `configs/scoring_models.json` 为准。

| 资源 | 来源 / revision | 当前用途 |
| --- | --- | --- |
| DINOv3-ViT-L-16 | `timm/vit_large_patch16_dinov3.lvd1689m` / `30c1109559f65dea34316b0d4842d35c5771fe11` | 保持性特征；L/16 |
| bert-base-uncased | `google-bert/bert-base-uncased` / `86b5e0934494bd15c9632b12f734a8a67f723594` | GroundingDINO文本编码 |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B` / `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 当前单票裁判 |
| gemma-4-31B-it | `google/gemma-4-31B-it` / `842da3794eaa0b77d5f08bae87a17459d91ff475` | 历史双裁判资源；当前单票不需要启用 |
| GroundingDINO-SwinT-OGC | [官方SwinT权重](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)；源码 `856dde20aee659246248e20734ef9ba5214f5e44` | 检测与局部区域 |

准备入口为 `scripts/download_scoring_models.py`、`scripts/install_scoring.sh`，完整环境及检查见 [评分环境](scoring-environment.md)。下载脚本的配置包含Gemma历史资源；准备当前单票任务时先检查已有资源，不因历史清单自动增加裁判或改变协议。

## EdiVal官方评分资源

同样存放于 `/home/chs/model/`，配置为 `configs/edival_scoring_models.json`。

| 资源 | 来源 / revision | 用途 |
| --- | --- | --- |
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` / `eed13092ef92e448dd6875b2a00151bd3f7db0ac` | 官方指令遵循判断 |
| RAHF | `C-Tianyu/RAHF` / `003923c4894db32b9d85da593e45977e8207f06e` | 官方视觉质量组件 |
| vit-large-patch16-384 | `google/vit-large-patch16-384` / `4b143e77059a54c70b348a76677ab9946f584e13` | RAHF视觉骨干 |
| t5-base | `google-t5/t5-base` / `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` | RAHF文本骨干 |
| HPSv3 | `MizzenAI/HPSv3` / `4f81e3e09edd82fe3c5f636444c721b592a735ca` | 图像偏好评分 |
| DINOv3-ViT-B-16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` / `5931719e67bbdb9737e363e781fb0c67687896bc` | 官方保持性特征；B/16，与MICE的L/16不同 |
| GroundingDINO-SwinT-OGC | 复用上节固定权重与源码 | 官方区域检测 |
| bert-base-uncased | 复用上节固定revision | 检测器文本编码 |

共八项资源。DINOv3 B/16配置 `existing_only=true`，当前由共享目录 `/media/vlm_model/asset_llm_ckpt/dinov3-vitb16-pretrain-lvd1689m/` 复制到自有模型目录并验证固定LFS身份，不能用L/16替代。换服务器先核查可用副本，不能假定共享路径存在。

准备入口为 `scripts/download_edival_scoring_models.py` 和 `scripts/install_edival_scoring.sh`。官方EdiVal代码固定commit `96d34b00d7ea2dc3de90f2bb01f292f6dec6294a`，服务器工具目录 `/home/chs/tools/EdiVal/`；HPSv3包版本 `1.0.0`，参考源码commit `bd0c5fcb5f587617b0169c07222ab78d01e2f3c2`。隔离环境、包版本及实际组件检查见 [官方评分环境](edival-scoring-environment.md)，正式入口见 [官方评分接口](edival-scoring.md)。

## ImgEdit评分边界

当前正式口径为 `imgedit-human-review-v1`，不需要下载自动裁判权重。作者发布的 `ImgEdit_Judge/` 属于可选对照资源，来自HF数据仓库 `sysuyy/ImgEdit` 的同一固定revision；已在 `/home/chs/model/ImgEdit-Judge-release/ImgEdit_Judge/`，下载入口 `scripts/download_imgedit_judge.py`。已有权重不代表正式评分采用该裁判。

## 候选数据与后续训练

以下用于数据调研、会话抽查或后续训练方案，不是当前exp0必需下载项；完整训练需求须待训练协议确定后单独维护。

| 候选资源 | 固定记录与现有范围 | 服务器目录 |
| --- | --- | --- |
| OmniIIEBench多轮部分 | HF `YamJoy/OmniIIEBench`，revision `334b76a6e3768ba90e98134d6730bc9cb22b18aa`；`multi_turn.tar.gz`与标注，260会话1131轮；已下载压缩包8053645375字节，SHA256 `cd752007d491dce72ea080c05cbe373e33293992b8a891e01a9f71028dea1c26` | `/home/chs/dataset/OmniIIEBench-multi-turn/` |
| VINCIE-10M | HF `leigangqu/VINCIE-10M`，revision `6067c43f4c3715af3ea78fba57b94847e8d4e3de`；仅9样本44帧35转变，未全量下载；视频帧转变标注不等于人工多轮编辑对话 | `/home/chs/dataset/VINCIE-10M-samples-seed20261006/` |
| MagicBrush | HF `osunlp/MagicBrush`，revision `1d8d4629150d18ca50afab66391866f2085be989`，抽取6会话 | `/home/chs/dataset/multiturn-focus-random-18-seed20261005/` |
| WEAVE | HF `WeiChow/WEAVE`，revision `b7d16e77dd2520920e9e2a9ef29cda192a2e2202`，抽取6会话 | 同上 |
| Pico-Banana | `apple-aiml-research/pico-banana-400k` 的多轮标注与Apple CDN图像，抽取6会话，seed `20261005`；当前记录无不可变上游commit，追溯以已采集原始标注与哈希为准 | 同上 |

候选说明与采集方法见 本地UMM研究目录的《数据集调研-多轮图像编辑》 和 本地UMM研究目录的样本可视化说明。Omni包含参考GT；若用于训练必须按完整会话划分，不能再把重叠样本作为独立评测。VINCIE chronological reader、当前目标loss与历史隔离仅是方案，尚未实现或训练。

## 服务器准备流程

先在本地确定实验协议和上述必需资源；到指定服务器确认hostname、空间、已有目录与下载manifest，再使用AGENT中的代理设置下载固定版本。校验文件与数据引用后安装相应隔离环境；实际GPU验收前按AGENT停止目标卡burn，验收或实验结束后恢复。最后把资源身份、方法、验收结论与产物位置写回本地说明，完整数据、权重和结果留服务器。
