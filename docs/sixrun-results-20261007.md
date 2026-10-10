# Lance 双协议六组实验结果

2026-10-09确认六组共用的v1区域观测器存在VAE缓存标签错位。下方推理与评分结果保留；历史attention的数值／hash验收不代表VAE空间语义正确。使用其VAE区域、集中度及空间组份额前须遵守[语义限制](attention-semantic-validation.md)。EdiVal正确观测已于2026-10-09完成全量补采，七项描述分析于2026-10-10完成；入口见[Attention分析](attention-analysis.md)。

## 实验设置

2026-10-07，EdiVal、MICE、ImgEdit 分别完成 bare-v2 与 chat-v1 全量推理，共 2644 会话、7928 轮。每轮保留真实历史与首次生成图像；chat 协议将先前指令置于 user、先前实际生成图置于 assistant。两版共用 image_edit system prompt。

EdiVal 两版采用官方 IF／CC／RAHF／HPS 评分。MICE 两版仅使用 Qwen3.6-27B 单票，四个数据分片不增加票数；CC 沿用原评分实现。不完整或矛盾回答保留首次证据与 null，独立列入人工待审，不重采样。

## 评分结果

| 输入协议 | EdiVal IF | MICE IF | MICE CC | MICE 累计 GA |
| --- | --- | --- | --- | --- |
| bare | 55.7692% (1716轮) | 35.9535% (2150/2160有效) | 67.7011% (1883/2160适用) | 35.8105% (2153/2160有效) |
| chat | 55.2448% (1716轮) | 35.0698% (2150/2160有效) | 67.4435% (1883/2160适用) | 35.3296% (2154/2160有效) |

以上 MICE 是本项目 Qwen 单票自动分数，人工校准尚未完成；不能与历史纳入人工复核的第一版分数混用。两版 IF 各有 7 轮源注释无效、3 轮裁判回答不完整；CC 各有 277 轮不适用。待审与不适用项不补零。

## EdiVal 分轮指标

数值为官方原始指标均值，括号内为有效项数；三轮总量各 572。RAHF 分别报告 plausibility 与 aesthetics，HPS 保留原始分值。本文未创造统一 CC／VQ 综合分数，也未将 HPS 原始值当作论文 VQ 差值。

| 协议 | 轮次 | 物体 DINOv3 | 物体 L1 | 背景 DINOv3 | 背景 L1 | RAHF plausibility | RAHF aesthetics | HPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bare | 1 | 0.782067 (453) | 0.867845 (453) | 0.917624 (507) | 0.876517 (507) | 0.576154 (572) | 0.612063 (572) | 4.886665 (572) |
| bare | 2 | 0.661574 (354) | 0.793905 (354) | 0.865571 (452) | 0.822868 (449) | 0.595717 (572) | 0.629703 (572) | 4.694264 (572) |
| bare | 3 | 0.609912 (261) | 0.763571 (261) | 0.836030 (380) | 0.787967 (379) | 0.604948 (572) | 0.634633 (572) | 4.404312 (572) |
| chat | 1 | 0.782067 (453) | 0.867845 (453) | 0.917624 (507) | 0.876517 (507) | 0.576154 (572) | 0.612063 (572) | 4.886665 (572) |
| chat | 2 | 0.661090 (354) | 0.794426 (354) | 0.864237 (452) | 0.821983 (449) | 0.596766 (572) | 0.629633 (572) | 4.710933 (572) |
| chat | 3 | 0.614021 (261) | 0.764819 (261) | 0.835579 (380) | 0.787887 (379) | 0.601906 (572) | 0.633531 (572) | 4.473459 (572) |

## Attention 存储

按已批准方案保存 FP32 文本 token、各历史图 ViT／VAE 8×8 区域、分组 mean／std／P10／P50／P90，保留 30 步、36 层、16 头。共 7928 份，184,085,621,704 字节（184.09 GB／约 171.44 GiB）。NPZ 保存于生产机本地；归档 receipt 保存 host、绝对路径、字节数、SHA256 与运行身份，attention 原件留在各生产机本地。

## 人工待审清单

bare 共 8 个组件：4 个回答不完整、4 个决策待审；chat 共 8 个组件：3 个回答不完整、5 个决策待审。不完整数是总待审数的子集。派生累计 GA 另受 bare 7 轮、chat 6 轮影响，不能再次加到组件数中。

| 协议 | 会话 | 轮次 | 组件 | 状态 | 原因 |
| --- | --- | --- | --- | --- | --- |
| bare | cm/04f5a82e7a72ac99 | 2 | IF | response_incomplete | ValueError: Empty or truncated judge response |
| bare | cm/07e76fa2f3fae8d5 | 3 | GA_prefix | pending_review | GA turn sequence invalid |
| bare | cm/09ada0942ee4348a | 2 | IF | response_incomplete | ValueError: Empty or truncated judge response |
| bare | cm/0cff8f57ea79a3a0 | 2 | IF | response_incomplete | ValueError: Empty or truncated judge response |
| bare | cu/00c1d95ed85d341f | 2 | GA_prefix | pending_review | GA turn sequence invalid |
| bare | cu/03ed26a0a6e055f7 | 3 | GA_prefix | pending_review | GA must stop at its first failure |
| bare | cu/0aed923fc2fdc254 | 2 | GA_prefix | response_incomplete | ValueError: Empty or truncated judge response |
| bare | cu/0c3d9d4819548338 | 3 | GA_prefix | pending_review | GA must stop at its first failure |
| chat | cm/01a71bb9ca3ce4ec | 3 | GA_prefix | pending_review | GA must stop at its first failure |
| chat | cm/067562644b03f407 | 1 | IF | response_incomplete | ValueError: Empty or truncated judge response |
| chat | cm/09ada0942ee4348a | 2 | IF | response_incomplete | ValueError: Empty or truncated judge response |
| chat | cm/0cdafcf8b151f85c | 3 | GA_prefix | pending_review | GA must stop at its first failure |
| chat | cm/0cff8f57ea79a3a0 | 2 | IF | response_incomplete | ValueError: Empty or truncated judge response |
| chat | cm/0ffa4acad897c98e | 2 | GA_prefix | pending_review | GA turn sequence invalid |
| chat | cu/04be8af256e1d1e5 | 3 | GA_prefix | pending_review | GA turn sequence invalid |
| chat | cu/05898e3273246e3e | 3 | GA_prefix | pending_review | GA must stop at its first failure |

完整首次回答、token 用量、完成原因与证据路径见0号机实验归档 `scoring/mice_{bare,chat}/pending_review.json`。此清单保留自动未决状态，尚未添加人工决定。

## 产物与验收

0号机实验归档：`/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/sixrun_20261007/experiment`。六组推理和四组评分各自的 validation 均 passed，completion 均 exit_code=0；归档队列 2698 项 completed，无失败。2026-10-07 12:11:59（Asia/Shanghai）最终独立复核 passed：全部 7928 份 NPZ 的 SHA256／数值／几何、2644 会话真实历史和运行身份、90 项冻结源码、四次评分原始证据与汇总、首次裁判回答缓存绑定均通过。最终证据为归档 `setup/independent_final_audit.json`，其中保存分机审计、评分审计和相关证据的 SHA256。四个 worker／supervisor 已退出，临时目录已清理；12:12 再次确认两机四张卡各自运行独立 burn。

冻结 runtime：两机 `/home/chs/exp0_attention/Lance-infer-on-EdiVal/runtime/sixrun_20261007_tokenregion_v3`。源码保持冻结；v2 原目录及并发锁迁移证据保留。运行、资源与恢复细节见 [六组实验说明](sixrun-20261007.md)。

独立验收脚本的 Python 整数键与 JSON 字符串键比较曾触发断言；单独的评分审计按 JSON 格式统一键后通过全部四组评分。该问题仅在复核脚本，未改动正式产物；原 stdout／traceback 和修正审计脚本保存在0号机归档 setup。

2026-10-07 12:13，最终验收与四卡burn恢复核实后，heartbeat `lance` 已通过应用工具暂停（PAUSED）。人工待审保持独立，不自动追加裁判或重采样。

完整归档迁移已逐文件比对通过；历史路径与hash原样保留，用同级 `../storage_migration/locations.json` 解析当前位置。环境、工具、共用资源及原有设施边界统一见 [存储布局](storage-layout-20261007.md)。所有原始产物留服务器，本地维护方案、结论和必要位置说明。
