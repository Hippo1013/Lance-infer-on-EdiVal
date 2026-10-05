# exp0 状态与交接

核对时间：2026-10-05 23:22，Asia/Shanghai。以下是时间快照，实时进度和终态由服务器产物证明。服务器项目为 `/home/chs/exp0_attention/Lance-infer-on-EdiVal`。

## 研究阶段

exp0已完成多轮推理管线与生成阶段分组attention的工程验收。MICE/ImgEdit bare全量已完成，MICE进入单票能力评分，EdiVal全量仍在推理。系统attention分析、干预及其效果评测尚未开展；不根据进度提前给出研究结论。

## 任务与产物

路径相对于服务器项目根目录。

| 任务 | 核对状态 | 产物根目录 |
| --- | --- | --- |
| MICE bare推理 | 720会话/2160轮/2160 attention，validation passed | `outputs/mice/full_bare_20261004_attention/` |
| ImgEdit bare推理 | 30会话/88轮/88 attention，validation passed | `outputs/imgedit/full_bare_20261004_attention/` |
| MICE Qwen单票v3，GPU0 | metrics 2160轮完成；Qwen已处理457/2160轮，尚无全量终态 | `outputs/scoring/full_qwen_once_pending_v3/` |
| EdiVal 512推理，GPU1 | 481/1716轮，160/572完整会话，481 attention；running | `outputs/edival/full_512_bare_20261005_attention/` |
| ImgEdit bare人工表 | 88/88轮已明确判定，统计与评分库留服务器 | `outputs/review/imgedit_bare_20261004/` |

MICE当前为 `mice-qwen-once-v3`，temperature0.6/seed42/context16384，每项VLM判断一票。双裁判v2已按用户调整停止，不能重启或自动追加Gemma/第二票。人工未决和不完整回答保留原尝试，不重采样；有效分母与人工状态独立报告。

EdiVal为bare-v2完整交错历史、prefix、512×512、30步及逐轮attention，仅GPU1，不开展EdiVal评分。ImgEdit采用 `imgedit-human-review-v1`，不能当作官方榜单或独立裁判校准；记录仍可由用户修改。

## 状态入口与终态条件

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
# 只读查询，不加载模型或触发burn启停。
cat outputs/scoring/full_qwen_once_pending_v3/job_status.json
cat outputs/scoring/full_qwen_once_pending_v3/judge_progress.json
cat outputs/edival/full_512_bare_20261005_attention/status.json
```

MICE的 `job_status.json` 表示编排阶段，`judge_progress.json` 表示逐轮进度。只有 `completion.json` 成功、`summary.json` complete、`validation.json` passed且覆盖720会话2160轮后，才能称自动评分完成；`pending_review.json` 的人工未决与不完整回答还须分别处理。`pending_components` 包含两类直接人工项，`incomplete_response_components` 是其子集，不能相加。

EdiVal只有 `completion.json` completed与 `validation.json` passed，且572会话1716轮全部源图/指令链、512输出、CFG/seed、attention及冻结源码通过后，才能称推理完成。查看器另在 `outputs/review/edival_512_bare_20261005/` 校验2288张HTTP图片并冻结manifest；启动检查不能替代这一终态。

| GPU | 活动任务 / tmux | 完成后的资源处理 |
| --- | --- | --- |
| 0 | `mice-score-qwen-once` | 本任务确认空闲后恢复本卡burn |
| 1 | `lance-edival-full` | 包装器确认空闲后恢复本卡burn |

monitor/watchdog保持原状；不能根据一项任务推定另一卡空闲。heartbeat `mice-v2` 每5分钟、`edival-gpu1` 每小时，仅完成、新故障或需处理时通知；这是定时兜底，不是即时唤醒。监控配置与运行中源码不属于本次整理的修改范围。

## 源码与部署边界

本地和GitHub保存完整开发源码；服务器共享目录用于MICE评分，EdiVal使用独立 `runtime/edival_512_20261005/`。服务器暂缺Git CLI，文件哈希核对不等于已执行git pull。

2026-10-05核对：MICE manifest的17个冻结文件在服务器与本地均匹配；EdiVal的24个冻结文件在runtime中全部匹配。当前本地 `scoring_qwen_once.py` 已迭代为评分v3，与EdiVal启动时冻结的旧版不同；该文件仍保留在runtime，不能为了目录一致改写它。

核对时服务器共享目录缺19个EdiVal文件；本次同步5个说明/验收文件后，仍有14个源码、配置、测试与页面文件留在独立runtime。另有4个源码/包配置差异：`pyproject.toml`、`src/lance_mice/dataset.py`、`src/lance_mice/runner.py`、`src/lance_mice/settings.py`。EdiVal入口、查看器和验收证据已在独立runtime；运行正常不需要覆盖共享目录。文档可单独同步，共享源码及runtime在活动任务结束前保持冻结。未来统一部署须先确认两项终态、进程退出与改动，再逐文件核对来源和哈希；不能直接全目录rsync或切换活动源码。

## 结果入口与后续顺序

MICE bare抽查为8768，ImgEdit bare人工表为8769，EdiVal动态查看为8770；SSH转发、恢复与数据身份见 [文档目录](README.md)。

后续先完成活动任务的终态复核及MICE人工清单，再按有效分母汇总能力缺陷，开展attention分析与干预设计。工程通过、替代裁判得分、人工判定和attention解释分别保留证据边界；本次整理不新增实验任务。
