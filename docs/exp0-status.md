# exp0状态与交接

核对时间：2026-10-05 23:50，Asia/Shanghai。快照不代表终态，实时状态以服务器产物为准。项目：`/home/chs/exp0_attention/Lance-infer-on-EdiVal`。

## 阶段与产物

当前完成推理管线及分组attention工程验收，MICE/ImgEdit bare全量通过，MICE单票能力评分与EdiVal推理进行中；attention系统分析、干预及效果评测尚未开展。

下列路径相对于服务器项目。

| 任务 | 核对状态 | 产物根目录 |
| --- | --- | --- |
| MICE bare推理 | 720会话2160轮/attention，validation passed | `outputs/mice/full_bare_20261004_attention/` |
| ImgEdit bare推理 | 30会话88轮/attention，validation passed | `outputs/imgedit/full_bare_20261004_attention/` |
| MICE单票v3，GPU0 | metrics2160轮完整；Qwen582/2160轮、running；14直接人工项含1不完整回答 | `outputs/scoring/full_qwen_once_pending_v3/` |
| EdiVal512，GPU1 | 594/1716轮，198/572完整会话，594 attention；running，不评分 | `outputs/edival/full_512_bare_20261005_attention/` |
| ImgEdit bare人工表 | 88/88轮明确判定，统计/评分库留服务器且可由用户修改 | `outputs/review/imgedit_bare_20261004/` |

MICE使用 [Qwen单票v3](scoring-qwen-once.md)，双裁判任务已按用户调整停止，不自动追加Gemma/第二票；未决与不完整保留首份证据、不重采样，人工状态独立。EdiVal采用bare-v2完整历史、prefix、512、30步及attention。ImgEdit为项目人工协议，不能当官方榜单。

## 状态与终态条件

只读查询，不加载模型或触发burn：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
cat outputs/scoring/full_qwen_once_pending_v3/job_status.json
cat outputs/scoring/full_qwen_once_pending_v3/judge_progress.json
cat outputs/edival/full_512_bare_20261005_attention/status.json
```

- MICE：completion成功、summary complete、validation passed，720/2160覆盖和证据通过后才称自动评分完成。pending_review.json中的标签未决/回答不完整另处理；pending_components包含不完整子集，不能相加。
- EdiVal：completion completed、validation passed，572/1716的源图/指令链、512输出、CFG/seed、attention和冻结源码全通过才称完成。查看器另校验2288张HTTP图并冻结 `outputs/review/edival_512_bare_20261005/` 的manifest，启动检查不替代终态。

GPU0的tmux为mice-score-qwen-once，GPU1为lance-edival-full。各任务结束确认本卡空闲后恢复burn，monitor/watchdog不动；不能由一项终态推定另一卡空闲。heartbeat mice-v2每5分钟、edival-gpu1每小时兜底，仅终态、新故障或需处理时通知，非即时唤醒。

## 源码与部署边界

本地/GitHub保存完整开发源码；服务器共享目录用于MICE，EdiVal运行于 `runtime/edival_512_20261005/`。服务器无Git CLI，逐文件同步不等于git pull。

2026-10-05实核：MICE manifest的17冻结文件双端匹配，EdiVal runtime的24冻结文件全部匹配。当前本地scoring_qwen_once.py为v3，EdiVal冻结的是启动时旧版，须保留各自身份。

共享目录尚缺14项EdiVal实现/配置/测试/页面，其中12运行文件已在runtime，2测试仅本地；另4项差异为pyproject.toml及src/lance_mice的dataset.py、runner.py、settings.py。完整文件清单在本地 `outputs/status/exp0_repository_sync_20261005.json`；当前无需覆盖共享目录。文档可单独同步，活动源码保持冻结。统一部署须先确认两项终态、进程退出和改动，再逐文件核对来源/哈希，不能全目录rsync或切换活动源码。

## 查看入口与后续

MICE bare8768、ImgEdit bare人工表8769、EdiVal动态8770；转发、恢复和身份见 [文档索引](README.md)。旧8765/8767数据独立保留。

先完成两项终态复核及MICE人工清单，再按有效分母汇总能力缺陷并设计attention分析/干预。工程通过、替代裁判得分、人工判定与attention解释分别保留证据边界。
