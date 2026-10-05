# MICE Qwen单次评分协议

## 范围与配置

当前 `mice-qwen-once-v3`对MICE bare-v2的CM/CU各360会话、720会话2160轮评分。仅GPU0、Qwen3.6-27B一票；结果经用户查看后再决定后续，不自动追加Gemma或第二票。GPU1由EdiVal任务管理。

| 项目 | 固定身份 |
| --- | --- |
| 输入 | `outputs/mice/full_bare_20261004_attention/` |
| 当前输出 / tmux | `outputs/scoring/full_qwen_once_pending_v3/` / mice-score-qwen-once |
| 采样 | temperature0.6、vote1、seed42、BF16、context16384、max_tokens2048、enable_thinking=False |
| 图像 | 原始RGB、无损PNG，不缩图替换 |

IF保持Edit-R2分支，检测分支共享，每个VLM判断调用一次Qwen；CC保持GroundingDINO＋DINOv3，与票数无关。GA按原图至当前轮完整图像/指令前缀和CM/CU模板逐前缀调用一次，保留 `GA_prefix_qwen`，累计最小值得 `GA_qwen`。精确正文/图序见 [提示词](scoring-prompts.md)，指标和官方差异见 [历史协议](scoring-protocol.md)。

分别报告IF_qwen、CC、GA_qwen和GA_prefix_qwen，按CM/CU、轮次及整体汇总；不合成总分或双裁判均值。空文本IF/无资格CC为null，报告各有效分母。

## 人工未决与故障边界

| 模型响应 | 策略与保存 |
| --- | --- |
| 正常结束但无一致显式YES/NO，或GA标签缺失/重复/顺序/结论矛盾 | completed-decision-pending-review-v1：pending_review/null，保留完整输入/raw及严格解析原因 |
| 实际返回但finish_reason=length，或正常stop空文本 | judge-response-incomplete-v1：response_incomplete/null，独立保留首份raw、token、finish及原错误 |
| CUDA、依赖、连接、输入/缓存/源码校验失败，或无实际响应证据 | 显式错误并停止，保留现场，按既有授权诊断修复续跑 |

合法0分仍为0。两类人工项都不猜答案、不补标签、不重采样、不增加票数；不完整尝试在 `response_attempts/qwen`，不能进入完整回答缓存或被替换，同请求重放直接返回原未决状态。

IF/GA独立：IF未决仍评分GA，GA未决保留IF/CC并继续后续独立前缀。未决前缀及后续累计GA保守为null，记录 `unresolved_prefixes`，即使已有0也显示依赖；人工确认前缀后重新计算累计值。

运行中保存 `pending_review_events.jsonl` 和 `judge_progress.json`，结束生成 `pending_review.json`（直接项/raw/原因/空白人工栏/累计依赖）及summary有效分母。标签待定与不完整回答分开统计；pending_components包含两类，incomplete_response_components为其子集，不能相加。自动遍历complete与manual_review_status=pending独立，不冒称人工已解决；未决从数值均值分母排除。

未来另开裁判评分同样沿用两类人工口径；停用的双裁判入口须先接入策略才能重新启用。

## 冻结与证据复用

独立入口 `scripts/score_mice_qwen_once.py`、`src/lance_mice/scoring_qwen_once.py`选单票配置，通用策略在 `scoring_pending.py`；原 `scoring/` IF规则/提示词及双裁判实现保留。所有入口、编排、策略和规则文件写入manifest源码哈希，运行中禁止修改。

prepare重新审计全量输入，只复用已停止且锁定的生产任务。核对旧manifest/固定profile源码、规则、模型revision、图像/标注/推理验证、采样/票/缓存身份和全部产物校验；保存每项来源SHA-256、重新绑定新指纹，迁移raw再解析，不复制裁判分数、不重跑已有调用。metric_reuse/judge_reuse记录数量，previous_error保留旧错误；完整回答与不完整尝试分别审计。

历史目录full_dual_v2、full_qwen_once_v1、full_qwen_once_pending_v2保留原manifest。v1因GA矛盾停止，v2因OCR截断停止，v3迁移已有2160轮测量、6060检测缓存、181已处理轮、305完整回答与1截断首尝试。CPU单票/累计/第二票拒绝及两类策略回归、真实无模型迁移通过；不表示裁判准确率已校准。

## 运行与终态

在服务器项目目录执行；活动任务不得重复启动。当前v3恢复命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chs/conda/envs/lance/bin/python scripts/full_mice_qwen_once_job.py \
  --inference outputs/mice/full_bare_20261004_attention \
  --output outputs/scoring/full_qwen_once_pending_v3 \
  --reuse-metrics outputs/scoring/full_qwen_once_pending_v2 --allow-full
```

阶段：全量CPU审计/迁移 → 已完成metrics复用 → GPU0 Qwen → 报告/人工清单 → CPU证据审计。子进程退出事件推进阶段，events.jsonl记录事件，job_status.json记录编排，judge_progress.json记录逐轮，completion.json记录最终退出；prepare/qwen/report/validate.log为阶段日志，外层为同名输出目录加.job.log。

GPU阶段只停止本卡burn，退出确认空闲后恢复；不操作GPU1、monitor/watchdog。heartbeat mice-v2每5分钟兜底，正常/新增人工项安静；发现中断按授权诊断修复续跑，不修改运行中源码、不重采样。此为定时检查，非即时唤醒。

完成须completion成功、summary complete、validation passed、720会话2160轮及单票raw/独立尝试/人工清单审计通过。汇报IF/CC/GA、有效分母和两类人工数量，恢复本卡空闲burn、暂停监控；人工解决状态独立报告。
