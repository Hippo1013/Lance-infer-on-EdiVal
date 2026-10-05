# MICE Qwen 单次评分协议

## 评分范围

用户于2026-10-05授权矛盾回答留到最后人工、继续GPU0，并要求中断时及时修复续跑；即时唤醒接口无法可靠实现时每5分钟检查。当前修复协议为 `mice-qwen-once-v3`。旧v1在21:01因GA标签矛盾停止；v2在21:54因一次OCR反复输出CSS、达到2048token后截断停止，已有181轮处理完成（其中3项标签待定）、305份完整原回答和全部2160轮共享测量。v3保留截断原文和技术原因，单独标“回答不完整，待人工”，不重采样该票，继续其他指标与样本；旧目录及manifest保持原始证据。

当前阶段先用 Qwen3.6-27B 对最新版 MICE bare v2 全量结果评一次，查看结果后再决定后续评分。此阶段仅使用GPU0；2026-10-05 GPU1已用于EdiVal独立全量推理，由EdiVal任务管理其资源。完成后不会自动启动 Gemma 或第二票。

输入为服务器项目的 `outputs/mice/full_bare_20261004_attention`，CM/CU各360会话，共720会话/2160轮。当前输出 `outputs/scoring/full_qwen_once_pending_v3`，tmux `mice-score-qwen-once`。此前v1/v2及full_dual_v2目录作为历史证据与测量来源保留。

## 指标与采样配置

- IF：保持原 Edit-R2 分支。检测分支共用一次；每个需 VLM 的判断只调用 Qwen 一次。
- CC：保持 GroundingDINO＋DINOv3 的对象/背景一致性计算，与裁判票数无关。
- GA：保持原图至当前轮完整图像和指令前缀、CM/CU 提示词；每个前缀只调用 Qwen 一次。保留原始 `GA_prefix_qwen`，按会话取历轮累计最小值得到 `GA_qwen`。
- Qwen：temperature=0.6，vote=1，seed=42，BF16，16384 上下文，max_tokens=2048，enable_thinking=False；原始 RGB 以无损 PNG 输入。

分数分别报告 `IF_qwen`、`CC`、`GA_qwen`、`GA_prefix_qwen`，按 CM/CU、轮次及整体汇总，报告有效分母和缺项状态。不生成双裁判均值，也不合成总分。空文本 IF、无资格 CC 仍为 null；运行错误停止对应阶段，保留完整原回答，不能计作编辑失败。

## 待定决策与人工复核

后续本轮评分遇到相同类型问题均按 `completed-decision-pending-review-v1` 处理：已正常结束的原回答没有一致的显式YES/NO，或GA的turn/final标签缺失、重复、次序或结论矛盾时，对该项保存 `status=pending_review, score=null`、解析原因与完整原始输入/回答。保持原严格解析器，不猜测答案，不补标签，不重采样；合法0分仍为0。

IF与GA分别处理：IF待定仍评分GA，GA待定保留已经得到的IF和CC，并继续所有后续轮次。保留后续独立GA前缀判断；有待定前缀时，该前缀及后续累计GA均保守地标为待定，记录 `unresolved_prefixes`，人工确定前缀后再重新计算累计值。即使此前累计值为0，待定依赖仍显式展示。

运行过程中写 `pending_review_events.jsonl` 和小量 `judge_progress.json`，结束生成 `pending_review.json`，列出直接待定项、原回答、原因、空白人工决策栏及累计GA依赖；`summary.json` 报告待定数量及各项有效分母。自动遍历完整2160轮可标 `complete`，人工解决状态另记 `manual_review_status=pending`，不宣称人工复核已完成。待定项从数值均值分母中排除。`judge_progress.json` 的 `pending_components` 包含标签待定及不完整回答，`incomplete_response_components` 是其中子集；两者不能相加。

v3另采用 `judge-response-incomplete-v1`：模型调用确实返回、但 `finish_reason=length` 或正常stop却无文本时，保留第一次原回答、token数、结束原因和原错误，单独记录 `response_incomplete/score=null`，与标签矛盾分开统计，加入最终人工清单。此类尝试保存在独立 `response_attempts/qwen`，不能进入完整回答缓存；同请求后续读该尝试直接返回“回答不完整”，不能重采样、增加票数或把它计为0。IF不完整仍继续GA，GA不完整仍继续后续样本，并显式保留累计依赖。

CUDA、检测依赖、输入/缓存校验、连接或没有实际模型响应证据的运行异常仍显式失败，不伪装成评分结果；监控发现后按既有授权诊断修复和GPU0续跑。未来另开裁判评分同样沿用这两类人工口径。停用的旧双裁判入口保留原冻结实现，重新启用前须接入策略，不能直接使用旧中断配置。

## 实现与测量复用

独立入口 `scripts/score_mice_qwen_once.py` 和 `src/lance_mice/scoring_qwen_once.py` 在本进程中选择单裁判/单票配置；原 `src/lance_mice/scoring/`、双裁判入口和规则源码未改。新 profile 的所有入口、编排和原始规则文件均写入 manifest 源码哈希，运行过程中禁止修改。

人工未决处理位于 `src/lance_mice/scoring_pending.py`，与原IF分支/提示词/采样配置分别纳入源码指纹。每次新prepare重新核对全量输入并锁定已停止的生产任务，校验manifest、固定旧profile源码、规则/模型/输入及所有产物校验值。v2曾迁移2160轮共享测量、103轮正常结果和180份完整原回答；v3从v2迁移181轮已处理结果（保留3项标签待定）、305份完整原回答和1份截断尝试，全部2160轮测量继续复用，每项保存来源SHA-256。旧错误行在 `previous_error.json` 留存，截断尝试单独保存，原采样和票身份不变，不重跑已有调用。

原任务停止时已有 879 轮检测/CC 完整结果，裁判回答为 0。新 prepare 重新核对全部输入，再锁定停止的旧任务，检查旧 manifest、所有原规则源码哈希、模型 revision、图像/标注及推理验证身份。每份复用测量和检测缓存均核对原指纹与校验值，重新绑定新指纹，并保留来源文件 SHA-256。`metric_reuse.json` 记录实际复用数量。其余轮次继续正常计算；没有导入或挑选旧裁判回答。

单票配置在独立进程中生效，不会更改旧源码或旧 manifest；这保证共享测量可以追溯，也保留了后续独立评分的边界。CPU 小型固定输入检查覆盖单票 seed 约束、单裁判报告、GA 的 1/0/1→1/0/0 累计规则和第二票拒绝；没有额外模型试评。CPU 检查不表示裁判准确率已校准。

## 运行与监控

首次v1于2026-10-05 20:21启动，v2于21:29启动，均已按上文记录退出。v3恢复命令如下；任务运行时不得重复启动：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chs/conda/envs/lance/bin/python scripts/full_mice_qwen_once_job.py \
  --inference outputs/mice/full_bare_20261004_attention \
  --output outputs/scoring/full_qwen_once_pending_v3 \
  --reuse-metrics outputs/scoring/full_qwen_once_pending_v2 --allow-full
```

编排：全量CPU输入审计及证据迁移 → 跳过已全部完成的共享metrics → GPU 0续完Qwen一票 → 单裁判报告与待定清单 → 全量CPU证据审计。GPU阶段前只停止已核对的GPU 0 burn，退出后仅确认并恢复本卡空闲burn；不操作GPU1的EdiVal任务，monitor和watchdog保持原状。

`events.jsonl` 根据子进程结束事件记录并推进阶段，`job_status.json` 保存状态，`completion.json` 为最终退出状态；阶段日志为prepare/qwen/report/validate.log。外层日志是 `outputs/scoring/full_qwen_once_pending_v3.job.log`。

应用heartbeat `mice-v2` 已改为每5分钟读取少量事件、进程与进度，正常或新增人工项时保持安静。这是定时兜底检查，不宣称真实即时事件唤醒；服务器阶段推进仍由子进程退出事件触发。发现中断立即定位并修复GPU0续跑，不停留在仅报告失败，不重采样已有回答，也不修改运行中源码。完成核对summary complete、validation passed、720会话2160轮、单票原回答、独立不完整尝试与人工清单，汇报IF/CC/GA/有效分母，并分别报告标签待定和回答不完整数量，恢复GPU0 burn，然后暂停监控；不自动追加Gemma或第二票。
