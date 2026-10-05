# MICE 双裁判评分协议与历史实现

## 实现与范围

2026-10-05 用户将当前阶段改为仅Qwen一票、人工未决项留到最后，并要求中断后修复续跑；没有可靠即时唤醒时每5分钟检查。当前采用 [Qwen 单次评分协议](scoring-qwen-once.md) v3，输出 `outputs/scoring/full_qwen_once_pending_v3`，标签矛盾与回答不完整分开记录，保留首份尝试且继续其他判断。先汇报这一轮结果，再决定后续评分。下文保留双裁判v2的旧规则与运行记录，旧入口尚未接入人工未决策略，不能直接用于后续新任务。

历史协议版本 `mice-dual-judge-v2`。该配置为用户指定的 temperature=0.6、每位裁判两次；按用户要求没有追加模型验证或试评。2026-10-05 用户授权全量，对 `outputs/mice/full_bare_20261004_attention` 启动评分，结果写入新的 `outputs/scoring/full_dual_v2`。此前 v1 的 12/120 轮验收属于 temperature=0、每位裁判一次的历史证据。完整提示词见 [裁判提示词](scoring-prompts.md)。

官方参考固定为 [Edit-R2 26b5582](https://github.com/yuxiaooye/Edit-R2/tree/26b55829246e1a67fc3c8d522324fee3d49cd954)：`rewards/reward_server/edival_reward_server.py`、`groundingdino_server.py`、`flow_grpo/edival_client.py`。选取的 IF 规则及 GA 提示词来源、源文件 SHA-256 和修改范围见 `src/lance_mice/scoring/NOTICE`。上游该版本未发现 LICENSE；未对这些片段宣称 Apache 许可，来源与许可范围在第三方说明中保留。

此实现评估已生成的 MICE 图像，不触发 Lance 推理。裁判是 Qwen3.6-27B 和 Gemma-4-31B-it；检测和 CC 共享 GroundingDINO Swin-T OGC / DINOv3 ViT-L/16。结果属于本项目双裁判协议，不能称为论文原裁判分数或其严格复现。

## 输入与计算

| 指标 | 输入 | 计算与保存 |
| --- | --- | --- |
| IF | 上一轮图、当前图、当前 task_type / formatted_instruction | 官方按编辑类型分支；VLM 依赖分别调用两位裁判；纯检测只算一次 |
| CC | 原图、当前图、当前 unchanged_objects / all_objects / bg_consistency | 共享检测、对象特征和背景特征，仅算一次 |
| GA | 原图至当前轮完整图像前缀、对应原始指令前缀 | CM 用全局约束提示词；CU 另给显式参照指令。分别保存裁判前缀判断及累计分数 |

IF 的 remove、position、count 使用纯检测。add、replace 使用视觉预检和检测；color、material 使用两图视觉判断；text 使用 OCR（替换文字先检测对象）；background 使用视觉判断，并按指令检查需保留的对象。保持上游阈值、框选择与比较规则；检测缓存按图像像素、对象文本和参数区分。

每个需要 VLM 的 IF 分支、每个 GA 前缀，对 Qwen、Gemma 各采样两次；单次结果和原始回答分别保存。每位裁判的分数先取两次算术平均，再取双裁判均值。因此 IF 与原始 GA 前缀均值等于四次有效评分之和除以 4。CC 和纯检测 IF 不重复执行。两位裁判须均有有效结果，不能只用可用的一位代替双裁判均值。CC 不重复计算。不把 IF、CC、GA 再合成一个总分。

GA 对每个前缀、每位裁判独立调用官方提示词两次，保存 `GA_prefix_votes_*`，两次均值为 `GA_prefix_*`，四次直接均值为 `GA_prefix_mean`；随后在每位裁判内部取前缀累计最小值，得到 `GA_*`，最后才求双裁判平均。例如一位的原始前缀判定为 1、0、1，累计值为 1、0、0；该不一致单独列出。显式累计是本项目的协议补充；上游客户端仅独立解析每次前缀回答。

## 裁判接口

采用离线 vLLM `LLM.chat`，无需部署三个常驻 HTTP 服务。两位裁判使用同一请求结构：图像按顺序置于 user content，后接官方提示词；各模型使用自己的聊天模板。每次请求独立，不在裁判间共享聊天历史。

固定 BF16、单 GPU、新运行的 Qwen 上下文 16384、Gemma 上下文 8192、一次最多 4 张图、并发 1、temperature=0.6、每位裁判两次投票（seed=42、43）、max_tokens=2048、`enable_thinking=False`。这是明确固定的项目设置；与论文同一裁判 Avg@4 相比，本项目为两个不同裁判各两次。GA 上游 max_tokens=1024，本项目保留 2048 输出预算。单卡模式已经过真实评分小样本验证。新 manifest 的 `context_limits` 记录模型运行容量；旧验收 manifest 无该字段时，保留原 8192 容量（`settings.max_model_len` 是兼容旧证据的默认值）。容量不同不改图像、提示词或采样参数；原始回答中的运行容量与来源仍保留，不将旧回答冒称新容量下重新生成。

评分使用保存的原始 RGB 像素，以无损 PNG 编码给裁判；不缩小到环境测试的 256 像素。上游部分 IF/CC HTTP 路径使用 JPEG，本文的无损输入也是已声明的差异。报告中的缩略图仅供浏览。

保留模型 revision、代码 SHA-256、元数据 SHA-256、每张输入图的文件及像素 SHA-256、提示词、原始回答、终止原因、输入/输出 token 数、判定及规则理由。IF 官方提示词要求只回答 YES/NO 或 OCR 文本，因而不会虚构模型未给出的解释；GA 的解释保存在原始标签回答中。

## CC 特征与缺项

沿用上游：目标图按原图尺寸 LANCZOS 对齐，对象区域使用原图检测框在两图同位置裁剪；对象相似度是 CLS cosine 与平均 patch cosine 各占 0.5；背景使用两图所有对象框并集遮罩，对保留 patch 的归一化特征作加权平均，再算 cosine。背景按 `bg_consistency` 选择是否进入均值。cosine 不擅自截断或重新缩放。

适配 timm 权重时明确排除 CLS＋4 register tokens，以真实空间网格处理背景。遮罩执行与 timm 图像相同的 resize / center crop 几何，再汇聚到 patch 网格；不会将前缀 token 当成空间 patch。这个修正与上游 HF 代码直接取 `features[:, 1:]` 不同，分数不得冒称逐位复现。

上游在 unchanged_objects 或 all_objects 为空时，两类 CC 均提前返回缺项。本协议保留这一资格规则并标记 `not_applicable`，不擅自补算另一分量。合法检测为空或背景无可用 patch 标记缺项原因；有一个有效分量则使用该分量，都无有效分量则为 `unmeasurable`。这些行的主 CC 是 null，汇总明确显示有效分母；同时保留 `upstream_missing_value=0.5` 供旧口径对照。运行错误绝不冒充缺项或 0.5。

## 异常与数据协议

原数据中有 7 条 CM 文字编辑的空目标文本，官方解析器不能解析。保持数据不变：对应 IF 标记 `invalid_annotation`、主分数 null，并保留 `upstream_invalid_format_value=0`。其 CC、GA 仍正常评估。报告保留所有会话，不静默丢弃整条样本。

YES/NO回答须有明确决策标记，GA须有唯一final标签、连续turn标签、在首次no停止且final与turn一致；严格拒绝矛盾或缺失结论。旧冻结双裁判v2把格式错误保存为错误并停止；当前和后续新任务使用 [人工未决策略](scoring-qwen-once.md#待定决策与人工复核)，正常回答的格式/标签矛盾记 `pending_review/null`，确已返回但空/截断的首份尝试记 `response_incomplete/null` 并保留原技术原因，两类分别报告，继续后续样本，最后人工。检测/特征/推理运行故障与完整性错误仍显式失败并由监控修复，不能计为编辑失败。正常回答和不完整首份尝试按身份恢复，不重采样择优。

汇总分别报告 CM/CU、各轮及整体，每项提供有效数量、总数量与缺项状态计数。双裁判分歧和 GA 前缀不单调情况单独列出。缺项会影响主分数分母，使用结果时必须同时报告覆盖率。

## 运行顺序

以下均在服务器项目目录运行，数据及输出保持分离。先读工作区 GPU 约定；只停止本次使用的 GPU 的 burn，等待 20–30 秒并检查显存释放及 pane 无子进程。不要触碰 monitor/watchdog。

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
/home/chs/conda/envs/lance/bin/python scripts/score_mice.py prepare \
  --selection smoke --output outputs/scoring/acceptance_new
# GPU 0 已按约定释放之后：
bash scripts/run_mice_scoring.sh 0 metrics --output outputs/scoring/acceptance_new
# 每个脚本结束会恢复该卡 burn；下一阶段前需再次按约定释放该卡。
bash scripts/run_mice_scoring.sh 0 qwen --output outputs/scoring/acceptance_new
bash scripts/run_mice_scoring.sh 0 gemma --output outputs/scoring/acceptance_new
/home/chs/conda/envs/lance/bin/python scripts/score_mice.py report \
  --output outputs/scoring/acceptance_new
```

`prepare` 对全部 720 会话 / 2160 轮进行 CPU 输入审计，但仅选定子集会送入评分。smoke 按固定哈希顺序和元数据覆盖所有编辑类型、CM/CU、空文本及 CC 缺项，不看生成质量。calibration 使用固定哈希顺序每个 split 的前 20 个会话，共 120 轮，供独立人工核对。选择名单在 manifest 中冻结。

`review.html` 含各会话图像、指令、得分、提示词和原始回答；`human_review.jsonl` 是留空的人工核对表，重复生成报告不覆盖已填写内容。程序通过不代表裁判准确率已校准。人工填写后，才能报告人机一致率。

全量使用新的输出目录 `outputs/scoring/full_dual_v2`，prepare 指定 `--selection all --allow-full`；其后 metrics、qwen、gemma、report 每阶段均须带 `--allow-full`。prepare 必须显式指定最新的 `--inference outputs/mice/full_bare_20261004_attention`，评分 CLI 的默认输入仍指向较早结果。单纯运行默认命令不会进入全量评分。

同一输出目录禁止覆盖 manifest；代码、参数、标注或输入变更后必须新建评分目录。恢复时重验指纹和保存结果校验值。评分源码与原推理源码相互独立；原推理输出和 attention 不写入。

## 双 GPU 并行裁判

默认可以按上节顺序用一张卡运行。若共享 metrics 阶段已完成，也可创建独立裁判 worker，让两个模型各用一张卡。worker 通过符号链接共享只读 manifest / metrics / detections，输出与运行锁分开；主报告通过稳定路径读取该 worker 的结果，不复制或重新计算分数。

```bash
/home/chs/conda/envs/lance/bin/python scripts/prepare_scoring_judge_worker.py \
  --output outputs/scoring/acceptance_new --judge gemma
# 两张卡各按 GPU 约定释放后，在两个 shell 中分别运行：
bash scripts/run_mice_scoring.sh 0 qwen --output outputs/scoring/acceptance_new
bash scripts/run_mice_scoring.sh 1 gemma --output outputs/scoring/acceptance_new/worker_gemma
# 两者均成功退出后，用主目录运行 report 与证据审计：
/home/chs/conda/envs/lance/bin/python scripts/score_mice.py report \
  --output outputs/scoring/acceptance_new
/home/chs/conda/envs/lance/bin/python scripts/validate_scoring_run.py \
  --output outputs/scoring/acceptance_new
```

worker 已存在时不再次创建；断点恢复直接重跑对应裁判命令。worker 的完整目录须随主目录一起保留，不能只搬走顶层软链接。阶段状态在各自工作目录的 `status_*.json` 中；逐轮完成记录即时落盘。

## 原始回答缓存

`requests/<judge>/` 按图像像素哈希、完整提示词、模型路径与固定 revision、生成设置、采样编号和对应 seed 绑定原始回答，保存校验值和产生该回答的代码指纹。只缓存正常结束的完整生成，不缓存截断或推理错误。重试尚未完成的轮次时，可复用已经完成的 IF 请求，再补齐缺失的 GA 请求；不会为格式问题反复生成并择优。

v1 阶段的格式修正使用 `scripts/replay_scoring_evidence.py` 从已固定的 v3/v4 证据迁移到最终 v5 实现。迁移限制在已知的源/目标代码哈希和本次验收/校准子集，重新核对元数据、选定图像、模型 revision 和生成参数。检测/CC 算法未改变的测量值可保留；裁判分数不直接复制，而由新版解析器重新解释同一份原始回答、重新计算 IF/GA 和均值。每条复用记录保留来源文件哈希及原代码指纹。此工具是本次修正的受限迁移入口，不是默认全量运行步骤。

各版原始回答和迁移来源应与 v5 结果一起保留，以供追溯。`--retry-errors` 不覆盖已完成的有效评分；若完整原始回答仍无法明确解析，缓存会保留它并继续报错，而不会换一个回答来取分。

## 上下文容量验证

`check_scoring_context.py` 使用 Qwen 原生 processor 计算全部 2160 个 GA 前缀长度，并先与 19 个真实 vLLM 请求 token 数逐一对齐。最大原图为 4000×2666，最大前缀是 `cm/0fbb4eb1135e8e84` 第三轮，输入 13070 token；加上 2048 输出预算小于 16384。因此 Qwen 新运行容量提高到 16384，不缩小或替换该图。`outputs/scoring/context_20261004/` 保存长度审计与最大前缀的单独真实测试。120 轮校准的原始调用采用 8192，完整未截断；它们的运行来源如实保留。

## 双次采样记录

每位裁判的逐轮 JSON 保存 `votes` 数组，编号为 1、2，各含 IF 与 GA_prefix 单次分数。`traces` 保存 vote、sampling_seed、temperature、完整提示词和原始回答。一次失败不能只平均剩余有效票；保留错误并停止该阶段。两次采样的缓存键不同，断点恢复也不会把同一个回答计作两票。

全量必须使用新的 `outputs/scoring/full_dual_v2` 目录；原 v1 温度 0 的结果不能复用为温度 0.6 的采样。历史 `replay_scoring_evidence.py` 只用于 v1 同参数的格式修正，参数变化会被拒绝。采样配置更新时没有重新运行测试、试评或全量准备；2026-10-05 的授权全量包含正式输入审计和结束后的证据审计。

## 全量任务编排

`scripts/full_mice_scoring_job.py` 只在新目录启动。先执行 prepare、共享 metrics，再创建隔离 Gemma worker，两个裁判各占一张卡；两个子进程均成功退出后生成报告并运行 `validate_scoring_run.py`。阶段推进等待进程退出事件，不使用模型轮询。运行期间禁止修改冻结源码。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chs/conda/envs/lance/bin/python scripts/full_mice_scoring_job.py \
  --inference outputs/mice/full_bare_20261004_attention \
  --output outputs/scoring/full_dual_v2 --allow-full
```

2026-10-05 的首次任务曾在 tmux `mice-score-full` 中运行，随后因用户改为 Qwen 单次而停止。不要重启上面的双裁判命令；当前任务见 [单次评分协议](scoring-qwen-once.md)。编排器只停止已核对的 GPU 0/1 burn 子进程，评分包装脚本退出后恢复本卡；monitor/watch_dog 不参与编排。完整评分输出是持久实验文件，独立于推理和 attention 目录。

任务目录保存 `events.jsonl`、`job_status.json`、`job_source_hashes.json`、各阶段日志和最终 `completion.json`；外层日志为 `outputs/scoring/full_dual_v2.job.log`。可以只读查询：

```bash
/home/chs/conda/envs/lance/bin/python scripts/full_mice_scoring_job.py \
  --inference outputs/mice/full_bare_20261004_attention \
  --output outputs/scoring/full_dual_v2 --status-only
```

应用 heartbeat `mice-v2` 已改为跟进当前 Qwen 单次任务，每5分钟读取少量阶段事件与任务存活状态，正常状态不发送例行进度。阶段衔接无需模型参与；完成、失败或需要处理时才汇报。完成后核对汇总、有效分母和验证结果，恢复空闲卡 burn，并暂停 heartbeat。此应用兜底检查仍使用监控 token，不宣称整个监控链条零 token。
