# EdiVal 官方多轮评分接口

本接口对已经完成的 Lance 512 bare 全量结果执行官方 `multipass` 评分。推理输入仍为完整真实交错历史，评分口径沿用固定官方版本 [96d34b0](https://github.com/TianyuCodings/EdiVal/tree/96d34b00d7ea2dc3de90f2bb01f292f6dec6294a)。模型、独立环境与安装检查见 [评分环境](edival-scoring-environment.md)。

2026-10-06 13:02（Asia/Shanghai），4会话12轮、全部9类任务的双卡验收完成，16张HPS图片与11次真实裁判调用通过来源校验，validation passed、completion completed/退出0。13:04:10全量任务在tmux `edival-score-full` 启动，结果根目录为 `outputs/scoring/edival_official_full_20261006/`，572会话1716轮。13:42:58全量完成、退出0，validation及独立只读复核passed；各阶段1716轮、HPS2288张。最终原始数值及有效分母见 [评分结果](edival-scoring-results.md)，累计IF、官方CC/O及VQ Δ见 [论文格式评分表](edival-paper-tables.md)。

## 图片映射与评分口径

| 项目 | 输入与规则 |
| --- | --- |
| IF | 上轮真实生成图 → 本轮生成图；第一轮使用原图。仅评当前指令，调用官方 `evaluate_instruction_following`，保留全部类别分支、阈值和 Qwen 提示词 |
| CC | 原图 → 本轮生成图，使用当前 CSV 行的 `unchanged_objects`、`all_objects`；完全沿用官方 `bg_consistency` 门控及 DINOv3 token 处理 |
| VQ | 每轮生成图与每个会话原图分别运行官方 RAHF 与传统图像质量函数；原图基线独立保留 |
| HPS | 官方 `update_hps_scores.py` 补评分口径：空文本、每张图片输出第一维、不四舍五入；每会话原图加三轮结果共4张 |
| 汇总 | 直接执行官方 `TaskRateCollector`，保留 `task_rate.json` 原始列表、`image_rate.json` 与逐会话结果；额外统计各项均值和有效分母，不构造综合总分 |

图片直接引用 `outputs/edival/full_512_bare_20261005_attention/run/edival/<index>/turn_0_input.png` 与 `turn_1.png` 至 `turn_3.png`，不复制或重编码推理结果。当前 CSV 的完整前缀由官方 parser 解析，取最后一条指令及类别。评分接口不会把历史指令再次发送给裁判。原图保存像素须匹配发布 ZIP；生成图、真实输入历史与原始指令须匹配推理 JSON 和会话指纹。图像与 attention 不回写。

## 双卡任务划分

按照数字索引排序后的会话交替分配，两卡各286会话、858轮。每张卡独立按 IF → metrics → HPS 三阶段运行，同卡阶段间退出模型进程并等待显存释放；两张卡无全局阶段屏障，可各自进入下一阶段。单卡任务完成后恢复该卡 burn；另一卡继续自己的任务。

IF 阶段在 `EdiVal-judge` 中加载一份完整 Qwen2-VL，并通过同卡的 `EdiVal` 子进程调用原样 GroundingDINO `predict`；图像张量以无损数组传输，检测框、分数与文本返回原样数值，`tensor_parallel_size=1`；metrics 阶段在 `EdiVal` 中加载 GroundingDINO、DINOv3-B 与 RAHF；HPS 阶段在 `EdiVal-hps` 中加载官方 reward 模型。双卡使用数据分片与模型副本，不将一份 Qwen 跨卡拆分；官方 loader 默认 TP=2 是部署参数，判分函数、权重、提示词和采样参数保持一致。Qwen固定温度0、最多1024 token，核心对象提取最多20 token，模型上下文2048、两张图上限、显存比例0.5；使用已通过的 vLLM 旧执行后端及 eager 模式。

## 验收与运行入口

持久源码为 `scripts/edival_scoring.py` 和 `scripts/full_edival_scoring_job.py`；任务固定部署于服务器 `/home/chs/exp0_attention/Lance-infer-on-EdiVal/runtime/edival_scoring_official_20261006_v2/`，活动任务期间不得修改。manifest 绑定四项评分代码/配置哈希、85项上游源码哈希、环境验收身份、原始 CSV/ZIP 身份、推理验证、逐图字节/像素哈希、逐轮指令与推理记录。

固定接口验收选取会话 `0,1,14,27`，12轮覆盖全部9类编辑任务，使用双卡和正式评分入口。通过后将其首份 IF/metrics/HPS 证据直接纳入全量运行；`seed-reuse.json` 保存来源与哈希。不会为了全量重新获取这些样本的裁判回答。

以下命令均在服务器执行。仅预检时保持 burn；GPU运行前按工作区规程释放对应两卡并核对 pane、子进程及显存。环境名必须保留大小写。

```bash
runtime=/home/chs/exp0_attention/Lance-infer-on-EdiVal/runtime/edival_scoring_official_20261006_v2
acceptance=/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/scoring/edival_official_acceptance_20261006_v2
full=/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/scoring/edival_official_full_20261006

/home/chs/conda/envs/EdiVal/bin/python -B "$runtime/scripts/edival_scoring.py" \
  prepare --indices 0,1,14,27 --output "$acceptance"
# 两卡已释放后运行；仅紧接全量启动时使用 keep-released。
/usr/bin/python3 -B "$runtime/scripts/full_edival_scoring_job.py" \
  --keep-released --output "$acceptance"

# 只有 acceptance 的 validation passed 后才能复用。
/home/chs/conda/envs/EdiVal/bin/python -B "$runtime/scripts/edival_scoring.py" \
  prepare --seed-from "$acceptance" --output "$full"
/usr/bin/python3 -B "$runtime/scripts/full_edival_scoring_job.py" --output "$full"
```

新输出目录须不存在；已经启动的任务拒绝直接重启，避免自动重新采样。两卡进程互斥锁位于 `outputs/scoring/.edival-scoring-gpu{0,1}.lock`，退出释放文件锁。运行时缓存使用各自 `mktemp` 等价的唯一系统临时目录，结束清理；模型加载中仍视为实际占卡，不能恢复 burn。主监督器只终止自己创建的进程。`monitor`、`watch_dog` 及其他任务不操作。

首次联用验收在加载阶段发现 Transformers5 的 BERT 接口不兼容 GroundingDINO，未产生裁判回答。原失败目录 `outputs/scoring/edival_official_acceptance_20261006/` 与初始源码保留；v2通过环境隔离的检测子进程处理，不修改官方BERT/检测源码。

## 评分证据与终态校验

- `manifest.json`、`preflight.json`：固定来源、输入与完整分片。
- `worker_<gpu>_<stage>.json`、`gpu_<gpu>.json`、`status.json`：模型加载、阶段进度及监督状态。
- `stages/if/<index>.json`：每轮 IF 分数、原因、实际裁判响应、实际 token IDs 和采样参数。
- `evidence/if_calls/`、`evidence/if_images/`：每次调用立即保存首份记录和官方实际发送的 JPEG 字节，包括检测裁剪图；验收复用的证据保持原始目录与身份。
- `stages/metrics/`、`stages/hps/`：官方质量/CC细节、HPS原始两维输出及组件间依赖哈希。
- `results/multipass/<index>_input_raw.json`、`task_rate.json`、`image_rate.json`：与官方 multipass 结构兼容的最终输出。
- `summary.json`、`validation.json`、`completion.json`：汇总、逐结果来源复核与退出状态。

异常不重采样，不用0替代技术失败。官方内部捕获的 VLM/检测错误由独立跟踪器记录，并使任务显式失败；真实“no”回答仍按原判分逻辑执行。官方允许的 CC 空值继续保留并列出数量，例如不适用背景一致性的行、没有可比较对象或没有未遮罩背景；不补值、不改变分母。有效数值须有限，HPS及质量字段须完整。

全量完成须同时满足：572会话、1716轮的三个阶段精确覆盖，2288张HPS图片，原图/结果及实际裁判图片哈希不变，IF→metrics→HPS依赖一致，官方原样汇总覆盖通过，`validation.status=passed`、`completion.state=completed` 与退出0。启动、首批验收及中间阶段均不作为全量完成。按2026-10-07存储约定，JSON清单、原始证据与完整产物均留服务器，本地仅记录汇总结论、协议与必要位置。

本聊天的 heartbeat `edival-gpu1` 在运行期间每5分钟检查少量状态，正常推进保持安静；全量完成并复核、汇总结论及汇报后暂停；该历史heartbeat现已暂停。终态与证据见 [评分结果](edival-scoring-results.md)。
