# 实验文档索引

当前阶段以 [exp0 状态与交接](exp0-status.md) 的带时间快照及服务器实时产物为准。历史部署与查看服务主要属于 `a800_0`；2026-10-07双协议实验同时使用 `a800_0`、`a800_1`，入口见下表。文档描述协议和入口；评分、图像及attention不入Git。

| 文档 | 用途与证据边界 |
| --- | --- |
| [双协议六组实验](sixrun-20261007.md) | 四卡动态队列、token／8×8区域attention、评分、监控与恢复 |
| [双协议六组实验结果](sixrun-results-20261007.md) | 全量7928轮、评分分母、184.09GB attention、16项人工待审与独立验收 |
| [实验存储布局](storage-layout-20261007.md) | /home/chs实体目录、全量归档位置、历史路径映射与本地结论边界 |
| [实验资源清单](experiment-resources.md) | 当前实验必需权重、数据、固定来源、准备入口与候选边界 |
| [实验源码版本](../versions/README.md) | 已完成推理与评分的确切源码对应关系 |
| [exp0 状态与交接](exp0-status.md) | 当前任务、产物定位、终态条件与部署差异 |
| [六组推理结果对照](sixrun-review.md) | 8775，最新 EdiVal／MICE／ImgEdit 的 bare/chat 全量真实会话 |
| [MICE第一版结果](mice-first-edition-results.md) | 720会话2160轮，31人工项补全、分母与CM/CU/轮次/类型统计 |
| [Qwen 单次评分](scoring-qwen-once.md) | 单票与首答策略；旧v3运行及新六组身份分开 |
| [评分协议](scoring-protocol.md) | 历史双裁判v2、IF/CC/GA规则、官方差异及缓存 |
| [裁判提示词](scoring-prompts.md) | 历史基础正文、新六组最小输出约束与图像顺序 |
| [评分环境](scoring-environment.md) | 固定模型、隔离环境、安装和历史GPU冒烟 |
| [评分工程验收](scoring-validation.md) | v1的12/120轮证据，不能当作当前裁判准确率 |
| [EdiVal 推理](edival-inference.md) | CSV前缀、ZIP原图、512尺寸与冻结运行 |
| [EdiVal 查看](edival-review.md) | 8770全量页面；572会话及2288张HTTP图片复核通过 |
| [EdiVal 官方评分环境](edival-scoring-environment.md) | 八项官方模型、EdiVal/独立裁判/HPS环境；GPU1组件检查通过；全量入口见评分接口 |
| [EdiVal 官方评分接口](edival-scoring.md) | 官方 multipass 函数、结果映射、双卡会话分片与评分证据 |
| [EdiVal 官方评分结果](edival-scoring-results.md) | 全量终态、逐轮分量、有效分母与独立复核 |
| [EdiVal 论文格式评分表](edival-paper-tables.md) | 累计IF、官方CC/O、VQ及两种Δ口径 |
| [MICE裁判失败分析](scoring-prompt-failure-analysis.md) | 31项输出失败及合法格式回答的语义边界 |
| [MICE提示词候选](scoring-prompt-candidates.md) | 已部署的最小修改及尚未采用的候选对照 |
| [MICE人工待定评分](mice-pending-human-review.md) | 8772、31项人工判断、规则/原证据/独立持久化 |
| [MICE 查看](mice-review.md) | 8768 bare抽查及8765历史页面 |
| [ImgEdit 人工评分](imgedit-human-review.md) | 8769 bare人工表及8767历史表、持久化与分母 |
| [Attention分析](attention-analysis.md) | v2前三项及后四项的源码、数据身份、CPU检查与验收入口 |
| [Attention语义验收](attention-semantic-validation.md) | 真实缓存身份、v1限制与v2必需门槛 |
| [源码同步](source-sync.md) | GitHub发布范围、克隆自检与服务器冻结部署边界 |
| [Attention 观测](attention-observation.md) | 新token／8×8区域字段、旧分组字段及解释限制 |
| [原生速度试跑](native-speed-probe.md) | 2026-10-04旧封装、单次粗略速度参考 |
| [伪文字诊断](text-artifact-diagnosis.md) | 封装消融与后续bare协议的证据边界 |

旧EdiVal独立运行的 [512验收](edival-validation-512-20261005.json) 是该版门槛；[768验收](edival-validation-20261005.json) 属于历史；[页面启动检查](edival-viewer-launch-20261005.json) 不代表全量终态。2026-10-06全量推理和HTTP图片复核均passed；完整证据在对应 `outputs/edival/full_512_bare_20261005_attention/` 与 `outputs/review/edival_512_bare_20261005/`，少量终态JSON已同步本地。完整图像、attention、逐次裁判证据及人工评分主库按约定保存在实验服务器，无需同步本地，源码与部署的同步边界见[源码同步](source-sync.md)。

阅读当前研究先读状态→Attention分析；核对能力基线读六组结果→六组协议→存储布局；历史评分与查看文档只解释各自独立运行。文档中的已同步小型JSON属于2026-10-06历史事实，2026-10-07之后不继续下载原始产物或清单。本地保留研究知识与必要源码，完整实验数据留服务器。
