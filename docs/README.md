# 实验文档索引

当前阶段以 [exp0 状态与交接](exp0-status.md) 的带时间快照及服务器实时产物为准。文档描述协议和入口；评分、图像及attention不入Git。

| 文档 | 用途与证据边界 |
| --- | --- |
| [exp0 状态与交接](exp0-status.md) | 当前任务、产物定位、终态条件与部署差异 |
| [Qwen 单次评分](scoring-qwen-once.md) | 当前单票v3、人工未决/不完整回答、续跑与监控 |
| [评分协议](scoring-protocol.md) | 历史双裁判v2、IF/CC/GA规则、官方差异及缓存 |
| [裁判提示词](scoring-prompts.md) | 完整正文、字段、图像顺序；票数以运行profile为准 |
| [评分环境](scoring-environment.md) | 固定模型、隔离环境、安装和历史GPU冒烟 |
| [评分工程验收](scoring-validation.md) | v1的12/120轮证据，不能当作当前裁判准确率 |
| [EdiVal 推理](edival-inference.md) | CSV前缀、ZIP原图、512尺寸与冻结运行 |
| [EdiVal 查看](edival-review.md) | 8770动态页面与结束后的HTTP图片复核 |
| [MICE 查看](mice-review.md) | 8768 bare抽查及8765历史页面 |
| [ImgEdit 人工评分](imgedit-human-review.md) | 8769 bare人工表及8767历史表、持久化与分母 |
| [Attention 观测](attention-observation.md) | 正向分组概率质量、数值检查与解释限制 |
| [原生速度试跑](native-speed-probe.md) | 2026-10-04旧封装、单次粗略速度参考 |
| [伪文字诊断](text-artifact-diagnosis.md) | 封装消融与后续bare协议的证据边界 |

EdiVal的 [512验收](edival-validation-512-20261005.json) 是当前门槛；[768验收](edival-validation-20261005.json) 属于历史；[页面启动检查](edival-viewer-launch-20261005.json) 不代表全量终态。完整运行证据在服务器对应 `outputs/` 中。
