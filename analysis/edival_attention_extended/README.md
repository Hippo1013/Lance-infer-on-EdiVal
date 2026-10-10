# EdiVal后四项注意力分析

本目录对已验收的`target-token-region-stats-v2`进行离线分析，不调用模型或评分器。研究方案维护于exp0父目录的`attention分析方案.md`；结果继续写入`attention分析结果.md`。本目录是持久实验源码，原始数据和数值产物只在服务器保存。

## 数据与计算范围

正式输入来自两机`outputs/attention_recollection/edival_semantic_20261009_v2`，以0号机已验收统计根的`metadata/source_manifest.jsonl.gz`绑定路径、生产机、历史与输出身份。任务标识为`edival_extended_20261010_v1`，输出根为`outputs/attention_analysis/edival_extended_20261010_v1`，图表根为`outputs/attention_reports/edival_extended_20261010_v1`。

每轮30步、36层、16个头；Query已平均。组均值、总体标准差与分位数支持分组Query参与程度，不能恢复Query空间位置或合并组的精确离散程度。文本按实际token，图片按正确映射的64格；图像起止标记独立统计。

## 程序职责

| 程序 | 职责 |
| --- | --- |
| `run.py`、`core.py`、`layout.py` | 预先固定原图组划分、两机分片计算、身份核对、过程与头指标、空间与token分布、同对象配对 |
| `transfer.py`、`merge.py` | 临时私网只读传输，派生文件逐项核对大小和哈希；凭据放系统临时目录 |
| `validation.py`、`check_real.py` | 独立标量数学用例及两机真实输入公式复算 |
| `aggregate.py` | 会话等权图谱、按原图聚类的区间、探索窗口冻结与复核 |
| `review_windows.py`、`finish_review.py` | 冻结窗口的原始记录敏感性、准确关注量加权、指令长度分层及候选归纳 |
| `plot.py` | 在服务器生成PNG／SVG和真实图片叠加图 |
| `validate_publication.py` | 全量覆盖、来源哈希、既有基线复现、分组与图谱口径核对 |

## 输出解释

`arrays/edival/<id>/turn_<t>.npz`包含`names`索引以及过程、层／头、步骤、标量和有效记录计数。`process`按头对有效值平均；`head`按步骤平均；`scalar`按全部有效记录平均，`weighted_scalar`在同一会话、同一对象和轮次内按对应关注量加权。两种均值回答不同问题；条件指标的有效记录数量可能随位置变化，因此不同平均顺序应明确标注。

`profile_names`索引空间分布、文本分布、头间差异与头贡献；`space_overall`、`space_step`和`space_layer`的第一路是记录等权条件比例，第二路是实际关注量加权。所有百分比均以未平滑的实际保存值计算；零分母为缺失，不是零关注比例。Top集合使用稳定索引处理并列，并以完整分布差异作为伴随指标。

`paired/`保持同一会话、同一真实对象、相同步骤／层／头，对齐比较总量、集中度、分布差异和高关注集合重合度。分布差异采用总变差距离，即对应元素份额差的绝对值之和的一半；0表示相同，1表示完全分离。

`summaries/full_atlas.npz`保存全部会话的过程、头及空间图谱，`exploration_atlas.npz`保存探索部分；`session_scalars.npz`与`tables/session_scalars.csv.gz`保留会话值。按原图组重采样2000次，会话仍等权。探索与复核各285个原图组，重复原图不跨组；这检验新增局部规律的稳定性，前三项已看过全量数据。

`metadata/frozen_candidates.json`必须在读入复核部分细粒度图谱前生成，窗口为固定连续3步×4层。复核不重新调整窗口。窗口关注量加权的准确结果在`window_variants.json`中；仅在每个过程位置内部加权的辅助曲线不能替代整个窗口的准确加权。

## 执行与验收

依次运行`run.py prepare`、两机`run.py compute --host a800_0/a800_1 --workers 8`、两机`check_real.py`、`merge.py`、`aggregate.py`、两机`review_windows.py`、`finish_review.py`、`plot.py`和`validate_publication.py`。执行前核对两机主机身份、既有环境、输出根与进程；重用任务目录时先核对来源与设计，不能覆盖未知结果。不要在已有任务结束前启动同一任务的另一实例。

默认CPU，设置数值库单线程。独立公式检查不冒称其他Codex复核；开始计算、生成图或脚本退出均不能代替最终验收。最终还需本地图表视觉检查、Markdown引用与哈希绑定、服务退出、凭据和临时目录清理及GPU终态核对。临时服务只在本任务期间开放固定私网入口，不提供目录浏览、凭据或远程执行。
