# Attention观测的语义验收

## 格式边界

`target-token-region-stats-v1`存在已复现的VAE缓存标签错位：缓存中的结束标记进入空间格，最后一个空间token进入其他上下文。其概率守恒和区域求和仍可能通过，不能作为正确空间语义的证据。六组实验共用此实现，因此各组v1 VAE空间数据都须遵守这一限制；本次只补采EdiVal，没有扩展其他组统计。

`target-group-mass-v1`整块图像分组本来包括起止标记，块内重排不改变整块求和，不能将它与空间token分组直接混用。旧文件与冻结runtime不改写。开发版本的v1区域观测器拒绝生成新观测，以防重复引入已知错误；历史读取与诊断保留。

`target-token-region-stats-v2`以真实缓存身份验证为前提：ViT的空间范围为`[start+1,end-1)`，标记为`start,end-1`；VAE的空间范围为`[start+2,end)`，标记为`start,start+1`。空间token不重不漏；标记作为独立观测保存到`image_marker_mean`，整体分配仍归入`context_other`。文本区域通道与新增标记分别计算，保持原文本均值的浮点求和布局。

目前v2的真实模型验收范围为EdiVal既定完整历史、512×512输出与单GPU模型。`SemanticLancePipeline`遇到不同目标查询布局会拒绝运行。不能未经验证就将它外推至新的分辨率、CFG并行、TP或SP。

## 身份验证

[cache_semantics.py](../src/lance_mice/cache_semantics.py)在原生gather前捕获post-RoPE Q/K只读副本，用精确BF16相等核对真实缓存追加、原缓存前缀、原生正分支Q/K及目标图查询。VAE trace使用实际gather索引，禁止仅从输入顺序猜测缓存顺序。小规模运行核对36层；全量每个去噪步核对首末层。索引、实现或目标布局变化必须重新通过小规模验收。

[scripts/validate_attention_semantics.py](../scripts/validate_attention_semantics.py)使用具有明确身份的缓存fixture，验证标记排除、空间token基数、文字通道分离和旧错误映射拒绝。`scripts/audit_lance_cache_order.py`另执行固定上游原生方法的CPU身份探针；CPU探针只证明顺序逻辑，不能代替真实模型运行。

旁路FP16 SDPA观测需要小规模FP32 softmax参考。v2对首末层、首末去噪步的实际查询抽查分组和全部细通道的绝对误差。计算精度导致的保存零值须与数学上的精确零区分。

## 统计口径

- 一条记录对应一个去噪步、一个层、一个head；输出图的1024个查询已平均。熵是在该平均分布上计算，不能解释为每个输出查询自身的熵。
- 图像起止标记不属于空间格；区域数量和token数量是不同的观察分辨率。文本与图像共同分配的百分比可以描述，token数量校正不等于语义功能校准。
- 熵、top份额和轮内二者比例均先在保存记录上计算，再平均记录和会话。新增上下文对原始查询的影响无法从查询平均数组完全消除；轮内二者比例只能拆分两个份额共同缩小的影响。
- 区分条件比例等权平均与实际关注量加权。极小关注量记录条件化后仍可能出现显著局部比例；同时查看原始总量、有效位置数和敏感性结果。
- 统计样本单位为会话，按重复原图聚类重采样；不是将层、头、去噪步或多轮记录当作独立样本。
- 缺失对象、结构性零和保存精度下的零分别表示。非线性计算、FP32存储误差、全部表主键和同一对象哈希需独立检验。

## 运行入口

本次任务身份为`edival_semantic_20261009_v2`，通过双机验证的runtime为`runtime/edival_semantic_20261009_v2_r2`。由`scripts/prepare_attention_recollection.py`从冻结六组runtime复制并作显式观测补丁，保留独立manifest。`scripts/recollect_edival_attention.py`使用首次保存的真实历史，逐轮要求输出RGB哈希完全复现。`scripts/supervise_attention_recollection.py`负责单卡任务、系统临时目录清理与对应burn恢复。

任何小规模身份或像素复现失败都会阻止全量。失败记录保留，不通过换样本、重采样或放宽误差门槛替代诊断。`analysis/edival_attention_semantic_v2`保留独立熵公式、全表等权口径、聚类bootstrap抽查、轮内同类对象比例及空间映射检查。科学产物与验收清单绑定最终哈希；报告只能引用已验收结果。

## 当前验收记录

2026-10-09 17:13完成EdiVal全部572会话1716轮与12组图文报告的最终验收。真实缓存身份、固定首次历史和保存图像的独立RGB核对通过；两机核心源码、模型环境及上游Bagel／Lance实现哈希一致。17张表、1716份派生数组、全部候选／科学清单，以及本地PNG／SVG与Markdown身份核对通过。独立公式涵盖18轮真实数组、全表均值及权重、3项聚类bootstrap区间和96组真实同类比例。阶段身份、统计实现与验收入口见[Attention分析](attention-analysis.md)。

独立保存图像／runtime核对为`scripts/validate_edival_recollection_completion.py`，各生产机证据在观测根`validation/recollection_independent_completion.json`；科学与成图核对为`scripts/validate_edival_analysis_publication.py`。0号机统计根`local_acceptance.json`绑定最终科学产物、双机身份核对、本地图文审阅和资源关闭。图11曾出现角落数字裁切，失败图与绘图版本保留在报告根`validation/visual_issue_r1`，修正后的12组图通过视觉验收。

防复发检查不能仅以“概率和为1”或“区域相加等于整组”替代身份验证。新观测格式、上游实现、查询布局或并行方式变化时，必须重新验证原生缓存身份和输出一致性；本次验收不能外推到其他bench或新的模型配置。
