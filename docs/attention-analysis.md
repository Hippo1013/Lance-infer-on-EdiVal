# EdiVal注意力分析实现

截至2026-10-10，EdiVal七项描述分析已完成：前三项为总体分配、同一对象跨编辑轮次变化和内部集中度；后四项为步骤／层变化、头差异与查询参与程度、图片空间变化、会话覆盖与候选复核。全部使用同一572会话1716轮，不按编辑成败筛选。当前没有评分关联、因果干预或训练结果。

本页提供公开仓库的源码与复现入口。完整中文研究方案、26张图的结果报告及最终PNG／SVG由本地UMM研究目录维护，不属于本仓库；原始观测、数值、图像、清单和验收日志留服务器。读取条件与解释限制见[观测说明](attention-observation.md)和[语义验收](attention-semantic-validation.md)。

## 数据身份与阶段产物

两机项目根均为`/home/chs/exp0_attention/Lance-infer-on-EdiVal`，同名路径不表示共享存储。研究采用`input_protocol=lance-history-chat-v1`，观测格式为`target-token-region-stats-v2`；不能将旧六组v1 VAE空间观测混入本次统计。

| 阶段 | 源码 | 服务器产物与验收入口 |
| --- | --- | --- |
| v2观测修复与补采 | `src/lance_mice/{cache_semantics,semantic_pipeline,attention_regions_v2}.py`，`scripts/{prepare_attention_recollection,recollect_edival_attention,supervise_attention_recollection}.py` | 两机`outputs/attention_recollection/edival_semantic_20261009_v2`；runtime为`runtime/edival_semantic_20261009_v2_r2`；各机`validation/recollection_independent_completion.json` |
| 前三项统计与图文，2026-10-09验收 | `analysis/edival_attention_semantic_v2/`，`scripts/build_edival_attention_report.py`及绘图／报告脚本 | 0号机`outputs/attention_analysis/edival_semantic_20261009_v2`，`local_acceptance.json`绑定最终科学清单与图文；图表根为`outputs/attention_reports/edival_semantic_20261009_v2` |
| 后四项，2026-10-10验收 | [analysis/edival_attention_extended](../analysis/edival_attention_extended/README.md) | 0号机`outputs/attention_analysis/edival_extended_20261010_v1`保存全量，1号机同名根保存其分片及复算；`completion.json`、`validation/science.json`、`validation/local_publication.json`、`validation/final_binding.json`；图表根为`outputs/attention_reports/edival_extended_20261010_v1` |

v2小规模双机4会话12轮、观测开关24次生成通过真实36层缓存身份及像素验证；全量输出逐像素复现首次输出。前三项17张表与1716份派生数组通过独立公式、等权汇总与聚类区间检查。后四项6864组配对、61776项既有标量复现、双机18轮独立公式及12组真实配对检查通过。当前阶段采用独立公式实现，不冒称另一个Codex的复核。

## 统计与复现边界

目标图Query已经平均，不能恢复输出Query与输入Key的空间对应。文本使用真实token，图片使用各自编码的64个空间格，非空间标记单列。熵、top份额与条件比例先逐记录计算，再按对象、轮次与会话的约定平均；统计样本为会话，按570个原图组重采样。记录等权与实际关注量加权分别保留，零分母与极小正值分别处理。

后四项探索／复核各285个原图组，重复原图不跨组；窗口固定连续3步×4层，在读取复核细粒度图谱前冻结。此前前三项已看过全量，因此此划分检验新增局部规律的稳定性。关注变化和局部偏向只能提出干预问题，不能直接证明遗忘或性能收益。

分析程序绑定既有任务路径、源清单与首次历史；GitHub源码不是服务器冻结runtime。首次运行前须准备服务器数据与清单、核对源码和环境身份，逐阶段执行；不要直接运行入口重建已完成结果或用CPU测试代替真实数据验收。

## 本地CPU检查

使用满足`pyproject.toml`的Python、NumPy与Pillow，在仓库根执行；不加载编辑模型或评分器：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B scripts/validate_attention_semantics.py
PYTHONDONTWRITEBYTECODE=1 python -B analysis/edival_attention_extended/validation.py
```

前三项数学检查会写验证JSON，须将测试根显式设为系统临时目录：

```bash
export UMM_TEST_ROOT=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=analysis/edival_attention_semantic_v2 python -B - <<'PY'
import os
from pathlib import Path
import tests
tests.ROOT = Path(os.environ['UMM_TEST_ROOT'])
print('Mathematical checks passed:', len(tests.run()))
PY
rm -r "$UMM_TEST_ROOT"
unset UMM_TEST_ROOT
```

缺少Torch时两项相关CPU测试会跳过；这些检查不证明GPU运行、全量数据覆盖或服务器部署同步。全量验收以绑定最终产物版本的上述回执为准。
