# 生成阶段 attention 观测

六组实验使用 `src/lance_mice/attention_regions.py`、格式 `target-token-region-stats-v1`；旧实验使用 `src/lance_mice/attention.py`、格式 `target-group-mass-v1`。读取时先检查protocol，不能沿用旧mass字段直接读取新NPZ。使用固定 vLLM-Omni `b742f86136d28423903a1213adde2a62a11067a2`，支持单卡和 DP2，每卡一个完整模型；暂不支持 CFG2 / TP / SP。

## VAE空间标签的已知缺陷

2026-10-09在EdiVal的1716份原始记录及两机固定版本实现中确认：`PackedAttentionMoT._forward_gen`预填充缓存顺序为`[past, start_marker, end_marker, vae_tokens...]`，`region_layout`却按输入顺序`[start_marker, vae_tokens..., end_marker]`划分空间位置。结束标记因此误入VAE左上格，最后一个真实空间token误入`context_other`，区域边界也有错位。VAE空间图、集中度、VAE组总量及含VAE的整图轨迹不能按预期空间语义解释。ViT的`_forward_und`缓存路径没有同一重排。

原概率和、区域求和与参考softmax检查均共用同一标签，无法发现这个身份错误；通过这些检查不代表缓存身份正确。CPU复现脚本为[scripts/audit_lance_cache_order.py](../scripts/audit_lance_cache_order.py)，诊断任务根为两机`outputs/attention_analysis/edival_region_bias_20261009_v1`。逐token信息没有保存在区域和中，不能通过平移热图精确修复。旧产物与冻结runtime保留，开发版已阻止v1产生新观测。六组共用同一v1实现，其VAE空间观测都存在这一限制；本次只补采与分析EdiVal，其他bench尚未修正。修复实现及后续必需门槛见[语义验收](attention-semantic-validation.md)。

## 当前EdiVal观测与验收

`target-token-region-stats-v2`通过双机固定4会话12轮、观测开关共24次生成验证，36层真实缓存身份、首次输出像素及未受影响数组完全一致。全量572会话1716轮也逐轮复现原输出，首末层每个去噪步核对实际Q/K、缓存与目标查询身份。2026-10-09 17:13完成全量统计、独立公式及本地图文验收，[七项分析](attention-analysis.md)的图文报告已更新，未改评分或开展干预。

正确的ViT空间范围为`[start+1,end-1)`，VAE为`[start+2,end)`；标记进入`context_other`并单独保存`image_marker_mean[30,36,16,N_image,2,2]`及`image_marker_key_positions`。观测runtime为两机`runtime/edival_semantic_20261009_v2_r2`，原件为`outputs/attention_recollection/edival_semantic_20261009_v2`，统计根为0号机`outputs/attention_analysis/edival_semantic_20261009_v2`。统计根`local_acceptance.json`绑定科学清单、双机独立像素／身份核对、模型环境、本地成图／报告和资源终态。

## 六组实验的token与区域字段

2026-10-07六组实验共7928份NPZ，184.09 GB。按CFG正分支、完整key softmax保存FP32统计；目标图query取平均，历史图区域内key求和，逐步／层／头保留，不对选中上下文二次归一。文本限原始历史和当前指令正文。设计上system／角色／分隔符与视觉非空间标记归入其他上下文；v1的VAE标记实际错位，必须遵守首节限制。

| 字段 | 形状与意义 |
| --- | --- |
| `text_mean` | `[30,36,16,N_text]`，每个指令key token的query均值 |
| `region_mean` | `[30,36,16,N_image,2,8,8]`，历史图ViT／VAE区域mass |
| `group_stats` | `[30,36,16,N_group,5]`，mean／population std／P10／P50／P90，顺序以stat_names为准 |
| 文本索引 | text_key_positions、text_token_ids、text_token_turns、text_token_offsets |
| 空间与分组索引 | region_token_counts、fine_key_channel、image_geometry_json、group_names、group_token_counts、token_group_ids |
| 观察身份 | protocol、timesteps、layer_ids、target_query_count；文件与运行身份绑定逐轮JSON和receipt |

当前代码检查FP32有限非负值、分位数顺序、区域mass守恒、完整组概率和、索引及几何；参考误差门槛5e-4、概率和门槛1e-3。两种协议开启／关闭观测的真实输出逐像素一致，六组全部覆盖与独立终态验收见 [实验说明](sixrun-20261007.md) 与 [结果](sixrun-results-20261007.md)。

区域格式仍聚合query空间，不能恢复生成图各位置与历史token的对应矩阵，也不能计算每个生成query自身的熵。它可以计算指令key token上的query平均分布熵；区域熵按空间格统计，不能等同于空间token熵。分布统计不能直接给出因果效应。本文字段表维护保存规格；六组实测容量见[六组结果](sixrun-results-20261007.md#attention-存储)。

## 历史分组格式的含义

生成第 n 轮图像时，以当前正在去噪的图像 latent token 为 query，对完整 key 集合计算注意力。每个去噪步、每层、每个 head，保存这些 query 对每个上下文段的**平均概率质量**。

第三轮输入为 `I0,T1,I1,T2,I2,T3`。保存的组为：

- `T1`、`T2`、`T3`：对应原始指令正文。
- `I0_vit/I0_vae`、`I1_vit/I1_vae`、`I2_vit/I2_vae`：原图和模型此前真实结果的两种视觉 token；图像组包括该段的模态边界 token。
- `context_other`：系统提示、分隔符及上下文后缀。旧有标签v1运行还包含历史说明和轮次标签；当前bare-v2不包含这些附加文本。
- `generation_markers`、`target_image`：当前生成标记及目标图像自身。

I1 的总注意力为 `I1_vit + I1_vae`，I2 同理。全部组的和约为 1；不会只把 T1/I1/T2/I2/T3 再归一化，掩盖其他 token 获得的注意力。每段 token 数同时保存，总质量除以 token 数可得到每 key token 的平均权重。

记录的是**正向条件分支**，CFG-negative 不与它混合。生成过程仍正常计算 CFG。保留 30 步、36 层、16 个 head，第三轮 `mass` 的形状为 `[30,36,16,12]`。query 空间维度已取均值；不保存完整逐像素 query×key 矩阵，也不据此宣称得到了 token 级熵或因果解释。

## 历史分组观测的计算

在上游 `PackedAttentionMoT._forward_gen` 的 `attn_noncausal_local` 调用前读取已经完成 Q/K 归一化、RoPE 和缓存拼接的真实 Q/K。固定版本把正向 CFG 分支放在 batch 0，其 key 包含历史缓存、当前标记和目标 latent。代码检查 key/query 长度和正向 padding mask，排除预填充调用，只选择目标 latent query。

设每个 key 的 value 为“属于哪个组”的 one-hot 向量，则 `softmax(QKᵀ/√d) @ one_hot` 正好给出每个 query 对各组的注意力和。旁路调用融合 SDPA，只在 Q/K 的 FP16 副本上计算这些统计；模型本身的 BF16 attention、V、速度场和 latent 更新保持原样。无需落盘巨大的完整概率矩阵。跨轮 prefix cache 不会删除历史 K/V，仍然能观测对它们的注意力。

每轮额外抽查首/末去噪步、首/末层、三个空间 query 的全部 head，与 FP32 直接 softmax 参考比较。最大绝对误差阈值为 `5e-4`，全组概率和误差阈值为 `1e-3`；不合格即停止，不静默保存错误权重。

工程验收：CM/CU 共六轮、每轮全部 step/layer/head 均有记录；最大参考绝对误差 `0.00020343`，最大概率和误差 `0.00007737`。六张图与关闭观测的既有基线逐像素一致。证据位于 `outputs/mice/attention_acceptance_20261004/summary.json`。

## 历史分组文件的读取

每轮保存 `turn_n.png`、`turn_n.json`、`turn_n.attention.npz`。JSON 含 NPZ 的 SHA-256、组名、token 数、抽查误差和平均质量；断点恢复检查 attention 文件完整性，拒绝缺失或篡改。

NPZ 字段：`mass[step,layer,head,group]`、`group_names`、`group_token_counts`、`token_group_ids`、`timesteps`、`layer_ids`、`target_query_count` 和 `protocol`。

```python
import numpy as np
with np.load("turn_3.attention.npz", allow_pickle=False) as a:
    names = a["group_names"].tolist()
    m = a["mass"]
    weights = {
        "T1": m[..., names.index("T1")],
        "I1": m[..., names.index("I1_vit")] + m[..., names.index("I1_vae")],
        "T2": m[..., names.index("T2")],
        "I2": m[..., names.index("I2_vit")] + m[..., names.index("I2_vae")],
        "T3": m[..., names.index("T3")],
    }
    print({k: float(v.mean()) for k, v in weights.items()})
```

attention 分布与编辑失败的关系是研究对象；相关性不能代替后续干预实验。
