# 生成阶段 attention 观测

实现：`src/lance_mice/attention.py`；格式版本 `target-group-mass-v1`。使用固定 vLLM-Omni `b742f86136d28423903a1213adde2a62a11067a2`，支持单卡和 DP2，每卡一个完整模型；暂不支持 CFG2 / TP / SP。

## 保存的含义

生成第 n 轮图像时，以当前正在去噪的图像 latent token 为 query，对完整 key 集合计算注意力。每个去噪步、每层、每个 head，保存这些 query 对每个上下文段的**平均概率质量**。

第三轮输入为 `I0,T1,I1,T2,I2,T3`。保存的组为：

- `T1`、`T2`、`T3`：对应原始指令正文。
- `I0_vit/I0_vae`、`I1_vit/I1_vae`、`I2_vit/I2_vae`：原图和模型此前真实结果的两种视觉 token；图像组包括该段的模态边界 token。
- `context_other`：系统提示、分隔符及上下文后缀。旧有标签v1运行还包含历史说明和轮次标签；当前bare-v2不包含这些附加文本。
- `generation_markers`、`target_image`：当前生成标记及目标图像自身。

I1 的总注意力为 `I1_vit + I1_vae`，I2 同理。全部组的和约为 1；不会只把 T1/I1/T2/I2/T3 再归一化，掩盖其他 token 获得的注意力。每段 token 数同时保存，总质量除以 token 数可得到每 key token 的平均权重。

记录的是**正向条件分支**，CFG-negative 不与它混合。生成过程仍正常计算 CFG。保留 30 步、36 层、16 个 head，第三轮 `mass` 的形状为 `[30,36,16,12]`。query 空间维度已取均值；不保存完整逐像素 query×key 矩阵，也不据此宣称得到了 token 级熵或因果解释。

## 观测位置和计算

在上游 `PackedAttentionMoT._forward_gen` 的 `attn_noncausal_local` 调用前读取已经完成 Q/K 归一化、RoPE 和缓存拼接的真实 Q/K。固定版本把正向 CFG 分支放在 batch 0，其 key 包含历史缓存、当前标记和目标 latent。代码检查 key/query 长度和正向 padding mask，排除预填充调用，只选择目标 latent query。

设每个 key 的 value 为“属于哪个组”的 one-hot 向量，则 `softmax(QKᵀ/√d) @ one_hot` 正好给出每个 query 对各组的注意力和。旁路调用融合 SDPA，只在 Q/K 的 FP16 副本上计算这些统计；模型本身的 BF16 attention、V、速度场和 latent 更新保持原样。无需落盘巨大的完整概率矩阵。跨轮 prefix cache 不会删除历史 K/V，仍然能观测对它们的注意力。

每轮额外抽查首/末去噪步、首/末层、三个空间 query 的全部 head，与 FP32 直接 softmax 参考比较。最大绝对误差阈值为 `5e-4`，全组概率和误差阈值为 `1e-3`；不合格即停止，不静默保存错误权重。

工程验收：CM/CU 共六轮、每轮全部 step/layer/head 均有记录；最大参考绝对误差 `0.00020343`，最大概率和误差 `0.00007737`。六张图与关闭观测的既有基线逐像素一致。证据位于 `outputs/mice/attention_acceptance_20261004/summary.json`。

## 文件与读取

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
