# 原生 Lance 小样本速度试跑

本文记录2026-10-04旧封装条件下的历史试跑，不是当前bare-v2的速度测量。入口：`scripts/native_speed_probe.py`。使用 [官方 Lance](https://github.com/bytedance/Lance) 固定提交 `4baeee086648996f6ab12e673cbe461b0b149997` 的真实模型、`ValidationDataset`、`validate_on_fixed_batch` 与 `validation_gen_KVcache`，不是 Omni 的单轮参考路径。

## 输入与采样

2026-10-04运行复用当时版本的 `history_segments`，每轮输入原图、所有历史原始指令和本模型生成的历史结果；顺序为 `I0,T1,I1,...,Tn`。保留官方 image-edit 系统提示、固定历史说明和轮次标签。数据加载为单进程、无提前读取，前一轮 PNG 写入后下一轮才读取，路径和像素哈希均存档。

原生数据接口通过最后一个元素推断系统提示的 vision_type；适配器明确设为 image。当前 CFG 正文定位选择最后一个完整 token 匹配，避免重复历史指令被当成本轮指令；历史及标签保留。这些适配不修改模型权重、原始指令或采样参数。

当前源码已随项目默认协议改为bare-v2；不能直接使用当前入口冒充旧输入条件或复现下述历史数值。精确复现须使用历史冻结源码/运行记录。

参数：BF16、30 步、shift 3.5、text CFG 4、无额外 image CFG、CFG 区间 `(0.4,1]`、global/min=0、768 档。目标噪声种子沿用项目按会话/轮次派生的 `noise` 域；原生图像编码使用单独的确定性编码域。原生 VAE 预处理/RNG、token 拼接、位置实现和 attention kernel 与 Omni 不完全相同，此试跑不建立数值等价。

原生 KV cache 用于同一轮去噪，各轮重新编码历史，不含项目 Omni 的跨轮 prefix cache。权重加载检查只允许官方固定位置嵌入和已单独严格加载的 ViT 权重缺项，其余缺项及意外权重均拒绝。

## 运行环境与复现

原生试跑与 Omni `lance` 环境隔离。此次临时 venv 只读复用现有 `dreamzero` 环境的 Python 3.11.15、Torch 2.8.0+cu128、torchvision 0.23.0+cu128、FlashAttention 2.8.3；在 venv 中安装 Transformers 4.49.0、Diffusers 0.29.1、decord、sk-video 和 headless OpenCV。共享环境及 Omni 环境依赖保持原状；临时 venv/源码在试跑后清理。

重新试跑需准备该固定提交的独立源码目录及兼容运行时；源码目录下的 `downloads` 指向 `/home/chs/model/Lance`，用于官方 VAE 路径。执行示例（`NATIVE_PYTHON` 和 `NATIVE_SOURCE` 指向所准备的环境和源码）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-} \
"$NATIVE_PYTHON" scripts/native_speed_probe.py \
  --native-source "$NATIVE_SOURCE" --output outputs/native/new_probe
```

模型常驻，固定 CM/CU 各一个三轮会话，先生成六轮预热，再从原图重新开始生成六轮计时。输出目录必须为空。只运行一次计时批次，用于了解大致速度，不作为稳定性能基准。

`runtime.json` 记录版本及完整调用参数，`system_prompt.json` 记录系统提示，`plan.json` / `native_inputs.json` 记录输入，`turns.json` 记录实际历史图像哈希、当前指令 token 区间、图像块数、输出哈希、耗时和峰值显存；`summary.json` 为汇总。

计时包括数据准备、图像哈希审计、原生编码/去噪/解码/保存及函数内部清理；排除模型加载、预热、外层循环清理和报告写入。GPU 0 执行试跑，GPU 1 持续 burn。此前 Omni 单卡测量时另一卡空闲，因此二者只作配置范围明确的速度参考，不能归因成纯框架加速比。

## 2026-10-04 试跑结果

产物：`outputs/native/probe_20261004/`，双端同名。原生单卡完成六轮预热和六轮正式计时；正式六轮合计 **87.279 秒**，平均 **14.547 秒/轮**，生成阶段峰值 PyTorch 分配显存 **18.473 GiB**。计时批次仅一次，不给出稳定性区间。

作为已有配置的参考，同样两个会话六轮，Omni 单卡 prefix 此前三次中位数 **47.064 秒**，DP2＋prefix 为 **23.806 秒**。在此次实际配置中，原生路径没有显示速度优势；保留 Omni DP2＋prefix 作为后续吞吐方案。两次实验的 GPU 1 占用、跨轮缓存、编码/RNG、实现与清理开销不同，不能把差值当作单一框架因素的加速比。
