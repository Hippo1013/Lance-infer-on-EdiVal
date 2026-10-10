# EdiVal 推理接口与输入协议

## 基准结构

EdiVal-Bench 包含 572 个三轮会话，共 1716 条编辑指令。发布数据没有 MICE 的 CM/CU 或 ImgEdit 的三种会话集划分；九种编辑类型属于逐轮任务标签，一个会话可以包含不同类型。内部路径中的 `edival/` 仅用于隔离 benchmark 身份。

依据：[官方论文](https://arxiv.org/html/2509.13399v3)、[固定版本生成代码](https://github.com/TianyuCodings/EdiVal/blob/96d34b00d7ea2dc3de90f2bb01f292f6dec6294a/generate.py)。原始文件与 SHA-256 身份见 `configs/edival_dataset.json`。

2026-10-05 直接检查服务器 `/home/chs/dataset/EdiVal`：

| 项目 | 核对结果 |
| --- | --- |
| CSV | 1716 行；同一 `image_index` 各有 `turns=1/2/3` 的累计前缀 |
| 会话 | 572 组；每组前两行的 `instructions` 与第三行的对应前缀完全一致 |
| ZIP | 606 张 JPEG；其中 572 张被 CSV 引用，34 张未参与正式会话 |
| 原图 | 572 张均成功解码，RGB、512×512；ZIP 全部成员 CRC 通过 |
| 双卡分片 | 每卡 286 个完整会话，858 轮；会话不交叉 |
| CSV 来源 | 与固定版本官方 GitHub CSV 的 SHA-256 完全一致 |

CSV 的 `instructions` 是 Python 列表字面量，使用 `ast.literal_eval` 解析，不是 JSON，也不执行 `eval`。原始自然语言中的引号、标点和措辞保持原样；不能逐行启动 1716 个会话，也不能遍历 ZIP 将额外 34 张图纳入推理。缺轮、重复行、前缀冲突和缺图均在模型加载前拒绝。

它具有跨轮对象依赖。例如会话 1 的第二轮添加鱼、第三轮将鱼的材质变成玻璃；会话 5 的第二轮添加鸟、第三轮将鸟替换为猫。论文的对象池也随编辑更新。很多指令依靠上一轮图像就能理解，对更早文字历史的显式依赖较弱。因此它可以观察连续编辑、内容保持与误差累积，但没有 MICE 式 CM/CU 分类或 ImgEdit 式版本回退专项，也不能由退化直接认定 attention 遗忘。

## 历史输入与推理设置

复用 `lance-history-bare-v2`：第三轮提供 `I0,T1,I1,T2,I2,T3`，其中 I1/I2 都是 Lance 保存后重新读入的实际 RGB 输出。正文仅由原始指令组成，图文间保留现有分隔；没有额外多轮说明、HISTORY/CURRENT EDIT 或 Turn 标签。保留现有官方图像编辑系统提示与模型模板。预览中的 `[IMAGE_n]` 只是图片位置标记，实际模型输入还包含视觉条件与特殊 token。

对象列表、`format_instructions`、`task_type`、背景一致性标志及评测答案不进入模型提示。每张历史图继续使用 ViT/VAE 条件；CFG-negative 仅移除当前指令正文，其余真实历史保留。种子按 `edival/<image_index>`、轮次和用途派生，会话结束清空缓存。

默认 DP2＋prefix，沿用 30 步、shift 3.5、text/image CFG 4/1、区间 `(0.4,1]`、global/min=0、BF16、seed42；支持 single、无缓存、断点恢复和 `--save-attention`。CSV 与整个 ZIP 的 SHA-256/大小加入 EdiVal 运行指纹，恢复时拒绝不同数据身份。`session.json` 同时保存 ZIP 路径、成员名和解码像素哈希。

按用户要求采用 EdiVal 官方 `generate.py` 的 512×512 尺寸：原图、Lance 目标生成、保存结果及历史回填均为 512×512。模型直接以 512 桶生成，不先生成 768 再缩小；MICE/ImgEdit 的默认 768 设置保留。ViT 条件编码仍使用 Lance 的既有设置。旧 768 工程检查仅为历史记录，本次全量和新验收都为 512。

## 官方模式与评测衔接

官方 `multipass` 示例以当前指令编辑上一轮结果；官方 `singlepass` 则每次从原图开始，把截至该轮的指令拼成一个提示。我们使用实际结果递进的完整交错历史，是本项目选择的 Lance 输入协议，不能称为官方 singlepass，也不能声称完全复现官方示例输入条件。

当前输出沿用 `edival/<image_index>/turn_0_input.png`、`turn_n.png/json` 与可选 attention 文件，方便既有历史链审计。官方评测器期望 `multipass/<index>_input_raw.png` 和 `<index>_input_raw_turn_<n>.png`；现有 [官方评分接口](edival-scoring.md) 直接引用原图和各轮PNG，将评分结果组织为官方multipass结构，无须复制或重编码推理图像，不能只改目录名。参考：[固定版本评测代码](https://github.com/TianyuCodings/EdiVal/blob/96d34b00d7ea2dc3de90f2bb01f292f6dec6294a/eval.py)。

EdiVal 的 IF/CC/VQ 与 MICE 的 IF/CC/GA 不是同一套完整评分协议。虽有共享思想和工具，背景一致性标志、逐轮对象池、聚合与 VQ 需要独立核对；本接口不启动或移植能力评分。

## 接口命令

服务器已部署独立运行源码 `runtime/edival_512_20261005/`，24个冻结源码/入口哈希与GPU1的512验收完全一致，终态核对仍全部匹配。本次部署未改动原共享项目，启动时核对其53个源码/入口与部署前一致；其他benchmark可在原项目继续独立开发，EdiVal运行只使用冻结目录。2026-10-05 21:18（北京时间）在 tmux `lance-edival-full` 启动一次全量，于2026-10-06 04:36完成，推理只用GPU1；输出 `outputs/edival/full_512_bare_20261005_attention/`，completion exit_code=0、validation passed。推理任务本身不评分；后续官方全量评分已完成，见 [评分结果](edival-scoring-results.md)。

以下 Python 命令在已配置的 Lance 环境、独立运行目录执行；正式全量包装器自行配置环境并在结束后恢复本次所用空闲 GPU 的 burn。

```bash
# 仅 CPU 数据审计及分片预览，无模型加载或解压。
python -B scripts/edival_data_check.py
python -B -m lance_mice.edival --selection all --dry-run

# 固定会话 0/1，共六轮：单卡 none 对 prefix＋attention 验收。
# 验证原图、完整历史、CFG、零 token 标签、KV 复用与逐像素一致性。
# 按工作区临时文件约定为这次检查提供系统临时输出目录。
edival_check_dir=$(mktemp -d /tmp/umm-edival-check.XXXXXX)
python -B scripts/edival_acceptance.py --gpu 1 --output "$edival_check_dir/acceptance"
# 保存需要交付的验收结论后清理；按工作区规则恢复本次使用卡的 burn。
rm -r -- "$edival_check_dir"

# 新一次 GPU1 全量，输出必须是新目录，验收源码/数据哈希必须一致。
bash scripts/run_edival_job.sh 1 /home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/edival/new_full \
  --acceptance docs/edival-validation-512-20261005.json
```

单卡可覆盖默认配置：`--profile single --gpus 1`；缓存可用 `--cache-mode none`，恢复必须保留原参数并加 `--resume`。输出目录需为新目录，不覆盖已有实验。

## 历史512验证范围

本地与服务器各 27 项相关 CPU 检查通过（9 项 EdiVal、15 项通用协议/历史与恢复、3 项 ImgEdit）。572 张源图解码、CRC、累计前缀和数据身份通过全量审计。双卡 dry-run 只验证分片；本次实际验收和全量均为 GPU1 单卡。

512 验收使用固定会话 0/1，两组各六轮，none 对 prefix＋attention 共 12 次真实生成；六轮结果逐像素一致。检查了实际生成尺寸、源图像素、完整历史/CFG、零 token 标签、后两轮 KV 前缀复用及全部 attention。最大 attention 参考误差 0.00022041798、概率和误差 0.00007402897，峰值 PyTorch 分配 15.782 GiB。24个冻结文件在验收与独立服务器runtime中一致；当前本地评分profile随后已升级v3，不要求用它替换EdiVal冻结版本，部署差异见 [exp0状态](exp0-status.md)；报告为 [512 验收记录](edival-validation-512-20261005.json)。历史 [768 验收](edival-validation-20261005.json) 不作为本次全量门槛。

全量入口 `scripts/full_edival_job.py` 要求这份 512 验收及完全相同的数据/源码身份；每 20 秒原子更新 `status.json`，结束后完整核对 572 个会话、1716 张输出和 1716 份 attention，并写 `validation.json` / `completion.json`。首次四会话十二轮已通过实际输入链、原始图像/指令、尺寸、attention、GPU1 进程环境及冻结源码复核，见服务器任务 `launch_check.json`；这不是全量终态。失败保留所有结果，不自动重采样或择优重试。

2026-10-06全量复核passed：572个原始源图/指令链、1716轮512输出/真实历史、CFG/seed、prefix复用、完整GPU1覆盖、全部attention哈希/分组/概率及24个冻结文件均通过。attention最大参考绝对误差0.00028109550、概率和误差0.00011336803；峰值PyTorch分配16954601472字节。终态时推理进程已退出，GPU1恢复原burn；查看器亦校验全部2288张HTTP图片，独立manifest已冻结。

本次总耗时26305.685秒，约7小时18分；终态检查发现GPU1 burn loop于2026-10-05 21:18:26再次启动，与Lance推理并行。启动后占卡复核漏掉该情况，故耗时不能作为独占GPU性能数据；产物全部通过工程审计，未重跑或择优采样。冻结源码保持原样，下次启动须在加载期和模型就绪后检查真实任务与burn进程，不能仅凭显存暂时为空判断空闲。

查看 `status.json` 不加载模型。带 `--status-only` 的 Python 入口仍需提供 `--acceptance` 参数；直接读取状态 JSON 更方便。attention 使用与前两 benchmark 相同的 `target-group-mass-v1`，保存全部 30 步、36 层、16 head 的概率分组统计，目标 query 空间取均值，不是完整逐像素矩阵。

## 结果查看页面

沿用既有四图布局：全部 572 会话的原图、三轮输出与英文原指令；30 个固定随机会话另有中文辅助译文，seed20261005。页面每 20 秒更新生成数量和 attention 数量，支持已生成/中文抽查筛选、搜索与放大对比。地址 <http://127.0.0.1:8770>，服务器仅监听回环，经 SSH 转发访问；使用方法见 [EdiVal 页面说明](edival-review.md)。

独立查看任务已于2026-10-06 04:36:57（北京时间）逐一校验2288张HTTP图片字节并冻结review manifest，查看validation passed。查看器与译文不修改模型输入或推理产物。
