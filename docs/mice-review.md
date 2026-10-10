# MICE 多轮编辑抽查页面

本页维护2026-10-04两次推理的独立抽查页。bare-v2入口为 <http://127.0.0.1:8768>；2026-10-07六组全量bare/chat对照使用 [8775查看器](sixrun-review.md)，不能用本页旧样本代表新实验。

## 页面身份与抽样

| 版本 | 推理目录 | 查看清单 | 翻译配置 | 服务与端口 |
| --- | --- | --- | --- | --- |
| bare-v2 | `outputs/mice/full_bare_20261004_attention/run` | `outputs/review/mice_30_bare_20261004` | `configs/mice_review_30_bare_zh.json` | mice-bare-review，8768 |
| 历史有标签v1 | `outputs/mice/full_20261004_attention/run` | `outputs/review/mice_20_20261004` | `configs/mice_review_20_zh.json` | mice-review，8765 |

seed20261004，按会话目录排序，用同一 `random.Random` 依次对CM、CU无放回抽样，各组内排序展示，不按质量筛选。bare每组15会话，共30会话90轮120图；v1每组10会话，共20会话60轮80图。两版清单和译文分别绑定推理身份。

## 内容与交互

页面直接读取服务器保存的RGB原图与三轮真实PNG结果，不重新推理、不使用GPU。支持类别筛选、中英指令／编号搜索、方向键切换、原图／上一轮放大比较与原始像素查看；宽屏四列、窄屏两列。

英文保留原始指令，中文仅供浏览，不进入模型输入；CU代词及歧义、指定文字的原语言保留。manifest记录样本及图片SHA256，服务启动重新校验，只开放页面、清单和已抽中图片，其他路径404。抽查不提供正式能力评分，不能由少量会话推断总体得分。

## 访问与服务恢复

端口已占用时先检查既有隧道及页面，勿重复启动。本机转发bare页面：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8768:127.0.0.1:8768 a800_0
```

服务器重启后先确认同名服务及端口未占用，再恢复。以下在0号机项目目录执行，显式传入bare身份，避免脚本默认恢复v1：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
/home/chs/conda/envs/lance/bin/python -B scripts/mice_review.py prepare \
  --run outputs/mice/full_bare_20261004_attention/run \
  --review outputs/review/mice_30_bare_20261004 \
  --translations configs/mice_review_30_bare_zh.json --per-split 15 --port 8768
tmux new-session -d -s mice-bare-review \
  'cd /home/chs/exp0_attention/Lance-infer-on-EdiVal && /home/chs/conda/envs/lance/bin/python -B scripts/mice_review.py serve --run outputs/mice/full_bare_20261004_attention/run --review outputs/review/mice_30_bare_20261004 --translations configs/mice_review_30_bare_zh.json --per-split 15 --port 8768 > outputs/review/mice_30_bare_20261004/server.log 2>&1'
```

停止仅本页：`tmux send-keys -t mice-bare-review:0.0 C-c`。恢复历史v1时将参数、tmux和端口全部按身份表切换，`--per-split 10`；隧道两端均用8765。查看器只监听服务器回环，重连后仍须核查实时服务状态。

## 源码与历史验证

实现：`scripts/mice_review.py`、`web/mice-review.html`，Python标准库，无新增依赖。2026-10-04 v1的三份功能源码双端SHA256一致，20会话指令与元数据匹配，80张HTTP图片哈希通过（69,883,564字节），非白名单404；浏览器筛选、搜索、放大、原图／原始像素及宽窄布局通过。这是v1验证记录，不替代bare或新六组的独立证据。
