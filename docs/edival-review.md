# EdiVal 全量结果查看

地址：<http://127.0.0.1:8770>。2026-10-06 04:36（北京时间）已完成全量推理与页面复核：572会话、1716轮/attention及2288张HTTP图片全部通过，独立manifest已冻结。服务器只监听回环地址，由本机 SSH 转发访问，图片与 attention 留在服务器。

## 数据与交互

页面包含本次完整 572 个三轮会话，四列依次显示原图与三轮真实输出，均为 512×512。推理期间每 20 秒更新完成轮数、会话数和 attention 数量；未生成的图片显示等待状态。支持全部/已生成/中文抽查筛选、编号或指令搜索、左右切换会话，以及上一轮/原图的放大比较。

所有英文指令来自本次固定运行清单，保留原始引号与标点。`configs/edival_review_30_zh.json` 用种子20261005均匀抽取30个会话，提供90条中文辅助译文；其余会话直接显示英文。译文与英文序列及本次 run fingerprint 绑定，不修正原指令矛盾，不进入模型输入。

查看器只读推理产物。本页面不提供评分表或调用评分；该历史运行后续的官方评分已完成，见 [评分结果](edival-scoring-results.md)。attention 与前两 benchmark 相同，为每轮30步×36层×16head的概率分组统计，目标query空间取均值；页面显示文件保存数量，完整有效性由推理复核检查。

## 服务与路径

服务器项目：`/home/chs/exp0_attention/Lance-infer-on-EdiVal`。

| 用途 | 项目内路径或 tmux |
| --- | --- |
| 固定推理源码 | `runtime/edival_512_20261005/`，运行中不修改 |
| 推理任务 | `outputs/edival/full_512_bare_20261005_attention/`，tmux `lance-edival-full` |
| 实际图片/attention | 上述目录的 `run/edival/<image_index>/` |
| 独立查看清单与日志 | `outputs/review/edival_512_bare_20261005/` |
| 查看服务 | tmux `edival-review`，127.0.0.1:8770 |
| 已完成的 HTTP 复核 | 原tmux `edival-review-finish`，终态为查看目录 `validation.json` passed |

查看服务已在服务器启动；需要本机重连时执行：

```bash
ssh -N -L 127.0.0.1:8770:127.0.0.1:8770 -o ExitOnForwardFailure=yes a800_0
```

服务器重启后，在确认8770空闲、现有同名服务未运行后恢复查看器：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal/runtime/edival_512_20261005
/home/chs/conda/envs/lance/bin/python -B scripts/edival_review.py serve \
  --run /home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/edival/full_512_bare_20261005_attention/run \
  --review /home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/review/edival_512_bare_20261005 \
  --translations configs/edival_review_30_zh.json --port 8770
```

上述命令只恢复查看服务，不恢复或重跑推理。

## 完整性证据

推理任务结束后先检查全部572个源图与指令链、1716轮真实历史、512尺寸、CFG/seed、attention哈希/概率及24个冻结源码，成功才写任务 `validation.json` passed 与 `completion.json` completed。任务包装器释放GPU1后恢复对应空闲burn，GPU0与monitor/watchdog不由此任务操作。

`scripts/edival_review_finish.py` 等待这一终态，再检查服务返回的manifest及全部2288张HTTP图片SHA-256，写独立查看目录 `manifest.json` 和 `validation.json`。出现错误时保存失败原因，不将不完整页面冻结为已完成清单。

启动阶段验证了双端三项查看器测试、浏览器筛选/搜索/中文指令/待生成状态/放大比较，以及本机SSH转发后的首批23张HTTP图片字节与尺寸。正式推理首四会话十二轮的数据链与attention复核见任务 `launch_check.json`。这些启动检查不代表全量已完成。
