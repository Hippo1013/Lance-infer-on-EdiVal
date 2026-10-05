# MICE 多轮编辑抽查页面

2026-10-04 无标签重跑新增入口：<http://127.0.0.1:8768>，tmux `mice-bare-review`。使用 `outputs/mice/full_bare_20261004_attention/run/`，seed `20261004`、CM/CU 各 15 会话，共 30 会话/90 轮/120 图；清单 `outputs/review/mice_30_bare_20261004/`，翻译 `configs/mice_review_30_bare_zh.json`。原 8765 页面保留。启动沿用下方命令，分别指定新 `--run`、`--review`、`--translations`、`--port 8768 --per-split 15`；页面统计随清单更新。

新页面本机转发（端口空闲时）：

```bash
ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:8768:127.0.0.1:8768 -L 127.0.0.1:8769:127.0.0.1:8769 a800x2
```

服务器浏览页直接读取 `outputs/mice/full_20261004_attention/run/` 的既有图像，不复制数据集、不重新推理、不使用 GPU。页面通过 SSH 转发访问，服务只监听服务器 `127.0.0.1:8765`。浏览器按需接收图片，不在本地保存一份数据集。

## 抽样与内容

- 对 CM、CU 各 360 个会话，按目录名排序；同一个 `random.Random(20261004)` 依次对 CM、CU 无放回抽取 10 个，展示时在各类别内排序。没有根据编辑质量筛选。
- 20 个会话共 80 张图片：保存的 `turn_0_input.png` 和 `turn_1.png` 至 `turn_3.png`。源图是推理时保存的 RGB 原图，不是预处理后的裁剪张量。
- 每轮显示原始英文及中文翻译；中文只供浏览，不回写推理输入。CU 的代词和原始歧义保留；指定添加的文字保持原语言。翻译文件以会话 fingerprint 绑定原版本。
- 支持类别筛选、中英指令/编号搜索、左右方向键切换、原图/上一轮对照、原始像素尺寸查看。宽窗口显示四列，窄窗口自动两列。
- 抽样清单及 80 张图片的 SHA-256 在 `outputs/review/mice_20_20261004/manifest.json`；服务启动会再次检查图像哈希。服务只开放页面、清单和已抽中图片，其他路径返回 404。

## 访问方式

本地 SSH 隧道存活时，在浏览器打开 <http://127.0.0.1:8765>。若本地重启或隧道断开，在本地终端重新执行以下命令并保持终端开启：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8765:127.0.0.1:8765 a800x2
```

若提示本地端口已占用，先直接打开页面确认既有隧道，勿重复启动。也可把 `-L` 的第一个 `8765` 改为其他空闲本地端口，浏览器访问相应端口。SSH 连接需保持服务器网络可达。

服务器启动命令（仅在 `tmux has-session -t mice-review` 确认不存在时启动）：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
/home/chs/conda/envs/lance/bin/python scripts/mice_review.py prepare
tmux new-session -d -s mice-review \
  'cd /home/chs/exp0_attention/Lance-infer-on-EdiVal && /home/chs/conda/envs/lance/bin/python scripts/mice_review.py serve > outputs/review/mice_20_20261004/server.log 2>&1'
```

停止页面服务：`tmux send-keys -t mice-review:0.0 C-c`。不会影响推理结果或其他 tmux 会话。该服务在服务器重启后需要手动启动。

## 源码与验证

- 服务与抽样：`scripts/mice_review.py`（Python 标准库，无新增依赖）。
- 浏览页面：`web/mice-review.html`。
- 固定 20 会话中文翻译：`configs/mice_review_20_zh.json`。
- 2026-10-04：双端三份源码 SHA-256 一致；20 会话的轮次元数据与指令匹配，全部 80 张图片经 HTTP 读取校验通过（69,883,564 字节）；非白名单路径返回 404。浏览器验证类别筛选、中文搜索、轮次放大对比、原图切换、原始像素尺寸及四列/窄屏布局。

此页面用于定性抽查，不提供正式能力评分；不能据这 20 个样本推断总体得分。
