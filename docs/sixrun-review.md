# 六组推理结果对照查看器

入口：<http://127.0.0.1:8775/?bench=edival>，按 EdiVal → MICE → ImgEdit 浏览。每个 bench 展示最新 sixrun_20261007 的 bare-v2／chat-v1 完整会话，支持类别筛选、会话／英文指令搜索、左右键切换与点击图片放大。英文原指令为实验依据；本页只读，不混用旧版人工评分，不提供 attention 可视化。

| Bench | 每版会话 | 每版轮次 | 入口 |
| --- | ---: | ---: | --- |
| EdiVal | 572 | 1716 | <http://127.0.0.1:8775/?bench=edival> |
| MICE | 720 | 2160 | <http://127.0.0.1:8775/?bench=mice> |
| ImgEdit | 30 | 88 | <http://127.0.0.1:8775/?bench=imgedit> |

两行输出分别来自该协议独立运行的真实历史，后续轮次使用该行的此前输出。展开“实验来源与真实输入记录”可查看 run／session fingerprint、PNG SHA256、逐轮 metadata 与 protocol。文件哈希与 RGB 像素哈希分开：生成历史的 input_hashes／output_hash 使用 Lance 的 RGB 像素哈希。

## 运行位置与访问方式

本地持久源码为 `scripts/sixrun_review.py`、`web/sixrun-review.html`；服务器部署在 `a800_0:/home/chs/exp0_attention/Lance-infer-on-EdiVal/` 同名位置。数据只读自 `outputs/sixrun_20261007/experiment/inference/`，图片与 metadata 不复制到本地。日志位于服务器 `outputs/review/sixrun_20261007/server.log`，tmux 为 `sixrun-review`，回环监听 8775。服务仅使用 CPU 与 Pillow，不加载推理／裁判模型。

启动时检查六组 completion／validation、plan/run 身份、完整会话数量、两版选择与原图一致、每轮指令、协议、图片像素哈希和完整历史链，并计算所有 PNG 文件哈希；读取图片时再次核对文件哈希。正式实验产物保持只读。

全量 PNG 核对使用4个CPU线程。部分源图带有较大的ICC配置块，Pillow的 `MAX_TEXT_CHUNK` 沿用已有MICE验收的4 MB上限，不改写或重新编码原图。

2026-10-07部署验收：六组2644会话7928输出轮、10572张原图／输出PNG引用全量核对通过；三个bench首尾会话的两版全部图片HTTP HEAD通过，未知图像路径返回404。浏览器已验证三bench切换、每版图片实际加载与各bench会话覆盖。完整核对摘要及服务日志保存在服务器的上述review目录。

本机访问须保留 SSH 转发；先核对实例和已有服务／端口，不重复启动：

```bash
ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8775:127.0.0.1:8775 a800_0
```

服务器服务退出后的恢复命令（先确认同名 tmux 和端口空闲）：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
tmux new-session -d -s sixrun-review \
  'cd /home/chs/exp0_attention/Lance-infer-on-EdiVal && /home/chs/conda/envs/lance/bin/python -B scripts/sixrun_review.py --base /home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/sixrun_20261007/experiment --port 8775 > outputs/review/sixrun_20261007/server.log 2>&1'
```
