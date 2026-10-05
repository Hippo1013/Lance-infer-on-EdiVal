# ImgEdit 人工评分器

2026-10-04 无标签重跑新增入口：<http://127.0.0.1:8769>，tmux `imgedit-bare-review`。使用 `outputs/imgedit/full_bare_20261004_attention/run/`，全 30 会话/88 轮/118 图；清单与独立评分库在 `outputs/review/imgedit_bare_20261004/`，翻译 `configs/imgedit_review_bare_zh.json`。原 8767 及旧评分保留。启动沿用下方命令，分别指定新 `--run`、`--review`、`--translations`、`--port 8769`；2026-10-05 23:23核对该bare表88/88轮均已明确判定；数值统计与评分记录留在服务器，本表仍可由用户修改。

新页面本机转发（端口空闲时）：

```bash
ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:8768:127.0.0.1:8768 -L 127.0.0.1:8769:127.0.0.1:8769 a800x2
```

## 数据与界面

访问地址：<http://127.0.0.1:8767>。服务直接读取服务器 `outputs/imgedit/full_20261004_attention/run/`，覆盖三类各 10 会话、共 88 轮；118 张图像包含原图与全部实际结果。无模型调用，无 GPU 使用。

- 88 条指令有中文辅助翻译；保留代词、原始歧义及指定文字（如 hello、Hello、111），英文原文可展开。翻译通过原文及会话 fingerprint 绑定本次推理。
- 每轮提供通过、不通过、待定、未评及备注；选择和备注自动保存。评分人选填，当前共用一份评分表，不是多名评委各自独立评分。
- 支持类别/评分状态筛选、中英文搜索、未完成会话跳转、任意原图/历史版本放大对比、原始像素查看。
- 统计按类别、轮次和完整会话汇总。JSON 包含协议、输入/图像身份、各项计数与评分；CSV 包含全部 88 轮、中文/英文指令、判定、备注和评分人。

## 评审口径

协议 `imgedit-human-review-v1`：结合截至当前轮次的原始指令、原图、实际生成历史，判断当前结果是否完成本轮要求并遵守仍然有效的历史要求。撤销、回溯或新指令覆盖旧要求时，按实际指令判断；不自动把前一轮失败传递至后续轮次。歧义或证据不足可标为待定并备注。

逐轮通过率 = 通过 /（通过 + 不通过）。未评与待定单列，不计入分母；分母为零时显示空值。会话全通过率仅统计所有轮次均已明确判定的会话。界面显示已评覆盖率，不能用部分已评样本冒充全量结果。

[官方多轮说明](https://github.com/PKU-YuanGroup/ImgEdit/blob/b79848168744c8db8389086d01fe3def1758aa45/Benchmark/Multiturn/Multiturn_readme.md)采用人工判断，但未公开可直接复现的完整表单及聚合实现。本工具的逐轮记录、待定项及上述汇总是项目补充，不声称复现官方榜单分数。旧8767页面对应有标签v1，存在已知提示封装干扰；当前8769对应bare-v2，按实际图像独立评审，不预填判断。

## 保存与恢复

服务器 `outputs/review/imgedit_human_20261004/` 保存：

- `manifest.json`：30 会话、原始英文、中文翻译、会话身份和 118 张图片 SHA-256。
- `ratings.sqlite3`：人工评分的持久主文件，独立于推理结果。以事务保存，逐轮版本号防止两个页面静默覆盖；冲突时保留编辑内容并提示刷新核对。
- `server.log`：页面服务日志。

页面显示“已保存”才代表写入服务器。断线会显示未保存，阻止站内跳转并在关闭前提示；恢复连接后点“重试保存”。服务重启后旧页面须刷新以重新获取会话令牌。导出前等待当前改动保存。未保存内容只在该浏览器页面内，关闭或强制刷新可能丢失。

本地 SSH 隧道（端口未被现有隧道占用时）：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8767:127.0.0.1:8767 a800x2
```

服务器服务仅监听 `127.0.0.1`，tmux `imgedit-review`。服务器重启后恢复：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
/home/chs/conda/envs/lance/bin/python -B scripts/imgedit_review.py prepare
# 仅在 tmux has-session -t imgedit-review 确认会话不存在时启动。
tmux new-session -d -s imgedit-review \
  'cd /home/chs/exp0_attention/Lance-infer-on-EdiVal && /home/chs/conda/envs/lance/bin/python -B scripts/imgedit_review.py serve > outputs/review/imgedit_human_20261004/server.log 2>&1'
```

停止仅该页面服务：`tmux send-keys -t imgedit-review:0.0 C-c`。不影响原图、推理结果或其他服务。评分库不得替换、清空或提交 Git。

## 验证记录

2026-10-04：4 项针对性测试通过（持久化与并发冲突、缺项分母、身份及字段检查、撤销判定）；118 张图像的 HTTP 字节哈希全部匹配原始文件，合计 108294675 字节。隔离评分表中验证了逐轮选择、中文备注、刷新恢复、类别/轮次统计、中文搜索、版本对比、JSON/CSV 全 88 行导出、无令牌写入拒绝及旧版本写入冲突。宽窗口四列、窄窗口两列且无横向溢出。正式评分表未写入测试判断。

源码：`scripts/imgedit_review.py`、`web/imgedit-review.html`、`configs/imgedit_review_zh.json`；测试：`tests/test_imgedit_review.py`。Python 标准库实现，无新增依赖。
