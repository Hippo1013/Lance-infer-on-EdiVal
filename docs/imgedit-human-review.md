# ImgEdit 人工评分器

bare-v2人工表入口为 <http://127.0.0.1:8769>，覆盖2026-10-04的30会话88轮。2026-10-05 23:23核对88/88轮均有明确判定；该表可由用户继续修改，历史快照不表示实时评分。2026-10-07六组结果使用 [8775只读对照页](sixrun-review.md)，本表不为新结果评分。

## 页面身份

| 版本 | 推理目录 | 清单与独立评分库 | 翻译配置 | 服务与端口 |
| --- | --- | --- | --- | --- |
| bare-v2 | `outputs/imgedit/full_bare_20261004_attention/run` | `outputs/review/imgedit_bare_20261004` | `configs/imgedit_review_bare_zh.json` | imgedit-bare-review，8769 |
| 历史有标签v1 | `outputs/imgedit/full_20261004_attention/run` | `outputs/review/imgedit_human_20261004` | `configs/imgedit_review_zh.json` | imgedit-review，8767 |

两版推理和评分库独立，禁止混用或用新准备覆盖旧库。

## 数据与界面

服务按上表直接读取对应服务器推理目录，覆盖三类各10会话、共88轮；118 张图像包含原图与全部实际结果。无模型调用，无 GPU 使用。

- 88 条指令有中文辅助翻译；保留代词、原始歧义及指定文字（如 hello、Hello、111），英文原文可展开。翻译通过原文及会话 fingerprint 绑定本次推理。
- 每轮提供通过、不通过、待定、未评及备注；选择和备注自动保存。评分人选填，当前共用一份评分表，不是多名评委各自独立评分。
- 支持类别/评分状态筛选、中英文搜索、未完成会话跳转、任意原图/历史版本放大对比、原始像素查看。
- 统计按类别、轮次和完整会话汇总。JSON 包含协议、输入/图像身份、各项计数与评分；CSV 包含全部 88 轮、中文/英文指令、判定、备注和评分人。

## 评审口径

协议 `imgedit-human-review-v1`：结合截至当前轮次的原始指令、原图、实际生成历史，判断当前结果是否完成本轮要求并遵守仍然有效的历史要求。撤销、回溯或新指令覆盖旧要求时，按实际指令判断；不自动把前一轮失败传递至后续轮次。歧义或证据不足可标为待定并备注。

逐轮通过率 = 通过 /（通过 + 不通过）。未评与待定单列，不计入分母；分母为零时显示空值。会话全通过率仅统计所有轮次均已明确判定的会话。界面显示已评覆盖率，不能用部分已评样本冒充全量结果。

[官方多轮说明](https://github.com/PKU-YuanGroup/ImgEdit/blob/b79848168744c8db8389086d01fe3def1758aa45/Benchmark/Multiturn/Multiturn_readme.md)采用人工判断，但未公开可直接复现的完整表单及聚合实现。本工具的逐轮记录、待定项及上述汇总是项目补充，不声称复现官方榜单分数。旧8767页面对应有标签v1，存在已知提示封装干扰；当前8769对应bare-v2，按实际图像独立评审，不预填判断。

## 保存与恢复

各版查看目录分别保存：

- `manifest.json`：30 会话、原始英文、中文翻译、会话身份和 118 张图片 SHA-256。
- `ratings.sqlite3`：人工评分的持久主文件，独立于推理结果。以事务保存，逐轮版本号防止两个页面静默覆盖；冲突时保留编辑内容并提示刷新核对。
- `server.log`：页面服务日志。

页面显示“已保存”才代表写入服务器。断线会显示未保存，阻止站内跳转并在关闭前提示；恢复连接后点“重试保存”。服务重启后旧页面须刷新以重新获取会话令牌。导出前等待当前改动保存。未保存内容只在该浏览器页面内，关闭或强制刷新可能丢失。

本地 SSH 隧道（端口未被现有隧道占用时）：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8769:127.0.0.1:8769 a800_0
```

服务只监听服务器回环。服务器重启后先确认同名服务与端口未占用，再在0号机恢复；显式传入bare身份，避免脚本默认恢复v1：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
/home/chs/conda/envs/lance/bin/python -B scripts/imgedit_review.py prepare \
  --run outputs/imgedit/full_bare_20261004_attention/run \
  --review outputs/review/imgedit_bare_20261004 \
  --translations configs/imgedit_review_bare_zh.json --port 8769
tmux new-session -d -s imgedit-bare-review \
  'cd /home/chs/exp0_attention/Lance-infer-on-EdiVal && /home/chs/conda/envs/lance/bin/python -B scripts/imgedit_review.py serve --run outputs/imgedit/full_bare_20261004_attention/run --review outputs/review/imgedit_bare_20261004 --translations configs/imgedit_review_bare_zh.json --port 8769 > outputs/review/imgedit_bare_20261004/server.log 2>&1'
```

停止仅该页：`tmux send-keys -t imgedit-bare-review:0.0 C-c`。恢复历史v1时，按身份表切换全部参数、tmux、日志路径及隧道两端端口。评分库不得替换、清空或提交Git。

## 验证记录

2026-10-04历史v1：4项针对性测试通过（持久化与并发冲突、缺项分母、身份及字段检查、撤销判定）；118 张图像的 HTTP 字节哈希全部匹配原始文件，合计 108294675 字节。隔离评分表中验证了逐轮选择、中文备注、刷新恢复、类别/轮次统计、中文搜索、版本对比、JSON/CSV 全 88 行导出、无令牌写入拒绝及旧版本写入冲突。宽窗口四列、窄窗口两列且无横向溢出。正式评分表未写入测试判断。

源码：`scripts/imgedit_review.py`、`web/imgedit-review.html`、`configs/imgedit_review_zh.json`；测试：`tests/test_imgedit_review.py`。Python 标准库实现，无新增依赖。
