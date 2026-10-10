# MICE人工待定项评分窗口

入口：<http://127.0.0.1:8772>，服务器tmux `mice-pending-review`。仅CPU，直接读取服务器图像；覆盖Qwen单票v3的31个直接人工项（28标签待定、3回答不完整），28会话、100张截至对应轮次的独立图像。仅展示当前轮及此前历史，不暴露未来结果。

## 评分规则与交互

页面顶部随IF/GA项切换规则。IF判断当前轮的指定编辑是否完成，比较上一轮与当前结果，按显式目标检查；不混入GA或CC。GA_prefix检查第1轮至当前轮完整前缀，所有轮次都成功为1，任意失败为0；CM全局约束、移除豁免及CU指代范围以本项裁判原始提示为准。任一项都可保持待定、添加备注；未评/待定为null。原始英文指令、显式目标、完整裁判提示及首份原回答可查看。

选择1分、0分或仍待定，点击「保存评分」或「保存并下一项」，出现「已保存」才写入服务器。图片可点击并排放大，选择原图/历史版本、切换原始像素。左侧显示已明确评分进度，支持IF/GA/未评/不完整筛选。关闭前未保存提示、站内跳转阻止、失败重试及并发版本冲突保护均保留。刷新后从服务器恢复。

31项是独立指标判断，不是31个完整会话。37轮累计GA依赖不需额外评分；人工确认原始前缀后再计算累计值。该窗口独立保存人工判定，不覆盖或自动回写原自动scores/summary；导出JSON包含来源指纹、源证据校验值、每项human_score及累计依赖，供后续汇总。

## 第一版汇总状态

2026-10-06已保存31/31项明确判定（7通过、24不通过），无待定／未评。独立固定快照与有效分数在`outputs/scoring/mice_first_edition_human_v1/`，汇总见 [第一版结果](mice-first-edition-results.md)。窗口后续修改不会自动更新该快照。

## 持久保存与恢复

服务器项目 `outputs/review/mice_qwen_pending_v3/` 保存manifest、ratings.sqlite3和server.log。评分库与自动证据独立，不清空、不提交Git；manifest绑定31项、原证据文件SHA-256与图像SHA-256。仅回环监听，写入要求页面令牌和条目版本。

```bash
# 本机隧道；若已有端口监听，先检查页面，不重复建立。
ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8772:127.0.0.1:8772 a800_0
```

服务重启（先确认tmux不存在）：

```bash
cd /home/chs/exp0_attention/Lance-infer-on-EdiVal
/home/chs/conda/envs/lance/bin/python -B scripts/mice_pending_review.py prepare
tmux new-session -d -s mice-pending-review \
  'cd /home/chs/exp0_attention/Lance-infer-on-EdiVal && /home/chs/conda/envs/lance/bin/python -B scripts/mice_pending_review.py serve > outputs/review/mice_qwen_pending_v3/server.log 2>&1'
```

## 验证记录

2026-10-06：3项持久化/并发冲突/身份与字段/导出空分数测试通过；隔离临时评分库完成HTTP保存、重新读取、导出、令牌拒绝和旧版本409检查，临时目录已清理。100张图像经本机隧道HTTP逐张SHA-256匹配。浏览器规则、完整历史、并排放大窗口显示正常；正式初始评分表31未评，没有测试分数。脚本及HTML双端哈希一致，原17份评分冻结源码未改动。

实现：`scripts/mice_pending_review.py`、`web/mice-pending-review.html`；针对性测试：`tests/test_mice_pending_review.py`。
