# MICE评分提示词候选方案

## 方案状态

2026-10-07用户确认采用最小修改版，已部署至六组冻结runtime并完成MICE bare/chat两组全量单票评分；实际正文见 [裁判提示词](scoring-prompts.md#六组实验采用的输出约束)。没有针对相同图像、相同judge随机身份单独开展受控提示词对照。GA保留原CM/CU标准、XML标签和解析器，仅追加文末输出契约，不添加答案示例；IF二值问题统一为单个小写yes/no，OCR仅增加逐区域转录一次与完成后停止。未采用JSON/first_failure、CM语义澄清或NO_TEXT接口。本次MICE仍仅Qwen3.6-27B单票，异常回答沿用v3保留首份证据、不重采样并继续执行的策略。旧冻结源码与历史结果保持独立，新运行须另建manifest和输出目录。下列内容保留为此前候选设计，不能整体当作已采用方案。

2026-10-06 CPU证据调查产生的候选稿，未调用GPU、未验证模型改进效果、未部署。用于用户审阅及未来独立实验，不用于重生成或替换full_qwen_once_pending_v3的已有票。原始标准与原始正文见scoring-prompts.md；原因分析见scoring-prompt-failure-analysis.md。

## GA输出约束候选

第一组实验仅在当前CM/CU完整提示词末尾追加以下内容，保留图像、完整当前前缀、评分标准、采样设置和解析器。这里的长度要求是提示词软约束，不是减少max_tokens；enable_thinking仍为False。

```text
输出前请完成当前轮的检查，再写该轮唯一的最终判定。不要先写暂定yes，再在解释中改判。

输出契约：
1. 每个被评估轮次恰好输出一次answer_turn_i和一次reason_turn_i，编号从1连续递增，不允许重复编号或补写修正标签。
2. 每轮reason只用一句简短说明，说明最终判定的依据；不要输出猜测分支、反复重读、推演草稿或“等等／修正／重新评估”。
3. 只要本轮有任一必需条件不满足，该轮的answer必须为no。不要因其他条件满足而输出yes。
4. 在首个no轮次停止；不得评估或输出后续轮次。answer_final必须为no。
5. 只有已提供的全部轮次都为yes，answer_final才能为yes。
6. 每个标签的结束标签必须与开始标签完全一致；特别是answer_turn_i不能用reason_turn_i结束。
7. 输出前检查轮次标签、理由与answer_final是否一致；仅输出检查后的一个版本，不要输出检查过程。

失败格式示例（仅示范格式，不代表任何样本的答案）：
<answer_turn_1>yes</answer_turn_1>
<reason_turn_1>该轮所有必需条件均满足。</reason_turn_1>
<answer_turn_2>no</answer_turn_2>
<reason_turn_2>所要求的数量未达到。</reason_turn_2>
<answer_final>no</answer_final>
```

格式示例有答案锚定风险；正式对照须另有“无示例、仅契约”的候选组，不凭当前31项通过率择优。对于只有1轮的输入，不追加2轮格式示例，改为按该请求轮数生成适用示例，或者统一不使用示例。

## CM全局约束澄清候选

第二组实验在输出约束候选基础上追加以下释义。它旨在落实现有正文中“新添加杯子应为玻璃”等例子的必要属性要求，但释义本身可能改变模型的实际判定分布，因此必须作为独立候选比较。不能只检查是否可以解析。

```text
全局约束检查包含两个独立必要条件：
A. 当前添加或修改的目标对象必须体现全局约束所要求的颜色或材质。
B. 不得把该约束扩散到未参与编辑的其他对象、背景或区域。
满足B但不满足A，仍是不通过；“only/exclusively”不表示目标对象可以不具备要求的属性。
移除轮次仅免于全局颜色／材质检查，仍必须成功移除正确对象。
当第一轮指令写明“With this rule active”或“From now on”时，该约束在第一轮的编辑中已经生效；不能因为subsequent一词而自行豁免该轮。
仅按指令和图像证据判断。不要为了避免早期失败、让任务看起来合理或推测数据集意图而放宽规则。
```

真实的属性冲突、空目标文字、对象描述和编辑目标的冲突不通过增加此段自动消解。需要先明确研究协议的优先级与标注政策；未经确认，不新增“局部指令优先”或“全局约束优先”的通用规则。

## IF二值输出候选

对象添加/替换的末句目前包含未闭合的引号。保留原问题与formatted_instruction，仅将输出要求改为：

```text
Return exactly one lowercase word: yes or no. Do not output explanations, punctuation, Markdown, or any other text.
```

该候选不把IF改为GA，也不把单个VLM回答当作完整IF最终分数；原检测／位置等后处理仍须保留。

## OCR输出候选

仍然识别当前图中的可读文字，再走原有文字比对分支；不向裁判泄漏目标答案，不改为“是否包含目标词”的二值问题。

```text
What readable text do you see in this image? Transcribe only the visible text content, nothing else. Visit each visible text region once in top-to-bottom, left-to-right order, then stop. Do not repeatedly transcribe the same region. If identical text appears in distinct visible regions, retain those actual occurrences. Do not invent unreadable text or continue after all visible regions have been transcribed. If there is no readable text, return an empty string.
```

最后一句“空字符串”与当前适配器把空响应作为response_incomplete的契约冲突。因此本稿不可直接部署：实验版须增加显式无文字状态并与“模型未回答”区分，或先删去最后一句，单独验证非空文字样本的重复退化。不能把空输出自动计0，也不能通过简单字符串去重丢掉真实重复文字。

## 输出协议替代方案

作为后续独立协议，可用唯一的first_failure整数（0表示全部通过，1..T表示第一失败轮）与短理由表示GA；程序由该字段推导最终二值结果，避免同时生成多轮判定和冗余final字段。采用结构约束只约束格式，不保证视觉或语义正确。必须建立新manifest、提示词身份、解析器与验证，保留原协议及本轮人工结果；不作为当前修复。

## 对照验证方案

31/31项人工评分已完成，已固定快照及指纹见 [第一版结果](mice-first-edition-results.md)。后续获准对照时，以该版快照明确来源。以31项原失败证据作为失败诊断集，另固定按CM/CU、轮次、任务类型分层的原成功对照集，包含上述已通过格式但语义有争议的案例。当前31项被人工窗口展示过原模型回答，人工结果应称为复核标签，不能当作盲评真值。

GPU试验必须先取得用户同意。运行前明确样本清单、候选数、GPU0使用时长与调用预算，在新diagnostics目录单独保存每次原回答、提示词、图像身份、采样参数及finish_reason。当前旧票仅作基线，人工答案不进入提示词；未来版本的试验回答不回写当前评分。

分别报告格式成功率、截断／重复率、与人工复核的一致率、合法基线样本的翻转率及人工仍待定数量。GA按first-failure和最终标签比较，不能仅凭最终no一致认定全程判断正确。提示词输出约束与语义澄清分组比较，不同时修改temperature、长度上限或模型。改善结论需同时考虑失败集和成功对照；不重复采样挑选最好的一次，不据单个案例宣布整体准确率提高。
