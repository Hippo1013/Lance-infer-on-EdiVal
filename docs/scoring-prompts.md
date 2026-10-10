# MICE 裁判提示词

下文完整基础正文属于2026-10-06单票v3与历史双裁判profile；2026-10-07六组实验已采用文末的最小输出约束修改。历史原正文与新运行分别绑定冻结源码，不将旧正文当作新prompt身份。Qwen3.6-27B与Gemma-4-31B-it没有模型专属评分标准；不同profile只决定模型、票数和采样身份。六组实验仅Qwen3.6-27B一票，不自动调用Gemma。

请求由一条 user 消息组成：先放对应图像，再放下面的文字；程序没有另加 system 消息。花括号字段由当前样本的标注/指令填入。历史Qwen单票v3为temperature=0.6、vote=1/seed42、context16384、max_tokens=2048、enable_thinking=False；历史双裁判v2为每位裁判2次、seed42/43。具体身份以manifest为准，见 [单票协议](scoring-qwen-once.md)。

来源：Edit-R2 固定提交 `26b55829246e1a67fc3c8d522324fee3d49cd954` 的 `rewards/reward_server/edival_reward_server.py` 与 `flow_grpo/edival_client.py`。本文从已有本地模板整理，不执行模型调用或重新验证。

## IF 提示词

移除、位置、数量三类使用检测规则，不向两位裁判发送评分提示词。其余类型如下；CC 也不调用这两位裁判。

### 对象添加

图片顺序：上一轮图、当前图。

```text
The first image is the original, and the second image reflects the changes made according to the editing instruction in subject addition. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Please respond with "yes" or "no.
```

### 对象替换

图片顺序：上一轮图、当前图。

```text
The first image is the original, and the second image reflects the changes made according to the editing instruction in subject replacement. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Please respond with "yes" or "no.
```

### 颜色修改

图片顺序：上一轮图、当前图。

```text
Look at the object in the image. Is the {object_name} {new_color}? Please answer only 'YES' or 'NO'.
```

### 材质修改

图片顺序：上一轮图、当前图。

```text
Is it possible that the {object_name} is made of {new_material}? Please answer only 'YES' or 'NO'.
```

### 文字识别

图片顺序：当前图。

```text
What text do you see in this image? Output only the text content, nothing else.
```

### 背景修改

图片顺序：当前图。

```text
Look at the background of this image. Does the background show [{background}]? Please answer only 'YES' or 'NO'.
```

对象添加/替换的 `{instruction}` 填入当前 `formatted_instruction`。文字类提示词只要求 OCR，程序再按规则比对目标文字；检测、位置和区域检查在评分程序中完成。正文的引号和标点按源码保留。

## GA 内容记忆提示词

用于 CM。图片按原图、第一轮结果、……、当前轮结果排列；指令也仅包含截至当前轮的前缀。

```text
你是一个专门评估多轮图像编辑任务的专家。你将获得一个由N张图像和N-1条编辑指令组成的序列。

**任务定义：**
该任务包含N-1个连续的编辑轮次。
- **第 i 轮**（i 从1到 N-1）：
  - **输入：** 图像 i
  - **指令：** 指令 i
  - **输出：** 图像 i+1

你的目标是从第1轮开始，按顺序评估每一轮编辑。

**每轮的评估标准：**
对于每一轮 i，根据指令 i 比较图像 i（输入）和图像 i+1（输出）。只有当一轮**同时满足**以下所有标准时，才算成功（"yes"）：
1. **指令遵循：** 图像 i+1 成功反映了指令 i 所要求的更改。
2. **全局约束遵循：** 如果第一个指令设定了影响整个会话的全局约束（例如，"后续编辑中添加或修改的对象必须是黄色"等），那么图像 i+1 必须遵循这个约束。具体规则如下：
     - **作用范围：** 全局约束的遵循仅需体现在当前轮编辑所涉及的物体上。例如当指令为"在桌子上添加一个杯子"且全局约束为"之后的编辑内容都应该是玻璃材质"时，那么成功的图像i+1上应该只有新添加的杯子是玻璃材质，而不能把桌子或其他物体也变成玻璃材质。
     - **移除类指令豁免：** 如果当前轮的指令是移除/删除某个物体（例如"remove the cup"、"去掉背景中的树"等），则该轮无需考虑全局约束——无论编辑结果如何，都视为全局约束被成功遵循。因为移除操作不涉及添加或修改物体，全局约束自然不适用。
     - **禁止整体色调替代：** 全局约束要求的是对编辑涉及的具体物体施加约束，而不是对图片整体进行色调变换。如果编辑后的图片只是将整张图片的色调转变为全局约束所规定的颜色（例如全局约束要求"黄色"，而图片整体被加上了黄色滤镜），但编辑涉及的具体物体并未真正体现该约束，则应判定为全局约束未被成功遵循。

**执行与输出逻辑：**
逐一评估各轮（第1轮，第2轮，...）。

- **如果第 i 轮成功（"yes"）：**
  输出：
  <answer_turn_i> yes </answer_turn_i>
  <reason_turn_i> 简要解释成功的原因。 </reason_turn_i>
  然后继续评估第 i+1 轮（如果存在）。

- **如果第 i 轮失败（"no"）：**
  输出：
  <answer_turn_i> no </answer_turn_i>
  <reason_turn_i> 对失败原因的解释（例如，"未能添加对象" 或 "成功添加了对象B，但意外删除了对象A"）。 </reason_turn_i>
  **立即停止。** 不要评估任何后续轮次。
  输出： <answer_final> no </answer_final>

- **如果所有轮次（1 到 N-1）都成功：**
  在评估完最后一轮后，输出： <answer_final> yes </answer_final>

**输入数据：**
**图像序列：**
- 图像 1: 初始图像
- 图像 2: 第1轮的结果
...
- 图像 N: 第N-1轮的结果

**指令：**
{instructions_formatted}

**回复格式：**
请严格按照上述类似XML的标签提供你的评估。不要在标签之外包含任何对话性文本。将标签中的 'i' 替换为实际的轮次编号（例如，<answer_turn_1>, <answer_turn_2>）。

```

`{instructions_formatted}` 的填充格式：

```text
第1轮：{第一轮原始指令}
第2轮：{第二轮原始指令}
第3轮：{第三轮原始指令}
```

第 1/2 轮评分只填已有轮次，不传未来指令。

## GA 内容理解提示词

用于 CU。图片顺序同上；每轮同时提供模型实际收到的原始指令和数据集的显式参照指令。

```text
你是一个专门评估多轮图像编辑任务中"内容理解（Content Understanding）"能力的专家。你将获得一个由N张图像和N-1条编辑指令组成的序列。

**任务背景：**
在多轮图像编辑的真实场景中，用户在建立对象上下文后，会自然地从使用完整对象名称过渡到使用代词（如"it"、"them"、"there"、"its"等）来指代前轮编辑过的对象。这要求编辑模型能够准确地进行"代词消解"——即理解代词指代的具体对象，并对该对象执行正确的编辑操作。

**任务定义：**
该任务包含N-1个连续的编辑轮次。
- **第 i 轮**（i 从1到 N-1）：
  - **输入：** 图像 i
  - **模型接收的指令：** 指令 i（可能包含代词，如"it"、"them"、"there"、"its"）
  - **显式参照指令：** 格式化指令 i（用方括号标明了代词所指代的具体对象，作为评估的客观参照）
  - **输出：** 图像 i+1

你的目标是从第1轮开始，按顺序评估每一轮编辑。

**每轮的评估标准：**
对于每一轮 i，根据指令 i 比较图像 i（输入）和图像 i+1（输出）。只有当一轮**同时满足**以下所有标准时，才算成功（"yes"）：

1. **代词消解正确（Content Understanding）：** 如果指令 i 中包含代词（如"it"、"them"、"there"、"its"等），图像 i+1 中被编辑的对象必须与格式化指令 i 中方括号内标注的对象一致。也就是说，模型必须正确理解代词指代的是哪个具体对象，并对该对象（而非其他对象）执行了编辑操作。具体规则如下：
   - **"it"/"them" 指代：** 代词指代前轮操作过的同一对象。例如，如果指令是"Remove it"，格式化指令是"Remove [red car]"，那么图像 i+1 中被移除的应该是红色汽车，而非其他对象。
   - **"its" 指代：** 所有格代词指代前轮操作过的对象的属性。例如，如果指令是"Change its color to blue"，格式化指令是"Change the color of [wooden brown door] to [blue]"，那么图像 i+1 中颜色变蓝的应该是木质棕色门。
   - **"there" 指代：** 空间代词指代前轮中某对象被移除后的位置。例如，如果指令是"Add a bench there"，格式化指令是"Add [bench] on the [left] of [flower bed]"，那么图像 i+1 中应该在之前花坛所在的位置添加了长椅。
   - **第1轮通常不含代词**（因为没有前文上下文），此时仅评估下述"指令遵循"标准。

2. **指令遵循（Instruction Following）：** 图像 i+1 成功反映了指令 i 所要求的更改——包括编辑类型正确（添加/删除/替换/颜色修改等）以及属性值正确（如颜色确实变为指定颜色）。

**执行与输出逻辑：**
逐一评估各轮（第1轮，第2轮，...）。

- **如果第 i 轮成功（"yes"）：**
  输出：
  <answer_turn_i> yes </answer_turn_i>
  <reason_turn_i> 简要解释成功的原因。 </reason_turn_i>
  然后继续评估第 i+1 轮（如果存在）。

- **如果第 i 轮失败（"no"）：**
  输出：
  <answer_turn_i> no </answer_turn_i>
  <reason_turn_i> 对失败原因的详细解释。请明确指出是代词消解错误还是指令遵循错误，例如：
    - 代词消解错误："代词'it'应指代[red car]，但模型错误地移除了蓝色卡车"
    - 指令遵循错误："正确识别了目标对象[wooden door]，但颜色未能成功修改为蓝色"
    - 两者皆错："代词'its'应指代[silver helmet]的属性，但模型修改了其他对象的颜色，且颜色也不正确" </reason_turn_i>
  **立即停止。** 不要评估任何后续轮次。
  输出： <answer_final> no </answer_final>

- **如果所有轮次（1 到 N-1）都成功：**
  在评估完最后一轮后，输出： <answer_final> yes </answer_final>

**输入数据：**
**图像序列：**
- 图像 1: 初始图像
- 图像 2: 第1轮的结果
...
- 图像 N: 第N-1轮的结果

**指令：**
{instructions_formatted}

**回复格式：**
请严格按照上述类似XML的标签提供你的评估。不要在标签之外包含任何对话性文本。将标签中的 'i' 替换为实际的轮次编号（例如，<answer_turn_1>, <answer_turn_2>）。

```

`{instructions_formatted}` 的填充格式：

```text
第1轮：
  - 模型接收的指令：{instruction[0]}
  - 显式参照指令：{formatted_instruction[0]}
第2轮：
  - 模型接收的指令：{instruction[1]}
  - 显式参照指令：{formatted_instruction[1]}
```

## 请求示例

以第一轮颜色修改为例，历史基础提示词的单票请求如下；历史Gemma及第二票使用相同正文：

```text
user:
  image 1: 原图
  image 2: 第一轮编辑结果
  text: Is the red car now blue? Answer YES or NO only.
```

上述颜色例子仅说明字段填充，不是一次新评分。完整 GA 正文均在上方列出。

## 六组实验采用的输出约束

2026-10-07 bare／chat两组MICE评分采用同一最小修改版。评分问题、检测／文字比对、CM／CU标准、XML解析器、图像顺序和单票采样保持原协议；没有采用JSON、first_failure、新属性优先级或答案示例。以下IF正文直接对应开发源码与六组冻结版。

### 对象添加输出约束

```text
The first image is the original, and the second image reflects the changes made according to the editing instruction in subject addition. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Return exactly one lowercase word: yes or no. Do not output explanations, punctuation, Markdown, or any other text.
```

### 对象替换输出约束

```text
The first image is the original, and the second image reflects the changes made according to the editing instruction in subject replacement. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Return exactly one lowercase word: yes or no. Do not output explanations, punctuation, Markdown, or any other text.
```

### 颜色修改输出约束

```text
Look at the object in the image. Is the {object_name} {new_color}? Return exactly one lowercase word: yes or no. Do not output explanations, punctuation, Markdown, or any other text.
```

### 材质修改输出约束

```text
Is it possible that the {object_name} is made of {new_material}? Return exactly one lowercase word: yes or no. Do not output explanations, punctuation, Markdown, or any other text.
```

### 文字识别输出约束

```text
What text do you see in this image? Output only the text content, nothing else. Visit each distinct visible text region once, in top-to-bottom, left-to-right order. Do not transcribe the same region repeatedly. If identical text appears in different visible regions, retain those actual occurrences. After all visible text regions have been transcribed, stop.
```

### 背景修改输出约束

```text
Look at the background of this image. Does the background show [{background}]? Return exactly one lowercase word: yes or no. Do not output explanations, punctuation, Markdown, or any other text.
```

### GA补充输出契约

CM与CU均在原完整正文末尾追加下段，不替换评估标准：

```text
补充输出要求：
1. 在输出某轮判定前完成该轮检查，不输出暂定判定。
2. 每个被评估轮次的answer_turn_i和reason_turn_i各输出一次；不重复编号，不补写修正版本。
3. 每轮理由仅用一句简短说明，与该轮最终判定一致。
4. 首个no轮次后停止，answer_final必须为no；只有全部已提供轮次均为yes，answer_final才能为yes。
5. 开始标签与结束标签必须正确配对。
6. 只输出最终标签结果，不输出推演、重读或自我修正过程。
```

新实验保留16个首次回答待审组件；原始回答不重采样，详见 [六组结果](sixrun-results-20261007.md)。这次全量完成证明不中断策略和产物覆盖，不单独证明提示词提高裁判准确性。
