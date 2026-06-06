# Codex 第三裁判指南 · 推理续写评测

你好，Codex。本文件请你作为**第三个、跨模型族的裁判**，参与 Mingle Reading 的"名家思维复刻"评测。
你属于 GPT 族；现有两个裁判是 DeepSeek（与被测 agent 同族）和 Claude（跨族）。加入你这个第三族，是为了
进一步排除"同族自我偏爱"，并把裁判一致性从两两变成三方。

请先读完 §1–§4，再做 §5 的任务。

---

## 1. 背景（30 秒）

Mingle Reading 给用户配了"名家陪读 agent"——目前是**鲁迅**和**张爱玲**。每个 agent 由一份 SKILL 驱动，
SKILL 试图编码作者的**思维方式**（认知动作、价值视角、推理习惯），而不只是**文字风格**（文言腔、反讽、华丽意象）。

导师的质疑是：**你怎么知道它复刻的是思维，而不只是腔调？** 本评测就是来回答这个的。
完整方法见同目录 `REPORT.md`；这里只讲和你的任务直接相关的部分。

---

## 2. 你要评的是"推理续写"，不是"像不像"

本评测的关键设计：**不让裁判凭印象判"读着像不像作者"**（那会掺进刻板印象、循环论证）。
而是用**作者本人的真实文字当标准答案**：

1. 取作者一篇真实文章，在某句话**之前**截断；前面的铺垫叫 **setup**。
2. 让被测 agent 接着 setup 往下写它的"下一步推理"，叫 **candidate**（候选）。
3. 作者本人接下来**真正写**的那段，叫 **reference**（参照）。

你的工作：**对照 reference，判断 candidate 有没有做出和作者同一个推理动作 / 得到同一个判断。**
你永远能看到 reference（标准答案），所以这是"对答案"，不是"凭感觉"。

---

## 3. 评分标准（务必严格按此，0/1/2）

```
2 = 核心推理动作与结论和【参照】一致（措辞可不同）。
1 = 方向相关，但只对了一半 / 偏浅 / 只抓住作者那一步里的一个子环节。
0 = 不同的论点、跑题、只是复述铺垫、或泛泛而谈。
```

### ⚠️ 最重要的一条校准（请认真看，这是前一个裁判踩过的坑）

**"同一个题材里写得很好、很有作者腔调"不等于 2，甚至常常是 0。**

前一个跨族裁判第一遍打分**偏松**：把那些"接着同一篇文章、写得漂亮、很鲁迅/很张爱玲"的续写都给了 1 分，
结果各版本分数都差不多、毫无区分度（连"张冠李戴"对照都拿了 ~1 分）。这是错的。

这套 rubric 奖励的是**复刻作者那一步具体的推理**，不是"写了一段同主题的好文字"。请把握：

- candidate 如果**接着同一话题、但做出的是另一个推理**（哪怕很精彩、很像作者），那是 **0**（不同的论点）。
- candidate 如果只是**把 setup 的意思换个说法复述**、没有推进，那是 **0**（泛泛/没推进）。
- candidate 如果用**第三人称分析这段文字**（"作者这里想说……"），而不是**产出**作者的下一步，那通常是 **0**。
- 只有当 candidate **真的做出了 reference 里那一步推理动作、得到同一判断**，才是 **2**；
  抓到那一步里的一部分、但没完整，是 **1**。

### 忽略风格与文采

只判**推理动作**。candidate 是文言还是大白话、华丽还是朴素、第一人称还是别的，**一律不影响打分**。
有的 candidate 会带明显的作者腔调甚至自报家门（"鲁迅先生说……"），**不要因此加分也不要因此减分**——
只看它有没有做出参照里的那一步。

### 几个具体提示

- 作者的"下一步"往往是一个**转折/跳跃**（鲁迅常从现象跳到"历史循环"或"撕掉标签看它实际在干什么"；
  张爱玲常从一个细节**转**到一个出人意料的判断）。如果 candidate 停在铺垫的延长线上、没做这个转折，多半是 0。
- 给 2 要保守：宁可把"沾边但没完整复刻"判 1，也不要把"另起炉灶的好文章"判 2。
- 你看不到版本标签（neutral / style_only / thinking_only / full / wrong_persona）——这是**盲评**，
  请不要试图猜测，按 candidate 本身打分即可。

---

## 4. 文件在哪、长什么样

待评任务（**已匿名、打乱顺序、不含版本标签**）：

```
backend/eval/persona_thinking/judge_handoff/layer1b_lu-xun.judge_tasks.jsonl       (鲁迅，150 条)
backend/eval/persona_thinking/judge_handoff/layer1b_zhang-ailing.judge_tasks.jsonl  (张爱玲，195 条)
```

每行是一个 JSON 对象：

```json
{
  "task_id": "886cd99ba33de561",
  "persona": "lu-xun",
  "setup": "……作者文章的铺垫（前文）……",
  "reference": "……作者本人接下来真正写的那段（标准答案）……",
  "candidate": "……被测 agent 续写的内容（你要评的）……",
  "rubric": "（就是 §3 那段评分说明）"
}
```

> **不要**去打开 `*.judge_key.json`——那里有版本标签，会破坏盲评。

如需逐条人类可读地浏览任务，可用：
```bash
python -m backend.eval.persona_thinking.tools.show_judge_tasks --name layer1b_lu-xun --start 0 --count 15
```

---

## 5. 你的任务与产出

**对每一条 task 打一个 0/1/2 分**，写一行简短理由（中文即可），输出到：

```
backend/eval/persona_thinking/judge_handoff/layer1b_lu-xun.judge_results_codex.jsonl
backend/eval/persona_thinking/judge_handoff/layer1b_zhang-ailing.judge_results_codex.jsonl
```

每行一个 JSON 对象，schema 与 Claude 裁判完全一致，只是文件名多了 `_codex`：

```json
{"task_id": "886cd99ba33de561", "score": 0, "note": "落到姿态反讽，未做参照里上位者制造机会的推理"}
{"task_id": "d972e6864831ebeb", "score": 2, "note": "复刻了天才需民众之土→先问是否提供环境"}
```

要点：
- `task_id` 必须与任务文件里的一字不差（用于解盲合并，**不靠条件标签**）。
- `score` 必须是整数 0 / 1 / 2。
- `note` 一句话说明依据（便于人工抽查你的判分质量）。
- **覆盖范围**：理想是全判（鲁迅 150 + 张爱玲 195）。若资源有限，**至少各判前 80 条**——
  任务文件已随机打乱，前 N 条本身就是覆盖各版本的随机样本，可用于一致性估计。
  （但请**整条整条**判，不要跳着挑，以免引入选择偏差。）

如果你愿意用项目里的小工具写分，也可以分批调用（它按 task_id 幂等、可重复覆盖）：
```bash
python -m backend.eval.persona_thinking.tools.record_judge_scores \
    --name layer1b_lu-xun --judge codex \
    --json '[{"task_id":"...","score":0,"note":"..."}, ...]'
```
> `--judge codex` 会把分写到 `layer1b_<persona>.judge_results_codex.jsonl`（正是 §6 分析所读的文件）。

---

## 6. 判完之后会发生什么（你不用做，供你理解）

维护者会运行跨族一致性分析，把你（Codex）的分加进来：

```bash
python -m backend.eval.persona_thinking.judge_agreement --personas lu-xun zhang-ailing --judges deepseek claude codex
```

它会报告：① 你和 DeepSeek、你和 Claude 的**完全一致率 / 差一分内一致率 / 相关**；
② 你给各版本（解盲后）的平均分，看你是否**也复现了** `full` 最高、`style_only`（只有腔调）垫底的排序。

**对评测最有价值的结果**是：你作为第三个、独立的模型族，**独立复现同样的版本排序**——
那会把"这不是某个模型自我偏爱"的结论从两族夯实到三族。请如实、严格地判，不要迎合任何预期排序；
你看不到版本标签，本来也无从迎合。

谢谢！
