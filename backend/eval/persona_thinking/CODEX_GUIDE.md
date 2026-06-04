# Guide for Codex: Persona Thinking Evaluation

This document brings you up to speed on the **Mingle Reading** persona-thinking evaluation project, and tells you exactly what task we need your help with.

---

## 1. What is Mingle Reading?

Mingle Reading is an AI-powered reading companion that pairs users with "celebrity reader agents" — currently **Lu Xun** (鲁迅) and **Zhang Ailing** (张爱玲), two of the most influential Chinese authors of the 20th century. While the user reads a book, the agent comments on the text from that author's perspective.

Each agent is driven by a `SKILL.md` file that encodes the author's **thinking style**: their cognitive moves, value lens, framing habits, and reasoning patterns — not just their surface prose style.

---

## 2. The Research Question

**The claim we need to prove**: the agent captures the author's *internal thinking style*, not just their surface writing style.

This matters because:
- A model can imitate Lu Xun's classical Chinese diction and ironic tone without ever employing his signature moves (e.g., treating every new phenomenon as a replay of historical cycles, or stripping the flattering label off a concept to reveal what it actually does).
- The thesis committee asked: "How do you know you're replicating thinking, not just style?"

---

## 3. How the Evaluation Works

### 3.1 Three Conditions

For each of 8 probes per author (contemporary scenarios the real author never wrote about — e.g., "algorithm-driven content feeds", "the 996 work culture"), the model generates a response under three conditions:

| Condition | System Prompt |
|-----------|---------------|
| `full` | Complete SKILL.md — both thinking frameworks AND surface style |
| `style_only` | Only surface language style instructions (diction, rhythm, register) — all cognitive/value instructions deliberately removed |
| `neutral` | Plain assistant, no persona |

### 3.2 Style Stripping

Each raw output is **rewritten into plain neutral Chinese** by a style-stripping LLM call: "change only *how* it's said, never *what* is said." After stripping, all three conditions look equally plain — the only remaining difference is *what cognitive moves were made*.

### 3.3 Rubric Scoring (your task)

A judge reads each style-stripped text and scores it against a **thinking rubric** for that author — dimensions like:

**Lu Xun (8 dimensions):**
- `LX1` 名实分离 — separating a concept's label from what it actually does
- `LX2` 二难推理 — dilemma framing ("if X then... but if not-X, also...")
- `LX3` 归谬法 — reductio ad absurdum on the opposing view
- `LX4` 隐喻去蔽 — creating or migrating a metaphor to reveal hidden structure
- `LX5` 历史循环 — reading the present as a replay of a historical pattern
- `LX6` 辩证张力 — holding two true-but-contradictory observations simultaneously
- `LX7` 弱者正义与独立批判 — defending the marginalized AND refusing to flatter "progressive" positions
- `LX8` 拒绝廉价安慰 — refusing to end on hope or easy resolution

**Zhang Ailing (6 dimensions):**
- `ZA1` 苍凉非壮烈 — bleakness without heroism; beauty that fades
- `ZA2` 参差对照 — juxtaposing contrasting registers to produce irony
- `ZA3` 物质即心理 — using concrete material details (fabric, food, furniture) to carry psychological weight
- `ZA4` 不愿承认的欲望 — naming desires people perform not-having
- `ZA5` 关系即算计 — reading intimacy as a negotiation of power and dependency
- `ZA6` 不说教 — refusing to moralize or prescribe; observing without verdict

Each dimension is scored **1–5**:
- 5 = this author's level; the move is present, precise, and non-trivial
- 4 = clearly present but slightly labored
- 3 = embryonic; there but shallow
- 2 = superficial trace only
- 1 = absent or actively contrary to this author's way of thinking

The judge **never knows which condition produced a text** (blind scoring). After scoring, we decode the condition labels and compare: if `full` scores significantly higher than `style_only` after style has been stripped, the gap can only be explained by thinking content — which proves the agent captured thinking, not style.

### 3.4 Key Metric

```
headline_gap = mean_thinking_total(full) − mean_thinking_total(style_only)
```

A clearly positive gap = evidence that SKILL.md's thinking frameworks add genuine value beyond surface style imitation.

---

## 4. Results So Far (from two judges: DeepSeek and Claude)

Both judges ran blind and produced consistent results:

| Author | Judge | full | style_only | neutral | gap |
|--------|-------|------|------------|---------|-----|
| Lu Xun | DeepSeek | 4.09 | 2.55 | 2.27 | **+1.54** |
| Lu Xun | Claude | 3.70 | 2.50 | 2.16 | **+1.20** |
| Zhang Ailing | DeepSeek | 4.34 | 3.55 | 2.17 | **+0.79** |
| Zhang Ailing | Claude | 4.15 | 3.38 | 2.19 | **+0.77** |

**Lu Xun**: the gap is large (+1.2 to +1.5). The `full` condition dominates on `LX5` (历史循环) and `LX4` (隐喻去蔽) — the most distinctively Lu Xun cognitive moves. Interestingly, `style_only` actually *loses* to `neutral` in 5/8 ELO matches, meaning surface style alone without the thinking framework makes the output worse.

**Zhang Ailing**: smaller but real gap (+0.77). The base model already has some affinity for Zhang Ailing-like sensibility (urban intimate scene analysis), so the baseline is higher. Still, `full` consistently wins forced-choice comparisons 87.5–100% of the time.

**Judge agreement (DeepSeek vs Claude)**: within ±1 point on 95–96% of dimension scores, Pearson r ≈ 0.82 — strong cross-family reliability.

---

## 5. Your Task: Score `gpt_judge_packet.json`

We need you (a GPT-family model) to score the same 48 items **independently and blindly**, to add a third data point to the cross-family agreement check and the ensemble.

### What to do

1. Read the file `backend/eval/persona_thinking/results/gpt_judge_packet.json`.
2. For each item in `items[]`:
   - Note the `persona` field — this tells you which rubric to use (`lu-xun` → 8 LX dims, `zhang-ailing` → 6 ZA dims)
   - Read `scenario` (the situation being analyzed) and `stripped_text` (the style-stripped analysis)
   - Use the `rubrics` and `anchors` sections in the same file as your scoring reference
   - Score each thinking dimension **1–5 integer** — focus only on cognitive moves, value lens, and framing; the style has already been stripped, so don't judge prose quality
3. Output a **single JSON object** covering all 48 items in this exact format:

```json
{
  "IT000": {
    "LX1_名实分离": 4,
    "LX2_二难推理": 2,
    "LX3_归谬法": 3,
    "LX4_隐喻去蔽": 4,
    "LX5_历史循环": 5,
    "LX6_辩证张力": 3,
    "LX7_弱者正义与独立批判": 3,
    "LX8_拒绝廉价安慰": 4
  },
  "IT001": {
    "ZA1_苍凉非壮烈": 3,
    "ZA2_参差对照": 4,
    "ZA3_物质即心理": 4,
    "ZA4_不愿承认的欲望": 3,
    "ZA5_关系即算计": 2,
    "ZA6_不说教": 4
  },
  ...
}
```

- Use the **exact dimension IDs** from the rubric (e.g., `LX1_名实分离`), not translations or abbreviations.
- Cover **all 48 item_ids** from the packet (IT000 through whatever the last one is).
- Return only the JSON, no explanation.

### Where to save

Save the output as:
```
backend/eval/persona_thinking/results/gpt_scores.json
```

Then the user will run:
```bash
python backend/eval/persona_thinking/tools/aggregate_judges.py
```

This will compute the three-judge ensemble and report your agreement with DeepSeek and Claude.

---

## 6. File Map (for context)

```
backend/eval/persona_thinking/
├── DESIGN.md                  methodology document (for thesis defense)
├── REPORT.md                  accessible narrative report of results
├── CODEX_GUIDE.md             this file
├── common.py                  shared plumbing (model calls, caching, data loaders)
├── conditions.py              system prompts for full/style_only/neutral conditions
├── style_strip.py             style-stripping LLM call
├── judge.py                   rubric scoring and forced-choice judging
├── stylometry.py              surface style feature measurement (control check)
├── run.py                     main evaluation orchestrator
├── rubrics/
│   ├── lu-xun.json            8 thinking dimensions with anchors
│   └── zhang-ailing.json      6 thinking dimensions with anchors
├── probes/
│   ├── lu-xun.jsonl           8 contemporary scenario probes
│   └── zhang-ailing.jsonl     8 contemporary scenario probes
├── corpus_anchors/
│   ├── lu-xun.json            real excerpts from Lu Xun's collected works
│   └── zhang-ailing.json      real excerpts from Zhang Ailing's works
├── results/
│   ├── result_20260531_162758.json   main run (DeepSeek judge)
│   ├── claude_scores.json            Claude's blind scores (already done)
│   ├── gpt_judge_packet.json         ← your scoring input
│   ├── gpt_scores.json               ← your scoring output (to be created)
│   ├── claude_judge_mapping.json     item_id → condition mapping + DeepSeek scores
│   ├── elo_report.json               ELO pairwise ranking results
│   └── ENSEMBLE_SUMMARY.md           multi-judge ensemble summary
└── tools/
    ├── make_judge_packet.py      generates blind scoring packets
    ├── analyze_cross_judge.py    Claude vs DeepSeek agreement analysis
    ├── aggregate_judges.py       multi-judge ensemble (DeepSeek + Claude + GPT)
    └── elo_pairwise.py           ELO ranking via pairwise forced-choice
```

---

## 7. Important Notes

- **Do not look at `claude_judge_mapping.json`** before scoring — it contains the condition labels (full/style_only/neutral) that would break the blind.
- The `stripped_text` fields have already had surface style removed. Do not penalize for plain writing.
- Score conservatively: a 5 means the text genuinely exhibits the move at the author's level, not just that the topic is related.
- For Lu Xun's `LX5` (历史循环): give a 5 only if the text explicitly maps a present phenomenon onto a specific historical pattern as Lu Xun himself did (e.g., seeing "content algorithm" as the modern "看客" / spectator dynamic from 狂人日记 or 药).
- For Zhang Ailing's `ZA1` (苍凉非壮烈): the affect should feel resigned and observational, not tragic or melodramatic. Melodrama is a 2, not a 5.
