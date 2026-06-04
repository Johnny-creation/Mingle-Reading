# 多裁判集成评测结果（Ensemble）

参与裁判：deepseek, claude, gpt
集成方式：各裁判各维度分取算数均值。

## 各裁判 + 集成：思维总均分

| 名家 | 条件 | Deepseek | Claude | Gpt | Ensemble |
|---|---||---||---||---||---|
| lu-xun | full | 4.086 | 3.703 | 4.172 | 3.987 |
| lu-xun | style_only | 2.547 | 2.5 | 2.594 | 2.547 |
| lu-xun | neutral | 2.266 | 2.156 | 2.344 | 2.255 |
| zhang-ailing | full | 4.344 | 4.146 | 4.458 | 4.316 |
| zhang-ailing | style_only | 3.552 | 3.375 | 3.771 | 3.566 |
| zhang-ailing | neutral | 2.167 | 2.188 | 2.688 | 2.347 |

## headline gap：full − style_only（风格剥离后思维增益）

| 名家 | Deepseek | Claude | Gpt | Ensemble |
|---||---||---||---||---|
| lu-xun | 1.539 | 1.203 | 1.578 | 1.44 |
| zhang-ailing | 0.792 | 0.771 | 0.687 | 0.75 |

## 裁判间一致性

### lu-xun

| 裁判对 | n | 平均绝对差 | ±1内 | 完全一致 | Pearson r |
|---|---|---|---|---|---|
| deepseek_vs_gpt | 192 | 0.549 | 0.932 | 0.474 | 0.819 |
| claude_vs_deepseek | 192 | 0.565 | 0.958 | 0.427 | 0.818 |
| claude_vs_gpt | 192 | 0.656 | 0.964 | 0.38 | 0.798 |

### zhang-ailing

| 裁判对 | n | 平均绝对差 | ±1内 | 完全一致 | Pearson r |
|---|---|---|---|---|---|
| deepseek_vs_gpt | 144 | 0.59 | 0.882 | 0.465 | 0.784 |
| claude_vs_deepseek | 144 | 0.493 | 0.938 | 0.5 | 0.826 |
| claude_vs_gpt | 144 | 0.625 | 0.889 | 0.486 | 0.767 |
