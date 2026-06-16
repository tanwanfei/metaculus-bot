# Metaculus 集成预测 Bot（AI Forecasting Benchmark）

为 Metaculus AI 预测基准赛（FutureEval / AIB）写的竞赛 bot，基于官方 `forecasting-tools` 框架，
实现了历届获奖复盘里**得分贡献最大的几个技巧**，而不是模板的“单模型单次预测”基线：

1. **集成/聚合**（Metaculus 官方分析里最大的单一得分杠杆）——每题多次预测，用框架自带、经过测试的聚合器合并（二元取中位数、多选归一化、数值分位数合并）。
2. **多模型多样性**——这些预测分散在**不同模型家族**上（GPT-4o + Claude），不是一个模型重复跑。获奖 bot 平均用 ~1.8 个不同模型，多样性能抵消单模型的系统性偏差。
3. **外部视角 → 内部视角提示**——每个预测 prompt 先强制走“参照类/基础概率”，再做个案推理，并对现状加权（这是好预测者的校准习惯）。
4. **带降级的深度研究**——新闻/搜索接地，多级自动降级：**AskNews DeepNews（多轮检索+推理的深度研究）→ 普通 AskNews 新闻 → 搜索模型 → 纯 LLM**。只要配了 AskNews key 就自动走 DeepNews（每题一次、整个集成共享），质量远高于单次新闻检索；研究永远不会让预测崩掉（失败就自动降级）。

> 每题预测次数 = `len(FORECASTER_MODELS) × RUNS_PER_MODEL`。默认 = 2 模型 × 2 次 = **每题 4 个预测聚合**。

---

## ⚡ 为什么默认零成本

比赛期间 Metaculus 通过它的 `metaculus/...` 代理**免费**提供 LLM 调用（用你的 `METACULUS_TOKEN` 计费到它头上）。
默认面板用的就是这个代理上的**两个不同厂商**模型，所以**只要有 METACULUS_TOKEN 就能跑一个有多样性的集成，零成本**，不需要 OpenAI / Anthropic / OpenRouter 的 key。
只有当你想用更新/更强的模型时，才需要自己的 key（见下面“调参”）。

---

## 你只需要准备 1 样东西（外加 1 样可选）

| | 是什么 | 在哪拿 | 必需？ |
|---|---|---|---|
| `METACULUS_TOKEN` | 你的 bot 账号 token | https://www.metaculus.com/futureeval/participate/ | **必需** |
| `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` | 配了就自动启用 **DeepNews 深度研究**（更强，但拿 key 需绑信用卡）。不配也行——会自动用免费的 metaculus 搜索代理做研究 | https://asknews.app/ | 可选 |

---

## 🚀 5 分钟上线（推荐：GitHub Actions，全自动，无需本地环境）

1. **把本文件夹做成一个 GitHub 仓库**（这个 `metaculus-bot/` 文件夹就是仓库根目录）：
   ```bash
   cd metaculus-bot
   git init && git add . && git commit -m "metaculus ensemble bot"
   # 在 GitHub 上新建一个空仓库，然后：
   git remote add origin <你的仓库地址>
   git push -u origin main
   ```
2. **加 secret**：仓库页 → `Settings → Secrets and variables → Actions → New repository secret`，
   至少加 `METACULUS_TOKEN`（名字必须完全一致，大写）。有 AskNews 就把那两个也加上。
3. **启用 Actions**：点仓库的 `Actions` 标签 → `I understand my workflows, go ahead and enable them`。
4. **先冒烟测试**：`Actions → Test bot (smoke test) → Run workflow`。约 3–5 分钟后，去你的 Metaculus bot
   主页确认预测有没有发上去（它会在 `bot-testing-area` 这个公开练习赛里预测）。
5. **完成**。`Forecast on AI tournament` 这个 workflow 已启用，每 20 分钟自动跑一次，自动接新题、跳过已预测过的题。

> 暂停：`Actions → Forecast on AI tournament → ··· → Disable workflow`。

---

## 🖥️ 本地运行（可选，调 prompt 时更快）

```bash
cd metaculus-bot
pip install -r requirements.txt
cp .env.example .env        # 然后编辑 .env 填入 METACULUS_TOKEN

python main.py --mode test_questions          # 冒烟测试（bot-testing-area）
python main.py --mode tournament              # 正赛：AIB + MiniBench
python main.py --mode tournament --no-publish # 干跑，只算不提交
```

---

## 🎛️ 调参（都在 `main.py` 顶部）

| 常量 | 作用 | 默认 |
|---|---|---|
| `FORECASTER_MODELS` | 集成面板用哪些模型 | 2 个免费代理模型（GPT-4o + Claude） |
| `RUNS_PER_MODEL` | 每个模型每题跑几次（越大聚合越强、越慢） | 2 |
| `MODEL_TEMPERATURE` | 采样温度（>0 才能产生可聚合的差异） | 0.4 |
| `MAX_CONCURRENT_LLM_CALLS` | 全局并发上限，防止触发限流 | 5 |
| `USE_ASKNEWS_DEEP_RESEARCH` | 总开关；没配 AskNews key 时自动忽略 | `True` |
| `ASKNEWS_DEEP_RESEARCH_MODEL` | DeepNews 用的模型（`deepseek-basic` 最省；可换 `claude-sonnet-4-6` / `gpt-5` / `o3` 更强更贵） | `deepseek-basic` |
| `ASKNEWS_DEEP_SEARCH_DEPTH` / `_MAX_DEPTH` | 检索-推理的轮数 / 上限（越大越深、越慢、越贵） | 2 / 4 |
| `ASKNEWS_DEEP_SOURCES` | 数据源；加 `"google"/"x"/"wiki"` 覆盖更广（可能需付费版） | `["asknews"]` |
| `MAX_CONCURRENT_RESEARCH_CALLS` | DeepNews 并发上限（慢+有限流，建议保持小） | 2 |

> **DeepNews 成本提示**：DeepNews 按调用消耗 AskNews 额度，每题一次（整个集成共享）。约 300–500 题 ≈ 300–500 次深度研究。想省额度就把深度调到 `1 / 1`、模型用 `deepseek-basic`；想要更强研究就加深度/换强模型。任何失败都会自动降级到普通检索，不会让预测崩掉。

想用自己的额度/更强模型，改 `FORECASTER_MODELS` 即可，例如：
```python
FORECASTER_MODELS = [
    "openrouter/openai/gpt-4o",                # 需要 OPENROUTER_API_KEY
    "openrouter/anthropic/claude-3.7-sonnet",  # 需要 OPENROUTER_API_KEY
]
```
**保持面板跨厂商**——多样性正是它有效的原因。改完务必先跑一次 `--mode test_questions` 确认模型名有效。

---

## 重要提醒

- **越早进越好**：Summer 2026（赛事 ID 33022）已在进行中，你从 leaderboard 中位、0 分开始累积，只对你入场后才结算的题计分。每晚一周就少一周可计分的题。
- **必须填 bot survey 才发奖**：去比赛页面完成获奖问卷，否则即使排名进钱也拿不到。
- **先 `--no-publish` 干跑看 prompt 输出**，满意了再正式提交。

## 文件结构
```
metaculus-bot/
├── main.py                       # bot 本体（集成 + 多模型 + 外部/内部视角）
├── requirements.txt
├── .env.example                  # 本地运行用；复制成 .env 填 key
├── .gitignore                    # 已忽略 .env，别把真 key 提交上去
└── .github/workflows/
    ├── run_bot.yaml              # 每 20 分钟跑正赛
    └── test_bot.yaml             # 手动触发的冒烟测试
```
