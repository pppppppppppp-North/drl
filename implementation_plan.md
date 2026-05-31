# Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data

## Project Proposal Implementation Plan

Audience: high school research project proposal  
Duration: two months, full-time  
Primary output: a reproducible experimental pipeline for training and evaluating deep reinforcement learning agents on Thai stock market data  
Compute plan: local development plus BistKA Mini-HPC for GPU sentiment modeling, vectorized RL training, and hyperparameter sweeps

## 1. Executive Summary

This project proposes a deep reinforcement learning (DRL) system that learns to optimize long-term trading profit in Thai stock markets using multi-source data. Instead of predicting only the next price or return, the project frames trading as a sequential decision problem. At every time step, an agent observes a market state, chooses a continuous portfolio weight, receives a reward based on profit and risk, and updates its policy to improve future decisions.

The project will start with a clean and manageable Thai equity universe, preferably SET50 and SET100 constituents, then expand only if data coverage is reliable. The experimental design will compare DRL agents against classical and machine learning baselines, including buy-and-hold, equal-weight rebalancing, momentum, mean-variance optimization, LSTM/XGBoost prediction-based strategies, and FinRL-style DRL baselines. PPO will be the main algorithm because the existing project materials already selected PPO and because PPO is stable for noisy sequential environments. A2C, DDPG, TD3, and SAC will be added as comparisons if time and compute allow.

The plan is intentionally structured as a project proposal and an engineering execution guide. It defines the research question, data sources, system architecture, experimental procedure, evaluation metrics, HPC usage, phase schedule, and expected outputs. The most important experimental requirement is walk-forward validation, because normal random train/test splits are invalid for financial time series. The second most important requirement is ablation testing, because the project claims to use multi-source data and must prove which sources actually help.

## 2. Research Goal and Questions

### 2.1 Main Goal

Build and evaluate a DRL trading framework that optimizes long-term profit and risk-adjusted return in Thai stock markets using technical, fundamental, macroeconomic, sentiment, market index, and market-flow data.

### 2.2 Research Questions

1. Can a continuous-action DRL agent learn profitable long-term portfolio allocation policies for Thai equities?
2. Does adding multi-source data improve out-of-sample trading performance compared with price-only and technical-indicator-only agents?
3. Which data sources contribute most to performance: technical indicators, macro variables, financial statements, Thai news sentiment, market index context, sector context, or investor-flow data?
4. Which DRL algorithm is most stable for this project under high noise and limited compute: PPO, A2C, DDPG, TD3, or SAC?
5. Does the learned policy generalize across different market regimes, including COVID-19 and large Thai market drawdowns?
6. Can BistKA HPC be used efficiently for reproducible training, hyperparameter tuning, and sentiment feature extraction?

## 3. Project Scope

### 3.1 Market Universe

The recommended primary universe is SET50 because it has higher liquidity, more stable corporate reporting, and better news coverage than smaller Thai stocks. SET100 is the recommended expansion universe. All SET-listed equities can be considered only after the pipeline works, because missing data, low liquidity, and survivorship bias become harder to control.

The first experimental universe should contain:

- SET50 liquid stocks with long available history.
- SET Index, SET50 Index, and SET100 Index as market context.
- Sector indices where available.
- A small pilot subset of 5 to 10 symbols for fast debugging.

### 3.2 Time Range

Use the longest possible period that can be collected consistently. The minimum acceptable daily experiment should target 10 years if available. If intraday data is available only for a shorter period, it should be used as a secondary experiment rather than forcing all sources into the same limited time range.

Recommended setup:

- Daily OHLCV: longest reliable period, ideally 2010 onward.
- Intraday or tick data: use only if licensed and stable; otherwise optional.
- News and sentiment: align to available news history.
- Macroeconomic data: daily, monthly, and quarterly variables, forward-filled only after official release dates.
- Financial statements: quarterly and annual fundamentals, lagged to avoid look-ahead bias.

### 3.3 Multi-Source Data

The project will attempt to collect and test:

- OHLCV prices and volume.
- Technical indicators such as RSI, MACD, Bollinger Bands, ATR, rolling volatility, moving averages, momentum, and volume ratios.
- SET/SET50/SET100 index returns and volatility.
- Sector index returns.
- Market breadth features such as number of advancing and declining stocks if available.
- Foreign, local institution, proprietary, and retail investor-flow data if available.
- Financial statements and ratios such as revenue growth, earnings growth, ROE, debt-to-equity, P/E, P/B, dividend yield, and market cap.
- Macroeconomic indicators from Bank of Thailand and other public sources, including policy rate, exchange rates, inflation, GDP indicators, tourism, exports, money supply, and bond yields.
- Thai and English financial news headlines.
- Thai financial sentiment features from WangchanBERTa or a smaller Thai sentiment model.
- Calendar features such as weekday, month, earnings season, holidays, and crisis-period flags.

## 4. Data Source Plan

The plan should prefer official and reproducible sources when possible.

### 4.1 Thai Market Data

Primary paid or official candidates:

- SET SMART Marketplace or SETSMART for historical price, index, fundamental, and corporate-action data.
- SET historical files where accessible.
- Licensed vendors if the school has access.

Fallback public candidates:

- Yahoo Finance through `yfinance` for daily OHLCV where Thai symbols are available.
- Stooq, Investing.com, Kaggle datasets, or other public datasets only after checking license and coverage.

Every collected dataset must be stored with:

- source name,
- download date,
- raw file path,
- license or access note,
- columns,
- frequency,
- coverage start and end dates,
- missing value percentage,
- known limitations.

### 4.2 Macroeconomic Data

Use Bank of Thailand statistics for Thai financial and economic variables. Candidate variables include exchange rates, policy rate, commercial bank rates, inflation-related indicators, financial market statistics, external sector indicators, tourism-related indicators, and real-sector indicators.

Important rule: macro data must be aligned by release date, not by the period it describes. For example, a quarterly GDP value for Q1 cannot be used inside the Q1 trading period before it was officially released.

### 4.3 News and Sentiment Data

Candidate news sources:

- SET news and corporate announcements.
- Thai financial news websites such as Kaohoon, Thunhoon, Bangkok Biz News, Prachachat, Krungthep Turakij, and The Standard Wealth, subject to allowed use.
- Company disclosures and annual reports.
- English financial news if available.

Sentiment extraction options:

- WangchanBERTa embeddings for Thai text.
- Fine-tuned WangchanBERTa classification into positive, neutral, and negative.
- Distant supervision using next-day or next-week abnormal returns to generate weak labels.
- A simpler dictionary or rule-based Thai sentiment fallback if GPU fine-tuning is too slow.

## 5. Modeling Framework

### 5.1 MDP Definition

The trading problem will be modeled as a Markov Decision Process:

- State: market features, stock features, sentiment features, macro features, and portfolio state.
- Action: continuous target portfolio weights.
- Reward: long-term profit and risk-adjusted performance.
- Transition: movement from one trading day or intraday bar to the next.
- Policy: neural network mapping observed state to portfolio allocation.

### 5.2 State Space

For each time step, the state should include:

- recent returns and normalized OHLCV,
- technical indicators,
- rolling volatility and drawdown,
- index and sector context,
- sentiment vector for each stock or aggregated market sentiment,
- macro features known at that time,
- current holdings and cash ratio,
- previous action,
- time features.

For a first working version, use a smaller state:

- 30-day lookback window,
- daily OHLCV and technical indicators,
- SET index features,
- one sentiment vector per stock-date,
- current position.

After the first working version, expand to fundamentals, macro, and investor flows.

### 5.3 Action Space

The user requested continuous actions. The recommended action is a vector of target portfolio weights.

For one stock:

- action `a_t` in `[-1, 1]`,
- positive means long exposure,
- zero means cash or no position,
- negative means short exposure only if the experiment explicitly allows shorting.

For a portfolio:

- action vector `w_t`,
- each element is a target weight for a stock,
- optional cash weight,
- weights are normalized by softmax or projection,
- long-only experiment: weights in `[0, 1]` and sum to at most 1,
- long-short experiment: weights in `[-w_max, w_max]` with leverage limit.

Because Thai retail stock shorting may not be realistic for all stocks, the main experiment should be long-only or long-cash. Long-short can be a research extension.

### 5.4 Reward Design

The reward must not only reward raw profit, because raw profit can encourage unstable or high-risk behavior. Use several reward variants:

1. Raw portfolio return.
2. Log portfolio return.
3. Return minus volatility penalty.
4. Differential Sharpe-style reward.
5. Sortino-style downside-risk reward.
6. Drawdown-penalized reward.
7. Hybrid reward: return + risk-adjusted component + trade quality bonus - transaction cost.

The main proposal reward:

`reward_t = alpha * log_return_t + beta * delta_sharpe_t - gamma * drawdown_penalty_t - lambda * turnover_t`

Transaction costs, slippage, and tax are not required by the user for the main scope, but a small turnover penalty should still be included as a regularizer. A no-cost version and a cost-aware version should both be tested.

### 5.5 Algorithms

Main algorithm:

- PPO with clipped objective, entropy regularization, and generalized advantage estimation.

Comparison algorithms:

- A2C: simple actor-critic baseline.
- DDPG: continuous control baseline.
- TD3: improved DDPG with twin critics and delayed updates.
- SAC: entropy-regularized continuous control baseline.
- DQN: optional only for a discrete buy/hold/sell comparison.

Baseline strategies:

- Buy-and-hold SET index.
- Buy-and-hold each stock.
- Equal-weight portfolio.
- Monthly rebalanced equal-weight portfolio.
- Moving-average crossover.
- Momentum strategy.
- Mean-variance optimization.
- Random policy.
- LSTM or XGBoost prediction followed by a simple trading rule.

## 6. Experimental Procedure

This section is the step-by-step experimental procedure. Each step must produce an artifact before the next step begins.

### Step 1: Create Project Repository Structure

Create this structure:

```text
project_root/
  config/
  data/
    raw/
    interim/
    processed/
    external/
  docs/
  logs/
  models/
  notebooks/
  reports/
  results/
  scripts/
  src/
    agents/
    data/
    envs/
    evaluation/
    features/
    models/
    sentiment/
    utils/
  tests/
```

Expected artifacts:

- `README.md`
- `requirements.txt` or `environment.yml`
- `config/default.yaml`
- empty folder structure

### Step 2: Build Data Manifest

For each source, create a manifest row:

- source,
- access method,
- frequency,
- date range,
- symbols,
- columns,
- license note,
- raw file path,
- missing value rate.

Expected artifacts:

- `data/data_manifest.csv`
- `docs/data_sources.md`

### Step 3: Collect Pilot Data

Collect a small pilot dataset:

- 5 Thai stocks,
- SET index,
- at least 3 years of daily OHLCV,
- one macro feature,
- one news or sentiment feature if possible.

Expected artifacts:

- `data/raw/prices_pilot.csv`
- `data/processed/prices_pilot.parquet`
- `data/processed/features_pilot.parquet`
- `notebooks/01_data_quality_check.ipynb`

### Step 4: Clean and Align Data

Perform:

- date parsing,
- symbol normalization,
- corporate action adjustment if available,
- duplicate removal,
- missing value report,
- frequency alignment,
- forward-fill only where logically allowed,
- no leakage from future data.

Expected checks:

- no duplicate `(date, ticker)` rows,
- no future macro information before release date,
- every feature has a known timestamp,
- train/validation/test split preserves time order.

### Step 5: Feature Engineering

Generate:

- returns: 1-day, 5-day, 20-day,
- rolling volatility,
- moving averages,
- RSI,
- MACD,
- Bollinger Bands,
- ATR,
- volume ratios,
- market index returns,
- sector-relative returns,
- macro changes,
- sentiment scores or embeddings.

Expected artifact:

- `data/processed/features_daily.parquet`

### Step 6: Build the Trading Environment

Implement a Gymnasium-compatible environment:

- `reset()`,
- `step(action)`,
- observation space,
- action space,
- portfolio accounting,
- reward calculation,
- episode termination,
- info dictionary with portfolio value, turnover, drawdown, and action.

Tests:

- action shape is valid,
- state shape is stable,
- no NaN observations,
- portfolio value updates correctly,
- environment does not use future data,
- episode ends at the correct point.

### Step 7: Implement Baselines

Before DRL training, implement:

- buy-and-hold,
- equal weight,
- random policy,
- momentum,
- moving average crossover.

Expected artifact:

- `results/baselines_metrics.csv`
- baseline equity curves

### Step 8: Implement PPO Correctly

The existing project report notes that PPO update logic is unfinished. The project must implement:

- rollout buffer,
- log probabilities,
- value estimates,
- GAE advantage calculation,
- clipped policy loss,
- value loss,
- entropy bonus,
- minibatch updates,
- checkpoint saving,
- TensorBoard or MLflow logging.

Validation:

- PPO can overfit a tiny synthetic environment,
- policy loss and value loss change during training,
- entropy does not immediately collapse to zero,
- action distribution is not always hold.

### Step 9: Train Technical-Only Agent

Train PPO using only price and technical features.

Outputs:

- trained model,
- training log,
- validation equity curve,
- metrics table,
- action distribution plot.

Success condition:

- model makes non-zero decisions,
- validation run completes without NaN,
- out-of-sample performance is compared with baselines.

### Step 10: Train Multi-Source Agent

Add sentiment, macro, fundamentals, and market context gradually.

Do not add all sources at once. Use staged experiments:

- price only,
- price + technical,
- price + technical + index,
- price + technical + macro,
- price + technical + fundamentals,
- price + technical + sentiment,
- all features.

Expected artifact:

- `results/ablation_metrics.csv`

### Step 11: Hyperparameter Tuning

Use Optuna or job arrays. Tune:

- learning rate,
- gamma,
- GAE lambda,
- PPO clip range,
- entropy coefficient,
- value loss coefficient,
- rollout length,
- batch size,
- network size,
- reward weights.

Use a small search first, then a narrower search.

### Step 12: Walk-Forward Validation

Use rolling windows. Example:

| Window | Train | Validation | Test |
|---|---|---|---|
| 1 | 2010-2016 | 2017 | 2018 |
| 2 | 2011-2017 | 2018 | 2019 |
| 3 | 2012-2018 | 2019 | 2020 |
| 4 | 2013-2019 | 2020 | 2021 |
| 5 | 2014-2020 | 2021 | 2022 |
| 6 | 2015-2021 | 2022 | 2023 |
| 7 | 2016-2022 | 2023 | 2024 |
| 8 | 2017-2023 | 2024 | 2025 |

Adjust years based on available data.

### Step 13: Crisis and Regime Testing

Test separately on:

- COVID-19 crash and recovery period,
- high-volatility Thai market periods,
- low-volatility sideways periods,
- bull market periods,
- bear market periods.

Report whether the agent:

- reduces drawdown,
- exits risky positions,
- overtrades,
- misses recoveries,
- depends too strongly on one data source.

### Step 14: Final Evaluation Metrics

Use all requested metrics:

- cumulative return,
- annualized return,
- annualized volatility,
- Sharpe ratio,
- Sortino ratio,
- maximum drawdown,
- Calmar ratio,
- win rate,
- turnover,
- average holding period,
- number of trades,
- profit factor,
- value at risk,
- conditional value at risk,
- beta to SET index,
- tracking error,
- information ratio,
- downside deviation,
- final portfolio value.

No formal statistical tests are required based on user instruction.

### Step 15: Final Report Figures

Prepare:

- data coverage chart,
- missing value heatmap,
- feature correlation map,
- model architecture diagram,
- training reward curve,
- entropy curve,
- validation equity curve,
- walk-forward equity curve,
- drawdown chart,
- action distribution,
- turnover chart,
- ablation bar chart,
- risk metric table,
- benchmark comparison table,
- crisis-period comparison chart.

## 7. Two-Month Phase Plan

### Phase 1: Foundation and Research Setup

Duration: Week 1

Goals:

- finalize scope,
- create repository,
- extract HPC rules,
- choose data sources,
- collect pilot data,
- build data manifest.

Tasks:

1. Confirm project universe: SET50 first, SET100 second.
2. Create project folder structure.
3. Prepare Python environment locally.
4. Prepare BistKA environments: `drl_env` and `llm_env`.
5. Collect pilot OHLCV data.
6. Create data-quality notebook.
7. Write first version of data manifest.
8. Define model and experiment configuration files.

Exit criteria:

- pilot data loads successfully,
- no major date alignment errors,
- one local test script runs end-to-end.

### Phase 2: Data Pipeline and Feature Engineering

Duration: Week 2

Goals:

- collect the longest practical daily dataset,
- produce clean feature tables,
- implement no-leakage splits.

Tasks:

1. Collect SET50/SET100 daily OHLCV.
2. Collect SET index and sector index data.
3. Collect macro features from BOT or other public sources.
4. Collect financial statement/fundamental features if accessible.
5. Convert raw CSV files to Parquet.
6. Generate technical indicators.
7. Align features by date.
8. Create train/validation/test split function.
9. Write data validation tests.

Exit criteria:

- `features_daily.parquet` exists,
- each column has documentation,
- no feature uses future data.

### Phase 3: Environment and Baselines

Duration: Week 3

Goals:

- create a reliable trading environment,
- implement simple baselines,
- verify reward and accounting.

Tasks:

1. Implement `ThaiStockTradingEnv`.
2. Add continuous action support.
3. Add portfolio accounting.
4. Add reward variants.
5. Add logging in `info`.
6. Implement buy-and-hold, equal weight, random, momentum, and moving average baselines.
7. Generate first baseline results.
8. Add unit tests for environment behavior.

Exit criteria:

- environment passes tests,
- baselines generate metrics and plots,
- action distribution can be plotted.

### Phase 4: PPO Completion and Technical-Only Agent

Duration: Week 4

Goals:

- complete PPO update logic,
- train first PPO agent,
- solve hold-only collapse.

Tasks:

1. Implement rollout buffer.
2. Implement advantage calculation.
3. Implement PPO clipped loss.
4. Add entropy regularization.
5. Add checkpoint saving.
6. Train on pilot data.
7. Train technical-only model on larger data.
8. Tune entropy coefficient and reward weights.
9. Confirm agent trades instead of always holding.

Exit criteria:

- PPO learns in synthetic test,
- PPO runs on Thai stock features,
- technical-only baseline has complete metrics.

### Phase 5: Sentiment and Multi-Source Fusion

Duration: Week 5

Goals:

- produce Thai sentiment features,
- add all major data groups through controlled ablations.

Tasks:

1. Collect Thai financial news or disclosure text.
2. Clean Thai text.
3. Generate WangchanBERTa embeddings or sentiment scores.
4. Cache sentiment features by date and ticker.
5. Merge sentiment with market data.
6. Add macro and fundamental features.
7. Train price + technical + sentiment model.
8. Train all-feature model.
9. Record ablation metrics.

Exit criteria:

- sentiment features are cached,
- all-feature observation space is stable,
- ablation table is created.

### Phase 6: Hyperparameter Search and Algorithm Comparison

Duration: Week 6

Goals:

- tune PPO,
- compare with A2C, DDPG, TD3, and SAC if feasible.

Tasks:

1. Define Optuna search space.
2. Run small local sanity search.
3. Launch HPC job array.
4. Compare PPO against A2C.
5. Add SAC or TD3 for continuous action comparison.
6. Select top configurations by validation Sharpe, drawdown, and stability.
7. Save all config files and seeds.

Exit criteria:

- best config is selected,
- algorithm comparison table exists,
- failed runs are documented.

### Phase 7: Walk-Forward and Stress Testing

Duration: Week 7

Goals:

- perform serious out-of-sample evaluation,
- test crisis and market-regime behavior.

Tasks:

1. Define walk-forward windows.
2. Launch one job per window.
3. Aggregate out-of-sample equity curves.
4. Compute final metrics.
5. Run crisis-period tests.
6. Run no-cost and cost-aware comparisons.
7. Run feature ablation again on selected windows.
8. Prepare final plots.

Exit criteria:

- walk-forward results exist,
- crisis results exist,
- final model is selected.

### Phase 8: Proposal Report and Presentation Outputs

Duration: Week 8

Goals:

- finalize proposal, figures, methodology, literature review, and reproducibility package.

Tasks:

1. Write final methodology.
2. Write literature review.
3. Prepare result tables.
4. Prepare project architecture diagram.
5. Prepare Gantt-style schedule table.
6. Export LaTeX and PDF.
7. Archive config, logs, and model summaries.
8. Offload important BistKA data before storage expiry if applicable.

Exit criteria:

- final plan PDF exists,
- literature review PDF exists,
- HPC notes and SLURM scripts exist,
- all final figures and tables are listed.

## 8. Gantt-Style Schedule

| Phase | Week | Main Work | Main Output |
|---|---:|---|---|
| 1 | 1 | setup, scope, HPC extraction, pilot data | repository, data manifest, HPC notes |
| 2 | 2 | data collection and features | processed feature table |
| 3 | 3 | trading environment and baselines | tested env, baseline metrics |
| 4 | 4 | PPO implementation and technical-only training | PPO model, technical-only results |
| 5 | 5 | sentiment and multi-source fusion | sentiment cache, ablation v1 |
| 6 | 6 | hyperparameter tuning and algorithm comparison | best configs, model comparison |
| 7 | 7 | walk-forward and stress testing | out-of-sample results |
| 8 | 8 | final proposal and documentation | plan PDF, literature review PDF |

## 9. HPC Execution Plan

Use local machine for:

- code editing,
- unit tests,
- tiny pilot experiments,
- plotting,
- report writing.

Use BistKA HPC for:

- WangchanBERTa embedding generation,
- model fine-tuning,
- long PPO training,
- Optuna or job-array sweeps,
- walk-forward windows.

Use the BistKA constraints from `hpc/HPC.md`:

- run compute only through SLURM,
- use project storage for large data,
- monitor quota with `myquota`,
- monitor credit with `mycredit`,
- avoid saving too many checkpoints,
- use `compute-devel` or `gpu4500-devel` for short debugging,
- use `compute-normal` for CPU RL training,
- use `gpu4500-normal` for sentiment modeling.

## 10. Expected Final Deliverables

1. A reproducible code repository.
2. Clean processed dataset or documented data pipeline.
3. Trading environment with tests.
4. PPO model implementation or Stable-Baselines3 equivalent.
5. Baseline results.
6. DRL model results.
7. Multi-source ablation results.
8. Walk-forward validation results.
9. Crisis-period analysis.
10. Final project proposal report.
11. Standalone literature review.
12. HPC usage appendix and SLURM scripts.

## 11. Success Criteria

The project is successful if:

- the pipeline runs end-to-end,
- the agent trains without NaN or shape errors,
- baselines and DRL agents are evaluated on the same periods,
- walk-forward validation is completed,
- ablation study shows whether each data source helps,
- final metrics and plots are reproducible from saved configs,
- the final report clearly explains limitations.

The project does not need to prove guaranteed profit. It needs to prove that the experimental process is correct, reproducible, and scientifically honest.

## 12. Key Limitations

- Financial markets are noisy and non-stationary.
- Backtest performance may not transfer to live trading.
- Public Thai market data may have missing coverage.
- News scraping may have legal and technical restrictions.
- Sentiment labels may be noisy.
- DRL can overfit strongly.
- Continuous actions can produce unrealistic turnover unless constrained.
- High school project time is short, so the first complete experiment is more important than a huge universe.

## 13. References

[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347

[2] X.-Y. Liu, H. Yang, J. Gao, and C. D. Wang, "FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance," ACM ICAIF, 2021. https://arxiv.org/abs/2111.09395

[3] Z. Zhang, S. Zohren, and S. Roberts, "Deep Reinforcement Learning for Trading," arXiv:1911.10107, 2019. https://arxiv.org/abs/1911.10107

[4] L. Lowphansirikul, C. Polpanumas, N. Jantrakulchai, and S. Nutanong, "WangchanBERTa: Pretraining transformer-based Thai Language Models," arXiv:2101.09635, 2021. https://arxiv.org/abs/2101.09635

[5] A. T. Rutherford, S. Chueykamhang, T. Bunditlurdruk, and N. Angsuwichitkul, "Aspect-Level Obfuscated Sentiment in Thai Financial Disclosures and Its Impact on Abnormal Returns," arXiv:2511.13481, 2025. https://arxiv.org/abs/2511.13481

[6] Stock Exchange of Thailand, "SMART Marketplace." https://www.set.or.th/th/services/connectivity-and-data/data/smart-marketplace

[7] Bank of Thailand, "Statistics." https://www.bot.or.th/en/statistics.html

[8] KVIS, "BistKA Mini-HPC Cluster Documentation," local file `HPC.pdf`, accessed from project folder.

