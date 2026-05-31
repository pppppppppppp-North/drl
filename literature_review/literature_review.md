# Literature Review

Project: **Deep Reinforcement Learning for Long-Term Profit Optimization in Thai Stock Markets Using Multi-Source Data**

## 1. Overview

This literature review supports a high school research proposal on using deep reinforcement learning (DRL) for long-term profit optimization in Thai stock markets. The review covers five connected areas:

1. reinforcement learning for sequential financial decisions,
2. continuous-action DRL algorithms for trading and portfolio allocation,
3. multi-source financial data,
4. Thai language sentiment modeling,
5. experimental validation for financial machine learning.

The central idea is that stock trading is not just a prediction problem. A model that predicts tomorrow's return can still lose money if it trades too often, ignores risk, or performs poorly during drawdowns. Reinforcement learning is attractive because it directly learns actions under a reward function. For this project, the reward can combine return, Sharpe ratio, Sortino ratio, drawdown control, and turnover penalties.

## 2. Reinforcement Learning for Financial Trading

Financial trading is naturally sequential. The result of today's action depends on the current market state, the existing portfolio, future price movement, and future decisions. This differs from supervised learning, where each prediction is often treated as independent. In trading, a model must decide not only whether a stock may rise, but also how much capital to allocate, whether to stay in cash, and whether risk is acceptable.

PPO is a strong candidate for this project because it is designed to update policies without moving too far from the previous policy. Schulman et al. introduced PPO as a practical policy-gradient method that keeps the advantages of more constrained policy optimization while being simpler to implement and tune [1]. This matters in stock trading because market data is noisy and unstable; aggressive policy updates can quickly destroy a learned strategy.

FinRL is especially relevant because it provides a full DRL framework for quantitative finance. Liu et al. describe FinRL as a full-stack framework with modular data, environment, and agent layers, including market frictions and reproducibility goals [2]. This supports the engineering design of the current project: data processors, trading environment, agent, evaluation, and experiment logging should be separated.

Zhang, Zohren, and Roberts applied DRL to trading across continuous futures and considered both discrete and continuous action spaces [3]. Their work is useful because this project also uses continuous action space. Continuous actions are closer to real portfolio allocation than buy/hold/sell labels, but they require stronger constraints and careful turnover control.

## 3. Continuous-Action Portfolio Optimization

Continuous action spaces allow the agent to output portfolio weights. This matches the user's requested design. For a single stock, the action can be a target exposure from 0 to 1 in a long-only experiment, or from -1 to 1 in a long-short experiment. For multiple stocks, the action is a vector of target weights.

Huang, Zhou, and Song study long-short portfolio optimization using DRL and mean Sharpe-ratio rewards [4]. Their work is relevant because it focuses on portfolio weights rather than simple buy/sell classification. However, Thai stock market shorting constraints may make a long-only or long-cash version more realistic for the main experiment. The long-short version can be kept as an extension.

For this project, continuous control algorithms should include PPO, A2C, DDPG, TD3, and SAC. PPO should remain the main model because the existing project files already use PPO and because PPO is easier to stabilize. SAC and TD3 can serve as stronger continuous-action comparisons if time permits.

## 4. Multi-Source Financial Data

Multi-source data is important because price alone may not explain all market behavior. The project should combine:

- OHLCV and technical indicators,
- index and sector context,
- macroeconomic indicators,
- financial statements and ratios,
- Thai financial news sentiment,
- investor-flow data if available.

The Stock Exchange of Thailand's SMART Marketplace is a strong candidate for official market data. The SET page describes API access to SET and TFEX data, including historical intraday trading data, end-of-day equity data, and company fundamental data [5]. This makes it suitable as the preferred source if access is available.

Bank of Thailand statistics are useful for macroeconomic features. The BOT statistics page lists economic, monetary, public finance, financial market, external sector, fiscal sector, real sector, and economic/financial indicator data, as well as BOT API service access [6]. These features can help the agent distinguish between stock-specific moves and macro-driven market regimes.

The main experimental challenge is not collecting many features. The challenge is proving that each feature helps. Therefore, the plan must include ablation studies. The experiment should compare price-only, technical-only, technical plus macro, technical plus sentiment, technical plus fundamentals, and all-feature models.

## 5. Thai Sentiment and Language Modeling

Thai financial text creates a special research opportunity because much financial NLP work focuses on English. WangchanBERTa is a strong Thai-language baseline. Lowphansirikul et al. pretrained a RoBERTa-style Thai model on a large Thai corpus and reported that WangchanBERTa outperformed multilingual baselines on Thai sequence and token classification tasks [7]. This supports using WangchanBERTa to encode Thai financial news, disclosures, and headlines.

A more finance-specific Thai sentiment direction is shown by Rutherford et al., who studied aspect-level obfuscated sentiment in Thai financial disclosures and its impact on abnormal returns [8]. Their work shows that Thai financial documents can contain subtle sentiment and that market reactions may depend on specific aspects of financial text. This supports the project's plan to test sentiment features separately rather than assuming all sentiment features are helpful.

For a high school project, full manual annotation of Thai financial sentiment may be too expensive. A practical approach is distant supervision. A headline can be weakly labeled positive if the stock has positive abnormal return after the headline, negative if it has negative abnormal return, and neutral if the movement is small. This is noisy but useful for creating a first financial sentiment classifier. Alternatively, embeddings can be used without labels and the DRL agent can learn whether the embedding is useful.

## 6. Reward Design and the Hold-Only Problem

The existing local project report identifies a hold-only convergence issue. This is common when transaction penalties or risk penalties make trading look dangerous early in training. If every exploratory trade loses money due to costs, the agent may learn that the safest action is never to trade.

The literature suggests several ways to reduce this problem:

- entropy regularization to encourage exploration,
- reward shaping to make useful actions visible earlier,
- curriculum learning with lower costs first,
- action distribution monitoring,
- turnover penalties that are not too large,
- comparing no-cost and cost-aware environments.

For this project, the reward should be tested in variants. A raw return reward is simple but unstable. A Sharpe-style reward is closer to the project goal but can be noisy when the rolling return variance is near zero. A hybrid reward is recommended:

`reward = alpha * return + beta * delta_sharpe - gamma * drawdown_penalty - lambda * turnover`

This reward aligns with long-term profit and risk control while still giving the agent enough signal to learn.

## 7. Evaluation and Validation

Financial machine learning must avoid look-ahead bias and overfitting. Random train/test split is not valid for stock time series. The project should use walk-forward validation:

1. train on an old period,
2. validate on the next period,
3. test on a later unseen period,
4. roll the window forward,
5. aggregate out-of-sample results.

The main metrics should include cumulative return, annualized return, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, turnover, win rate, profit factor, beta, tracking error, information ratio, VaR, and CVaR. Statistical tests are not required by the user's scope, but the report should still avoid making claims that are stronger than the evidence.

The most important comparisons are:

- DRL vs buy-and-hold SET index,
- DRL vs equal weight,
- DRL vs momentum,
- DRL vs mean-variance optimization,
- multi-source DRL vs technical-only DRL,
- PPO vs A2C/SAC/TD3 if compute allows.

## 8. Research Gap

The project's research gap is the combination of:

- Thai stock market focus,
- long-term profit optimization rather than price prediction,
- continuous-action portfolio weights,
- multi-source features,
- Thai financial language modeling,
- reproducible HPC-based experimental pipeline.

Many studies use DRL for US or global assets. Fewer projects focus on Thai equities with Thai-language data. The project can therefore contribute by demonstrating a transparent experimental framework, even if the final model does not beat every baseline.

## 9. Practical Implications for the Project Plan

The literature leads to these design decisions:

1. Use PPO as the main algorithm because it is stable and already fits the existing project direction.
2. Use continuous portfolio weights, but begin with long-only constraints for realism.
3. Use WangchanBERTa for Thai sentiment features, but cache embeddings before RL training.
4. Use official SET and BOT sources when possible.
5. Use walk-forward validation.
6. Include ablation studies for every major data source.
7. Track drawdown, turnover, and action distribution, not only profit.
8. Treat the final result as a research system, not as trading advice.

## References

[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347

[2] X.-Y. Liu, H. Yang, J. Gao, and C. D. Wang, "FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance," ACM ICAIF, 2021. https://arxiv.org/abs/2111.09395

[3] Z. Zhang, S. Zohren, and S. Roberts, "Deep Reinforcement Learning for Trading," arXiv:1911.10107, 2019. https://arxiv.org/abs/1911.10107

[4] G. Huang, X. Zhou, and Q. Song, "Deep Reinforcement Learning for Long-Short Portfolio Optimization," arXiv:2012.13773, revised 2025. https://arxiv.org/abs/2012.13773

[5] Stock Exchange of Thailand, "SMART Marketplace." https://www.set.or.th/th/services/connectivity-and-data/data/smart-marketplace

[6] Bank of Thailand, "Statistics." https://www.bot.or.th/en/statistics.html

[7] L. Lowphansirikul, C. Polpanumas, N. Jantrakulchai, and S. Nutanong, "WangchanBERTa: Pretraining transformer-based Thai Language Models," arXiv:2101.09635, 2021. https://arxiv.org/abs/2101.09635

[8] A. T. Rutherford, S. Chueykamhang, T. Bunditlurdruk, and N. Angsuwichitkul, "Aspect-Level Obfuscated Sentiment in Thai Financial Disclosures and Its Impact on Abnormal Returns," arXiv:2511.13481, 2025. https://arxiv.org/abs/2511.13481

