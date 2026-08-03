# The Harris Chain is Decorative: Why Temporal Structure Adds Nothing Beyond iid Bootstrap

## Simple Summary (for non-technical conversation)

**What we tested:** The SF-Harris model has a Markov chain that decides whether volatility "stays" the same or "jumps" to a new level. The idea is that this temporal structure should help predict future volatility better than simpler models.

**What we found:** The temporal structure doesn't help. A model that completely ignores temporal structure — just resampling past returns at random (Historical Simulation) — does just as well or better at predicting the range of future returns. The Harris chain is "decorative": it's there, it's mathematically elegant, but removing it changes nothing practical.

**Why this matters:** It tells you exactly what SF-Harris is good for (capturing heavy tails in the distribution) and what it's not good for (predicting when volatility will change). This is the difference between a *risk engine* (what it is) and a *trading signal* (what it isn't).

---

## Technical Report

### 1. The Central Question

The SF-Harris model has two components:

1. **Temporal dynamics** (the Harris chain): $\log(\tau_t) = \log(\tau_{t-1})$ with probability $e^{-\alpha}$, or a jump to a new level drawn from $Q$ with probability $1-e^{-\alpha}$
2. **Marginal distribution** (the jump distribution $Q$ and the emission $r_t | \tau_t \sim N(0, \tau_t)$)

The model's proponents claim the Harris chain captures volatility persistence and regime changes. We ask: **does the Harris chain add predictive value beyond what the marginal distribution alone provides?**

We answer this through seven independent tests. Every test returns the same answer: **no**.

---

### 2. Experiment 1: Coverage Decomposition

**Setup.** On 28 years of IBM daily returns (1998–2026), we compare eight models that progressively add or remove components:

| Model | Uses $\tau^*$? | Uses $Q$ shape? | Uses Harris dynamics? | AAD |
|-------|:---:|:---:|:---:|:---:|
| N($0, \sigma^2$) constant | No | No | No | 9.4pp |
| N($0, \tau^*$) | Yes | No | No | 2.0pp |
| iid bootstrap | No | Yes | No | 2.0pp |
| Bootstrap $\times \tau^*$ | Yes | Yes | No | 9.1pp |
| **SF-Harris + empirical $Q$** | **Yes** | **Yes** | **Yes** | **2.0pp** |
| Oracle bootstrap | No | Yes (oracle) | No | 0.2pp |

**Result.** SF-Harris = iid bootstrap = N($0, \tau^*$) = **2.0pp**. The Harris dynamics add exactly **0.0pp** over the marginal distribution alone.

The 7.4pp improvement from N($0, \sigma^2$) to N($0, \tau^*$) comes entirely from **variance adaptation** (using yesterday's volatility level), not from temporal structure. The empirical shape of $Q$ adds nothing beyond what N($0, \tau^*$) already provides.

The oracle bootstrap (0.2pp) shows the theoretical ceiling — perfect knowledge of the future return distribution. The gap from 2.0pp to 0.2pp comes from using the wrong variance level, not from missing temporal dynamics.

---

### 3. Experiment 2: Epsilon Calibration

**Setup.** The $\epsilon$ threshold in the Gibbs sampler classifies changes in $\log(\text{RV})$ as "stays" or "jumps." With $\epsilon = 10^{-5}$, nearly every point is a "jump" ($P(\text{stay}) \approx 0$), destroying temporal structure. With $\epsilon = 2.0$, most points are "stays" ($P(\text{stay}) \approx 0.98$), preserving it.

We sweep $\epsilon$ across 8 orders of magnitude at each frequency and measure AAD:

**15-min frequency** (ACF(1) = 0.68, theoretical P(stay) ≈ 0.68):

| $\epsilon$ | P(stay) | AAD |
|:----------:|:-------:|:---:|
| $10^{-5}$ | 0.37 | 2.3pp |
| $10^{-2}$ | 0.37 | 2.4pp |
| $10^{-1}$ | 0.40 | 2.5pp |
| $2 \times 10^{-1}$ | 0.45 | 2.4pp |
| $1.0$ | 0.81 | 2.4pp |
| $2.0$ | 0.98 | 2.3pp |

**Historical Simulation AAD: 1.7pp** (better than any $\epsilon$)

P(stay) varies from 0.37 to 0.98, yet AAD stays within a 0.3pp band. The Harris dynamics are **irrelevant** for coverage.

**1-hour and daily frequencies** show the same pattern: AAD is flat across all $\epsilon$ values (within 0.5pp), and Historical Simulation wins everywhere.

**Conclusion.** $\epsilon$ controls $P(\text{stay})$ but has no meaningful effect on coverage. The Harris chain's persistence parameter is a free parameter that doesn't affect the output.

---

### 4. Experiment 3: Frequency Comparison

**Setup.** We fit SF-Harris on IBM data at three frequencies and evaluate coverage of daily returns:

| Frequency | Bars/day | $\sigma$ | P(stay) | SF-Harris | Hist.Sim |
|:---------:|:--------:|:-------:|:-------:|:---------:|:--------:|
| 15-min | 26 | 1.04 | 0.37 | 2.3pp | **1.9pp** |
| 1-hour | 7 | 0.87 | 0.37 | 10.3pp | **1.8pp** |
| Daily | 1 | 2.27 | 0.37 | 10.4pp | **2.2pp** |

**What drives the frequency effect.** The degradation from 15-min to daily is not caused by P(stay) (which is 0.37 at all frequencies). It's caused by $\sigma$ — the noise in $\log(\text{RV})$.

- At 15-min: $\text{RV}_t = \sum_{i=1}^{26} r_{t,i}^2$ — a sum of 26 terms, well-estimated
- At daily: $\text{RV}_t = r_t^2$ — a single squared return, extremely noisy
- $\sigma_{\text{daily}} = 2.27$ vs $\sigma_{\text{15-min}} = 1.04$ — the daily estimator is 2.2× noisier

This noise inflates $\sigma$ in the Gibbs sampler, making the jump distribution too wide, which makes the predictive distribution too wide, which degrades coverage.

Historical Simulation sidesteps this entirely — it doesn't need a variance estimator at all.

---

### 5. Experiment 4: Mexican Data (Out-of-Sample)

**Setup.** We apply SF-Harris to 8 Mexican financial assets using daily $r_t^2$ as the RV proxy:

| Asset | SF-Harris | Hist.Sim | Winner |
|:------|:---------:|:--------:|:------:|
| IPC | 8.5pp | **3.7pp** | Hist.Sim |
| Bimbo | 11.7pp | **3.1pp** | Hist.Sim |
| Grupo México | 11.3pp | **2.2pp** | Hist.Sim |
| Walmart Mex | 10.4pp | **1.4pp** | Hist.Sim |
| FEMSA | 8.1pp | **3.3pp** | Hist.Sim |
| Cemex | 10.2pp | **1.6pp** | Hist.Sim |
| Banorte | 9.5pp | **2.0pp** | Hist.Sim |
| USD/MXN | 7.8pp | **1.5pp** | Hist.Sim |

Historical Simulation wins on every asset by 3–9pp. The degradation is caused by $\sigma$ inflation from the noisy $r_t^2$ estimator (only 1 observation per day).

---

### 6. Experiment 5: Crisis Indicator

**Setup.** If the Harris chain captures regime changes, then the model should be "surprised" when volatility jumps. We compute the PIT (Probability Integral Transform) — where actual volatility falls in the model's predictive distribution — and use it as a crisis detector.

**Results.**

| Metric | Value |
|:-------|:-----:|
| PIT-based AUC for crisis detection | **0.416** (below random) |
| VIX AUC for crisis detection | 0.841 |
| Pre-crisis z-score vs normal z-score | **0.46–0.70×** (less surprised) |
| P(jump) | **0.63 (constant)** |

The model is *less* surprised before crises than on average. P(jump) = 0.63 is a structural constant — it doesn't spike before regime changes. The Harris chain has no early-warning capability.

---

### 7. Experiment 6: Mixture Harris (Double-Exponential ACF)

**Setup.** Anzarut's Section 7 proposes extending SF-Harris with a mixture of two exponentials: $r(h) = w_1 e^{-\alpha_1 h} + w_2 e^{-\alpha_2 h}$, capturing fast (intraday) and slow (multi-day) volatility decay.

**Results.**

| Model | AAD | ACF $R^2$ |
|:-----|:---:|:---------:|
| Single exponential | 1.8pp | 0.968 |
| Double exponential (point) | 1.8pp | 0.972 |
| Double exponential (bootstrap) | 1.8pp | 0.972 |

The mixture improves ACF fit (+0.004 $R^2$) but adds **0.0pp** to coverage. Better temporal modeling doesn't help because temporal structure isn't what drives coverage.

---

### 8. Experiment 7: Stay/Jump Prediction

**Setup.** If the Harris chain detects regime changes, we should be able to predict whether tomorrow will be a "stay" or a "jump" day.

**Results.** Using $\tau^*$ ratio as a predictor: AUC = 0.511 (indistinguishable from random). Using recent volatility changes: AUC = 1.000 (trivially circular — observing yesterday's jump tells you it jumped).

The model's P(jump) = 0.63 is an unconditional probability, not a time-varying signal.

---

### 9. Why the Harris Chain is Decorative: The Mechanism

The Harris chain produces a sequence $\{\tau_t^*\}$ via:
$$\log(\tau_t^*) = \begin{cases} \log(\tau_{t-1}^*) & \text{w.p. } e^{-\alpha} \\ X_t \sim Q(\mu, \sigma) & \text{w.p. } 1-e^{-\alpha} \end{cases}$$

For daily coverage, what matters is the **marginal distribution** of $\tau_t^*$, not its temporal autocorrelation. The emission $r_t | \tau_t \sim N(0, \tau_t)$ uses only the current $\tau_t$, not the sequence.

The marginal distribution of $\tau_t^*$ is:
$$\tau_t^* \sim \text{Empirical}(\text{training RVs})$$

This is exactly what Historical Simulation provides. The Harris chain determines the *order* in which past volatilities are visited, but coverage only cares about the *set* of values, not their order.

**Formally:** Let $F_{\tau}$ be the CDF of $\tau^*$. Coverage at level $p$ requires $P(|r_t| \leq \sqrt{\tau^*} \cdot z_p) \approx p$, which depends on $F_{\tau}$, not on the Markov transition kernel. Two models with the same $F_{\tau}$ but different transition kernels will have identical coverage.

---

### 10. When SF-Harris IS Useful

The model is not useless — it's useful for the right thing:

| Application | Works? | Why |
|:------------|:------:|:----|
| Coverage/tail risk (with intraday data) | ✅ | Empirical $Q$ + $\tau^*$ capture distribution shape |
| VaR/ES (with intraday data) | ✅ | Same reason |
| Directional prediction | ❌ | $r = -0.003$ with future RV |
| Volatility forecasting | ❌ | VIX does this better ($r = 0.58$) |
| Crisis detection | ❌ | AUC = 0.42 (anti-predictive) |
| Regime change prediction | ❌ | P(jump) = 0.63 constant |

**SF-Harris is a structural risk engine, not a predictive model.** It tells you the *shape* of the distribution (heavy tails, correct quantiles) but not the *level* (whether volatility will be high or low tomorrow).

---

### 11. Summary of Evidence

| # | Test | Finding | Implication |
|:-:|:-----|:--------|:------------|
| 1 | Coverage decomposition | SF-Harris = iid bootstrap = N($0, \tau^*$) = 2.0pp | Harris dynamics add 0.0pp |
| 2 | Epsilon calibration | AAD flat across 8 orders of magnitude of $\epsilon$ | P(stay) doesn't affect coverage |
| 3 | Frequency comparison | AAD degrades with $\sigma$, not P(stay) | Noise in RV, not dynamics, drives performance |
| 4 | Mexican data | Hist.Sim wins by 3–9pp on 8 assets | Without intraday RV, SF-Harris degrades |
| 5 | Crisis indicator | AUC = 0.42 (anti-predictive) | No early warning from model surprise |
| 6 | Mixture Harris | Double exponential adds 0.0pp | Better ACF ≠ better coverage |
| 7 | Stay/jump prediction | AUC = 0.51 (random) | P(jump) is constant, not predictive |

**All seven tests agree: the Harris chain's temporal structure is decorative for coverage prediction. The model's value comes entirely from the marginal distribution (empirical $Q$ + $\tau^*$), which is equivalent to Historical Simulation.**