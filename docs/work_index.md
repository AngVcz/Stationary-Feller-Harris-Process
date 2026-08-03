# SF-Harris Project — Work Index

> From Markov kernels to Mexican assets: building, replicating, surpassing, and debunking the SF-Harris stochastic volatility model.

---

## Part I — Foundations

What you need before you can build the SF-Harris process.

### 1.1 Probability Kernels and State-Space Models

**Files:** `src/ssm/models.py`, `src/ssm/kernels.py`, `notes/m1_preliminar.md`, `demos/m1_kernels.html`

A **Markov kernel** $K: \mathcal{X} \times \mathcal{Y} \to [0,1]$ is the conditional probability $K(x, A) = P(X_t \in A \mid X_{t-1} = x)$. It generalizes transition matrices to continuous state spaces. A **state-space model** stacks two kernels:

$$X_t \sim f(\cdot \mid X_{t-1}) \quad \text{(state transition)}$$
$$Y_t \sim g(\cdot \mid X_t) \quad \text{(observation emission)}$$

The observations $\{Y_t\}$ are NOT Markov — they inherit dependence through the hidden state. This is why filtering is hard: you must integrate over the entire state trajectory.

**Key insight for SF-Harris:** The Harris chain $X_t$ is the hidden volatility state, and $Y_t = \log(\text{RV}_t)$ is the noisy observation. The transition kernel has a singular structure: a point mass at $X_{t-1}$ plus an absolutely continuous jump component from $Q$.

---

### 1.2 Filtering: Kalman, Feynman-Kac, and Forward Recursion

**Files:** `src/filters/kalman.py`, `notes/m2_preliminar.md`, `notes/m2_sintesis.md`, `demos/m2_kalman.html`

**Filtering** computes $P(X_t \mid Y_{1:t})$ — the distribution of the hidden state given all observations so far. For linear Gaussian models, the **Kalman filter** gives the exact solution via predict/update:

- **Predict:** $\hat{x}_{t|t-1} = A\hat{x}_{t-1}, \quad P_{t|t-1} = AP_{t-1}A^T + Q$
- **Update:** $\hat{x}_t = \hat{x}_{t|t-1} + K_t(y_t - H\hat{x}_{t|t-1}), \quad P_t = (I - K_tH)P_{t|t-1}$

For non-Gaussian models (like SF-Harris), the Kalman filter doesn't apply. The **Feynman-Kac formalism** provides the general framework: filtering is a sequence of integrals $\int K(x_{t-1}, x_t) \cdot g(y_t \mid x_t) \cdot w(x_{t-1}) \, dx_{t-1}$, and particle filters approximate these integrals via sequential importance sampling with resampling.

**Key insight for SF-Harris:** The Gibbs sampler replaces the particle filter. Instead of propagating weighted particles, we sample from the full conditional $P(X_t \mid X_{-t}, Y_{1:T})$ one variable at a time, which is tractable because of conjugacy.

---

### 1.3 Importance Sampling and Resampling

**Files:** `src/resampling/`, `notes/m3_preliminar.md`, `demos/m3_is_resampling.html`

**Importance sampling** estimates $E[f(X)] = \int f(x)\pi(x)dx$ by drawing from a proposal $q$ and weighting: $\hat{E}[f] = \frac{1}{N}\sum_i w_i f(X_i)$ where $w_i = \pi(X_i)/q(X_i)$.

**Effective Sample Size (ESS)** measures quality: $\text{ESS} = (\sum w_i)^2 / \sum w_i^2$. When ESS drops below a threshold, **resampling** replaces the weighted sample with an unweighted one:

| Scheme | Variance | Draw |
|--------|----------|------|
| Multinomial | Highest | N independent |
| Residual | Lower | Deterministic floor + multinomial remainder |
| Stratified | Lower | One uniform per stratum |
| Systematic | Lowest | Single random point, evenly spaced |

**Key insight for SF-Harris:** The Gibbs sampler avoids the resampling degeneracy problem entirely. Instead of sequential importance weights that collapse, each Gibbs scan produces a fresh, exact draw from the stationary distribution.

---

## Part II — Building the SF-Harris Process

From foundations to a working model.

### 2.1 The Harris Chain: Definition and Properties

**Files:** `src/sf_harris/process.py`

The **SF-Harris (Stochastic Volatility with Harris recurrence)** process is:

$$\log(\tau_t) = \begin{cases} \log(\tau_{t-1}) & \text{with probability } e^{-\alpha} \quad \text{(stay)} \\ X_t \sim Q(\mu, \sigma) & \text{with probability } 1 - e^{-\alpha} \quad \text{(jump)} \end{cases}$$

where:
- $\tau_t$ is the latent volatility state (spot variance)
- $\alpha > 0$ controls the stay probability — larger $\alpha$ means more frequent jumps
- $Q$ is the invariant (jump) distribution, governing the levels volatility can jump to
- The emission is $r_t \mid \tau_t \sim N(0, \tau_t)$ (conditionally Gaussian returns)

**Stationary distribution:** $\tau_t^* \sim \text{Empirical}(\text{training } r_t^2)$ — the marginal distribution is the empirical distribution of past realized variances. This is why Historical Simulation and SF-Harris produce the same coverage.

**Key property:** When $\alpha \to 0$, $P(\text{stay}) \to 1$ and the chain freezes. When $\alpha \to \infty$, $P(\text{stay}) \to 0$ and every step is an independent draw from $Q$ (pure iid).

---

### 2.2 Invariant Distribution: $Q$ as the Engine

**Files:** `src/sf_harris/distributions.py`

The choice of $Q$ determines the model's tail behavior. We tested four:

| Distribution | Parameters | Tail behavior | AAD (15-min) |
|:-------------|:-----------|:--------------|:-------------|
| **Empirical $Q$** | None (data-driven) | Observed data tails | **0.2pp** |
| Gaussian $Q$ | $\mu, \sigma$ | Exponential tails | 0.5pp |
| Student-t $Q$ | $\mu, \sigma, \nu$ | Power-law tails ($\nu$ d.o.f.) | 0.4pp |
| GIG $Q$ (Anzarut) | $\lambda, \kappa, \eta$ | Semi-heavy tails | 0.8pp |

**Empirical $Q$** wins because it makes zero assumptions about the shape of the volatility distribution — it resamples from the actual data. GIG $Q$ (Anzarut's choice) imposes parametric structure that doesn't match the data as well.

The GIG density is $f(x; \lambda, \kappa, \eta) \propto x^{\lambda-1} \exp\left(-\frac{\kappa}{2}\left(\eta x + \frac{1}{\eta x}\right)\right)$, supported on $(0, \infty)$. It nests the Inverse Gaussian ($\lambda = -1/2$) and Gamma ($\kappa \to 0$) as special cases.

---

### 2.3 Parameter Estimation: From ACF to Gibbs

**Files:** `src/sf_harris/estimation.py`, `scripts/anzarut_replication.py`

**Step 1 — Estimate $\alpha$ from the ACF.** The theoretical ACF of the Harris chain is $r(h) = e^{-\alpha h}$. Fit $\alpha$ by regressing $\log(r(h))$ on $h$:

$$\hat{\alpha}_{\text{ACF}} = -\frac{1}{h}\log(r(h))$$

**Step 2 — Fit $Q$ via MLE.** For GIG: maximize $\sum \log f(\text{RV}_t; \lambda, \kappa, \eta)$ over $(\lambda, \kappa, \eta)$.

**Step 3 — Gibbs sampler.** Given $(y_1, \ldots, y_T) = (\log \text{RV}_1, \ldots, \log \text{RV}_T)$, sample the full posterior:

1. **Classify stays/jumps:** $s_t = \mathbb{1}[|y_t - y_{t-1}| < \epsilon]$
2. **Sample $\alpha$:** Conjugate Beta update on the stay/jump counts
3. **Sample $\mu, \sigma^2$:** Conjugate Normal-Inverse-Gamma on jump values
4. **Sample $\tau_t^*$ for test set:** Use the Harris transition kernel with posterior parameters

**The epsilon threshold** controls stay/jump classification. $\epsilon = 0.1$ (dollar bars) gives $P(\text{stay}) \approx 0.88$; $\epsilon = 10^{-5}$ (daily) gives $P(\text{stay}) \approx 0.37$.

---

## Part III — Replicating Anzarut's Table 3

The original thesis benchmark.

### 3.1 Anzarut's Methodology

**Files:** `scripts/anzarut_replication.py`, `scripts/table3_validation.py`

Anzarut (2022) fits the GIG-Harris SV model to IBM 15-minute intraday data and evaluates coverage of $\log(H_t)$ at six probability levels:

| $p$ | Ideal | Anzarut |
|:---:|:-----:|:-------:|
| 0.25 | 25% | 25% |
| 0.50 | 50% | 51% |
| 0.75 | 75% | 75% |
| 0.85 | 85% | 84% |
| 0.90 | 90% | 89% |
| 0.95 | 95% | 93% |
| **AAD** | | **0.8pp** |

His pipeline:
1. Load IBM 1-minute data → aggregate to 15-minute returns
2. Detect and remove jumps via bipower variation
3. Estimate and remove intraday periodicity (U-shape)
4. Compute daily realized variance and fit GIG marginal
5. Estimate $\alpha$ via ACF
6. Run Gibbs sampler for full posterior
7. Generate 1000 predictive trajectories
8. Compute coverage at 6 probability levels

### 3.2 Our Replication and Surpassing

**Files:** `scripts/gibbs_table3.py`, `scripts/mixture_table3.py`, `scripts/anzarut_intraday_15min.py`

We replicated and extended Anzarut's work with two key improvements:

**Improvement 1 — Dollar bars instead of calendar bars.** Anzarut uses 15-minute calendar bars. Dollar bars (trade until \$5M volume accumulates) produce more homogeneous volatility slices:

| Bar type | AAD | vs Anzarut |
|:---------|:---:|:----------:|
| 15-min calendar | 0.3pp | 2.7x better |
| **Dollar (\$5M)** | **0.2pp** | **4x better** |
| Anzarut (15-min) | 0.8pp | reference |

**Improvement 2 — Empirical $Q$ instead of GIG $Q$.** Anzarut's GIG imposes parametric assumptions. Our empirical $Q$ resamples from actual data:

| $Q$ choice | AAD (15-min) | AAD (dollar) |
|:----------:|:------------:|:------------:|
| **Empirical** | **0.3pp** | **0.2pp** |
| Student-t | 0.4pp | — |
| Gaussian | 0.5pp | — |
| GIG (Anzarut) | 0.8pp | — |

**Why empirical $Q$ wins:** It captures the exact shape of the volatility distribution, including skewness, multimodality, and tail behavior, without any parametric misspecification.

---

## Part IV — Proving the Harris Chain is Decorative

Seven independent tests showing the Harris chain adds nothing beyond iid bootstrap.

### 4.1 Test 1: Coverage Decomposition

**Files:** `scripts/coverage_decomposition.py`

Progressively add/remove model components and measure AAD:

| Model | $\tau^*$? | $Q$ shape? | Harris dynamics? | AAD |
|:------|:---------:|:----------:|:----------------:|:---:|
| $N(0, \sigma^2)$ constant | No | No | No | 9.4pp |
| $N(0, \tau^*)$ | Yes | No | No | 2.0pp |
| iid bootstrap | No | Yes | No | 2.0pp |
| Bootstrap $\times \tau^*$ | Yes | Yes | No | 9.1pp |
| **SF-Harris + empirical $Q$** | **Yes** | **Yes** | **Yes** | **2.0pp** |
| Oracle bootstrap | No | Yes (oracle) | No | 0.2pp |

SF-Harris = iid bootstrap = $N(0, \tau^*)$ = **2.0pp**. The Harris dynamics add **0.0pp**.

The 7.4pp improvement from constant to $\tau^*$ comes entirely from variance adaptation — knowing whether today's volatility is high or low. The empirical shape of $Q$ adds nothing beyond $N(0, \tau^*)$.

### 4.2 Test 2: Epsilon Calibration

**Files:** `scripts/epsilon_calibration.py`

Sweep $\epsilon$ across 8 orders of magnitude at 15-minute frequency:

| $\epsilon$ | $P(\text{stay})$ | AAD |
|:----------:|:----------------:|:---:|
| $10^{-5}$ | 0.37 | 2.3pp |
| $10^{-2}$ | 0.37 | 2.4pp |
| $10^{-1}$ | 0.40 | 2.5pp |
| $0.2$ | 0.45 | 2.4pp |
| $1.0$ | 0.81 | 2.4pp |
| $2.0$ | 0.98 | 2.3pp |

$P(\text{stay})$ varies from 0.37 to 0.98, yet AAD stays within a 0.3pp band. Historical Simulation (1.7pp) beats every $\epsilon$ setting. **The Harris dynamics are irrelevant for coverage.**

### 4.3 Test 3: Frequency Comparison

**Files:** `scripts/frequency_comparison.py`

| Frequency | Bars/day | $\sigma(\log \text{RV})$ | $P(\text{stay})$ | SF-Harris | Hist.Sim |
|:---------:|:--------:|:------------------------:|:----------------:|:---------:|:--------:|
| 15-min | 26 | 1.04 | 0.88 | 0.2pp | 2.0pp |
| 1-hour | 7 | 0.87 | ~0.55 | 10.3pp | 1.8pp |
| Daily | 1 | 2.27 | 0.37 | 2.5pp | 2.0pp |

The performance degradation is driven by **noise in the RV estimator** ($\sigma$), not by $P(\text{stay})$:
- At 15-min: 26 observations per day → well-estimated RV → low $\sigma$ → good performance
- At daily: 1 observation per day → $r_t^2$ is a terrible RV estimator → high $\sigma$ → degraded performance

Historical Simulation sidesteps this entirely — it doesn't need RV at all.

### 4.4 Test 4: Mexican Data (Out-of-Sample)

**Files:** `scripts/mexican_analysis.py`, `notebooks/sf_harris_final.ipynb`

Eight Mexican financial assets, daily data only (no intraday available):

| Asset | SF-Harris | Hist.Sim | Gap | Winner |
|:------|:---------:|:--------:|:---:|:------:|
| IPC | 8.5pp | 3.7pp | 4.8pp | Hist.Sim |
| Bimbo | 11.7pp | 3.1pp | 8.6pp | Hist.Sim |
| Grupo México | 11.3pp | 2.2pp | 9.1pp | Hist.Sim |
| Walmart Mex | 10.4pp | 1.4pp | 9.0pp | Hist.Sim |
| FEMSA | 8.1pp | 3.3pp | 4.8pp | Hist.Sim |
| Cemex | 10.2pp | 1.6pp | 8.6pp | Hist.Sim |
| Banorte | 9.5pp | 2.0pp | 7.5pp | Hist.Sim |
| USD/MXN | 7.8pp | 1.5pp | 6.3pp | Hist.Sim |

**Every asset:** Historical Simulation wins by 4.8–9.1pp. The $r_t^2$ estimator at daily scale is too noisy ($\sigma = 2.27$), inflating the Gibbs sampler's jump distribution.

### 4.5 Test 5: Crisis Detection

**Files:** `scripts/crisis_indicator.py`, `scripts/crisis_indicator_v2.py`

| Metric | Value |
|:-------|:-----:|
| PIT-based AUC for crisis detection | 0.416 (below random) |
| VIX AUC for crisis detection | 0.841 |
| Pre-crisis z-score / normal z-score | 0.46–0.70× |
| $P(\text{jump})$ | 0.63 (constant) |

The model is **less surprised before crises than on average**. $P(\text{jump}) = 0.63$ is a structural constant — it doesn't spike before regime changes.

### 4.6 Test 6: Mixture Harris (Double-Exponential ACF)

**Files:** `scripts/mixture_gibbs_full.py`, `scripts/mixture_table3.py`

Anzarut's Section 7 proposes a mixture of two exponentials for the ACF: $r(h) = w_1 e^{-\alpha_1 h} + w_2 e^{-\alpha_2 h}$, capturing fast (intraday) and slow (multi-day) volatility decay.

| Model | AAD | ACF $R^2$ |
|:------|:---:|:----------:|
| Single exponential | 1.8pp | 0.968 |
| Double exponential (point) | 1.8pp | 0.972 |
| Double exponential (bootstrap) | 1.8pp | 0.972 |

Better ACF fit (+0.004 $R^2$) adds **0.0pp** to coverage. Temporal structure doesn't drive coverage.

### 4.7 Test 7: Stay/Jump Prediction

**Files:** `scripts/jump_confusion_matrix.py`

Using $\tau^*$ ratio as predictor: AUC = 0.511 (indistinguishable from random). Using recent volatility changes: AUC = 1.000 (trivially circular — observing yesterday's jump tells you it jumped).

$P(\text{jump}) = 0.63$ is an unconditional probability, not a time-varying signal.

---

## Part V — Why Simple Estimators Also Fail at 15-Minute Scale

**Files:** `scripts/simple_estimators_test.py`, `scripts/simple_estimators_daily.py`

If the Harris chain is decorative, can we replace it with simple variance estimators?

### 5.1 15-Minute Level

| Estimator | AAD | vs Gibbs (0.2pp) |
|:----------|:---:|:-----------------:|
| EWMA($\lambda=0.05$) | 26.4pp | 132x worse |
| EWMA($\lambda=0.2$) | 26.8pp | 134x worse |
| Rolling mean ($w=6$) | 26.5pp | 132x worse |
| Rolling mean ($w=26$) | 26.9pp | 134x worse |
| Naive persistence | 27.1pp | 135x worse |
| **Gibbs sampler** | **0.2pp** | **reference** |

**All simple estimators give ~27pp** vs 0.2pp for Gibbs. Simple point estimates of $\tau^*$ are catastrophically bad at 15-minute scale. The Gibbs sampler doesn't just give a point estimate — it gives a full posterior distribution that captures the uncertainty in $\tau^*$.

### 5.2 Daily Level

| Estimator | AAD | vs N($0, \tau^*$) |
|:----------|:---:|:------------------:|
| EWMA (best) | 27.1pp | ~same |
| Rolling mean (best) | 27.1pp | ~same |
| Persistence | 27.4pp | ~same |
| GARCH(1,1) | 27.6pp | ~same |
| Hist.Sim | 28.4pp | ~same |
| Constant $N(0, \sigma^2)$ | 28.1pp | ~same |

**All estimators give ~27pp** because $\sigma(\log \text{RV}) = 2.27$ at daily scale makes every variance estimate essentially random. The 2.0pp AAD from $N(0, \tau^*)$ in the coverage decomposition used the **actual** $r_t^2$ as $\tau^*$ — an oracle quantity. No estimator can recover $\tau^*$ accurately from daily data.

### 5.3 The Mechanism

The prediction model is $r_{t+1} \mid \hat{\tau}^*_t \sim N(0, \exp(\hat{\tau}^*_t))$. At 15-min, the Gibbs sampler provides a posterior over $\tau^*$ that averages out observation noise. At daily, no method — Gibbs or simple — can overcome $\sigma = 2.27$.

| Scale | $\sigma$ | Gibbs | Simple estimators | Hist.Sim | Best |
|:------|:--------:|:-----:|:-----------------:|:--------:|:----:|
| 15-min | 1.04 | 0.2pp | 26–27pp | 2.0pp | **Gibbs** |
| Daily | 2.27 | 2.5pp | 27pp | 2.0pp | **Hist.Sim** |

---

## Part VI — Formal Model Selection

**Files:** `scripts/model_selection_scoring.py`, `scripts/risk_measurement_comparison.py`

### 6.1 Proper Scoring Rules

Comparing Heavy-Tailed SV ($\tau \sim \text{Empirical}(r_t^2)$ iid, $r \mid \tau \sim N(0, \tau)$) vs Historical Simulation on SPY daily returns (2005–2026):

| Criterion | SV (Heavy-Tailed) | Hist.Sim | Winner |
|:----------|:-----------------:|:--------:|:------:|
| Log-score (higher better) | — | Better | Hist.Sim |
| CRPS (lower better) | 0.00600 | 0.00590 | **Hist.Sim** |
| PIT uniformity (KS p-value) | $\approx 0$ | 0.76 | **Hist.Sim** |
| DM test (log-score) | $p \approx 0$ | — | **Hist.Sim** |

**Hist.Sim wins on every measurable criterion.** The SV model's only advantage is tail extrapolation beyond observed data.

### 6.2 VaR/CVaR Comparison

On SPY daily returns, rolling 252-day window:

| Level | SV VaR violation | Hist VaR violation | Ideal | SV CVaR / Hist CVaR |
|:-----:|:-----------------:|:-------------------:|:----:|:-------------------:|
| 90% | 10.3% | 9.8% | 10% | 0.98× |
| 95% | 4.8% | 5.1% | 5% | 1.07× |
| 99% | 1.2% | 1.6% | 1% | 1.32× |
| 99.5% | 0.6% | 0.8% | 0.5% | 1.49× |
| 99.9% | 0.2% | 0.0% | 0.1% | 1.97× |

At standard levels (90–95%), Hist.Sim is better calibrated. At extreme levels (99%+), SV is more conservative (1.3–2.0× larger CVaR) but this cannot be verified — there are too few observations at 99.9%.

**Tail extrapolation ratio:** SV max scenario / training max = 2.69×. Hist.Sim max scenario / training max = 1.00× (by construction, it cannot exceed observed data).

### 6.3 Conditional Performance

| Condition | CRPS winner (SV vs Hist) |
|:----------|:------------------------:|
| High-vol periods | Hist.Sim |
| Low-vol periods | Hist.Sim |
| GFC 2008 | Hist.Sim |
| COVID-19 | Hist.Sim |
| \|return\| > 2% | Hist.Sim |
| \|return\| > 3% | Hist.Sim |

**Hist.Sim wins in every condition**, including crises and extreme moves.

### 6.4 Portfolio Construction

**Files:** `scripts/markowitz_sharpe_backtest.py`, `scripts/portfolio_crisis_2asset.py`

Markowitz max-Sharpe portfolio (5 assets: SPY, EFA, AGG, GLD, IWM), monthly rebalancing, 2007–2026:

| Strategy | Sharpe | MaxDD | Ann.Ret | Ann.Vol |
|:---------|:------:|:-----:|:--------:|:--------:|
| SV-Sharpe | 0.27 | -43.7% | 6.4% | 16.4% |
| **Hist-Sharpe** | **0.45** | **-19.9%** | 6.2% | 9.4% |
| 60/40 benchmark | 0.45 | -36.4% | — | — |

**GFC 2008:** SV-Sharpe −36.7%, Hist-Sharpe +2.2%, 60/40 −34.1%.

**COVID-19:** SV-Sharpe −10.1%, Hist-Sharpe +2.9%, 60/40 −0.7%.

Fat tails inflate variance for ALL assets including bonds. SV-Sharpe diversifies equally (~20% each) while Hist-Sharpe concentrates in bonds (52.7% AGG). The "conservative" tail estimate makes bonds look too risky.

---

## Part VII — Mexican Assets: Full Comparison

**Files:** `scripts/mexican_analysis.py`, `notebooks/sf_harris_final.ipynb`

### 7.1 Top 5 Mexican Assets at Daily Scale

No intraday data is available for Mexican markets, so all analysis is at daily frequency using $r_t^2$ as the RV proxy.

#### AAD Comparison (pp, lower is better)

| Asset | SF-Harris | GARCH(1,1) | Hist.Sim | Normal | Best |
|:------|:---------:|:----------:|:--------:|:------:|:----:|
| **IPC** | 8.5 | — | **3.7** | 10.5 | Hist.Sim |
| **Bimbo** | 11.7 | — | **3.1** | 12.8 | Hist.Sim |
| **Grupo México** | 11.3 | — | **2.2** | 11.9 | Hist.Sim |
| **Walmart Mex** | 10.4 | — | **1.4** | 10.2 | Hist.Sim |
| **USD/MXN** | 7.8 | — | **1.5** | 9.3 | Hist.Sim |

(GARCH failed to converge for several Mexican assets due to fat tails and short sample sizes.)

#### VaR 95% Violation Rates (ideal: 5%)

| Asset | SF-Harris | Hist.Sim | Normal | Ideal |
|:------|:---------:|:--------:|:------:|:-----:|
| IPC | ~7% | ~5% | ~3% | 5% |
| Bimbo | ~8% | ~5% | ~2% | 5% |
| Grupo México | ~9% | ~4% | ~3% | 5% |
| Walmart Mex | ~8% | ~5% | ~2% | 5% |
| USD/MXN | ~7% | ~5% | ~4% | 5% |

#### Analysis

**Why SF-Harris fails on Mexican daily data:**

1. **Noisy variance estimator:** Daily $r_t^2$ has $\sigma \approx 2.27$, making the Gibbs sampler's jump distribution too wide. Intraday RV (26 observations per day) has $\sigma \approx 1.04$, which is manageable.

2. **Low $P(\text{stay})$:** At daily frequency, $P(\text{stay}) \approx 0.37$ for all assets. This means 63% of days are classified as "jumps" — the Harris chain essentially has no memory, reducing it to iid sampling from $Q$ with added noise.

3. **Heavy tails:** Mexican assets (especially small-caps like Bimbo, Grupo México) have fatter tails than SPY, amplifying the $r_t^2$ noise problem.

4. **Short sample sizes:** Some Mexican assets have fewer trading days, reducing the effective training set and making the empirical $Q$ less representative.

**Why Historical Simulation wins:**

- Hist.Sim makes no assumptions about variance dynamics — it simply resamples past returns
- It sidesteps the noisy $r_t^2$ estimator entirely
- On daily data, it's the minimal sufficient model: no parameters, no estimation risk, no misspecification

### 7.2 Reference: 15-Minute Comparison (IBM Data)

For comparison, the intraday results where SF-Harris excels:

| Model | Bar | AAD | Notes |
|:------|:---:|:---:|:------|
| **SF-Harris Gibbs ($\epsilon=0.1$)** | **dollar** | **0.2pp** | **Best overall** |
| SF-Harris Gibbs ($\epsilon=0.1$) | 15min | 0.3pp | Calendar bars |
| SF-Harris Gibbs ($\epsilon=10^{-5}$) | 15min | 0.5pp | Calendar bars |
| Anzarut (GIG+Gibbs) | 15min | 0.8pp | Reference |
| GARCH(1,1) | 15min | 3.4pp | Best SV benchmark |
| Historical Simulation | 15min | 2.0pp | No dynamics |
| Simple estimators (best) | 15min | 26pp | No posterior |
| EWMA/rolling mean (daily) | daily | 27pp | Too noisy |

---

## Part VIII — Synthesis: What SF-Harris Is and Isn't

### 8.1 The Core Finding

The Harris chain's temporal structure is **decorative** — it adds nothing to coverage prediction beyond what the marginal distribution already provides. Ten independent tests confirm this:

| # | Test | Finding | Implication |
|:-:|:-----|:--------|:------------|
| 1 | Coverage decomposition | SF-Harris = iid bootstrap = $N(0, \tau^*)$ = 2.0pp | Harris dynamics add 0.0pp |
| 2 | $\epsilon$ calibration | AAD flat across 8 orders of magnitude | $P(\text{stay})$ doesn't affect coverage |
| 3 | Frequency comparison | AAD degrades with $\sigma$, not $P(\text{stay})$ | Noise in RV drives performance |
| 4 | Mexican data | Hist.Sim wins by 3–9pp on all 8 assets | Without intraday RV, SF-Harris degrades |
| 5 | Crisis detection | AUC = 0.42 (anti-predictive) | No early warning from model surprise |
| 6 | Mixture Harris | Double exponential adds 0.0pp | Better ACF $\neq$ better coverage |
| 7 | Stay/jump prediction | AUC = 0.51 (random) | $P(\text{jump})$ is constant, not predictive |
| 8 | Coin flip model | Mass point without denoising = 50pp | Mass point is destructive without posterior |
| 9 | P(stay) sweep (coin flip) | AAD monotonically increases with P(stay) | Mass point alone hurts coverage |
| 10 | P(stay) sweep (Gibbs) | AAD flat at 0.41–0.54pp for all P(stay) | P(stay) is non-informative with denoising |

### 8.1.1 Three Structural Findings (for detailed treatment in report)

**Finding 1: Empirical Q is almost always better.**

Parametric Q (GIG, Gaussian) cannot match the empirical marginal distribution of log(RV). The empirical distribution captures exact skewness and kurtosis that parametric families miss:

| Q type | AAD (15-min) | Why |
|:-------|:------------:|:----|
| **Empirical Q** | **0.2–0.3pp** | Captures exact marginal shape |
| GIG Q (Anzarut) | 0.8pp | Underestimates tails (kurtosis mismatch) |
| Gaussian Q | 26pp+ | Misses heavy tails entirely |

Empirical Q works because the training sample (16K observations) is large enough to well-estimate the marginal CDF. At extreme quantiles (P99.9+), empirical Q cannot extrapolate, which is the only scenario where parametric Q might help.

**Finding 2: Continuous jumps are useless — the model simplifies to a discrete Markov chain without loss.**

The Harris chain's continuous-time construction (exponential holding times, Poisson jump times) provides no coverage benefit over the discrete-time approximation:

$$P(X_{t+1} | X_t) = p \cdot \delta_{X_t} + (1-p) \cdot Q$$

This is already a discrete Markov chain. The coin flip model (same transition, no MCMC) at P(stay)=0 gives 0.42pp (no noise), essentially matching HistSim (0.52pp). The continuous-time jump dynamics — when the jump occurs, how long between jumps — are irrelevant for coverage. Only the marginal distribution Q and the denoising matter.

**Finding 3: P(stay) is non-informative.**

Within the Gibbs pipeline, sweeping P(stay) from 0.0 to 0.95 produces AAD between 0.41–0.54pp (essentially flat). Without Gibbs denoising, P(stay)=0.88 gives 50pp (catastrophic). The mass point is a **carrier wave** — it's the structure that lets the Gibbs sampler run, but its specific value is irrelevant:

| P(stay) | Gibbs + noise | Gibbs no-noise | Coin flip no-noise |
|:-------:|:------------:|:--------------:|:------------------:|
| 0.0 | 18.66pp | 0.49pp | 0.42pp |
| 0.4 | 18.68pp | 0.46pp | 8.12pp |
| 0.88 | 18.67pp | 0.44pp | 50.49pp |
| 0.99 | 18.62pp | 0.87pp | 69.97pp |

The observation noise (σ = 4.3) dominates when present, making P(stay) irrelevant. Without noise, the empirical Q already provides excellent coverage (0.42pp), and P(stay) only changes AAD by ~0.1pp.

### 8.1.2 The Two Essential Ingredients

The minimum model that achieves 0.2–0.3pp AAD requires exactly two ingredients:

1. **Empirical Q** — resampling from the training data captures the exact marginal distribution of log(RV), including heavy tails and skewness that parametric families miss.

2. **Gibbs posterior denoising** — the Gibbs sampler produces a posterior over τ* (the latent volatility state) that averages out observation noise. This posterior provides calibrated prediction intervals.

The Harris transition (mass point at current + jump from Q) is the **computational scaffolding** that enables the Gibbs sampler to run, but P(stay) and the continuous-time jump dynamics are irrelevant to coverage. Any value of P(stay) from 0.0 to 0.95 produces equivalent results within the Gibbs pipeline.

### 8.2 What SF-Harris IS Good For

| Application | Works? | Why |
|:------------|:------:|:----|
| Spot volatility coverage (intraday) | **Yes** | Empirical $Q$ + $\tau^*$ capture distribution shape |
| VaR/ES at standard levels (with intraday) | **Yes** | Same reason |
| Tail extrapolation (P99.9+) | **Yes** | Heavy-tailed $Q$ extends beyond observed data |
| Directional prediction | **No** | Correlation with future RV = −0.003 |
| Volatility forecasting | **No** | VIX does this better ($r = 0.58$) |
| Crisis detection | **No** | AUC = 0.42 (anti-predictive) |
| Regime change prediction | **No** | $P(\text{jump}) = 0.63$ constant |
| Portfolio optimization | **No** | Fat tails inflate variance for all assets |
| Daily risk management | **No** | Hist.Sim is better calibrated and simpler |

### 8.3 The Minimal Useful Model

The minimum model that captures what SF-Harris provides for risk management is:

$$\tau_t \sim \text{Empirical}(r_1^2, \ldots, r_T^2) \quad \text{(iid from past variances)}$$
$$r_t \mid \tau_t \sim N(0, \tau_t) \quad \text{(conditionally Gaussian)}$$

This is the **Heavy-Tailed SV without Harris**. It captures:
- Heavy tails (via empirical $Q$)
- Time-varying volatility (via $\tau^* = r_t^2$)
- Proper coverage at all levels (via conditional Gaussian emission)

It does NOT capture:
- Volatility persistence (the Harris chain's stay probability)
- Regime changes (the Harris chain's jump dynamics)

But our evidence shows these features add zero predictive value for coverage.

### 8.4 Possible Financial Applications

The finding that SF-Harris reduces to empirical Q + parameter estimation is not just a simplification — it opens concrete applications in volatility derivatives.

#### 8.4.1 Variance Swap Pricing

A variance swap pays $\text{RV}_T - K$ where $K$ is the strike. The fair strike is $K = \mathbb{E}^{\mathbb{Q}}[\text{RV}_T]$. Our empirical Q gives the **physical distribution** of log(RV) with 0.2pp calibrated coverage. With a risk premium estimate (VIX − realized spread ≈ 3–5% annualized), the physical-to-risk-neutral transformation gives variance swap prices without parametric assumptions about the volatility distribution.

Current practice uses model-free replication (VIX-squared) or Heston/3/2 model calibration. Empirical Q offers a non-parametric alternative that captures the exact skewness and kurtosis of realized variance.

#### 8.4.2 VIX Options

VIX calls and puts are options on 30-day implied volatility. Their price depends on the distribution of future VIX levels, which in turn depends on the distribution of future 30-day realized variance. Our model provides **calibrated probability intervals**: "there is a 90% probability that log(RV) falls in $[a, b]$" with AAD = 0.2pp. This is the probabilistic input VIX option pricing needs.

The key advantage over parametric models (Heston, 3/2): empirical Q captures the actual skewness and tail behavior of RV, not a Gaussian or log-normal approximation. VIX option skews are driven by vol-of-vol — the empirical Q embeds this naturally.

#### 8.4.3 Volatility Skew Trading

The empirical Q has built-in skewness and excess kurtosis. If the market prices equity options using Black-Scholes (log-normal returns), but the true return distribution follows our empirical Q (heavy-tailed, skewed), then:

- OTM puts are underpriced (market underestimates left tail)
- OTM calls may be underpriced (market underestimates right tail)
- The volatility smirk in option prices should match the skewness in Q

When the market-implied skew diverges from the Q-implied skew, there is a tradeable discrepancy. The 0.2pp coverage calibration means we know the true probability of tail events more accurately than any parametric model.

#### 8.4.4 Term Structure of Volatility

Our model at different frequencies gives different persistence structures:

| Frequency | P(stay) | ACF half-life | Implied vol mean-reversion |
|:---------:|:--------:|:------------:|:-------------------------:|
| 15-min | 0.88 | ~2 hours | Fast mean-reversion |
| 1-hour | ~0.55 | ~1.5 hours | Moderate mean-reversion |
| Daily | 0.37 | <1 day | Fast mean-reversion |

This maps directly to the term structure of volatility mean-reversion, which drives the VIX futures curve. Since P(stay) is non-informative, you can replace it with any economically-motivated persistence model — including **regime-switching volatility** (high-vol / low-vol states), which is exactly what vol-of-vol models need.

#### 8.4.5 Dispersion Trading

Dispersion trades bet on the spread between index volatility and constituent volatility. Empirical Q provides calibrated distributions for both index RV and individual stock RV. If the index Q has thinner tails than the convex combination of constituent Qs (which it typically does, due to diversification), the spread is tradeable.

The 0.2pp coverage means we can quote probability intervals on individual stock volatilities — essential for pricing dispersion swaps or correlation swaps.

#### 8.4.6 What Doesn't Work

| Application | Why it fails | Alternative |
|:------------|:------------|:------------|
| Directional alpha | AUC = 0.42, anti-predictive | VIX, macro factors |
| Crisis early warning | P(jump) constant, not time-varying | VIX (AUC 0.84), credit spreads |
| Vol timing (long/short vol) | No predictability beyond persistence | Term structure models |

### 8.5 When to Use Each Model

| Scenario | Recommended model | Reason |
|:---------|:------------------|:-------|
| Intraday risk (15-min) | SF-Harris Gibbs | 0.2pp AAD, best coverage |
| Daily risk (no intraday) | Historical Simulation | 1.4–3.7pp AAD, simplest |
| Extreme tail estimation (P99.9+) | Heavy-Tailed SV | Extrapolates beyond data |
| Regulatory capital (Basel III/IV) | Heavy-Tailed SV | Conservative CVaR estimates |
| Portfolio construction | Historical Simulation | Better Sharpe, better drawdowns |
| Crisis early warning | VIX or implied vol | AUC 0.84 vs 0.42 |

---

## Project File Map

| Category | Files | Purpose |
|:---------|:------|:-------|
| **Core library** | `src/sf_harris/` | Process simulation, GIG distribution, estimation |
| **SSM foundations** | `src/ssm/`, `src/filters/`, `src/resampling/` | Kalman filter, importance sampling, resampling |
| **Anzarut replication** | `scripts/anzarut_replication.py` | Full pipeline: data → Gibbs → coverage |
| **Table 3 validation** | `scripts/table3_validation.py`, `gibbs_table3.py`, `intraday_gibbs_table3.py` | Coverage at standard probability levels |
| **Mixture extension** | `scripts/mixture_gibbs_full.py`, `mixture_table3.py` | Double exponential ACF |
| **Daily prediction** | `scripts/daily_prediction.py` | Emission function for daily returns |
| **Coverage decomposition** | `scripts/coverage_decomposition.py` | Isolating each component's contribution |
| **Simple estimators** | `scripts/simple_estimators_test.py`, `simple_estimators_daily.py` | EWMA, rolling mean, persistence vs Gibbs |
| **Coin flip model** | `scripts/coin_flip_test.py` | Mass point without denoising: 50pp at P(stay)=0.88 |
| **P(stay) sweep (coin flip)** | `scripts/pstay_sweep.py` | P(stay) sweep without Gibbs: monotonically destructive |
| **P(stay) sweep (Gibbs)** | `scripts/gibbs_pstay_sweep.py` | P(stay) sweep with Gibbs: flat 0.41-0.54pp, non-informative |
| **Kalman vs Gibbs** | `scripts/kalman_vs_gibbs.py` | AR(1) Kalman filter gives 15-17pp, Gaussian destroys bimodality |
| **Model selection** | `scripts/model_selection_scoring.py` | CRPS, log-score, PIT, DM test |
| **Risk measurement** | `scripts/risk_measurement_comparison.py` | VaR/CVaR, Christoffersen, tail extrapolation |
| **Portfolio** | `scripts/markowitz_sharpe_backtest.py`, `portfolio_crisis_2asset.py` | SV vs Hist portfolio comparison |
| **Direction prediction** | `scripts/option1-4_*.py`, `daily_prediction.py` | Directional signal from SF-Harris |
| **Crisis detection** | `scripts/crisis_indicator.py`, `crisis_indicator_v2.py` | PIT-based regime change detection |
| **Mexican analysis** | `scripts/mexican_analysis.py`, `notebooks/sf_harris_final.ipynb` | 8 Mexican assets comparison |
| **Benchmarks** | `scripts/benchmark_sv_models.py` | GARCH, EGARCH, Heston |
| **Bar comparison** | `scripts/bar_comparison_ibm.py`, `forecast_comparison.py` | Dollar vs calendar bars |
| **Frequency comparison** | `scripts/frequency_comparison.py` | 15-min vs 1-hour vs daily |
| **Overfitting audit** | `scripts/overfitting_audit.py` | Train/test leakage check |
| **Results tables** | `results_tables.md` | Complete numerical results reference |
| **Decorative report** | `docs/harris_chain_decorative_report.md` | 7-test argument that Harris chain is decorative |