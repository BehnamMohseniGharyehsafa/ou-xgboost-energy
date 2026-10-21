# Mathematical methodology

## 1. Target, origin and information set

Let $d$ denote target UTC midnight. All supplied prices with timestamps $t<d$ are assumed available then. At regular cadence $h$ hours, the target is

```math
y_d=\frac{1}{m}\sum_{j=0}^{m-1}P_{d+jh},\qquad m=24/h.
```

The code uses complete UTC days and observation/interval-start labels. It checks that every price in a target's expected grid is present. The information-set assumption must be checked against actual publication times; timestamps alone do not establish availability.

## 2. Causal features

For day $d$, the tree sees daily means lagged by 1, 2, 3, 7, 14, 21 and 28 calendar days; rolling means and sample standard deviations over 3, 7, 14 and 28 days ending at $d-1$; and the previous complete day's last price, high, low, range and within-day price-change standard deviation. The last statistic uses population standard deviation (`ddof=0`); rolling daily standard deviations use `ddof=1`.

Day-of-week and month sine/cosine features use the target calendar date, which is known in advance. No current-day observed price summary enters the feature vector. Calendar gaps remain in the index, so a seven-day lag always means seven calendar days. Any row whose required historical inputs are missing is excluded. No interpolation, whole-sample quantile clipping, standardization or global feature selection occurs.

This creates 24 features. The first 28 complete days provide warm-up rather than training rows.

## 3. Tree model

XGBoost minimizes squared loss with histogram trees, depth 5, learning rate 0.05, 250 rounds, row and feature subsampling 0.8, minimum child weight 2 and L2 regularization 1. The fallback is scikit-learn histogram gradient boosting, depth 6, learning rate 0.06, 250 iterations and minimum leaf size 20. These are different estimators; fallback results must be labelled accordingly.

Both use a fixed random seed. Tree splits do not require standardized features. Scikit-learn's early stopping is explicitly disabled to avoid an implicit randomly selected validation subset. The implementation does not perform hyperparameter search or choose the best model on the evaluation period.

## 4. OU drift estimation

The working process is

```math
dP_t=\kappa(\theta-P_t)\,dt+\sigma\,dW_t.
```

For historical consecutive observations in the window $[d-W,d)$, form

```math
r_i=\frac{P_{i+1}-P_i}{\Delta t_i}=a+bP_i+\epsilon_i.
```

The implementation fits weighted least squares with objective

```math
\min_{a,b}\sum_i\Delta t_i\,(r_i-a-bP_i)^2.
```

Weighting by $\Delta t_i$ corresponds to the Euler noise variance of the derivative being proportional to $1/\Delta t_i$. On a regular grid all weights are equal. Time is measured in hours, giving $\widehat\kappa=-\widehat b$ in hour$^{-1}$ and $\widehat\theta=\widehat a/\widehat\kappa$ in price units.

Pairs separated by more than 1.5 declared time steps are excluded. With more than 100 eligible pairs, increments above the window's 99th percentile of absolute changes are excluded. At least 20 retained pairs and a varying predictor are required. Constant prices, non-finite estimates, or $\widehat\kappa\leq10^{-10}$ are flagged as invalid. The method uses actual elapsed times; it does not assign a median time step to missing observations.

This is a **trimmed Euler approximation**, not exact maximum likelihood, an unbiased finite-sample estimator, or a jump-diffusion fit. Trimming changes the estimation target. Seasonal raw prices and jumps can distort the estimate. Diffusion $\sigma$ is not estimated, because only the conditional mean is used.

## 5. Future daily expectation

Given last observation $P_0$ at $t_0$, a constant future level $\theta_d$ yields

```math
\mathbb E[P_{t_0+s}\mid P_0]=\theta_d+(P_0-\theta_d)e^{-\kappa s}.
```

Let $\delta=d-t_0\geq0$. Integrating from target midnight for $H=24$ hours gives

```math
\mathbb E\!\left[\frac1H\int_{\delta}^{\delta+H}P_{t_0+s}\,ds\mid P_0\right]
=\theta_d+(P_0-\theta_d)A,
```

```math
A=e^{-\kappa\delta}\frac{1-e^{-\kappa H}}{\kappa H}.
```

For hourly observations ending at 23:00, $\delta=1$ hour. This explicitly accounts for the distance between the final observation and midnight. `expm1` avoids cancellation for small $\kappa H$, and the $\kappa=0$ limit returns $P_0$. Negative speeds are invalid rather than silently clamped. The continuous integral is an approximation to the discrete target in section 1.

## 6. Conditioned OU, historical OU and blend

The original model idea is retained by setting $\theta_d=f_d$, where $f_d$ is the tree forecast. Consequently

```math
o_d=f_d+(P_0-f_d)A,\qquad
\widehat y_d=wf_d+(1-w)o_d=f_d+(1-w)A(P_0-f_d).
```

This is a dynamically damped correction of the tree forecast toward the latest observed price. A faster mean reversion or a larger $w$ shrinks the correction. The two components are dependent because they share $f_d$. For valid positive speeds, $A\in[0,1]$, so the hybrid lies between $f_d$ and $P_0$.

The independent historical-level baseline substitutes $\widehat\theta$ from the drift fit for $f_d$. Both OU forecasts are missing if estimation fails; the final blend then uses $f_d$, with `blend_fallback=True` and an explicit `ou_status`.

## 7. Testable guarantees and limits

Tests perturb prices at and after an origin and verify that its feature vector and forecast do not change. Separate tests verify the analytic expectation against numerical integration and recover known Euler drift coefficients. These establish implementation properties, not empirical market accuracy. Publication delays, retrospectively revised data and market-specific calendars require additional domain-specific handling.
