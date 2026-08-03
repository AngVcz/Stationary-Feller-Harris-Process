"""Stay/jump confusion matrix for SF-Harris volatility model.

Can we predict WHEN volatility will jump (regime change)?
Defines 'jump' as |log(RV_t) - log(RV_{t-1})| > threshold and evaluates
predictors: unconditional model, SF-Harris tau*, recent change.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from anzarut_replication import (
    load_ibm_data, compute_15min_returns, detect_and_remove_jumps,
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris
)

df = load_ibm_data(start_date="1998-01-01", end_date="2026-12-31")
returns = compute_15min_returns(df)
returns_clean = detect_and_remove_jumps(returns, n_passes=2, top_pct=0.001)

rv_df = returns_clean.to_frame("return")
rv_df["rv_15min"] = rv_df["return"]**2
rv_df["date"] = pd.to_datetime(rv_df.index.date)
daily_rv_all = rv_df.groupby("date")["rv_15min"].sum()
daily_rv_all = daily_rv_all[daily_rv_all > 0]

close_daily = df["close"].resample("D").last().dropna()
daily_log_ret = np.log(close_daily).diff().dropna()

common_daily = daily_rv_all.index.intersection(daily_log_ret.index)
daily_rv = daily_rv_all.loc[common_daily].values
daily_ret = daily_log_ret.loc[common_daily].values
daily_log_spot = np.log(daily_rv)
daily_log_spot = daily_log_spot[np.isfinite(daily_log_spot)]

n = len(daily_log_spot)
split = int(n * 0.8)
train_log = daily_log_spot[:split]
test_log = daily_log_spot[split:]
test_ret = daily_ret[split:split+len(test_log)]
test_rv = daily_rv[split:split+len(test_log)]
n_test = len(test_log)

# SF-Harris predictions
daily_alpha = estimate_alpha(train_log)
rng_gibbs = np.random.default_rng(111)
gibbs_result = gibbs_gig_harris(train_log, alpha_init=daily_alpha["alpha_acf"],
                                  epsilon=1e-5, n_iter=5000, burn_in=2000, rng=rng_gibbs)
rng_sim = np.random.default_rng(222)
sim_log = simulate_predictive_sf_harris(
    train_log, test_log, gibbs_result, gibbs_result["alpha"],
    Q_type="empirical", n_sim=2000, rng=rng_sim
)
n_test_actual = min(n_test, sim_log.shape[1])
sim_rv = np.exp(sim_log[:, :n_test_actual])
test_ret = test_ret[:n_test_actual]
test_rv = test_rv[:n_test_actual]
tau_median = np.median(sim_rv, axis=0)

# Log-RV changes
log_rv_changes = np.diff(daily_log_spot[split:split+n_test_actual])
abs_changes = np.abs(log_rv_changes)

alpha_val = float(np.mean(gibbs_result["alpha"])) if isinstance(gibbs_result["alpha"], np.ndarray) else gibbs_result["alpha"]
p_jump_model = 1 - np.exp(-alpha_val)

print("=" * 70)
print("STAY/JUMP CONFUSION MATRIX FOR SF-HARRIS VOLATILITY MODEL")
print("=" * 70)

print(f"\n  Dataset: {n} days, Test: {n_test_actual} days")
print(f"  SF-Harris P(jump) = {p_jump_model:.3f}, P(stay) = {1-p_jump_model:.3f}")
print(f"  Mean |delta log-RV|: {np.mean(abs_changes):.4f}")
print(f"  Median |delta log-RV|: {np.median(abs_changes):.4f}")

for thresh_name, thresh_val in [("median", np.median(abs_changes)),
                                  ("P75", np.percentile(abs_changes, 75)),
                                  ("P90", np.percentile(abs_changes, 90))]:
    is_jump = abs_changes > thresh_val
    n_jumps = np.sum(is_jump)
    n_stays = np.sum(~is_jump)

    print(f"\n{'='*70}")
    print(f"  THRESHOLD: |delta log-RV| > {thresh_val:.4f} ({thresh_name})")
    print(f"  Jumps: {n_jumps} ({n_jumps/len(is_jump):.1%}), Stays: {n_stays} ({n_stays/len(is_jump):.1%})")
    print(f"{'='*70}")

    # 1. Unconditional model
    print(f"\n  1. UNCONDITIONAL MODEL (always predict jump, P={p_jump_model:.3f}):")
    tp = n_jumps
    fp = n_stays
    fn = 0
    tn = 0
    acc = (tp + tn) / len(is_jump)
    print(f"     Predict Jump: TP={tp}, FP={fp}")
    print(f"     Predict Stay: FN={fn}, TN={tn}")
    print(f"     Accuracy={acc:.3f} (majority class baseline)")

    # 2. SF-Harris tau* ratio
    print(f"\n  2. SF-HARRIS tau* RATIO PREDICTOR:")
    tau_ratio = tau_median[1:] / tau_median[:-1]
    min_len = min(len(tau_ratio), len(is_jump))
    pred_score = np.abs(np.log(tau_ratio[:min_len]))
    actual = is_jump[:min_len]

    if np.std(pred_score) > 1e-10:
        auc = roc_auc_score(actual, pred_score)
        fpr, tpr, thresholds = roc_curve(actual, pred_score)
        j = tpr - fpr
        best_idx = np.argmax(j)
        best_thresh = thresholds[best_idx]
        pred_opt = pred_score > best_thresh
        tp = np.sum(actual & pred_opt)
        tn = np.sum(~actual & ~pred_opt)
        fp = np.sum(~actual & pred_opt)
        fn = np.sum(actual & ~pred_opt)
        acc = (tp + tn) / min_len
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        print(f"     AUC = {auc:.3f}")
        print(f"     Optimal |log(tau_ratio)| > {best_thresh:.3f}:")
        print(f"     {'':>20} Predict Jump  Predict Stay")
        print(f"     {'Actual Jump':>20}  {tp:>11}  {fn:>12}")
        print(f"     {'Actual Stay':>20}  {fp:>11}  {tn:>12}")
        print(f"     Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    # 3. Recent change predictor
    print(f"\n  3. RECENT |log-RV CHANGE| PREDICTOR:")
    recent = np.abs(np.diff(daily_log_spot[split-1:split+n_test_actual-1]))
    recent = recent[1:min_len+1] if len(recent) >= min_len else recent
    min_len3 = min(len(recent), len(is_jump))
    pred_score3 = recent[:min_len3]
    actual3 = is_jump[:min_len3]

    if np.std(pred_score3) > 1e-10 and min_len3 > 50:
        auc3 = roc_auc_score(actual3, pred_score3)
        fpr3, tpr3, thresh3 = roc_curve(actual3, pred_score3)
        j3 = tpr3 - fpr3
        best_idx3 = np.argmax(j3)
        best_thresh3 = thresh3[best_idx3]
        pred_opt3 = pred_score3 > best_thresh3
        tp3 = np.sum(actual3 & pred_opt3)
        tn3 = np.sum(~actual3 & ~pred_opt3)
        fp3 = np.sum(~actual3 & pred_opt3)
        fn3 = np.sum(actual3 & ~pred_opt3)
        acc3 = (tp3 + tn3) / min_len3
        prec3 = tp3 / (tp3 + fp3) if (tp3 + fp3) > 0 else 0
        rec3 = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else 0
        f13 = 2*prec3*rec3/(prec3+rec3) if (prec3+rec3) > 0 else 0
        print(f"     AUC = {auc3:.3f}")
        print(f"     Optimal |lagged change| > {best_thresh3:.3f}:")
        print(f"     {'':>20} Predict Jump  Predict Stay")
        print(f"     {'Actual Jump':>20}  {tp3:>11}  {fn3:>12}")
        print(f"     {'Actual Stay':>20}  {fp3:>11}  {tn3:>12}")
        print(f"     Accuracy={acc3:.3f}, Precision={prec3:.3f}, Recall={rec3:.3f}, F1={f13:.3f}")

    # 4. Random baseline
    print(f"\n  4. RANDOM BASELINE:")
    p_j = n_jumps / len(is_jump)
    print(f"     Random AUC = 0.500")
    print(f"     Majority class accuracy = {max(p_j, 1-p_j):.3f}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"\n  The SF-Harris model gives P(jump) = {p_jump_model:.3f} as an UNCONDITIONAL")
print(f"  probability. It does NOT vary by day -- every day has the same")
print(f"  {p_jump_model:.0%} chance of a volatility jump.")
print(f"\n  This means the model CANNOT predict WHICH specific days will see")
print(f"  regime changes in volatility. The P(jump) = {p_jump_model:.3f} is a")
print(f"  structural parameter (alpha = {alpha_val:.4f}), not a time-varying signal.")
print(f"\n  The confusion matrix above shows whether tau* or recent changes")
print(f"  can predict jumps better than random. AUC > 0.5 would indicate")
print(f"  some predictive power for volatility regime changes.")