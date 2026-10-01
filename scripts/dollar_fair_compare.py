"""Fair dollar-vs-calendar comparison: leak-free rolling-threshold dollar bars (~26/day)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from anzarut_replication import (
    load_ibm_data, compute_15min_returns, split_returns_by_date,
    estimate_alpha, gibbs_gig_harris, simulate_predictive_sf_harris, compute_coverage,
    build_dollar_bars_rolling,
)
from anzarut_intraday_15min import PROB_LEVELS

df = load_ibm_data()
dol = build_dollar_bars_rolling(df, lookback_days=30, bars_per_day=26)
ndays = len(dol.index.to_series().dt.date.unique())
print("bars_total=" + str(len(dol)) + " bars_per_day=" + str(round(len(dol)/ndays, 2)) + " span=" + str(dol.index[0].date()) + "->" + str(dol.index[-1].date()))

daily_dv = df["dollar_volume"].groupby(df.index.normalize()).sum()
thr_series = (daily_dv.shift(1).rolling(30, min_periods=1).mean()/26).dropna()
print("threshold_mean=" + str(round(thr_series.mean())) + " first3=" + str([round(v) for v in thr_series.head(3)]) + " last3=" + str([round(v) for v in thr_series.tail(3)]))

ret = compute_15min_returns(df)
train_ret, _ = split_returns_by_date(ret, 0.8)
cutoff = pd.Timestamp(train_ret.index.date.max())
print("cutoff=" + str(cutoff.date()))
dol_log = np.log((dol**2).clip(lower=1e-20)).replace([np.inf,-np.inf],np.nan).dropna()
tr_full = dol_log[dol_log.index <= cutoff].values
te_full = dol_log[dol_log.index > cutoff].values

def run_arm(clean):
    if clean:
        q01, q99 = np.percentile(tr_full, 0.5), np.percentile(tr_full, 99.5)
        tr = tr_full[(tr_full>q01)&(tr_full<q99)]
        te = te_full[(te_full>q01)&(te_full<q99)]
        nrm_tr, nrm_te = len(tr_full)-len(tr), len(te_full)-len(te)
    else:
        tr, te = tr_full, te_full
        nrm_tr = nrm_te = 0
    alpha = estimate_alpha(tr)
    g = gibbs_gig_harris(tr, alpha_init=alpha["alpha_acf"], epsilon=0.1,
                         n_iter=5000, burn_in=2000, rng=np.random.default_rng(55))
    sim = simulate_predictive_sf_harris(tr, te, g, g["alpha"], Q_type="empirical",
                                        n_sim=2000, rng=np.random.default_rng(321))
    cov = compute_coverage(te, sim, PROB_LEVELS)
    aad = float(np.mean([abs(cov[p]-p*100) for p in PROB_LEVELS]))
    levels = [{"p":p,"nominal":p*100,"coverage":float(cov[p]),"deviation":float(cov[p]-p*100)} for p in PROB_LEVELS]
    return {"aad":aad,"alpha_acf":float(alpha["alpha_acf"]),"alpha_post":float(np.mean(g["alpha"])),
            "p_stay":float(np.exp(-np.mean(g["alpha"]))),"n_train":len(tr),"n_test":len(te),
            "n_rm_train":nrm_tr,"n_rm_test":nrm_te,"levels":levels}

res = {"bars_total":len(dol),"bars_per_day":len(dol)/ndays,
       "span":[str(dol.index[0].date()),str(dol.index[-1].date())],
       "threshold_mean":float(thr_series.mean()),"cutoff":str(cutoff.date()),
       "clean":run_arm(True),"no_clean":run_arm(False)}
print("RESULT_JSON " + json.dumps(res))