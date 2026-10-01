"""Count cells with AAD<0.8 (both IS and OOS) by bar-type, and pairwise win/loss
dollar vs calendar by (pct, bpd-tier index)."""
import json
from pathlib import Path

d = json.loads(Path("docs/grid_is_oos.json").read_text())
pcts = d["pcts"]
cal = d["calendar"]
dol = d["dollar"]

# --- 1. cells with both IS<0.8 and OOS<0.8 ---
def filt(rows):
    return [r for r in rows if r["aad_tr"] < 0.8 and r["aad_te"] < 0.8]

cal_ok = filt(cal)
dol_ok = filt(dol)
print(f"cells with IS<0.8 AND OOS<0.8:  calendar={len(cal_ok)}  dollar={len(dol_ok)}  total={len(cal_ok)+len(dol_ok)}")

# --- 2. pairwise win/loss: match by pct and bpd-tier index ---
cal_bpd = sorted({r["bpd"] for r in cal})
dol_bpd = sorted({r["bpd"] for r in dol})
print(f"\ncalendar bpd tiers: {cal_bpd}")
print(f"dollar   bpd tiers: {dol_bpd}")

def by_pct_bpd(rows, bpd):
    return {(r["pct"], r["bpd"]): r for r in rows}

cm = by_pct_bpd(cal, cal_bpd)
dm = by_pct_bpd(dol, dol_bpd)

# win criterion = OOS AAD (aad_te) lower = wins
dol_wins = cal_wins = ties = 0
dol_win_cells = []
cal_win_cells = []
for pct in pcts:
    for i in range(min(len(cal_bpd), len(dol_bpd))):
        c = cm.get((pct, cal_bpd[i]))
        dd = dm.get((pct, dol_bpd[i]))
        if not c or not dd:
            continue
        co, do = c["aad_te"], dd["aad_te"]
        if do < co - 1e-9:
            dol_wins += 1
            dol_win_cells.append((pct, cal_bpd[i], dol_bpd[i], co, do))
        elif co < do - 1e-9:
            cal_wins += 1
            cal_win_cells.append((pct, cal_bpd[i], dol_bpd[i], co, do))
        else:
            ties += 1

print(f"\n--- pairwise by (pct, bpd-tier), criterion = OOS AAD lower wins ---")
print(f"dollar wins: {dol_wins}   calendar wins: {cal_wins}   ties: {ties}")

# repeat with both-in-sub08 filter on the pair
dol_w2 = cal_w2 = 0
for pct in pcts:
    for i in range(min(len(cal_bpd), len(dol_bpd))):
        c = cm.get((pct, cal_bpd[i]))
        dd = dm.get((pct, dol_bpd[i]))
        if not c or not dd:
            continue
        if not (c["aad_tr"] < 0.8 and c["aad_te"] < 0.8 and dd["aad_tr"] < 0.8 and dd["aad_te"] < 0.8):
            continue
        if dd["aad_te"] < c["aad_te"] - 1e-9:
            dol_w2 += 1
        elif c["aad_te"] < dd["aad_te"] - 1e-9:
            cal_w2 += 1
print(f"\n--- restricted to pairs where BOTH cells are sub-0.8 (IS&OOS) ---")
print(f"dollar wins: {dol_w2}   calendar wins: {cal_w2}")

# also: among the sub-0.8 population, how many dollar cells beat the calendar median OOS
import statistics
cal_oos = [r["aad_te"] for r in cal_ok]
dol_oos = [r["aad_te"] for r in dol_ok]
print(f"\nOOS AAD among sub-0.8 cells:  calendar median={statistics.median(cal_oos):.3f} mean={statistics.mean(cal_oos):.3f}")
print(f"                            dollar   median={statistics.median(dol_oos):.3f} mean={statistics.mean(dol_oos):.3f}")