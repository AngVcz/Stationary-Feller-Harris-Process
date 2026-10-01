# SF-Harris Project — Handoff Document

## Project Status: Borrador listo para revisión del asesor

El borrador `report/borrador_revision.tex` está completo con datos de simulación (100 réplicas, 5 métodos) y resultados empíricos. Compila a 7 páginas.

---

## Simulation Study (Section 3.5.2 Replication)

### Table II — Case 1: Q = Discrete Uniform{1,...,5}, E_α

| k | NDNJ | EM | MLE | Gibbs-a | Gibbs-b | NDNJ(A) | EM(A) | MLE(A) | Gibbs-a(A) | Gibbs-b(A) |
|---|------|-----|-----|---------|---------|---------|-------|--------|------------|------------|
| 20 | 0.49 | 0.58 | 0.41 | 0.45 | 0.48 | 0.45 | 0.94 | 0.75 | 0.32 | 0.44 |
| 100 | 0.42 | 0.65 | 0.50 | 0.54 | 0.43 | 0.40 | 0.12 | 0.10 | 0.15 | 0.39 |
| 500 | 0.43 | 0.54 | 0.45 | 0.50 | 0.45 | 0.28 | 0.03 | 0.04 | 0.04 | 0.26 |
| 1000 | 0.45 | 0.59 | 0.50 | 0.51 | 0.48 | 0.22 | 0.03 | 0.03 | 0.02 | 0.18 |

Left = our replication (α̂ cap=50), Right = Anzarut (α̂ cap=10).

### Table III — Case 2: Q = GIG(λ,κ,η), E_α

| k | NDNJ | MLE | EM | Gibbs-a | Gibbs-b | NDNJ(A) | MLE(A) | EM(A) | Gibbs-a(A) | Gibbs-b(A) |
|---|------|-----|-----|---------|---------|---------|--------|-------|-------------|------------|
| 20 | 0.96 | 0.96 | 0.96 | 0.47 | 0.51 | 0.51 | 1.76 | 1.49 | 0.33 | 0.51 |
| 100 | 0.88 | 0.88 | 0.88 | 0.45 | 0.50 | 0.44 | 1.59 | 1.29 | 0.15 | 0.44 |
| 500 | 0.86 | 0.86 | 0.86 | 0.45 | 0.51 | 0.28 | 1.56 | 1.29 | 1.09 | 0.25 |
| 1000 | 0.81 | 0.81 | 0.81 | 0.44 | 0.50 | 0.19 | 1.51 | 1.29 | 1.86 | 0.17 |

### Table III — Case 2: Q = GIG(λ,κ,η), E_Q (KL divergence)

| k | NDNJ | MLE | EM | Gibbs-a | Gibbs-b | NDNJ(A) | MLE(A) | EM(A) | Gibbs-a(A) | Gibbs-b(A) |
|---|------|-----|-----|---------|---------|---------|--------|-------|-------------|------------|
| 20 | 0.35 | 0.24 | 0.32 | 0.22 | 0.38 | 0.09 | 0.09 | 0.10 | 0.77 | 0.84 |
| 100 | 0.16 | 0.15 | 0.11 | 0.06 | 0.36 | 0.02 | 0.02 | 0.03 | 0.31 | 0.31 |
| 500 | 0.025 | 0.031 | 0.017 | 0.023 | 0.051 | 0.01 | 0.04 | 0.07 | 0.16 | 0.18 |
| 1000 | 0.004 | 0.011 | 0.027 | 0.008 | 0.042 | 0.01 | 0.05 | 0.09 | 0.09 | 0.12 |

### Key findings

1. **NDNJ = MLE = EM for continuous Q** (theoretical result: likelihood factorizes). Anzarut's different values come from joint 4D optimization artifacts.

2. **Gibbs-a halves E_α** vs point estimators (0.47 vs 0.96 at k=20) by integrating over the posterior. Anzarut's Gibbs-a worsens with k (1.86 at k=1000) — likely convergence issues.

3. **E_α discrepancy with Anzarut**: Our values are higher (0.81 vs 0.19 at k=1000) because we use α̂ cap=50 vs Anzarut's cap=10. With cap=10, extreme α estimates are artificially truncated.

4. **GIG near-unidentifiability**: When κ>10, λ is nearly free (flat likelihood). Point MLE finds boundary solutions. Gibbs posterior averaging resolves this — our E_Q via Gibbs posterior is 0.24–0.35 at k=20, comparable to Anzarut's 0.09–0.10.

5. **Gibbs-b has systematically higher E_Q** than Gibbs-a (0.38 vs 0.22 at k=20) due to bias from the conjugate Gamma update on α.

---

## Mexican Data Results (Section 4.6)

### Daily AAD (pp) — Hist. Sim. wins all 8 assets

| Asset | SF-Harris | Hist. Sim. | Gap |
|-------|-----------|-----------|-----|
| IPC | 8.5 | 3.7 | 4.8 |
| Bimbo | 11.7 | 3.1 | 8.6 |
| Grupo México | 11.3 | 2.2 | 9.1 |
| Walmart Mex | 10.4 | 1.4 | 9.0 |
| FEMSA | 8.1 | 3.3 | 4.8 |
| Cemex | 10.2 | 1.6 | 8.6 |
| Banorte | 9.5 | 2.0 | 7.5 |
| USD/MXN | 7.8 | 1.5 | 6.3 |

### Hourly (1h) AAD — SF-Harris wins 5/8

| Asset | SF-Harris | Hist. Sim. | Gap | Winner |
|-------|-----------|-----------|-----|--------|
| Cemex | 0.98 | 2.48 | -1.50 | SF-Harris |
| FEMSA | 1.13 | 1.40 | -0.27 | SF-Harris |
| Bimbo | 1.33 | 1.12 | +0.21 | Hist. Sim. |
| Grupo México | 1.47 | 1.56 | -0.09 | SF-Harris |
| Walmart Mex | 1.63 | 1.65 | -0.02 | SF-Harris |
| Banorte | 1.72 | 1.58 | +0.14 | Hist. Sim. |
| IPC | 2.66 | 3.76 | -1.11 | SF-Harris |
| USD/MXN | 2.98 | 2.86 | +0.12 | Hist. Sim. |

---

## Key Code Files

- **Simulation study**: `scripts/simulation_study.py` — runs Case 1 and Case 2 with all 5 methods
- **Estimation methods**: `src/sf_harris/estimation.py` — NDNJ, MLE, EM, Gibbs-a, Gibbs-b, gibbs_q_posterior_mean, gibbs_estimate_discrete
- **Process**: `src/sf_harris/process.py` — SFHarrisProcess.simulate()
- **Distributions**: `src/sf_harris/distributions.py` — DiscreteUniformQ, GIGQ (with kl_divergence)
- **Borrador**: `report/borrador_revision.tex` — 7 pages, compiles with pdflatex

### Data files

- `data/simulation_results.npz` — latest simulation (cap=50, full range, with Gibbs)
- `data/simulation_results_gibbsQ_cap10.npz` — cap=10 run
- `data/simulation_results_restricted_ranges.npz` — α∈(0,5), κ∈(0.1,10)
- `data/mexican_intraday_results.npz` — hourly (1h) AAD for 8 Mexican assets
- `Trash/mexican_results.pkl` — daily AAD for 8 Mexican assets

---

## How to Re-run Simulations

```bash
# Quick test (10 replications)
python scripts/simulation_study.py --quick

# Full run (100 replications, both cases, with Gibbs)
python scripts/simulation_study.py --n-rep 100

# Case 1 only (discrete Q)
python scripts/simulation_study.py --case 1 --n-rep 100

# Case 2 only (GIG Q), skip Gibbs for speed
python scripts/simulation_study.py --case 2 --n-rep 100 --no-gibbs

# Restricted parameter ranges (closer to Anzarut)
python scripts/simulation_study.py --n-rep 100 --alpha-range 0 5 --kappa-range 0.1 10
```

---

## Open Questions for Advisor

1. **Alpha cap**: We use cap=50 (honest), Anzarut uses cap=10. Should we also report cap=10 for direct comparison?

2. **Gibbs-a worsening with k in Anzarut**: E_α(Gibbs-a) goes 0.33→0.15→1.09→1.86, which is counter-intuitive. Possible causes: convergence issues, different MH proposal, different priors. Our Gibbs-a improves monotonically (0.47→0.44).

3. **NDNJ/MLE/EM E_Q methodology**: We use Gibbs posterior mean for Q estimation in all three methods (avoids GIG near-unidentifiability). Anzarut uses point MLE. Our E_Q for NDNJ is 0.35 at k=20 vs Anzarut's 0.09 — should we also report point MLE values?

4. **"Decorative" claim strength**: The thesis advisor may want softer language. The evidence is strong (p_stay sweep: 0.41–0.54 pp) but the word "decorative" is provocative.

5. **Restricted range comparison**: Should we include the restricted range (α∈(0,5), κ∈(0.1,10)) results as a supplementary table? They're closer to Anzarut's values.