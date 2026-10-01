# Handout — Investigación: barras-dollar vs calendario, AAD, y frecuencia

**Proyecto:** SF-Harris stochastic volatility (Anzarut 2024, IIMAS-UNAM) — servicio social / tesis, Ángel Vences Adame.
**Período de datos:** IBM 2012–2014 (test/train = 0.79×, volumen cayendo).
**Fecha:** 2026-08-08.

> Este handout resume lo **verificado** en la sesión sobre por qué el viejo titular
> "dollar = MEJOR, AAD 0.30 pp" aparecía, qué significa realmente, y qué tan robusto es.
> Todo número aquí proviene de corridas Gibbs (5000 iter / 2000 burn-in, ε=0.1 para dollar,
> 1e-5 para calendario; n_sim=2000; seeds 55/321 o 42/123) sobre series split-by-date
> (corte 2014-05-28), a menos que se indique lo contrario.

---

## 1. Pregunta original

El script `anzarut_intraday_15min.py` reportaba que las **barras-dollar** daban AAD ≈ **0.30 pp**,
mejor que las barras-calendario 15-min (≈ 0.52). ¿Eso es legítimo, o un artefacto (fuga /
frecuencia / degeneración)? Hay **tres fuga distintas** en juego y se investigaron por
separado.

---

## 2. Resultados verificados (AAD en pp; menor = mejor calibración)

### 2a. Atribución del viejo titular 0.30 (umbral constante global, leak)

| Brazo | Regla de umbral | Barras/día | AAD | ¿Fuga? |
|---|---|---|---|---|
| **leaky** (viejo titular) | media $vol 15-min sobre **toda** la serie | 69.7 | **0.299** | **SÍ** (incluye test) |
| trainonly | media $vol 15-min sobre **sólo train** | 67.6 | 0.515 | no |
| causal (adaptativa) | rolling 30-d shift(1), target 130/d | 66.2 | 1.099 | no |
| const:26 | umbral constante = $vol diario train / 26 | 15.7 | 1.345 | no |
| const:13 | umbral constante = $vol diario train / 13 | 8.15 | 1.725 | no |

**Conclusión:** el viejo 0.30 = **constante-honesto 0.515 @ ~70/día  −  0.216 pp de fuga**.
Es decir, era **mayormente honesto**, no mayormente artefacto. La fuga (umbral global que
incluye volumen de test) baja el umbral ~4 % y lo empareja con la frecuencia de barras de
test → AAD se ve ~0.2 pp mejor. **No** era degeneración (los retornos eran de tamaño real:
|r| mediana ≈ 3e-4, std log r² ≈ 10.7) ni era puro artefacto de regla-constante.

### 2b. Curva AAD × frecuencia (lo que importa de verdad)

| Construcción | 8/día | 13–16/día | 24–30/día | 54/día | 66–70/día |
|---|---|---|---|---|---|
| **Dollar adaptativa** (causal) | **0.362** | — | 0.502 / 0.655 | 0.834 | 1.099 |
| **Dollar constante** (leak-free) | 1.725 | 1.345 | — | — | 0.515 |
| **Calendario** (clock) | — | 0.668 (30-min) | 0.521 (15-min) | — | 0.542 (5-min = 78/día) |

- **Dollar adaptativa: AAD sube con la frecuencia** (grueso mejor). 0.362 @ 8.5/día → 1.099 @ 66/día.
- **Dollar constante: AAD baja con la frecuencia** (fino mejor). 1.725 @ 8/día → 0.515 @ 67/día.
- **Cruz ~50/día**: debajo de 50/día gana la adaptativa; arriba gana la constante. **Reglas opuestas, ninguna domina universalmente.**
- **Calendario: plano** (~0.52) en 13–78/día → robusto a la frecuencia (porque agrega a RV diaria).

---

## 3. El hallazgo robusto (lo que sobrevive a todo)

> **`p_stay ≈ 0.37–0.39` (plano) en TODOS los puntos** — dollar/calendario, limpio/sucio,
> constante/adaptativa, 8 a 78 barras/día — mientras el AAD oscila 5× (0.36 → 1.7).

Esto significa que **el AAD está midiendo la geometría de construcción de barras, no la
skill de pronóstico del modelo.** Comparar AAD entre construcciones (dollar vs calendario,
o reglas de umbral) **no es robusto**: el modelo subyacente no cambia (p_stay idéntico),
solo cambia cómo se cortan las barras, y eso mueve el AAD.

Mecanismo de la persistencia-per-barra: `p_stay` se infiere por-barra y queda clavado en
~0.38 porque el ACF lag-1 de `log(r²)` está dominado por el **ruido de emisión** (cada
barra es un solo `r²`, 1 g.l. → σ_obs ≈ 2), no por la señal latente. Entonces la
**persistencia física** del régimen = `1/(1−0.38) ≈ 1.6 barras × duración_de_barra`, que
**escala con la frecuencia**. Barras finas → el modelo "cree" que el régimen cambia cada
pocos minutos (mismatch con el régimen lento de IBM) → mala cobertura. Barras gruesas →
persistencia física más larga → mejor empareje → mejor AAD. El calendario evita esto
agregando los retornos intradía a RV diaria (1 obs limpia/día, ~26 g.l., σ_obs ≈ 1.04).

---

## 4. ¿"State of the art" = adaptativa de 8 barras/día?

El mejor AAD medido del dollar es **adaptativa @ 8.5/día (0.362)** — pero:

1. **Es el borde de lo medido, no un óptimo confirmado.** La curva adaptativa es
   monótona (mejora al grueso hasta 8.5/día). No se midió < 8/día; podría seguir bajando
   (4/2/1/día) o hacer upturn (muy pocas observaciones). "8 es el óptimo" es afirmación de
   borde.
2. **8/día ≈ barras de ~49 min (≈ horarias) = frecuencia BAJA.** Que el mejor punto de un
   método "de alta frecuencia" salga en casi-horarias es sospechoso y subraya que el AAD
   premia ruido de emisión bajo, no utilidad.
3. **AAD = calibración (cobertura), no utilidad/sharpness.** Un AAD bajo no implica el
   modelo decida mejor.

### ¿Cuándo 8 barras/día le gana a 80?
- **Horizonte de decisión diario/multihorario**: VaR, capital, márgenes, rebalanceo
  lento — 80 predicciones/día son resolución que no se puede usar y sólo añaden salto.
- **Régimen que se mueve a escala ~diaria** (caso IBM): 80/día submuestrea-aliasa un
  proceso lento; 8/día lo empareja (p_stay × duración → ~78 min).
- **Cuando no se puede actuar** más de unas veces/día.

### ¿Cuándo 80 le gana a 8?
- **Decisiones intradía** (hedging, detección de crash a las 10am): 8/día lo ves horas tarde.
- **Régimen que SÍ se mueve intradía**: 8/día hace aliasing.
- **Cuando se necesita sharpness** por-barra, no sólo cobertura.

**Adaptativa @ 8/día ≠ barras de 1h ingenuas:** el umbral adapta al volumen → barras
homocedásticas a lo largo del drift de volumen del test; un reloj de 1h fijo no lo haría. Y
8 obs/día > 1 RV-diaria, por eso le gana al calendario 0.52: observa el latente 8× más
seguido con cada obs razonablemente limpia.

---

## 5. Diagnósticos estructurales (sin Gibbs, `mech_struct.py`)

- Tendencia de volumen: 2012 = 691 M, 2013 = 710 M, 2014 = 639 M (test/train = 0.79×, **cae**).
- corr(conteo de barras/día, $vol diario): constante **0.953**, adaptativa **0.876**.
- Shift `log(r²)` train→test: constante **+0.04** (0.02σ), adaptativa **+0.17** (0.09σ).
- Autocorr lag-1: constante 0.091, adaptativa 0.095 (idéntica → no es persistencia).
- La adaptativa sobre-cubre **uniformemente** (todas las desviaciones positivas, pico en
  p=0.75 centro, no colas) → firma de **distribution-shift** (shift 0.17 al romper
  homocedasticidad), no de jump-mis-specification.

---

## 6. Pendientes / decisiones abiertas

- **Enmarque del titular (sin decidir):** cuatro opciones estaban sobre la mesa —
  (a) liderar con calendario 15-min (0.52/0.46) + caveat dollar;
  (b) dollar honesto 0.515 + caveat;
  (c) addendum sin tocar el titular;
  (d) buscar una métrica de skill robusta a ε y regla-de-barra (pinball/CRPS/coverage
  de un solo nivel). El usuario pidió aclarar antes de elegir.
- **Fix de producción de STEP 5b** en `scripts/anzarut_intraday_15min.py` (líneas ~380–411):
  sigue usando el umbral global viejo. La "construcción correcta" libre de fuga es
  frecuencia/regla-dependiente (constante mejor ≥50/día, adaptativa ≤50/día), así que la
  elección depende de la frecuencia titular que se decida.
- **Medir < 8/día** (4, 2, 1 adaptiva) para confirmar/descartar el upturn del sweet-spot
  (contestaría si 8 es óptimo o sólo borde). ~3 min.
- **Propagar** cualquier titular cambiado al LaTeX / PPTX **solo con confirmación** del usuario.

---

## 7. Inventario de scripts de la sesión

| Script | Qué hace |
|---|---|
| `scripts/anzarut_replication.py` | módulo compartido; `build_dollar_bars_rolling` (causal, shift(1), leak-free) añadido antes en la cadena. |
| `scripts/freq_sweep.py` | barrido parametrizado de frecuencia. Uso: `python freq_sweep.py <dollar\|calendar> <int>`. |
| `scripts/_dilution_arm.py` | un brazo por arg: `leaky\|trainonly\|causal\|const:N`. Imprime `RESULT_JSON {threshold, bpd, aad, levels, abs_r_median, std_logr2, n_train, n_test, tr_mean, te_mean, arm}`. |
| `scripts/_arm_{trainonly,causal,const13,const26}.txt` | salidas `RESULT_JSON` (coverage por nivel capturada). |
| `scripts/mech_struct.py` | diagnósticos sin-Gibbs (tendencia de volumen, correlaciones, shifts, autocorr). |
| `scripts/dollar_fair_compare.py` | comparación justa: `build_dollar_bars_rolling` lookback 30 / 26 bpd → 23.86/día, clean 0.5018, no-clean 0.7361. |
| `scripts/ablation_jumpcleaning.py` | run() 15-min (0.5212/0.4601); run_dollar() viejo (0.2993/1.6599). |
| `scripts/verify_dilution.py` | superseded por `_dilution_arm.py` (la versión 3-arm secuencial hacía timeout). |

**Trabajo paralelo (otro plan, no esta sesión):** el plan `composed-shimmying-bonbon.md`
corrige la fuga train/test en `detect_and_remove_jumps` + `estimate_periodicity` en los 3
scripts titulares (`coverage_decomposition.py`, `anzarut_intraday_15min.py`,
`anzarut_replication.py __main__`) y recalcula esos 3 titulares. Los ~12 scripts de prueba
restantes conservan la fuga (AADs absolutos provisionales, conclusiones relativas robustas).

---

## TL;DR

- El viejo **0.30 pp** del dollar = constante-honesto **0.515** @ ~70/día  **menos 0.216 pp de fuga** → mayormente honesto.
- **`p_stay` plano (0.38) con AAD oscilando 5×** → el AAD mide geometría de barras, no skill → comparar AAD entre construcciones no es robusto.
- Dollar adaptativa: grueso mejor (0.362 @ 8/día). Dollar constante: fino mejor (0.515 @ 70/día). Cruz ~50/día. Calendario: plano 0.52.
- "8 barras/día" ≈ horario = frecuencia baja; es el borde medido, no óptimo confirmado; AAD = calibración no utilidad.
- La receta "correcta" libre de fuga depende de la frecuencia objetivo → el fix de STEP 5b espera la decisión de enmarque.