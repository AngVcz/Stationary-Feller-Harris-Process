# Estudio comprensivo y verificado del modelo SF-Harris y del trabajo de tesis de Ángel Vences Adame

**Síntesis integrada de 7 lectores de subsistema y 3 críticos adversariales (matemática, estadística, código)**
IIMAS-UNAM · Licenciatura en Actuaría · Servicio social / Tesis

> Generado por un workflow de 11 agentes (7 lectores + 3 críticos + síntesis) sobre el repositorio completo.

---

## 1. Qué es el modelo SF-Harris

El SF-Harris es un modelo de volatilidad estocástica (SV) cuyo estado latente de volatilidad sigue una **cadena de Markov de Harris en tiempo discreto** con una estructura particular: un punto de masa en el valor anterior ("stay") más un salto desde una distribución invariante Q ("jump").

### 1.1 Definición matemática precisa

**Cadena de Harris (estado latente).** Sea `x_t = log(τ_t)` la volatilidad latente (log-varianza). La transición es

```
x_t | x_{t-1} ~ (1 - e^{-α}) Q + e^{-α} δ_{x_{t-1}}
```

donde:
- `stay_prob = e^{-α}`: probabilidad de permanecer en el valor anterior (punto de masa).
- `jump_prob = 1 - e^{-α}`: probabilidad de saltar, dibujando un valor nuevo desde Q.
- `Q` es la distribución **invariante (estacionaria)** de la cadena.

**Simulación (`process.py:simulate`).** En cada paso: con probabilidad `e^{-α}` se copia exactamente `obs[t-1]`; con probabilidad `1-e^{-α}` se dibuja de `Q.sample()`. La inicialización es `obs[0] = Q_sample`.

**Invariancia de Q (verificada algebraicamente).** Si `x_{t-1} ~ Q`, entonces
```
P(x_t ∈ A) = (1-e^{-α}) Q(A) + e^{-α} Q(A) = Q(A)
```
Mezclando `(1-e^{-α})Q + e^{-α}δ_{x_{t-1}}` sobre `x_{t-1} ~ Q` se obtiene Q. **La marginal estacionaria de `x_t` es Q por construcción del kernel**, no por un teorema derivado. Este es un punto central: la afirmación "marginal = Q" es cierta *por diseño*, no por inferencia.

**ACF teórica.** Bajo estacionariedad (`x ~ Q` con media `μ_Q`, varianza `σ_Q²`) y saltos independientes del pasado (Q_sample fresco en cada salto):
```
Cov(x_t, x_{t-1}) = e^{-α} σ_Q²   ⇒   Corr(1) = e^{-α}
```
Por la propiedad de Markov, `Corr(h) = e^{-α h}` (un solo exponencial). De aquí el estimador
```
α = -log(ACF(1)) = -log(ρ₁)
```
implementado en `scripts/anzarut_replication.py:estimate_alpha` (con piso `max(ρ₁, 0.01)`).

**Emisión condicional gaussiana.** La observación (retorno) está dada por
```
r_t | τ_t ~ N(0, τ_t)
```
Es decir, `sim_ret = √τ · z` con `z ~ N(0,1)`. La emisión usa **solo `τ_t` actual**, no la trayectoria. Esta emisión **no vive en la librería núcleo** (`src/sf_harris/`), sino en los scripts (`anzarut_replication.py:simulate_predictive`).

**Marginal estacionaria del modelo con Q empírica.** Al fijar `Q = Empírica(log RV de entrenamiento)` (centrada), la marginal de `τ` coincide con la empírica de RV (≈ varianza realizada). **Esto equivale a Historical Simulation** (resamplear los RV observados). Precisión importante: la cadena vive en `log(RV)`, así que la marginal de `log(τ)` es `Empírica(log RV)`; exponentiando, `τ ~ Empírica(RV)`. No debe confundirse `Empirical(RV)` (RV bien estimada, p. ej. a 15-min con 26 obs/día) con `Empirical(r_t²)` (retorno cuadrado crudo, proxy ruidosísimo a frecuencia diaria con 1 obs/día). El reporte a veces escribe "Empirical(r_t²)" cuando en el régimen intradía debería decir "Empirical(RV)".

### 1.2 Los ingredientes del modelo

| Ingrediente | Rol | Implementación |
|---|---|---|
| **Q invariante** | Distribución estacionaria de los saltos; fija la marginal predictiva | `DiscreteUniformQ` ({1,…,m}), `GIGQ(λ,κ,η)`, o empírica (resamplear train) |
| **α (de la ACF)** | Controla la persistencia temporal: `P(stay)=e^{-α}` | `α = -log(ρ₁)`; estimado por NDNJ/MLE/EM/Gibbs |
| **Gibbs sampler con conjugación** | Inferencia bayesiana de (α, μ, σ) y denoising de τ* | `estimation.py` (Gibbs-a, Gibbs-b) y `scripts:gibbs_gig_harris` (con epsilon, NIG) |

**Distribución GIG.** `GIGQ(λ,κ,η)` con densidad
```
f(x) = η^λ / (2 K_λ(κ)) · x^{λ-1} exp(-(κ/2)(ηx + 1/(ηx))),  x > 0
```
Parametrización interna `(χ,ψ)`: `χ = κη`, `ψ = κ/η`, `ω = √(χψ) = κ`. Muestreo vía `scipy.stats.geninvgauss(λ, b=κ)` y `X = Z/η`. Divergencia KL analítica entre GIGs vía momentos `E[X]`, `E[1/X]`, `E[log X]` (este último por derivada numérica de `log K_λ` respecto a λ) con fallback Monte Carlo.

**Verificación importante (falso positivo desmentido).** Un lector marcó los momentos `E[X]` y `E[1/X]` en `kl_divergence` como invertidos. **La verificación Monte Carlo (400k muestras, η≠1) confirma que el código es CORRECTO**: `E[X]` código=0.605 vs MC=0.606; `E[1/X]` código=2.44 vs MC=2.44; `E[log X]` código=−0.691 vs MC=−0.690; KL entre dos GIG distintas código=0.0678 vs MC=0.0669. El único defecto es que el **docstring** de `GIGQ` (línea 46) tiene `χ` y `ψ` intercambiados respecto al código — bug documental, no de cálculo.

### 1.3 Estimadores de los parámetros

| Método | Fórmula / mecanismo | Notas |
|---|---|---|
| **NDNJ** | `p̂ = n_changes/n_total`; `α = -log(1-p̂) = -log(n_stay/n_total)` | Idéntico a MLE continuo en este código |
| **MLE continuo** | `α = -log(n_stay/n_total)`, cap 50 | Analítico por factorización de la verosimilitud |
| **MLE discreto** | Minimización numérica en (0.01, 50) | Resuelve saltos ocultos (Q con átomos) |
| **EM discreto** | E-step: `E[no_jump] = n_stay · p_stay/(p_stay + p_jump·pmf_same)`; M-step: `α = -log(E[no_jump]/n)` | |
| **Gibbs-a** | MH random-walk sobre `π(α|data) ∝ (1-e^{-α})^{n_jumps} e^{-α(n_stays+c)}`, prior `Exp(c)` | Adaptación de paso (target 0.44) |
| **Gibbs-b** | `α ~ Gamma(n_jumps+1, 1/(j_m+c))` con `j_m = max(jump_times[-1], 1.0)` | Heurística sesgada (ver §5) |

**Resultados clave de la factorización.** Para Q **continua** (sin átomos), `P_Q(x_t = x_{t-1}) = 0`, así que "observar mismo" ⟺ no-salto y "observar cambio" ⟺ salto. La verosimilitud factoriza:
```
L(α) = e^{-α·n_stay} (1-e^{-α})^{n_change} · [términos q(x_t) independientes de α]
```
El argmax es analítico: `α̂ = -log(n_stay/n_total)`. **NDNJ = MLE = EM en E_α para Q continua** — pero en parte porque el código enruta los tres a la misma función (`mle_alpha_continuous` para α; `gibbs_q_posterior_mean` para Q), no porque sean tres implementaciones que convergen por separado.

Para Q **discreta** (con átomos, Caso 1: Uniform{1,…,5}) hay saltos ocultos (el proceso salta pero cae en el mismo valor): `P(obs same) = e^{-α} + (1-e^{-α})/m`. NDNJ queda sesgado a la baja; MLE/EM resuelven numéricamente. **La igualdad NDNJ=MLE=EM NO se mantiene para Q discreta.**

---

## 2. Arquitectura del trabajo: los 7 subsistemas

El proyecto se organiza en siete subsistemas que se encadenan desde los fundamentos pedagógicos hasta el reporte académico, siguiendo la cadena conceptual "kernel de Markov → filtrado recursivo → aproximación Monte Carlo → modelo SF-Harris → inferencia Gibbs → validación empírica → tesis decorativa → aplicaciones → reporte".

| # | Subsistema | Propósito | Encadena con |
|---|---|---|---|
| 1 | **Fundamentos SSM/Kalman/resampling** | Enseñar progresivamente kernel de Markov → filtrado Kalman → IS/ESS/resampling. Piezas reutilizables: AR(1)/SSM lineal-gaussiano, filtro de Kalman, ESS, 4 esquemas de resampling. Justifica por qué Gibbs reemplaza al filtro de partículas. | → Núcleo |
| 2 | **Librería núcleo SF-Harris** | Implementa la cadena de Harris, distribuciones Q (DiscreteUniform, GIG), estimadores (NDNJ/MLE/EM/Gibbs-a/Gibbs-b), verosimilitudes. Sustento matemático. | → Replicación |
| 3 | **Replicación Anzarut + estudio de simulación** | Replica Sección 3.5.2 (Tablas II/III) y Sección 4 (pipeline IBM, Tabla 3 de cobertura). Valida estimadores y el pipeline empírico completo. | → Decorativa + Reporte |
| 4 | **Tesis "cadena Harris decorativa" (10 tests)** | La afirmación más fuerte y central: la cadena de Harris no añade poder predictivo de cobertura más allá de Q empírica + denoising Gibbs. 10 tests de ablación, 3 hallazgos estructurales, 2 ingredientes esenciales. | → Aplicaciones + Reporte |
| 5 | **Aplicaciones financieras** | 9 aplicaciones: activos mexicanos (diario/intradía), selección de modelos (log-score, CRPS, PIT, DM), VaR/CVaR, portafolios Markowitz/CVaR, variance swaps, dollar vs calendar bars, predicción diaria con emisión. | → Reporte |
| 6 | **Reporte LaTeX** | Narrativa académica de tesis/servicio social. Argumenta la tesis "decorativa" con pruebas de ablación. Incluye borrador de revisión con el estudio de simulación. | (salida) |
| 7 | **Notas, tests y demos** | Capa pedagógica (M1-M3 sobre Chopin) y de verificación (tests unitarios de bloques constructivos). | (soporte transversal) |

**El hilo conductor.** M1 formaliza el núcleo de transición `K(x,·)`; M2 muestra que cuando K y el núcleo de observación G son lineal-gaussianos, el filtrado tiene solución cerrada (Kalman); M3 introduce IS/ESS/resampling como maquinaria para el caso general no lineal/no gaussiano — justo donde vive SF-Harris (saltos GIG, Q discreto/continuo). La decisión de usar Gibbs en lugar de un filtro de partículas se justifica: (a) el objetivo es la posterior sobre parámetros, no la distribución filtrante en línea; (b) hay conjugación explícita; (c) el promedio posterior resuelve la casi-no-identificabilidad del GIG que degeneraría los pesos de un PF (ESS→1). `scripts/kalman_vs_gibbs.py` pone a prueba directamente esta decisión.

---

## 3. La tesis central "decorativa"

La afirmación más fuerte del trabajo es que **la cadena de Harris es decorativa para predicción de cobertura**: su estructura temporal (el punto de masa `P(stay)·δ_{X_t}` y la construcción de tiempo continuo) no aporta poder predictivo más allá de lo que ya da la distribución marginal Q empírica más el denoising bayesiano (Gibbs). El modelo se reduce a dos ingredientes esenciales.

### 3.1 Los 10 tests de ablación

| Test | Qué prueba | Resultado reportado |
|---|---|---|
| 1 | Descomposición de cobertura | SF-Harris = iid bootstrap = N(0,τ*) = 2.0pp; las dinámicas de Harris añaden 0.0pp; los 7.4pp de mejora sobre N(0,σ²) vienen de adaptación de varianza (τ*), no de estructura temporal |
| 2 | Calibración de epsilon | P(stay) varía de 0.37 a 0.98 barriendo ε 8 órdenes; AAD stays en banda 0.3pp; Hist.Sim (1.7pp) supera a todo ε → P(stay) no afecta cobertura |
| 3 | Frecuencia | La degradación de 15-min a diario la causa σ (ruido en estimador RV), no P(stay) (0.37 en todas las frecuencias) |
| 5 | Crisis | AUC=0.416 (anti-predictivo) para detección de crisis basada en PIT; VIX da 0.841; P(jump)=0.63 constante |
| 7 | Stay/Jump | AUC=0.511 (random) prediciendo saltos con ratio τ*; P(jump)=0.63 es probabilidad incondicional, no señal temporal |
| 8 | Coin flip | Punto de masa sin denoising = 50pp a P(stay)=0.88; el punto de masa es destructivo sin posterior del Gibbs |
| 9 | P(stay) sweep sin Gibbs | Sin Gibbs, AAD aumenta monótonamente con P(stay) (destructivo): 0.42→65.39 sin ruido, 18.51→10.12 con ruido |
| 10 | P(stay) sweep con Gibbs | Con Gibbs, AAD flat 0.41-0.54pp para todo P(stay) → P(stay) no-informativa con denoising |
| (Kalman) | Filtro de Kalman falla | AAD=16pp; la asunción gaussiana destruye la bimodalidad de la predictiva (masa en estado actual y en Q al saltar) |
| (Mexicanos) | Datos mexicanos diarios | Hist.Sim supera a SF-Harris en 8/8 activos (gaps 4.8-9.1pp) |

### 3.2 Los 3 hallazgos estructurales

1. **Q empírica casi siempre mejor** (0.2-0.3pp vs GIG 0.8pp, Gaussian 26pp). La forma empírica de Q domina a la paramétrica en el rango probado.
2. **Saltos continuos inútiles → cadena discreta sin pérdida.** El modelo siempre se implementó como cadena discreta (transición un paso); el esqueleto discreto en la rejilla de enteros es representación exacta del proceso continuo con retención `Exponential(α)`.
3. **P(stay) no-informativa dentro del pipeline Gibbs** (0.41-0.54pp flat); el punto de masa es "carrier wave" (andamio computacional).

### 3.3 Los 2 ingredientes esenciales

El modelo mínimo útil es:
1. **Q empírica**: `τ ~ Empirical(r²)` (resamplear datos de entrenamiento) — equivalente a Historical Simulation.
2. **Denoising Gibbs**: estimación bayesiana de (μ, σ) para centrar/escalar la predictiva.

Receta mínima (Cap. 5): ajustar (μ, σ) por Gibbs/ML sobre datos centrados, muestrear iid de `{x_i} + μ̂`, añadir `N(0, σ̂)`.

### 3.4 POR QUÉ P(stay) resulta no-informativa

La razón matemática —que los reportes no enuncian claramente— es estructural:

> **La cobertura es una propiedad MARGINAL del predictivo, y P(stay)=e^{-α} solo gobierna la estructura CONJUNTA/temporal.** Bajo estacionariedad, la marginal predictiva = Q = Empírica, INDEPENDIENTE de P(stay). Por tanto la cobertura no puede depender de P(stay) salvo en horizontes cortos donde la mezcla no ha ocurrido.

La no-informatividad es **esperable estructuralmente, no un hallazgo empírico sorpresivo**. Dos modelos con la misma F_τ (CDF marginal de τ*) pero distinto kernel dan cobertura idéntica. Al fijar Q=empírica, la marginal queda anclada al entrenamiento y P(stay) solo reorganiza el orden temporal de los valores, no su distribución marginal.

**Conflación importante a corregir.** En el coin-flip sin denoising (Tests 8-9), P(stay) **sí es informativa — siempre destructiva** (0.42pp en P=0 vs 50pp en P=0.88). Llamarla "no-informativa" mezcla dos regímenes distintos: "sin efecto dentro del Gibbs" (cobertura marginal) frente a "siempre perjudicial sin denoising" (coin-flip). P(stay)=0 es óptimo en el coin-flip; no es que "no importe", es que siempre perjudica. Además, la flatness es específica a Q empírica (la marginal queda fijada); con Q paramétrica (GIG) P(stay) podría interactuar con la estimación de Q y tener efecto.

---

## 4. Hallazgos clave

### 4.1 Replicación vs Anzarut (estudio de simulación)

**Caso 2 (Q=GIG), Tabla III — E_α:**

| k | NDNJ=MLE=EM | Gibbs-a | Gibbs-b |
|---|---|---|---|
| 20 | 0.9561 | 0.4706 | — |
| 500 | 0.8603 | — | — |
| 1000 | 0.8055 | 0.4368 | — |

- **NDNJ=MLE=EM son bit-idénticos** en E_α para Q continua (factorización de la verosimilitud). Los tres llaman a `mle_alpha_continuous`.
- **Gibbs-a reduce E_α a la mitad** vs estimadores puntuales: 0.47 vs 0.96 en k=20; 0.44 vs 0.81 en k=1000.
- **Gibbs-b tiene E_Q sistemáticamente mayor** que Gibbs-a por sesgo del update Gamma conjugado.

**Discrepancia con Anzarut (cap de α):**

| Cap | NDNJ k=1000 | Anzarut |
|---|---|---|
| 50 (código actual) | 0.8055 | 0.19 |
| 10 (NPZ no reproducible) | 0.2173 | 0.19 |

La única corrida que replica a Anzarut (cap=10) **no puede regenerarse con el código actual** (cap=50 hardcodeado en todas partes).

### 4.2 Replicación empírica (Tabla 3 de Anzarut)

La replicación a 15-min supera la Tabla 3 de Anzarut y a los benchmarks SV:

| Modelo | AAD (pp) |
|---|---|
| **Ours (dollar bars)** | **0.2-0.3** |
| Anzarut | 0.8 |
| GARCH | 3.4 |
| EGARCH | ~70 |
| Heston | ~54 |

**Caveat de granularidad.** El "win" sobre Anzarut es a nivel 15-min, mientras que la agregación diaria degrada a 9.47pp. `results_tables.md` mezcla ambas escalas sin siempre aclararlo.

### 4.3 Resultados mexicanos

| Frecuencia | Resultado |
|---|---|
| **Diario** | Hist.Sim gana 8/8 activos (SF-Harris 8-12pp vs Hist 1.4-3.7pp); GARCH no converge por colas pesadas/muestras cortas |
| **Intradía (1h)** | SF-Harris gana 5/8 activos (Cemex 0.98 vs 2.48, FEMSA 1.13 vs 1.40, IPC 2.66 vs 3.76) |

**P(stay)≈0.37 en TODOS los activos** a frecuencia diaria.

**Caveat crítico.** El "SF-Harris gana 5/8 en intradía" NO usa el modelo SF-Harris real: `gibbs_simple` estima μ/σ por NIG simple y α por Exponential crudo, sin umbral ε ni clasificación stay/jump; `simulate_predictive` hace iid Q-empírica + ruido, no el kernel stay/jump. Realmente testa "Q-empírica + denoising", que es el modelo mínimo esencial de la propia tesis — apoya la tesis decorativa pero está mal etiquetado.

### 4.4 Selección de modelo y riesgo

- **Colas pesadas**: SV Heavy-Tailed es más conservador en colas 99%+ (CVaR ratio SV/Hist ~1.14-1.33x; genera escenarios ~2.7x más allá del máximo de entrenamiento vs 1.0x para Hist.Sim).
- **Portafolio**: Hist-Sharpe (0.45) supera SV-Sharpe (0.27) en backtest Markowitz 5-activos mensual.
- **Dollar vs calendar bars**: dollar bars producen mejores pronósticos (RMSE h=1: 0.419 vs 0.804; ACF R² 0.986 vs 0.926).

---

## 5. Verificación: lo que sostiene y lo que no

Integrando los hallazgos de los tres críticos adversariales (matemática, estadística, código).

### 5.1 Afirmaciones sólidamente probadas

| Afirmación | Estatus | Evidencia |
|---|---|---|
| Q es la distribución invariante; la marginal estacionaria es Q | **CORRECTA** (por construcción del kernel) | Verificada algebraicamente: `(1-e^{-α})Q + e^{-α}Q = Q` |
| ACF(1) = e^{-α} del proceso latente | **CORRECTA** | Derivación: `Cov = e^{-α}σ_Q²` bajo estacionariedad + saltos independientes |
| NDNJ=MLE=EM en E_α para Q continua | **CORRECTA** (con condición precisa) | Factorización de la verosimilitud; argmax analítico. Condición: Q atomless ⟹ saltos observables ⟹ factorización |
| Esqueleto discreto = representación exacta del proceso continuo en la rejilla | **CORRECTA** (por construcción) | `P(no salto en una unidad) = e^{-α}` codifica fielmente la tasa continua |
| La divergencia KL de distributions.py es correcta | **CORRECTA** (falso positivo desmentido) | Verificación MC (400k, η≠1): KL analítico 0.0678 vs MC 0.0669. Solo el docstring tiene χ/ψ intercambiados |
| Gibbs-a reduce E_α a la mitad vs estimadores puntuales | **CORRECTA** | 0.47 vs 0.96 (k=20); 0.44 vs 0.81 (k=1000) |

### 5.2 Afirmaciones con salvedades

**A.3 — Actualización de α: DESCRIPCIÓN ERRÓNEA + APROXIMACIÓN SESGADA.** La afirmación "actualización conjugada Beta para α" no describe el código: **ambas implementaciones usan una actualización GAMMA** (tiempo-continuo Poisson), no Beta. La conjugada EXACTA en tiempo discreto es: `m|α ~ Binomial(n, 1-e^{-α})`; con conjugada `Beta(a+m, b+n-m)` sobre `p = 1-e^{-α}` y luego `α = -log(1-p)`. El Gamma es aproximación válida solo para α pequeño; para α grande, `Bin(n, 1-e^{-α})` satura en n mientras `Poisson(α·n)` excede n, divergiendo.

Además, ambas implementaciones ignoran la **censura a derecha** tras el último salto: usan como escala el tiempo del último salto `t_m` (script) o `j_m = jump_times[-1]` (librería) en lugar de `T = n` (ventana total). Como `t_m < T`, se subestima la exposición y se **sobrestima α** — sesgo consistente con el "E_Q sistemáticamente mayor de Gibbs-b" reportado.

**A.3b — Actualización NIG: APROXIMADA E INCONSISTENTE.** (a) El paso `σ²|μ,data` omite el término de acoplamiento `0.5·κ₀(μ-μ₀)²` de la condicional exacta NIG. (b) **Inconsistencia de modelo**: el paso NIG trata `log_rv` como iid `N(μ,σ²)`, estimando la MARGINAL empírica de log(RV) como Normal — contradice que Q debería ser GIG o empírica. El NIG no modela Q; μ/σ solo centran/escalan la predictiva empírica, siendo en gran medida redundante para la predicción.

**A.5 — "P(stay) no-informativa": EVIDENCIA MAL ATRIBUIDA + RAZÓN NO ARTICULADA.**
- (1) Existe una explicación matemática más simple que los reportes no enuncian: la cobertura es propiedad **marginal**, P(stay) gobierna lo **conjunto/temporal**; bajo estacionariedad la marginal = Q = Empírica, independiente de P(stay). Es **esperable estructuralmente**, no un hallazgo empírico sorpresivo.
- (2) El sweep de P(stay) (`gibbs_pstay_sweep.py`) es una simulación forward con P(stay) **sobrescrito**, no el pipeline Gibbs real. Los "0.41-0.54pp planos" provienen de **Approach B SIN ruido observacional** (líneas 134-144, `obs_noise=False`), i.e. Q empírica pura. **Approach A (con ruido, el régimen del Gibbs real) da ~18.66pp plano para TODO P(stay)**. El reporte cita el régimen sin ruido (0.41-0.54pp) como evidencia, pero el Gibbs real con ruido opera en 18pp.
- (3) **Conflación**: en el coin-flip sin denoising, P(stay) **es informativa — siempre destructiva**.
- (4) La flatness es **específica a Q empírica** (marginal fijada al entrenamiento); con Q paramétrica (GIG) P(stay) podría tener efecto.

**A.6 — "Saltos continuos inútiles": CIERTA POR CONSTRUCCIÓN, EVIDENCIA MAL ATRIBUIDA.** El esqueleto discreto es representación exacta del proceso continuo en la rejilla — la claim es cierta por construcción, no por el experimento citado. **No existe implementación en tiempo continuo contra la cual comparar**, así que "no pierde nada vs la continua" es no-falsificable en este proyecto.

**Marginal = Empirical(r_t²): TAUTOLÓGICA + IMPRECISIÓN TERMINOLÓGICA.** Cierto solo porque Q se FIJA a la empírica. Imprecisión: la cadena vive en log(RV), así que la marginal de τ es `Empirical(RV)`, NO `Empirical(r_t²)` (retorno cuadrado crudo).

### 5.3 Fugas de datos y comparaciones injustas

**FUGA DE DATOS [bloqueante].** El pipeline empírico tiene fuga de información train/test:
- `detect_and_remove_jumps(returns)` se ejecuta sobre TODA la serie antes del split 80/20 (`anzarut_replication.py:559`).
- `estimate_periodicity(returns)` (U-shape intradía) se calcula sobre TODA la serie antes del split (`:566`).
- `dollar_threshold` se calcula sobre el df completo en `anzarut_intraday_15min.py:385`, mientras `table3_validation.py` y `gibbs_table3.py` sí evitan la fuga.

La estructura de saltos y la pauta intradía del test filtran al entrenamiento, **contaminando el headline AAD=0.2pp** y todos los scripts que importan estas funciones.

**COMPARACIÓN INJUSTA MEXICANOS DIARIOS.** "Hist.Sim gana 8/8" no es apples-to-apples: SF-Harris es **fixed-origin horizonte-largo** (simula TODOS los n_test pasos desde `last_val=train[-1]` en un solo disparo, sin re-estimar), mientras Hist.Sim es iid bootstrap (cada punto marginal). Además SF-Harris usa emisión `N(0,τ)` con colas Gaussianas delgadas, perdiendo la cola pesada empírica que Hist.Sim preserva.

**AUC=0.416 DE CRISIS: MODELO DESCALIBRADO.** El número titular proviene de `crisis_indicator.py` v1, que entrena en pre-GFC (1998-2007) y testea 2008-2026 con modelo congelado. El docstring de v2 admite que v1 tenía "PIT mean=0.36, not 0.50" (descalibrado). Un AUC<0.5 de un modelo descalibrado no es evidencia robusta de anti-predicción. v2 (rolling window, refit anual) es metodológicamente correcta pero el reporte lidera con el número de v1.

**AAD=2.5pp DE RETORNO: ARTEFACTO DEL RUIDO DE EMISIÓN.** La cobertura de retornos diarios (2.5pp) se reporta como MEJOR que la cobertura de RV diaria (13pp), lo cual es contraintuitivo. La emisión `N(μ+β·τ*, τ*)` añade ruido Gaussiano que ensancha mecánicamente los intervalos predictivos, enmascarando la mala calibración de τ*.

### 5.4 Tests circulares o con problemas

| Test | Problema | Impacto |
|---|---|---|
| **Test 1 (Descomposición)** | `aad5 = aad1` (línea 108): el Modelo 5 (N(0,τ*)) se DECLARA idéntico al Modelo 1 (SF-Harris) por fiat, no se computa. La descomposición "τ* aporta 7.4pp, empirical shape aporta 0.0pp" se deriva de esta igualdad circular. Además Modelo 6 "oracle" remuestrea de `test_ret` (fuga). Y usa ε=1e-5 (P(stay)=0.37, cadena destruida) como strawman. | La claim "SF-Harris = N(0,τ*) = 2.0pp" es trivialmente cierta por construcción |
| **Test 7 (Stay/Jump)** | Bug off-by-one en `jump_confusion_matrix.py:133-137`: `recent = recent[1:min_len+1]` alinea el predictor con el target del MISMO instante, haciendo `pred_score3[j] == is_jump[j]`. El AUC=1.0 es artefacto de alineación, no "circularidad de ayer" como dice el reporte. | El AUC=0.511 del ratio τ* (hallazgo relevante) está bien computado |
| **Test 10 (P(stay) sweep)** | NO barre P(stay) en el pipeline Gibbs real. Approach A (con ruido) da 18.66pp plano; los 0.41-0.54pp vienen de Approach B (sin ruido = Q empírica pura). | "P(stay) no-informativa con denoising Gibbs" no está demostrada en el régimen del Gibbs real |
| **Test 2 (ε calibration)** | La flatness cross-ε puede ser trivial: Q_type='empirical' remuestrea train siempre, fijando la marginal independientemente de P(stay). Específica a Q empírica, no general. | Apoya la tesis pero el mecanismo es "Q empírica fija la marginal", no "P(stay) no importa" |
| **Comparación estimadores simples** | `simple_estimators_test.py` reporta 26pp para EWMA/rolling (versión Gauss) pero OMITE la versión EmpQ (bootstrap de residuales) que podría ser mucho mejor. | Exagera la ventaja del Gibbs |

### 5.5 "Decorativo": ¿afirmación que excede la evidencia?

El framing "la cadena de Harris es decorativa" sobrepasa la evidencia en un punto crucial: **el Gibbs sampler —que ES el motor de inferencia de la cadena Harris— añade valor enorme a 15-min** (0.2pp vs 26pp de estimadores simples). El reporte a veces concede esto ("2 ingredientes: Q empírica + denoising Gibbs") pero el titular "Harris chain is decorative" sugiere que toda la cadena es inútil, cuando el denoising Gibbs (parte de la cadena) es uno de los 2 ingredientes esenciales.

La distinción correcta es: **las DINÁMICAS TEMPORALES (P(stay)) son decorativas para cobertura** (propiedad marginal), pero el **motor de inferencia (Gibbs) no lo es**. El framing debería ser "la dinámica temporal de la cadena es decorativa para cobertura bajo estacionariedad con Q fija", no "la cadena es decorativa".

### 5.6 No-reproducibilidad del cap de α

| Archivo | E_α(NDNJ, k=20) | Cap |
|---|---|---|
| table3_comparison.csv | 1.03 | ¿? |
| simulation_results.npz | 0.9561 | 50 |
| simulation_results_gibbsQ_cap10.npz | 0.2267 | 10 |
| _restricted.npz | 0.3986 | ¿? |

Cuatro valores distintos para la misma celda. El código actual (cap=50 hardcodeado en 11 líneas distintas) solo puede regenerar `simulation_results.npz`. **La corrida que replica a Anzarut (cap=10, 0.22) no es reproducible desde el repositorio actual.**

---

## 6. Fortalezas y debilidades

### 6.1 Del modelo

| Fortalezas | Debilidades |
|---|---|
| Estructura elegante: punto de masa + salto desde Q invariante, con ACF analítica `e^{-αh}` | La dinámica temporal (P(stay)) es decorativa para cobertura bajo estacionariedad con Q fija — es estructural, no un hallazgo |
| Conjugación explícita vuelve al Gibbs eficiente | Actualización de α usa Gamma (tiempo continuo) en lugar de Beta (tiempo discreto) exacto; ignora censura a derecha → sesgo |
| Promedio posterior resuelve la casi-no-identificabilidad del GIG | El "posterior" sobre (λ,χ,ψ) es en realidad la verosimilitud sin prior explícito (prior plano impropio) |
| Q empírica equivale a Historical Simulation — robusto | Q empírica no extrapola más allá de datos observados (P99.9+); GIG podría ganar en colas extremas |
| Esqueleto discreto es representación exacta del proceso continuo en la rejilla | Emisión N(0,τ) con colas Gaussianas delgadas pierde la cola pesada empírica que Hist.Sim preserva |

### 6.2 Del trabajo

| Fortalezas | Debilidades |
|---|---|
| Cadena conceptual coherente: fundamentos → núcleo → replicación → tesis → aplicaciones → reporte | Fuga de datos train/test en el pipeline empírico (saltos, periodicidad, umbral dollar sobre toda la serie) |
| Estudio de simulación que valida estimadores (Tablas II/III) | NDNJ/MLE/EM NO son 3 métodos distintos en Q continua — son bit-idénticos; la "Tabla III" realmente compara 3 estimadores, no 5 |
| 10 tests de ablación que delimitan qué es SF-Harris y qué no es | Tests circulares (Test 1), con bug (Test 7), strawman (ε=1e-5), y que no barren el Gibbs real (Test 10) |
| Aplicaciones financieras concretas (VaR, portafolios, variance swaps) | Cap. 6 (Aplicaciones) es en gran parte especulativo, sin backtest ni evidencia numérica |
| Delimitación honesta de regímenes (15-min vs diario) | Inconsistencia 15-min: work_index reporta SF-Harris gana; decorative_report reporta Hist.Sim gana. Ningún documento reconcilia |

### 6.3 Del código

| Fortalezas | Debilidades |
|---|---|
| Librería núcleo bien tipada (numpy.typing), con docstrings matemáticos, APIs coherentes | **Crisis de reproducibilidad**: IBM.txt, CSV/NPZ de resultados y pickle del notebook NO están versionados; rutas absolutas hardcodeadas a `C:\Users\angve\...` |
| 85/85 tests pasan; primitivas de resampling (4 esquemas) estándar y limpias | Los estimadores que sostienen TODAS las claims comparativas (gibbs_*, _mle_gig) NO tienen tests unitarios |
| Verosimilitudes y momentos verificados (KL correcto, falso positivo desmentido) | Tests rotos/tautológicos: `assert not np.array_equal(...) or True`; test sin aserciones |
| Demanda técnica BAJA en primitivas (resampling, ESS, importance, kernels) | Demanda técnica ALTA en scripts: 38 scripts redundantes (diag_gig_mle≈diag_mle_gig, 4 scripts *_table3, crisis_indicator v1 obsoleto) |
| | MCMC sin diagnósticos de convergencia: no hay R-hat, ESS, ni inspección de cadenas; solo burn-in fijo |
| | `count_transitions` detecta stays por IGUALDAD EXACTA de floats: la librería núcleo es INUTILIZABLE sobre datos reales continuos (n_stay=0 → α→cap 50); el umbral ε, la actualización y la emisión viven en el script monolítico |
| | Código muerto: `_get_jump_values` definida pero NUNCA usada (docstring miente "Used by NDNJ"); `_gig_log_lik` duplica `_gig_log_lik_params` |

---

## 7. Preguntas abiertas y recomendaciones

### 7.1 Para el autor

**Urgentes (defensa de tesis):**

1. **Corregir la fuga de datos.** Recalcular periodicity y detectar saltos usando SOLO la porción de entrenamiento antes del split 80/20. Reportar el AAD tras corregir la fuga para cuantificar el impacto real sobre el headline 0.2pp. Unificar el tratamiento del umbral dollar entre scripts.

2. **Reformular la Tabla III como comparación de 3 estimadores**, no 5. Reportar la identidad teórica NDNJ=MLE=EM para Q continua como hallazgo, no como 3 columnas separadas. Eliminar el ruido MC compartiendo el mismo rng-state o reportar intervalos de incertidumbre.

3. **Resolver la no-reproducibilidad del cap de α.** Hacer el cap un parámetro explícito y versionado. Regenerar TODOS los archivos con un único cap documentado, o reportar ambos regímenes (cap=10 y cap=50). Resolver la contradicción en `table3_notes.txt`.

4. **Corregir la actualización de α.** Reemplazar la actualización Gamma por la conjugada Beta EXACTA en tiempo discreto: `m ~ Bin(n, 1-e^{-α})`, `p = 1-e^{-α}`, posterior `Beta(a+m, b+n-m)`, muestrear `p` y transformar `α = -log(1-p)`. Si se retiene Gamma, corregir la censura usando `T=n` como escala, no `t_m`.

5. **Reformular "P(stay) no-informativa" honestamente.** Como: "La cobertura es una propiedad marginal; P(stay) afecta solo la dependencia temporal (conjunta), no la marginal estacionaria = Q; por ello es estructuralmente no-informativa para cobertura bajo estacionariedad y Q fija". Distinguir "no-informativa dentro del Gibbs" de "destructiva sin denoising". Ejecutar el sweep DENTRO del pipeline Gibbs real. Reportar honestamente que la flatness ocurre en ambos regímenes pero con magnitudes muy distintas (0.4pp sin ruido vs 18pp con ruido).

6. **Reconciliar la inconsistencia 15-min.** work_index (SF-Harris gana 0.2pp vs 2.0pp) vs decorative_report (Hist.Sim gana 1.9pp vs 2.3pp). Un documento debe explicar qué mide cada uno.

7. **Declarar que "decorativo" se refiere a la dinámica temporal (P(stay)), no a toda la cadena.** El Gibbs (parte de la cadena) es uno de los 2 ingredientes esenciales.

8. **Corregir bugs de tests.** Bug off-by-one en Test 7 (`jump_confusion_matrix.py:133-137`); circularidad en Test 1 (`aad5 = aad1`); strawman ε=1e-5; Test 10 no barre el Gibbs real. Repetir la descomposición con ε=0.1 (P(stay)=0.88, cadena preservada).

9. **Reportar el AUC de crisis de v2** (rolling, recalibrado) como cifra principal. Etiquetar v1 como modelo congelado descalibrado.

10. **Versionar datos y rutas.** Versionar IBM.txt y VIX_History.csv (o documentar su obtención con un script de descarga determinista). Eliminar rutas absolutas hardcodeadas. Mover CSV/NPZ a `data/results/` versionados.

**Metodológicas:**

11. Añadir diagnósticos de convergencia MCMC: R-hat (≥1.1), ESS, trazas con ≥2 cadenas; reportarlos junto a las medias posteriores. Justificar el prior sobre (λ,χ,ψ) (no prior plano impropio).

12. Implementar Newey-West HAC en el test de Diebold-Mariano (lag ≈ n^(1/3)); usar distribución t con gl apropiados. Re-evaluar las conclusiones de significancia.

13. Comparar SF-Harris y Hist.Sim en el mismo régimen (ambos fixed-origin O ambos rolling 1-paso-adelante). Probar emisión empírica (r|τ resampleado) vs Gaussiana para aislar el efecto de colas. Re-estimar GARCH rolling.

14. Reportar media ± SE (o IC95%) para cada celda del estudio de simulación; reportar nº de fallbacks por celda; aumentar a ≥500 réplicas.

15. Validar que `simulate_egarch`/`simulate_heston` producen escalas correctas antes de comparar (los 30-40pp sugieren benchmarks rotos).

**De presentación:**

16. Reconciliar el conteo de tests (¿10 u 11?) y enumerarlos explícitamente.

17. Incluir el estudio de simulación (Sección 3.5.2 de Anzarut) en el reporte principal, no solo en el borrador.

18. Incluir el matiz "SF-Harris gana 5/8 en intradía 1h" en el reporte principal.

19. Corregir la definición de Harris-recurrencia (Cap.2): añadir la condición de minorización (ψ-cimiento).

20. Corregir referencias bibliográficas: eliminar `geman2015` (placeholder); corregir `anderson1960` (mashup fabricado); resolver la discrepancia de año de la tesis fuente (2024 vs 2017).

21. Definir formalmente `τ*` (denoised) al introducirlo, no tardíamente.

22. Etiquetar correctamente el "SF-Harris gana 5/8 intradía" como "Q-empírica+denoising vs Hist.Sim intradía" (no usa el modelo SF-Harris real). Esto APOYA la tesis decorativa, pero debe etiquetarse honestamente.

### 7.2 Para el asesor

1. **Exigir la corrección de la fuga de datos antes de aprobar el reporte.** Es el hallazgo metodológico más grave: contamina el headline AAD=0.2pp.

2. **Pedir que la tesis "decorativa" se reformule con la distinción correcta** entre "dinámica temporal decorativa" (P(stay)) y "motor de inferencia no-decorativo" (Gibbs).

3. **Verificar que la claim "replica a Anzarut" sea reproducible** desde el repositorio actual.

4. **Solicitar diagnósticos MCMC** antes de aceptar las medias posteriores como estimadores.

5. **Evaluar si el estudio de simulación debe estar en el cuerpo principal.**

6. **Considerar el alcance de la tesis.** Delimitar lo probado (dinámica temporal decorativa) de lo propuesto (aplicaciones especulativas).

7. **Reconocer el valor pedagógico.** La cadena conceptual (M1-M3, fundamentos → núcleo → Gibbs) es coherente y bien construida; las primitivas de resampling, Kalman y SS están correctamente implementadas. El estudiante internalizó la teoría de Chopin.

---

## Síntesis valorativa final

El trabajo demuestra, con evidencia mixta pero dirección correcta, que **la dinámica temporal de la cadena de Harris (P(stay)) es decorativa para predicción de cobertura bajo estacionariedad con Q fija** — y esta es una observación estructural correcta (la cobertura es propiedad marginal, P(stay) gobierna lo conjunto), no un hallazgo empírico sorpresivo. El modelo se reduce a dos ingredientes: Q empírica (≈ Historical Simulation) y denoising bayesiano (Gibbs). El esqueleto discreto es representación exacta del proceso continuo en la rejilla observada.

Sin embargo, la evidencia que sostiene esta tesis tiene problemas reales: fuga de datos en el pipeline empírico, tests circulares (Test 1), bug off-by-one (Test 7), un sweep de P(stay) que no opera en el régimen del Gibbs real (Test 10), una "Tabla III" que en realidad compara 3 estimadores (no 5), una replicación de Anzarut no reproducible desde el código actual, y un framing "decorativo" que sobrepasa la evidencia al sugerir que toda la cadena es inútil cuando el Gibbs (parte de ella) es esencial.

De las 6 afirmaciones centrales del modelo, 3 son matemáticamente correctas con matices (invariancia de Q, factorización NDNJ=MLE=EM, ACF e^{-αh}), 1 es una descripción errónea del código con sesgo de aproximación (actualización de α como Beta cuando es Gamma sin censura), y 2 tienen la dirección correcta pero atribuyen mal la evidencia y no articulan la razón estructural (P(stay) no-informativa, saltos continuos inútiles — ambas ciertas por construcción, no por los experimentos citados). La divergencia KL es correcta (falso positivo desmentido por verificación Monte Carlo).

La librería núcleo está razonablemente bien estructurada y tipada (85/85 tests), pero los scripts experimentales son la fuente principal de deuda técnica y de los hallazgos metodológicos graves. La crisis de reproducibilidad (datos no versionados, rutas hardcodeadas, resultados no regenerables) debe resolverse antes de la defensa.

Para el estudiante de actuaría que defiende esta tesis: el núcleo intelectual —entender por qué la cobertura es una propiedad marginal y por qué la dinámica temporal no aporta bajo estacionariedad— es sólido y valioso. Los puntos ciegos están en la ejecución: fuga de datos, comparaciones injustas, tests circulares, framing que excede la evidencia, y MCMC sin diagnósticos. Corregir estos puntos elevaría el trabajo de "observación correcta con evidencia problemática" a "demostración rigurosa".