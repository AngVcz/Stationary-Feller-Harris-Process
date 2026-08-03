# M2 — Síntesis: Filtrado, Predicción y el Ciclo Predict/Update

## Lo que aprendiste

- El **filtrado** p(x_t | y_{0:t}) se calcula con un ciclo recursivo predict/update:
  - **Predict** (Chapman-Kolmogorov): propaga la creencia filtrada hacia adelante usando el núcleo M
  - **Update** (Bayes): incorpora la observación multiplicando por G y normalizando

- La recursión **arranca desde la prior** p(x_0). No necesitas calcular nada "desde cero" — cada paso usa el resultado del anterior.

- El **formalismo Feynman-Kac** envuelve la recursión de filtrado en un marco unificado que produce filtrado, predicción, verosimilitud y suavizado como casos especiales.

- El **filtro de Kalman** es el caso especial donde todo se calcula en forma cerrada porque la familia gaussiana es cerrada bajo predict (convolución) y update (producto de gaussianas).

## Conexión con lo que ya sabes

| Concepto nuevo | Equivalente en M1 |
|---------------|-------------------|
| Predict = Chapman-Kolmogorov | Núcleo M propagando una distribución |
| Update = Bayes con núcleo G | Núcleo G actuando como verosimilitud |
| p(x_0) = prior | Distribución inicial del AR(1) |
| Ganancia de Kalman K_t | Cuánto confiar en la observación vs. la predicción |
| Log-verosimilitud | Normalización del update de Bayes (subproducto) |

## Las cinco preguntas de inferencia

| Problema | Usa M | Usa G | Usa L | Código |
|----------|:-----:|:-----:|:-----:|--------|
| Filtrado | ✓ | ✓ | ✗ | `predict()` + `update()` |
| Predicción | ✓ | ✗ | ✗ | Solo `predict()` |
| Suavizado fijo | ✓ | ✓ | ✓ (ventana) | M6 |
| Suavizado completo | ✓ | ✓ | ✓ (completo) | M6 (FFBS) |
| Verosimilitud | ✓ | ✓ | ✗ | `filter().log_likelihoods` |

## Lo que viene después (M3)

El siguiente módulo introduce **importance sampling y resampling** — la técnica que permite implementar el ciclo predict/update cuando las distribuciones NO son gaussianas. En vez de representar p(x_t | y_{0:t}) con una media y covarianza, la representas con **partículas** y **pesos**.

La intuición clave: el filtro de partículas hace exactamente lo mismo que el filtro de Kalman (predict/update), pero con muestras en vez de gaussianas.

## Pregunta de autoevaluación

Si tienes un SSM no lineal donde X_t = f(X_{t-1}) + ε_t con f no lineal, ¿por qué no puedes usar las ecuaciones del filtro de Kalman directamente? ¿Qué parte se rompe?