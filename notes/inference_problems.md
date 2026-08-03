# Problemas de Inferencia en Modelos de Espacio de Estados

Cada problema pregunta algo distinto sobre el estado oculto. La diferencia está en **qué** quieres saber y **cuándo** (con qué información).

## Resumen rápido

| Problema | Pregunta | Información usada | Complejidad |
|----------|----------|-------------------|-------------|
| Filtrado | ¿Dónde está X_t ahora? | Y_{0:t} (pasado + presente) | Recursivo, O(t) |
| Predicción | ¿Dónde estará X_{t+h}? | Y_{0:t} (solo pasado) | Recursivo, O(h) |
| Suavizado fijo | ¿Dónde estuvo X_{t-l}? | Y_{0:t} (pasado + presente + reciente) | Recursivo + buffer, O(l) |
| Suavizado completo | ¿Cuál fue toda la trayectoria? | Y_{0:t} (toda la historia) | Requiere_backward pass, O(t) |
| Verosimilitud | ¿Qué tan probables son las observaciones? | Y_{0:t} | Subproducto del filtrado |

---

## 0. ¿De dónde sale p(x_{t-1} | y_{0:t-1})? — La cadena de recurrencia

La recursión de filtrado no calcula p(x_{t-1} | y_{0:t-1}) "desde cero" en cada paso. Es **recursiva**: la filtrada del paso anterior es el insumo para el predict del paso siguiente.

La cadena arranca desde la **prior** p(x_0), que tú eliges (en Kalman: `x0_mean`, `x0_cov`):

```
t=0:  Tienes p(x₀) ← prior (la eliges tú)
      │
      ├─ update → p(x₀ | y₀) ∝ G(x₀, y₀) · p(x₀)      ← primera filtrada
      │
t=1:  ├─ predict → p(x₁ | y₀) = ∫ M(x₀, x₁) p(x₀ | y₀) dx₀
      │              ↑ usa la filtrada del paso anterior
      ├─ update  → p(x₁ | y₀:₁) ∝ G(x₁, y₁) · p(x₁ | y₀)
      │              ↑ esta se convierte en la filtrada para el siguiente predict
t=2:  ├─ predict → p(x₂ | y₀:₁) = ∫ M(x₁, x₂) p(x₁ | y₀:₁) dx₁
      │                              ↑ usa la filtrada del paso anterior
      └─ ...
```

**En código**, esto es exactamente lo que hace tu `KalmanFilter`:

```python
for t in range(T):
    x_pred, P_pred = self.predict()        # usa self._x_filt, self._P_filt
    x_filt, P_filt = self.update(y_t, P_pred)  # produce nueva filtrada
    self._x_filt = x_filt   # se guarda para el siguiente predict
    self._P_filt = P_filt   # se guarda para el siguiente predict
```

`self._x_filt` y `self._P_filt` **son** p(x_{t-1} | y_{0:t-1}). Después de cada update, se guardan. En el siguiente predict, se usan. La única distribución que necesitas "desde fuera" es la prior p(x₀).

---

## 1. Filtrado: p(x_t | y_{0:t})

**¿Qué quieres?** La distribución del estado actual dadas todas las observaciones hasta ahora.

**¿Cuándo?** Monitoreo en tiempo real, tracking, navegación. Si quieres saber "¿dónde está el objeto ahora?", necesitas filtrado.

**¿Cómo se calcula?** Ciclo predict/update (ver sección 0 para de dónde viene cada término):

```
p(x_t | y_{0:t-1}) = ∫ M(x_{t-1}, x_t) p(x_{t-1} | y_{0:t-1}) dx_{t-1}    [Predict: Chapman-Kolmogorov]

p(x_t | y_{0:t}) ∝ G(x_t, y_t) · p(x_t | y_{0:t-1})                        [Update: Bayes]
```

- **Predict** usa el núcleo **M** (dinámica del modelo)
- **Update** usa el núcleo **G** (observación)

**En Kalman:** x̂_{t|t}, P_{t|t} — exactamente lo que ya implementamos. Cada paso de `predict()` + `update()` te da la distribución filtrada.

**¿Qué núcleos necesitas?** Solo M y G. Solo hacia adelante (forward).

---

## 2. Predicción de estado: p(x_{t+h} | y_{0:t})

**¿Qué quieres?** La distribución del estado futuro, h pasos adelante.

**¿Cuándo?** Predecir la trayectoria de un huracán, forecasting financiero, planificación.

**¿Cómo se calcula?** Aplica predict h veces, sin update:

$$p(x_{t+1} | y_{0:t}) = \int M(x_t, x_{t+1}) p(x_t | y_{0:t}) dx_t$$

$$p(x_{t+2} | y_{0:t}) = \int M(x_{t+1}, x_{t+2}) p(x_{t+1} | y_{0:t}) dx_{t+1}$$

Y así sucesivamente. Sin observaciones futuras, solo propagas la incertidumbre hacia adelante con **M**.

**En Kalman:** Solo `predict()`, sin `update()`. La covarianza crece en cada paso (la incertidumbre aumenta porque no tienes nuevas observaciones).

**¿Qué núcleos necesitas?** Solo **M**. Solo hacia adelante (forward).

**¿Por qué no G?** Porque no tienes observaciones futuras. Solo tienes la dinámica del modelo.

---

## 3. Suavizado de rezago fijo: p(x_{t-l:t} | y_{0:t})

**¿Qué quieres?** La distribución de estados recientes (ventana de tamaño l), usando observaciones hasta el presente.

**¿Cuándo?** Detección de fallas con ligero retraso,GPS con corrección retrospectiva de los últimos l pasos.

**¿Cómo se calcula?** Filtrado forward + backward pass sobre una ventana de tamaño l:

1. Filtrado forward hasta t → p(x_t | y_{0:t})
2. Backward pass con el núcleo backward L_{t+1}(x_{t+1}, x_t) sobre los últimos l pasos

El núcleo backward es:

$$L_{t+1}(x_{t+1}, x_t) = p(x_t | x_{t+1}, y_{0:t}) = \frac{M(x_t, x_{t+1}) p(x_t | y_{0:t})}{\int M(x'_t, x_{t+1}) p(x'_t | y_{0:t}) dx'_t}$$

**En Kalman:** Se puede hacer con un buffer de tamaño l. No necesitas guardar toda la historia.

**¿Qué núcleos necesitas?** M (forward) + L (backward sobre la ventana).

---

## 4. Suavizado completo: p(x_{0:t} | y_{0:t})

**¿Qué quieres?** La distribución de toda la trayectoria, usando todas las observaciones.

**¿Cuándo?** Reconstrucción histórica (¿cuál fue la trayectoria real del sistema?), estimación de parámetros vía EM (necesitas E[X_t X_{t-1}' | y_{0:t}] para actualizar A).

**¿Cómo se calcula?** Forward filtering + backward sampling (FFBS):

1. **Forward pass:** Filtrado estándar → guarda p(x_t | y_{0:t}) para todo t
2. **Backward pass:** Desde t hacia atrás, muestrea:

$$x_t \sim p(x_t | x_{t+1}, y_{0:t}) \propto p(x_t | y_{0:t}) \cdot M(x_t, x_{t+1})$$

Esto usa el **núcleo backward** L que conecta cada paso con el siguiente.

**En Kalman:** El suavizado completo se llama Rauch-Tung-Striebel (RTS) smoother. Es un backward pass que corrige las estimaciones filtradas usando información futura.

**¿Qué núcleos necesitas?** M (forward) + L (backward completo).

**¿Por qué es mejor que el filtrado?** Porque el filtrado en t solo usa Y_{0:t}. El suavizado usa Y_{0:T} con T > t, así que las observaciones futuras retroalimentan información sobre estados pasados.

---

## 5. Cálculo de verosimilitud: p(y_{0:t})

**¿Qué quieres?** La probabilidad marginal de las observaciones bajo el modelo.

**¿Cuándo?** Estimación de parámetros (MLE), comparación de modelos, diagnóstico.

**¿Cómo se calcula?** Es un **subproducto del filtrado**. En cada update:

$$p(y_t | y_{0:t-1}) = \int G(x_t, y_t) \cdot p(x_t | y_{0:t-1}) dx_t$$

Esto es la normalización del update de Bayes. Y la verosimilitud total:

$$p(y_{0:t}) = \prod_{s=0}^{t} p(y_s | y_{0:s-1})$$

**En Kalman:** Ya lo implementamos — `log_likelihoods` en `FilterResult`. Cada paso calcula:

$$\log p(y_t | y_{0:t-1}) = -\frac{1}{2}\left(m \log(2\pi) + \log |S_t| + v_t' S_t^{-1} v_t\right)$$

donde v_t es la innovación y S_t su covarianza.

**¿Qué núcleos necesitas?** M (predict) + G (update). Solo forward — la verosimilitud es gratuita.

---

## ¿Cuál elegir?

| Situación | Problema | Método |
|-----------|----------|--------|
| "¿Dónde está el objeto ahora?" | Filtrado | Forward only (predict + update) |
| "¿Dónde estará en 5 pasos?" | Predicción | Forward, solo predict (sin update) |
| "¿Dónde estuvo hace 3 pasos, considerando lo que sé ahora?" | Suavizado fijo | Forward + backward (ventana corta) |
| "¿Cuál fue la trayectoria completa?" | Suavizado completo | Forward + backward completo (FFBS) |
| "¿Qué tan bueno es el modelo?" | Verosimilitud | Subproducto del filtrado |

**Regla práctica:**

- Si solo necesitas el presente → **filtrado** (más barato, O(t))
- Si necesitas el futuro → **predicción** (igual de barato, solo M)
- Si necesitas el pasado reciente → **suavizado fijo** (O(t + l), buffer pequeño)
- Si necesitas toda la trayectoria → **suavizado completo** (O(t) pero dos pasadas, guarda toda la historia)
- Si necesitas comparar modelos o estimar parámetros → **verosimilitud** (gratis con el filtrado)

**Núcleos que usa cada problema:**

| Problema | M (forward) | G (observación) | L (backward) |
|----------|:-----------:|:----------------:|:------------:|
| Filtrado | ✓ | ✓ | ✗ |
| Predicción | ✓ | ✗ | ✗ |
| Suavizado fijo | ✓ | ✓ | ✓ (ventana) |
| Suavizado completo | ✓ | ✓ | ✓ (completo) |
| Verosimilitud | ✓ | ✓ | ✗ |

La predicción es la más simple (solo M). El suavizado completo es el más caro (forward + backward). El filtrado y la verosimilitud son la misma pasada — la verosimilitud es "gratis" porque viene de la normalización del update.