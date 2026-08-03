# M2 — Respuestas Ideales: Filtrado, Predicción y el Ciclo Predict/Update

## 0. ¿De dónde sale p(x_{t-1} | y_{0:t-1})?

**Pregunta:** ¿Cómo se calcula la distribución filtrada que entra al predict?

**Respuesta ideal:**

No se calcula "desde cero" — es **recursiva**. La filtrada del paso anterior es el insumo para el predict del paso siguiente. La cadena arranca de la prior p(x_0):

```
p(x₀)                                           ← prior (tú la eliges)
  → update → p(x₀ | y₀)                         ← primera filtrada
  → predict → p(x₁ | y₀) = ∫ M · p(x₀ | y₀)  ← usa la filtrada anterior
  → update  → p(x₁ | y₀:₁)                      ← nueva filtrada
  → predict → p(x₂ | y₀:₁) = ∫ M · p(x₁ | y₀:₁)  ← usa la filtrada anterior
  → ...
```

En el código: `self._x_filt` y `self._P_filt` almacenan p(x_{t-1} | y_{0:t-1}). Después de cada update, se guardan. En el siguiente predict, se usan. La única distribución que necesitas desde fuera es la prior p(x_0).

---

## 1. Paso de predicción usando notación de núcleos

**Pregunta:** Escribe el predict step usando la notación de núcleos de M1.

**Respuesta ideal:**

$$p(x_t \mid y_{0:t-1}) = \int M(x_{t-1}, x_t) \, p(x_{t-1} \mid y_{0:t-1}) \, dx_{t-1}$$

Desglose:

- **p(x_{t-1} | y_{0:t-1})** — distribución filtrada del paso anterior (el "insumo")
- **M(x_{t-1}, x_t)** — núcleo de transición (la dinámica del modelo)
- **∫ ... dx_{t-1}** — marginalizar sobre todos los estados anteriores posibles

Esto es **Chapman-Kolmogorov**: propagar una distribución a través de un núcleo. No usa Bayes ni el núcleo G — es pura dinámica.

**En Kalman:** x̂_{t|t-1} = A x̂_{t-1|t-1}, P_{t|t-1} = A P_{t-1|t-1} A' + Q. La integral se resuelve en forma cerrada porque la convolución de gaussianas es gaussiana.

---

## 2. ¿Por qué "multiplicar por G" da la regla de Bayes?

**Pregunta:** Si el update es "multiplicar por G(x_t, y_t)", ¿por qué esto da Bayes?

**Respuesta ideal:**

Porque G(x_t, y_t) **es** la verosimilitud, y la normalización completa la regla de Bayes.

Bayes dice:

$$p(x_t \mid y_{0:t}) = \frac{p(y_t \mid x_t, y_{0:t-1}) \, p(x_t \mid y_{0:t-1})}{p(y_t \mid y_{0:t-1})}$$

En un SSM, dado que Y_t es condicionalmente independiente de Y_{0:t-1} dado X_t:

$$p(y_t \mid x_t, y_{0:t-1}) = p(y_t \mid x_t) = G(x_t, y_t)$$

Entonces:

$$p(x_t \mid y_{0:t}) = \frac{G(x_t, y_t) \cdot p(x_t \mid y_{0:t-1})}{p(y_t \mid y_{0:t-1})}$$

El numerador es "predict × G" y el denominador es la constante de normalización. El update step hace dos cosas:

1. **Multiplica** la predictiva por G → numerador (distribución no normalizada)
2. **Normaliza** → divide por p(y_t | y_{0:t-1}) = ∫ G(x_t, y_t) p(x_t | y_{0:t-1}) dx_t

El paso 2 es clave: la constante de normalización **es la verosimilitud** p(y_t | y_{0:t-1}). Por eso el filtro de Kalman calcula la log-verosimilitud como subproducto.

**En Kalman:** La ganancia de Kalman K_t y la covarianza actualizada P_{t|t} son el resultado de normalizar el producto de dos gaussianas. El producto de gaussianas es gaussiano, y los parámetros se calculan algebraicamente.

---

## 3. ¿Por qué p(x_t | y_{0:t}) se mantiene gaussiana?

**Pregunta:** En el caso lineal-Gaussiano, ¿por qué la distribución filtrada se mantiene gaussiana después de cada predict/update?

**Respuesta ideal:**

Porque la familia gaussiana es **cerrada** bajo las dos operaciones del ciclo predict/update:

**Predict (convolución gaussiana):**

Si p(x_{t-1} | y_{0:t-1}) = N(μ, Σ) y la transición es x_t = A x_{t-1} + v_t con v_t ~ N(0, Q), entonces:

$$p(x_t \mid y_{0:t-1}) = N(A\mu,\, A\Sigma A' + Q)$$

La convolución de gaussianas es gaussiana. La media se transforma linealmente, la covarianza se transforma y se suma Q.

**Update (producto de gaussianas):**

Si la predictiva es N(μ_pred, Σ_pred) y la verosimilitud es N(y_t; H x_t, R), el posterior es gaussiano:

$$p(x_t \mid y_{0:t}) = N(\mu_{filt},\, \Sigma_{filt})$$

donde:
- μ_filt = μ_pred + K(y_t - H μ_pred)
- Σ_filt = (I - KH) Σ_pred
- K = Σ_pred H'(H Σ_pred H' + R)^{-1}

El producto de dos gaussianas (normalizado) es gaussiano. Esto es una propiedad algebraica: el exponente cuadrático se combina en otro exponente cuadrático.

**¿Cuándo se rompe?**

Si el modelo NO es lineal-Gaussiano, al menos una de estas dos operaciones produce una distribución no gaussiana:

- Si la transición es no lineal (X_t = f(X_{t-1}) + ε_t): la convolución de una gaussiana con una transformación no lineal no es gaussiana
- Si la verosimilitud no es gaussiana (Y_t | X_t ~ Poisson(exp(X_t))): el producto de una gaussiana con una Poisson no es gaussiana

En esos casos, necesitas el **filtro de partículas** (M4) para representar la distribución filtrada con muestras y pesos.

---

## 4. Los cinco problemas de inferencia

**Filtrado** p(x_t | y_{0:t}): Usa M y G (forward only). Es lo que ya implementamos — predict + update en cada paso.

**Predicción** p(x_{t+h} | y_{0:t}): Usa solo M (forward sin update). Solo propagas la incertidumbre hacia adelante. En Kalman: solo `predict()`, sin `update()`.

**Suavizado fijo** p(x_{t-l:t} | y_{0:t}): Usa M, G y L (forward + backward en ventana). Corrige estimaciones recientes usando observaciones actuales.

**Suavizado completo** p(x_{0:t} | y_{0:t}): Usa M, G y L (forward + backward completo). Reconstruye toda la trayectoria. En Kalman: RTS smoother.

**Verosimilitud** p(y_{0:t}): Usa M y G (forward). Subproducto del filtrado — la constante de normalización del update. En Kalman: ya lo tenemos en `filter().log_likelihoods`.

**Núcleo backward** L_{t+1}(x_{t+1}, x_t) = p(x_t | x_{t+1}, y_{0:t}): Invierte la flecha del tiempo condicionalmente. No es simplemente "M al revés" — combina M con la distribución filtrada. Solo se necesita para suavizado.

---

## 5. ¿Cómo se elige/estima M?

M es parte de la **especificación del modelo**. No se infiere durante el filtrado.

- Si el modelo es AR(1) + ruido: M está determinada por φ y σ_v (parámetros que tú eliges)
- Si no conoces los parámetros: necesitas **estimación** (M7 — MLE, EM)
- En HMMs con estados discretos: la matriz de transición P se estima contando transiciones (si los estados son observados) o con EM (si son latentes)

El filtrado asume que M y G son conocidas. La estimación de parámetros es un problema separado que usa la verosimilitud que el filtro calcula como subproducto.