# M2 — Lectura Preliminar: Filtrado, Predicción y el Ciclo Predict/Update

## Lo que ya sabes de M1

En M1 aprendiste:
- Un **núcleo de probabilidad** K(x, A) generaliza la matriz de transición a espacios continuos
- Un **SSM** tiene dos núcleos: M (dinámica) y G (observación)
- El **problema de filtrado** es recuperar p(x_t | y_{1:t}) — el estado oculto dado lo que observaste
- (Y_t) **no es Markov** — el estado oculto X_t crea puentes de dependencia entre observaciones

Ahora vamos a construir la **solución** al problema de filtrado para el caso lineal-Gaussiano.

---

## Paso 1 — El problema de filtrado, bien planteado

Queremos calcular la **distribución de filtrado**:

$$p(x_t \mid y_{1:t})$$

Es decir: dadas todas las observaciones hasta el tiempo $t$, ¿cuál es la distribución del estado $X_t$?

**¿Por qué no simplemente aplicar Bayes?**

Podrías pensar: "Aplico Bayes una vez y listo." Pero hay un problema: $p(x_t \mid y_{1:t})$ **cambia en cada instante**. Cada nueva observación $y_t$ modifica tu creencia sobre el estado. No basta con calcular una distribución estática — necesitas una **recursión** que actualice tu creencia paso a paso.

> **Intuición:** Piensa en un GPS que te da una lectura cada segundo. No recalculas desde cero cada vez — tomas tu última estimación, la propagas un segundo hacia adelante, y la corriges con la nueva lectura. Ese ciclo es exactamente el filtro.

**Conexión con CTMCs:**

En una CTMC, la distribución estacionaria satisface $\pi = \pi P$. Eso es una ecuación estática: la distribución no cambia en el tiempo.

El filtrado es el **análogo dinámico**: en vez de buscar una distribución fija, buscas una distribución que **se actualiza** con cada nueva observación. La recursión de filtrado es como $\pi = \pi P$, pero con una corrección bayesiana en cada paso.

---

## Paso 2 — El ciclo predict/update

La recursión de filtrado tiene **dos pasos** que se alternan:

### Predict (predicción)

$$p(x_t \mid y_{1:t-1}) = \int M(x_{t-1}, x_t) \, p(x_{t-1} \mid y_{1:t-1}) \, dx_{t-1}$$

**¿Qué hace?** Toma tu creencia filtrada del tiempo $t-1$ y la propaga un paso hacia adelante usando la dinámica del modelo (el núcleo $M$).

**Intuición:** La incertidumbre **aumenta** — estás prediciendo sin nueva información. Es como abrir los ojos en la niebla: sabes dónde estabas, pero ahora hay más incertidumbre sobre dónde estás.

**Conexión con M1:** $M(x_{t-1}, x_t)$ es exactamente el núcleo de transición que ya implementaste en `gaussian_kernel`. Para el AR(1): $M(x_{t-1}, x_t) = \mathcal{N}(x_t;\, \phi x_{t-1},\, \sigma_v^2)$.

### Update (actualización)

$$p(x_t \mid y_{1:t}) \propto G(x_t, y_t) \cdot p(x_t \mid y_{1:t-1})$$

**¿Qué hace?** Multiplica tu predicción por la verosimilitud de la observación $y_t$ (el núcleo $G$) y normaliza.

**Intuición:** La incertidumbre **disminuye** — la observación te da información nueva. Es como abrir los ojos en la niebla y ver una silueta: ahora sabes más que antes.

**Conexión con M1:** $G(x_t, y_t)$ es el núcleo de observación. Para el modelo lineal-Gaussiano: $G(x_t, y_t) = \mathcal{N}(y_t;\, x_t,\, \sigma_w^2)$.

### El ciclo completo

```
tiempo 0:     p(x_0)          ← distribución a priori (sin observaciones)
               │
               ▼
tiempo 1:  Predict ──► p(x_1 | y_{1:0}) = ∫ M(x_0,x_1) p(x_0) dx_0
               │
               ▼
           Update ───► p(x_1 | y_{1:1}) ∝ G(x_1, y_1) · p(x_1 | y_{1:0})
               │
               ▼
tiempo 2:  Predict ──► p(x_2 | y_{1:1}) = ∫ M(x_1,x_2) p(x_1 | y_{1:1}) dx_1
               │
               ▼
           Update ───► p(x_2 | y_{1:2}) ∝ G(x_2, y_2) · p(x_2 | y_{1:1})
               │
              ...
```

> **Intuición clave:** El filtrado es un **ciclo infinito** de predecir (la dinámica difunde tu creencia) y actualizar (la observación la concentra). Es como un péndulo: la dinámica te aleja de la certeza, la observación te devuelve.

**¿Por qué esto es Chapman-Kolmogorov + Bayes?**
- Predict = Chapman-Kolmogorov (propagar una distribución a través de un núcleo de transición)
- Update = Bayes (multiplicar por la verosimilitud y normalizar)

Todo filtro — Kalman, de partículas, de rejilla — sigue este patrón. Lo que cambia es **cómo representas** la distribución y **cómo calculas** la integral y la normalización.

---

## Paso 3 — ¿Por qué Feynman-Kac?

El ciclo predict/update nos da la distribución de filtrado como densidad. Pero hay dos cosas que **no podemos obtener directamente** de este ciclo:

### 1. La verosimilitud $p(y_{1:t})$

Para estimar parámetros (M7: MLE), necesitas la verosimilitud del modelo:

$$p(y_{1:t}) = p(y_1) \cdot p(y_2 \mid y_1) \cdot p(y_3 \mid y_{1:2}) \cdots p(y_t \mid y_{1:t-1})$$

Cada factor $p(y_t \mid y_{1:t-1})$ es la **constante de normalización** del update:

$$p(x_t \mid y_{1:t}) = \frac{G(x_t, y_t) \, p(x_t \mid y_{1:t-1})}{p(y_t \mid y_{1:t-1})}$$

El denominador es exactamente lo que necesitamos, pero el ciclo predict/update solo nos da el numerador.

### 2. La conexión con importance sampling

En M4 implementaremos el filtro de partículas. La idea central del particle filter es **muestrear** de la distribución predictiva y **re-pesar** por la verosimilitud. Esto es exactamente importance sampling:

$$w_t^{(i)} \propto \frac{p(x_t^{(i)} \mid y_{1:t})}{q_t(x_t^{(i)})}$$

donde $q_t$ es la distribución de propuesta (el predict) y el objetivo es la distribución de filtrado. Esto es un **cambio de medida** — exactamente lo que viste en M1 con los pesos de importancia.

### El formalismo FK resuelve ambos problemas

El modelo de Feynman-Kac envuelve la recursión de filtrado en un marco unificado:

1. **Distribución de filtrado** $p(x_t \mid y_{1:t})$: la distribuición objetivo
2. **Distribución predictiva** $p(x_t \mid y_{1:t-1})$: la distribución de propuesta
3. **Pesos** proporcionales a $G(x_t, y_t)$: la verosimilitud como cambio de medida
4. **Constante de normalización** $p(y_t \mid y_{1:t-1})$: la verosimilitud incremental

La distribución de filtrado se obtiene normalizando las partículas ponderadas. La verosimilitud se obtiene como **subproducto** de la normalización.

> **Intuición:** FK es como un **motor** que toma el ciclo predict/update y produce cuatro cosas a la vez: filtrado, predicción, verosimilitud, y (más adelante) suavizado. Es el marco matemático que unifica todo.

---

## Paso 4 — Puente de notación: FK → Kalman

Ahora viene lo importante: cada ecuación del filtro de Kalman es un **caso especial** de la recursión FK cuando todas las distribuciones son gaussianas.

| Concepto FK | Equivalente Kalman | Intuición |
|------------|-------------------|-----------|
| $\varphi_t$ (distribución predictiva) | $\hat{x}_{t \mid t-1},\, P_{t \mid t-1}$ | Dónde creemos que está el estado antes de ver $y_t$ |
| $G(x_t, y_t)$ (núcleo de observación) | Verosimilitud $\mathcal{N}(y_t;\, H x_t,\, R)$ | Qué tan probable es lo que observamos |
| Normalización $\varphi_t \cdot G$ | Update con ganancia de Kalman $K_t$ | Cuánto confiar en la observación |
| Distribución filtrada normalizada | $\hat{x}_{t \mid t},\, P_{t \mid t}$ | Nuestra mejor estimación después de ver $y_t$ |

### Ecuaciones del filtro de Kalman

**Predict:**

$$\hat{x}_{t \mid t-1} = A \, \hat{x}_{t-1 \mid t-1}$$

$$P_{t \mid t-1} = A \, P_{t-1 \mid t-1} \, A^\top + Q$$

**Update:**

$$K_t = P_{t \mid t-1} \, H^\top \left( H \, P_{t \mid t-1} \, H^\top + R \right)^{-1}$$

$$\hat{x}_{t \mid t} = \hat{x}_{t \mid t-1} + K_t \left( y_t - H \, \hat{x}_{t \mid t-1} \right)$$

$$P_{t \mid t} = \left( I - K_t H \right) P_{t \mid t-1}$$

### ¿Por qué funciona en forma cerrada?

El filtro de Kalman funciona porque la **familia gaussiana es cerrada** bajo las operaciones del ciclo predict/update:

1. **Predict = convolución gaussiana:** Si $p(x_{t-1} \mid y_{1:t-1})$ es gaussiana y $M$ es lineal-gaussiana, entonces $p(x_t \mid y_{1:t-1})$ también es gaussiana. (La convolución de gaussianas es gaussiana.)

2. **Update = Bayes con verosimilitud gaussiana:** Si la predictiva es gaussiana y la verosimilitud es gaussiana, el posterior también es gaussiano. (El producto de dos gaussianas, normalizado, es gaussiano.)

3. **La distribución queda completamente caracterizada por su media y varianza** — no necesitas guardar toda la función de densidad, solo dos números (para el caso escalar) o dos matrices (para el caso vectorial).

> **Intuición:** El filtro de Kalman es "barato" porque las gaussianas se representan con media y covarianza. Las operaciones predict/update se reducen a álgebra lineal — no hay integrales que calcular numéricamente. Para distribuciones no gaussianas, necesitas representar la densidad de otra forma (partículas en M4, rejillas, etc.).

### ¿Qué pasa cuando el modelo NO es lineal-Gaussiano?

| Componente | Kalman (lineal-Gaussiano) | Filtro de partículas (general) |
|-----------|---------------------------|-------------------------------|
| Distribución | Gaussiana (media + covarianza) | Partículas + pesos |
| Predict | Álgebra lineal | Propagar cada partícula |
| Update | Ganancia de Kalman | Multiplicar pesos por verosimilitud |
| Normalización | Fórmula cerrada | Sumar pesos y dividir |
| Verosimilitud | Subproducto de S | Producto de normalizaciones |

El filtro de partículas (M4) es la **generalización** del filtro de Kalman a modelos no lineales/no gaussianos. Pero la estructura es la misma: predict → update → normalizar.

---

## Preguntas antes de leer Chopin Capítulo 5

1. **Escribe el paso de predicción usando la notación de núcleos de M1.** Es decir, escribe $p(x_t \mid y_{1:t-1})$ en términos del núcleo $M$ y la distribución filtrada del tiempo anterior.

2. **Si el update es "multiplicar por $G(x_t, y_t)$", ¿por qué esto da la regla de Bayes?** Pista: piensa en qué es el numerador y qué es el denominador en la fórmula de Bayes.

3. **En el caso lineal-Gaussiano, ¿por qué $p(x_t \mid y_{1:t})$ se mantiene gaussiana después de cada paso de predict/update?** Pista: qué propiedades de las gaussianas garantizan que predict (convolución) y update (producto con verosimilitud gaussiana) preservan la forma gaussiana?

Si alguna de estas preguntas no está clara después de intentar, pregunta antes de abrir Chopin.