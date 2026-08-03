# M3 — Lectura Preliminar: Importance Sampling, ESS y Resampling

## Lo que ya sabes de M2

En M2 implementaste el filtro de Kalman: predict/update en forma cerrada porque las distribuciones son gaussianas. La verosimilitud sale como subproducto de la normalización.

Pero el Kalman **solo funciona** cuando el modelo es lineal-Gaussiano. ¿Qué pasa cuando no lo es?

La respuesta: necesitas **representar las distribuciones de otra forma**. En vez de media y covarianza (2 números), usas **partículas y pesos** (N muestras). Eso es el filtro de partículas (M4).

Antes del filtro de partículas, necesitas entender tres herramientas: **importance sampling**, **ESS**, y **resampling**.

---

## Paso 1 — Integración Monte Carlo y el problema de las integrales

El filtrado requiere calcular integrales en cada paso:

$$p(x_t \mid y_{0:t}) = \frac{G(x_t, y_t) \int M(x_{t-1}, x_t) p(x_{t-1} \mid y_{0:t-1}) dx_{t-1}}{\int G(x_t, y_t) \left[\int M(x_{t-1}, x_t) p(x_{t-1} \mid y_{0:t-1}) dx_{t-1}\right] dx_t}$$

Cada una de esas integrales es, en general, **intratable analíticamente**. Necesitamos aproximarlas.

**Integración Monte Carlo:** Si tienes muestras $x^{(1)}, ..., x^{(N)}$ de una distribución π, entonces:

$$E_\pi[f(X)] = \int f(x) \pi(x) dx \approx \frac{1}{N} \sum_{i=1}^N f(x^{(i)})$$

**Intuición:** En vez de calcular la integral, sacas muestras y promedias. Por la ley de los grandes números, el promedio converge a la integral.

**Problema:** Necesitas muestras de π. ¿Qué haces si no puedes muestrear directamente de π?

---

## Paso 2 — Importance Sampling: Cambio de medida

Si no puedes muestrear de π (la "target"), pero puedes muestrear de q (la "proposal"), usas **importance sampling**:

$$E_\pi[f(X)] = E_q\left[f(X) \cdot \frac{\pi(X)}{q(X)}\right]$$

El cociente $w(x) = \frac{\pi(x)}{q(x)}$ se llama **peso de importancia**.

**Intuición:** Es como si estuvieras encuestando, pero algunos votos cuentan más. Las muestras de q que caen donde π es grande (región importante) reciben pesos altos. Las que caen donde π es chica reciben pesos bajos.

**En la práctica:**

1. Muestrea $x^{(i)} \sim q$ para $i = 1, ..., N$
2. Calcula pesos $w^{(i)} = \frac{\pi(x^{(i)})}{q(x^{(i)})}$
3. Normaliza: $W^{(i)} = \frac{w^{(i)}}{\sum_j w^{(j)}}$
4. Aproxima: $E_\pi[f(X)] \approx \sum_{i=1}^N W^{(i)} f(x^{(i)})$

**Conexión con M2:** En el filtro de Kalman, el update hace exactamente esto:
- La predictiva $p(x_t | y_{0:t-1})$ es la proposal $q$
- La filtrada $p(x_t | y_{0:t})$ es la target $\pi$
- El peso es la verosimilitud: $w \propto G(x_t, y_t)$

En Kalman, todo se resuelve en forma cerrada. En el filtro de partículas, lo resolvemos con muestras y pesos.

---

## Paso 3 — Effective Sample Size (ESS): ¿Cuándo funciona IS?

Importance sampling funciona bien cuando q y π son similares. Cuando son muy diferentes, la mayoría de los pesos son casi cero y unas pocas partículas dominan. Pierdes eficiencia.

El **ESS (Effective Sample Size)** mide cuántas muestras "equivalentes" tienes:

$$ESS = \frac{\left(\sum_{i=1}^N w^{(i)}\right)^2}{\sum_{i=1}^N \left(w^{(i)}\right)^2}$$

O con pesos normalizados $W^{(i)}$:

$$ESS = \frac{1}{\sum_{i=1}^N \left(W^{(i)}\right)^2}$$

**Interpretación:**

- ESS = N: todos los pesos son iguales → IS funciona tan bien como muestrear directamente de π
- ESS = 1: un solo peso domina → IS es inútil (una partícula cuenta, las demás no)
- ESS ≈ N/2: razonable — la mitad de las muestras "cuentan"

**Regla práctica:** Si ESS < N/2 (o ESS < algún umbral), necesitas resampling.

---

## Paso 4 — Resampling: Reiniciar los pesos

Cuando el ESS es bajo, las partículas con pesos altos se "clonan" y las de pesos bajos se descartan. Esto se llama **resampling**.

**Idea:** Reemplazar el conjunto ponderado $\{(x^{(i)}, W^{(i)})\}$ por un conjunto no ponderado $\{\tilde{x}^{(j)}\}$ donde cada $\tilde{x}^{(j)}$ es elegido entre las $x^{(i)}$ con probabilidad $W^{(i)}$.

Después del resampling: todos los pesos son $1/N$ (iguales), y las partículas están concentradas en las regiones importantes.

**Hay 4 esquemas principales:**

### 4.1 Multinomial resampling

El más simple: muestrea N veces de las partículas con probabilidades $W^{(1)}, ..., W^{(N)}$.

- Cada $\tilde{x}^{(j)}$ se muestrea independientemente
- Fácil de entender e implementar
- Mayor varianza — puede clonar partículas irrelevantes o perder partículas buenas

### 4.2 Residual resampling

1. Toma $\lfloor N \cdot W^{(i)} \rfloor$ copias de cada partícula (la parte determinística)
2. Las copias restantes se muestrean de forma multinomial con los residuos

- Menos varianza que multinomial puro
- Cada partícula con peso alto garantiza al menos una copia

### 4.3 Stratified resampling

1. Divide el intervalo [0,1] en N estratos de tamaño 1/N
2. En cada estrato, genera un punto aleatorio uniforme
3. Asigna partículas según dónde cae cada punto en la distribución acumulada de pesos

- Menor varianza que multinomial
- Garantiza que cada región del espacio de pesos está representada

### 4.4 Systematic resampling

1. Genera un solo número aleatorio $u \sim U(0, 1/N)$
2. Los puntos de muestreo son $u, u + 1/N, u + 2/N, ..., u + (N-1)/N$
3. Asigna partículas según la distribución acumulada

- La menor varianza de los 4 esquemas
- Solo usa un número aleatorio — muy eficiente
- Es el esquema más usado en la práctica

**Comparación rápida:**

| Esquema | Varianza | Complejidad | Uso práctico |
|---------|----------|-------------|---------------|
| Multinomial | Alta | O(N log N) | Pedagógico, no en práctica |
| Residual | Media | O(N) | Ocasional |
| Stratified | Baja | O(N) | Buen default |
| Systematic | Muy baja | O(N) | **El más usado** |

---

## Paso 5 — Conexión con el filtro de partículas (M4)

El filtro de partículas combina todo:

1. **IS:** Representar p(x_t | y_{0:t}) con partículas y pesos
2. **ESS:** Detectar cuándo los pesos degeneran
3. **Resampling:** Reiniciar los pesos cuando ESS es bajo

El ciclo es:

```
Para cada partícula i = 1, ..., N:
  1. Propagar: x_t^{(i)} ~ M(x_{t-1}^{(i)}, ·)          [Predict — usa el núcleo M]
  2. Pesar: w_t^{(i)} = G(x_t^{(i)}, y_t)                [Update — usa el núcleo G]
  3. Normalizar: W_t^{(i)} = w_t^{(i)} / Σ w_t^{(j)}    [Normalización]

Si ESS < umbral:
  4. Resampling: reemplazar {x_t^{(i)}, W_t^{(i)}} por {x̃_t^{(j)}, 1/N}
```

Es exactamente el ciclo predict/update del Kalman, pero con partículas en vez de gaussianas.

---

## Antes de leer Chopin Ch.8-9

Asegúrate de poder responder:

1. ¿Qué es importance sampling y por qué necesitas cambiar de medida?
2. Si los pesos de importancia son todos iguales, ¿cuánto vale el ESS? ¿Y si un solo peso domina?
3. ¿Por qué el resampling reduce la varianza? ¿Cuál es el costo?
4. ¿Qué diferencia hay entre resampling multinomial y systematic? ¿Cuál prefieres y por qué?

Si alguna de estas preguntas no está clara después de leer estas notas, pregunta antes de abrir Chopin.