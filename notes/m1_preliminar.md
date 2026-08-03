# M1 — Lectura Preliminar: Modelos de Espacio de Estados y Núcleos de Markov

## Prerrequisitos que ya conoces

Ya trabajaste con **cadenas de Markov de tiempo continuo (CTMCs)**. Sabes que:

- Un proceso de Markov tiene la propiedad de que el futuro depende del pasado **solo a través del presente**
- Las CTMCs se definen por sus **tasas de transición** (generador Q)
- La distribución estacionaria π satisface πQ = 0

Estos conceptos son la base de todo lo que viene. Lo que cambia es el **lenguaje** y la **generalidad**.

## De CTMCs a núcleos de probabilidad

Una CTMC tiene tasas de transición. Pero en tiempo discreto, o en espacios de estado continuos (como ℝᵈ), necesitamos algo más general: **núcleos de probabilidad** (probability kernels).

**Intuición:** Un núcleo de probabilidad K(x, ·) es como una **matriz de transición**, pero en vez de filas con probabilidades discretas, tienes una **función** que para cada estado x te da una distribución de probabilidad sobre el siguiente estado.

| Concepto que ya conoces | Generalización en Chopin |
|--------------------------|--------------------------|
| Matriz de transición P   | Núcleo de probabilidad K  |
| P(i,j) = probabilidad   | K(x, A) = P(X_{t+1} ∈ A \| X_t = x) |
| Distribución estacionaria π | Medida invariante π tal que πK = π |
| CTMC con generador Q    | Proceso de Markov con núcleo K |

**Punto clave:** Un núcleo K(x, ·) es una **medida de probabilidad** para cada x fijo, y una **función medible** para cada conjunto A fijo. Esta dualidad es lo que hace potente el formalismo.

## ¿Qué es un modelo de espacio de estados?

Un **modelo de espacio de estados (SSM)** tiene dos partes:

1. **Estado oculto** X_t que evoluciona como proceso de Markov (lo que NO observamos directamente)
2. **Observación** Y_t que depende de X_t (lo que SÍ observamos)

```
X_0 → X_1 → X_2 → X_3 → ...   (estado oculto, Markov)
 ↓      ↓      ↓      ↓
Y_0    Y_1    Y_2    Y_3       (observaciones, cond. indep. dado X)
```

**Ejemplo concreto que ya debes poder imaginar:** Piensa en un AR(1) con ruido de observación.

- Estado: X_t = φ·X_{t-1} + ε_t (proceso Markov)
- Observación: Y_t = X_t + η_t (ruido de medición)

¡Este es un SSM! El AR(1) es la **dinámica oculta** y el ruido es la **observación**.

## ¿Por qué necesitamos todo esto?

El problema central: **filtrado**. Dadas las observaciones Y_{0:t}, ¿cuál es la distribución del estado X_t?

- Si el modelo es lineal-Gaussiano: **Filtro de Kalman** (solución exacta, M2)
- Si no: **Filtro de partículas** (solución aproximada, M4)

Todo el libro de Chopin construye herramientas para resolver este problema en modelos cada vez más generales.

## Conceptos clave a identificar en Chopin Ch.2-4

Al leer, presta atención a estos conceptos y cómo se relacionan con lo que ya sabes:

1. **Definición 2.1 (SSM):** Dos núcleos — M (dinámica) y G (observación). Equivalen a la matriz de transición y la matriz de emisión en un HMM.

2. **Núcleo de probabilidad (§4.1):** Formalización de "dada una condición, obtengo una distribución". Piensa en K(x, dy) como P(X_{t+1} ∈ dy | X_t = x).

3. **Cambio de medida (§4.2):** Técnica para "re-pesar" probabilidades. Es como cambiar de variables en integración — mismo espacio, diferente densidad de referencia.

4. **Núcleos hacia atrás (backward kernels, §4.3):** K_t→(t-1) dado Y_t. Es la distribución del estado anterior dado el actual. Invertir el tiempo es clave para suavizado (smoothing, M6).

5. **Propiedad de Markov para SSMs (§4.5):** El proceso conjunto (X_t, Y_t) NO es Markov, pero X_t SÍ es Markov, y Y_t es condicionalmente independiente dado X_t.

## Antes de leer Chopin

Asegúrate de poder responder estas preguntas:

1. ¿Qué es un núcleo de probabilidad y en qué se diferencia de una función de distribución?
2. ¿Cómo se relaciona un SSM con un HMM?
3. ¿Por qué el problema de filtrado no tiene solución exacta en modelos no lineales?

Si alguna de estas preguntas no está clara después de leer estas notas, pregunta antes de abrir Chopin.