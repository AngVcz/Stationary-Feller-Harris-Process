# M1 — Respuestas Ideales: Conceptos Clave de Chopin Ch.2-4

## 1. SSM Definition (Def 2.1): Kernels M y G vs. forma funcional

**Pregunta:** ¿Cuál es la diferencia entre escribir la observación como un núcleo G(x_t, dy_t) vs. escribir Y_t = X_t + η_t? ¿Son equivalentes?

**Respuesta ideal:**

Son equivalentes **en el caso lineal-Gaussiano**, pero no en general.

En nuestro código, `Y_t = X_t + η_t` con `η_t ~ N(0, σ²)` es una forma **paramétrica específica**. El núcleo de observación correspondiente es:

$$G(x_t, dy_t) = \mathcal{N}(y_t;\, x_t,\, \sigma_w^2)$$

Es decir, dado `x_t`, la observación `y_t` se distribuye normal centrada en `x_t`. La forma funcional Y_t = X_t + η_t es simplemente una **realización** de este núcleo.

Pero el formalismo de núcleos es más general. Hay modelos donde no existe una representación aditiva:

- **Observaciones de conteo:** Y_t | X_t ~ Poisson(exp(X_t)) — no hay ruido aditivo, la distribución es discreta.
- **Observaciones categóricas:** Y_t | X_t ~ Categorical(softmax(X_t)) — el espacio de observación no es ℝ.
- **Observaciones con ruido no aditivo:** Y_t = h(X_t) · ε_t donde el ruido es multiplicativo.

**Por qué importa:** Chopin usa núcleos (no ecuaciones funcionales) porque el formalismo cubre todos estos casos sin cambiar nada. La recurrencia de filtrado p(x_t | y_{1:t}) se escribe igual. Cuando implementemos el filtro de partículas en M4, el núcleo G será una función que evalúa p(y_t | x_t) — y funciona igual si Y_t = X_t + η_t o si Y_t | X_t ~ Poisson(exp(X_t)).

**Enlace con lo que ya sabes:** Es lo mismo que pasar de una matriz de transición P (caso discreto finito) a un núcleo K(x, dy) (caso general). La matriz P(i,j) solo funciona cuando el espacio de estados es {1,...,N}. El núcleo funciona siempre.

---

## 2. Probability kernels (§4.1): K(x, A) vs. k(x, x')

**Pregunta:** ¿Cuál es la diferencia entre un núcleo K(x, A) y una densidad de transición k(x, x')? ¿Cuándo puedes pasar de uno al otro?

**Respuesta ideal:**

Un **núcleo de probabilidad** K : X × 𝒳 → [0,1] es una función de **dos argumentos**:

1. **Para x fijo**, K(x, ·) es una **medida de probabilidad** en (X, 𝒳). Satisface K(x, X) = 1.
2. **Para A fijo**, K(·, A) es una **función medible** X → [0,1].

Esto es la generalización directa de una matriz de transición: P(i, ·) era una distribución de probabilidad en las columnas, y P(·, j) era una función de las filas.

Una **densidad de transición** k(x, x') es una función de dos puntos que satisface:

$$K(x, A) = \int_A k(x, x')\, dx'$$

**Condiciones para que k exista:**

- K(x, ·) debe ser **absolutamente continua** respecto a una medida dominante λ (típicamente Lebesgue en ℝ^d, o conteo en espacios discretos).
- Es decir, si λ(A) = 0, entonces K(x, A) = 0 (para todo x).
- Cuando esto se cumple, k(x, x') = dK(x, ·)/dλ existe por Radon-Nikodym.

**Cuando k no existe:**

Si el espacio de estados tiene componentes discretas y continuas (por ejemplo, mezclas de Dirac + densidad), puede que no exista una densidad con respecto a Lebesgue. El núcleo K(x, A) siempre está bien definido, pero la densidad k(x, x') puede no existir.

**En la práctica:**

| Objeto | Dominio | Cuándo existe | Ejemplo |
|--------|---------|--------------|---------|
| K(x, A) | Punto × conjunto medible | Siempre | P(X_{t+1} ∈ A \| X_t = x) |
| k(x, x') | Punto × punto | Cuando hay medida dominante | N(x'; φx, σ²) para AR(1) |

**Enlace con tu código:** Tu función `gaussian_kernel(x_prev, sigma, x_curr, phi)` calcula k(x, x') — la densidad. Para obtener K(x, A), integrarías k sobre A. El test `test_kernel_integrates_to_one` verifica que ∫_ℝ k(x, x') dx' = 1, lo cual confirma que k es una densidad válida para el núcleo.

---

## 3. Change of measure (§4.2): ¿Por qué cambiar de medida?

**Pregunta:** ¿Por qué no siempre usar la medida "original"? ¿Cuándo es necesario cambiar?

**Respuesta ideal:**

Cambiar de medida es la técnica fundamental detrás de **importance sampling** y todo SMC. La idea:

**Problema:** Queremos calcular E_π[f(X)] bajo una distribución π (la "target"), pero:

- No sabemos muestrear directamente de π
- π tiene propiedades computacionales desfavorables
- La distribución π misma es lo que queremos estimar (como en filtrado: p(x_t | y_{1:t}))

**Solución:** Muestreamos de una distribución q (la "proposal") que sí sabemos muestrear, y re-pesamos:

$$E_\pi[f(X)] = E_q\left[f(X) \cdot \frac{d\pi}{dq}(X)\right]$$

donde dπ/dq es la **derivada de Radon-Nikodym** (en el caso discreto, simplemente el cociente de probabilidades).

**Cuándo es necesario:**

1. **Filtrado:** p(x_t | y_{1:t}) no se puede muestrear directamente — es una distribución posterior condicional. Muestreamos de la predicción p(x_t | y_{1:t-1}) y re-pesamos por p(y_t | x_t).

2. **Eficiencia:** A veces π tiene regiones de alta probabilidad donde q pone poco peso (o viceversa). Cambiar de medida con una buena q puede reducir varianza.

3. **Propiedades matemáticas:** Bajo una medida alternativa, los procesos pueden ser **martingalas**, lo que permite aplicar teoremas de convergencia. Chopin usa esto extensivamente en los capítulos de convergencia (Ch.11).

**Enlace con M3:** El peso de importancia w(x) = dπ/dq(x) es exactamente el cambio de medida. Cuando implementes importance sampling, cada partícula x^(i) muestreada de q tiene peso w^(i) ∝ π(x^(i))/q(x^(i)). Normalizar estos pesos (tu función `normalize_kernel`) produce una aproximación de π.

**El punto clave:** En el bootstrap filter (M4), la proposal es la dinámica del modelo M y el cambio de medida viene de la verosimilitud G(x_t, y_t). Específicamente:

$$p(x_t | y_{1:t}) \propto G(x_t, y_t) \cdot \int M(x_{t-1}, x_t)\, p(x_{t-1} | y_{1:t-1})\, dx_{t-1}$$

El cambio de medida es p(y_t | x_t) = G(x_t, y_t), que re-pesa las partículas de la predicción.

---

## 4. Backward kernels (§4.3): ¿Para qué sirven?

**Pregunta:** ¿Por qué Chopin define núcleos hacia atrás? ¿Qué problema requiere forward kernels sola no pueden resolver?

**Respuesta ideal:**

Los núcleos forward M(x_{t-1}, dx_t) resuelven el problema de **filtrado**: estimar p(x_t | y_{1:t}) recursivamente hacia adelante en el tiempo.

Pero hay problemas que requieren información del **futuro**:

### Smoothing: p(x_t | y_{1:T}) con T > t

Queremos estimar el estado en un tiempo *interior* usando *todas* las observaciones, no solo las pasadas. Esto es más preciso que el filtrado porque incorpora observaciones futuras.

La descomposición clave usa el núcleo backward L_{t+1}:

$$p(x_t | y_{1:T}) = p(x_t | y_{1:t}) \int L_{t+1}(x_{t+1}, x_t)\, \frac{p(x_{t+1} | y_{1:T})}{p(x_{t+1} | y_{1:t})}\, dx_{t+1}$$

donde el **núcleo backward** es:

$$L_{t+1}(x_{t+1}, x_t) = p(x_t | x_{t+1}, y_{1:t})$$

Es la distribución del estado anterior dado el estado actual y las observaciones pasadas. Invierte la flecha del tiempo condicionalmente.

### Por qué los núcleos forward no bastan:

El filtro forward produce p(x_t | y_{1:t}) — la distribución usando solo Y_{1:t}. Pero p(x_t | y_{1:T}) ≠ p(x_t | y_{1:t}) en general, porque las observaciones Y_{t+1:T} contienen información adicional sobre X_t a través de la cadena de Markov:

```
X_t → X_{t+1} → ... → X_T
       ↓              ↓
      Y_{t+1}        Y_T
```

Y_{t+1:T} informa sobre X_{t+1:T}, que a su vez informa sobre X_t (porque X_t es el "ancestro" en la cadena de Markov). Los núcleos backward cuantifican esta conexión.

### Aplicaciones:

- **Suavizado fijo (fixed-interval):** Estimar toda la trayectoria dadas todas las observaciones. Esencial en aplicaciones donde necesitas reconstruir la historia.
- **EM algorithm (M7):** Estimación de parámetros requiere la distribución suavizada E[X_t X_{t-1} | y_{1:T}].
- **FFBS (M6):** Forward Filtering Backward Sampling usa núcleos backward para muestrear trayectorias completas de la distribución suavizada.

**Enlace con tu proyecto:** En M11 (capstone), cuando estimes parámetros del modelo SF-Harris para volatilidad estocástica, vas a necesitar la distribución suavizada, no solo la filtrada. Los núcleos backward son el ingrediente matemático que hace esto posible.

---

## 5. Markov property for SSMs (§4.5): ¿Es (Y_t) Markov?

**Pregunta:** ¿La secuencia de observaciones Y_t por sí sola es un proceso de Markov?

**Respuesta ideal:**

**No.** (Y_t) no es Markov en general.

### Demostración por qué:

Recordemos la estructura del SSM:

```
... → X_{t-2} → X_{t-1} → X_t → ...
        ↓           ↓         ↓
      Y_{t-2}    Y_{t-1}   Y_t
```

Las observaciones satisfacen **independencia condicional**: Y_s ⊥ Y_t | X_s, X_t (dado el estado oculto, las observaciones son independientes).

Pero al marginalizar X, las observaciones quedan **dependientes**. La razón:

1. Y_t depende de X_t (por la observación).
2. X_t depende de X_{t-1} (por la dinámica de Markov).
3. X_{t-1} también generó Y_{t-1}.

Entonces Y_{t-1} lleva información sobre X_{t-1}, que lleva información sobre X_t, que lleva información sobre Y_t. La cadena de dependencia es:

$$Y_{t-1} \xrightarrow{\text{informa sobre}} X_{t-1} \xrightarrow{\text{predice}} X_t \xrightarrow{\text{genera}} Y_t$$

**¿Y si condicionamos en Y_{t-1}?** Aun así, Y_{t-2} sigue teniendo información sobre Y_t a través de la misma cadena:

$$Y_{t-2} \rightarrow X_{t-2} \rightarrow X_{t-1} \rightarrow X_t \rightarrow Y_t$$

Condicionar en Y_{t-1} **bloquea parcialmente** la dependencia (porque Y_{t-1} ya informa sobre X_{t-1}), pero no la elimina del todo, porque Y_{t-2} da información adicional sobre X_{t-2} que Y_{t-1} no captura completamente.

### ¿Cuándo SÍ es Markov?

El único caso donde (Y_t) es Markov es cuando el estado oculto X_t se puede recuperar perfectamente de una sola observación (por ejemplo, si σ_w = 0, observación sin ruido). En ese caso, Y_t = X_t (casi seguro), y el proceso observado hereda la propiedad de Markov de X_t.

### Consecuencia práctica:

Dado que (Y_t) no es Markov, **no puedes modelar las observaciones directamente como un proceso de Markov**. Necesitas el modelo de estado oculto + las ecuaciones de filtrado para hacer inferencia. Esto es precisamente por lo que los HMMs y SSMs existen como modelos separados de las cadenas de Markov observadas.

**Enlace con series de tiempo:** Si ajustas un AR(1) directamente a las observaciones Y_t (sin modelo de estado oculto), estás asumiendo implícitamente que (Y_t) es Markov. En un SSM con ruido de observación, esto es una aproximación. La calidad de la aproximación depende de la relación señal-ruido: si σ_w >> σ_v (mucho ruido de observación), la estructura de Markov se degrada más.

---

## Resumen: De CTMCs a SSMs — el mapa conceptual

```
Lo que ya sabes           La generalización en Chopin
─────────────────         ────────────────────────────
Matriz de transición  →   Núcleo de probabilidad K(x, dy)
Cadena de Markov      →   Proceso de Markov en espacio general
Distribución estác.   →   Medida invariante π tal que πK = π
HMM (estados discretos) → SSM (estados continuos)
Observación directa   →   Observación con ruido via núcleo G
Filtro de Kalman      →   Filtro de partículas (M4)
```

**Lo que llevas a M2:**
- Los núcleos son la generalización de matrices de transición a espacios continuos.
- El cambio de medida conecta distributiones inaccesibles con proposals muestreables.
- Los núcleos backward abren la puerta al smoothing (M6).
- (Y_t) no es Markov — por eso necesitamos el formalismo de SSMs.
- Tu código `models.py` + `kernels.py` implementa exactamente los núcleos M y G del caso lineal-Gaussiano.