# M1 — Síntesis: Modelos de Espacio de Estados y Núcleos de Markov

## Lo que aprendiste

- Un **núcleo de probabilidad** K(x, A) generaliza la idea de matriz de transición a espacios continuos.
  Es una función de dos argumentos: para x fijo, es una distribución de probabilidad; para A fijo, es una función medible.

- Un **modelo de espacio de estados (SSM)** tiene dos componentes:
  - Estado oculto X_t que evoluciona como proceso de Markov (núcleo M)
  - Observación Y_t que depende de X_t (núcleo G)
  - Los Y_t son condicionalmente independientes dado los X_t

- La **propiedad de Markov** se aplica al proceso {X_t}, no al proceso conjunto {(X_t, Y_t)}.

## Conexión con lo que ya sabes

| Concepto nuevo              | Equivalente en CTMCs                |
|-----------------------------|--------------------------------------|
| Núcleo de probabilidad K    | Matriz de transición P               |
| Medida invariante π         | Distribución estacionaria π          |
| SSM                         | HMM en espacio continuo              |
| Filtrado p(x_t | y_{1:t})  | Distribución condicional instantánea |
| Predicción p(x_{t+1} | y_{1:t}) | Un paso de la cadena hacia adelante |

## Lo que viene después (M2)

El siguiente módulo introduce el **formalismo de Feynman-Kac**, que es la herramienta matemática que conecta los núcleos de probabilidad con el filtrado secuencial. Si entiendes que p(x_t | y_{1:t}) = (peso × transición) / (normalización), ya tienes la intuición.

## Pregunta de autoevaluación

Si tienes un AR(1) con φ=0.9 y observas Y₁ = 2.3, ¿puedes escribir la densidad del filtrado p(X₁ | Y₁ = 2.3) como un producto de un núcleo y una función de peso?