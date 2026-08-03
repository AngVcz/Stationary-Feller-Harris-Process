# Estudio de Simulación SF-Harris: Sección 3.5.2

## Descripción

Replicación del estudio de simulación descrito en la Sección 3.5.2 de la tesis de la Dra. Michelle Anzarut. Se evalúan cuatro métodos de estimación para el proceso SF-Harris:

- **NDNJ**: Estimador no paramétrico basado en densidad (Nonparametric Density-based Non-Jump)
- **MLE**: Estimador de máxima verosimilitud
- **EM**: Algoritmo Esperanza-Maximización
- **Gibbs-a / Gibbs-b**: Muestreador de Gibbs (dos configuraciones)

## Configuración

### Caso 1: Q = Uniforme Discreta {1, ..., 5}
- Parámetro a estimar: α (dependencia)
- α ~ U(0, 30)
- Tamaños de muestra: k = 20, 100, 500, 1000
- 100 réplicas por configuración
- Métrica de error: E_α = (1/100) Σ |α_i - α̂_i| / 30

### Caso 2: Q = GIG(λ, κ, η)
- Parámetros a estimar: α, λ, κ, η
- α ~ U(0, 30), λ ~ U(-5, 5), κ ~ U(0.1, 50), η ~ U(0.1, 4)
- Tamaños de muestra: k = 20, 100, 500, 1000
- 100 réplicas por configuración
- Métricas de error: E_α y E_Q (divergencia KL)

## Notas sobre el Proceso SF-Harris

El proceso SF-Harris es una cadena de Markov en tiempo discreto con:
- P(x_t = x_{t-1}) = e^{-α}  (no salto)
- P(x_t ~ Q) = 1 - e^{-α}  (salto/regeneración)

Para Q continua (GIG), cada cambio observado corresponde exactamente a un salto.
Para Q discreta (Uniforme), existen saltos ocultos donde el proceso salta pero cae en el mismo valor.

## Resultados

_Los resultados se generarán al ejecutar `python scripts/simulation_study.py`_

## Archivos

- `src/sf_harris/process.py`: Simulador del proceso SF-Harris
- `src/sf_harris/distributions.py`: Distribuciones Uniforme Discreta y GIG
- `src/sf_harris/estimation.py`: Métodos de estimación (NDNJ, MLE, EM, Gibbs)
- `scripts/simulation_study.py`: Script principal del estudio de simulación
- `tests/test_sf_harris.py`: Pruebas unitarias