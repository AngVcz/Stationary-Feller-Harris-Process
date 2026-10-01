# SF-Harris Stochastic Volatility — Servicio Social

**Autor:** Ángel Vences Adame, IIMAS-UNAM, Licenciatura en Actuaría

Modelado de volatilidad estocástica usando procesos SF-Harris y métodos SMC (Sequential Monte Carlo). El proyecto replica y extiende los resultados de Anzarut (2023) para la predicción de volatilidad intradía y diaria en activos financieros.

## Estructura

```
src/            — Código fuente: proceso SF-Harris, filtros Kalman, resampling, kernels SSM
scripts/        — Estudios numéricos: replicación de tablas, backtests, diagnósticos
tests/          — Tests unitarios (pytest)
notebooks/      — Notebook final con resultados sobre activos mexicanos
demos/          — Demos HTML interactivos (módulos 1-3 y experimentos; sf-harris_demos.html es la demo principal)
docs/           — Documentación y hallazgos en texto
results/        — Salidas de experimentos: jsons y logs
figures/        — Figuras generadas
data/           — Datos procesados (CSV)
report/         — Reporte LaTeX (borrador)
presentacion/   — Presentación del servicio social (SF-Harris_SS.pptx)
```

## Cómo ejecutar

```bash
pip install -e ".[dev]"
pytest
python scripts/simulation_study.py --quick
```

## Resultado principal

La cadena Harris es decorativa para predicción de cobertura: el modelo SF-Harris se reduce a 2 ingredientes (Q empírica + denoising Gibbs), y la estructura de saltos continuos no mejora la predicción direccional.
