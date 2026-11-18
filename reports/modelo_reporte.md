# Modelo: reporte de selección

Generado: 2026-11-18T02:41:34

- Entrenamiento 2006–2020: 3997 partidos
- Validación 2021–2023: 851 partidos
- Variables: 18
- **Modelo elegido:** `{"tipo": "logistica", "C": 0.003, "calibrar": false}`

Las temporadas 2024–2025 no se evalúan aquí a propósito.

## Validación, todos los partidos (851)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 55.0% | 0.6885 | 0.2477 |
| Elo | 61.6% | 0.6505 | 0.2291 |
| Logística (mejor C) | 63.2% | 0.6382 | 0.2243 |
| Boosting (mejor) | 64.0% | 0.6356 | 0.2231 |

## Validación, partidos con moneyline (851)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Elo | 61.6% | 0.6505 | 0.2291 |
| Logística (mejor C) | 63.2% | 0.6382 | 0.2243 |
| Boosting (mejor) | 64.0% | 0.6356 | 0.2231 |
| Las Vegas | 65.0% | 0.6184 | 0.2152 |

## Rejilla: regresión logística

| C | Acierto | Log loss |
|---|---|---|
| 0.003 | 63.2% | 0.6382 |
| 0.01 | 63.8% | 0.6392 |
| 0.03 | 63.8% | 0.6400 |
| 0.1 | 63.2% | 0.6404 |
| 0.3 | 63.3% | 0.6406 |
| 1.0 | 63.2% | 0.6406 |
| 3.0 | 63.1% | 0.6407 |

## Rejilla: boosting

| Árboles | Profundidad | Acierto | Log loss |
|---|---|---|---|
| 50 | 2 | 63.1% | 0.6379 |
| 100 | 2 | 64.0% | 0.6356 |
| 200 | 2 | 63.8% | 0.6360 |
| 400 | 2 | 63.3% | 0.6384 |
| 50 | 3 | 63.6% | 0.6386 |
| 100 | 3 | 64.0% | 0.6376 |
| 200 | 3 | 64.3% | 0.6390 |
| 400 | 3 | 63.5% | 0.6488 |

## Coeficientes de la logística (variables estandarizadas)

Positivo favorece al local.

| Variable | Coeficiente |
|---|---|
| elo_diff | +0.259 |
| diff_qb | +0.134 |
| diff_off_epa | +0.107 |
| diff_off_pass_epa | +0.103 |
| diff_off_success | +0.091 |
| diff_qb_cambio | +0.091 |
| home_qb_change | -0.085 |
| diff_def_epa | -0.069 |
| neutral | -0.067 |
| diff_off_rush_epa | +0.066 |
| diff_def_pass_epa | -0.063 |
| div_game | -0.060 |
| away_bye | -0.046 |
| rest_diff | +0.033 |
| diff_def_success | -0.031 |
| diff_def_rush_epa | -0.024 |
| away_qb_change | +0.019 |
| home_bye | +0.010 |

## Calibración del modelo elegido

- Log loss sin calibrar: 0.6382
- Log loss calibrando cada temporada con las otras dos: 0.6397
- Platt sobre toda la validación: pendiente a = 0.964, desplazamiento b = -0.058 (a = 1 y b = 0 significa que ya estaba calibrado)
- ¿Se aplica calibración? **No**

### Aciertos por rango de confianza (validación)

| Confianza | Partidos | Confianza media | Acierto real |
|---|---|---|---|
| 50–60% | 344 | 54.8% | 56.1% |
| 60–70% | 274 | 64.8% | 60.9% |
| 70–80% | 165 | 74.6% | 72.1% |
| 80–100% | 68 | 83.3% | 86.8% |

### Rangos de confianza: logística (mejor C)

| Confianza | Partidos | Confianza media | Acierto real |
|---|---|---|---|
| 50–60% | 344 | 54.8% | 56.1% |
| 60–70% | 274 | 64.8% | 60.9% |
| 70–80% | 165 | 74.6% | 72.1% |
| 80–100% | 68 | 83.3% | 86.8% |

### Rangos de confianza: boosting (mejor)

| Confianza | Partidos | Confianza media | Acierto real |
|---|---|---|---|
| 50–60% | 353 | 55.3% | 56.7% |
| 60–70% | 248 | 64.4% | 66.1% |
| 70–80% | 176 | 74.4% | 64.2% |
| 80–100% | 74 | 84.2% | 91.9% |

Gráfica: `reports/calibracion.png`
