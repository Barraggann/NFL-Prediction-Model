# Experimento: variables de cruce ataque-defensa

Generado: 2026-11-12T18:12:37

- Entrenamiento 2006–2020: 3997 partidos
- Validación 2021–2023: 851 partidos
- Regla: adoptar solo si el log loss de validación mejora más de 0.002.
- El cruce aditivo ya está en el modelo; aquí se prueba la interacción (producto).

## Regresión logística

| C | Base: acierto | Base: log loss | Con cruces: acierto | Con cruces: log loss |
|---|---|---|---|---|
| 0.003 | 63.2% | 0.6382 | 63.5% | 0.6380 |
| 0.01 | 63.8% | 0.6392 | 63.7% | 0.6389 |
| 0.03 | 63.8% | 0.6400 | 63.9% | 0.6397 |
| 0.1 | 63.2% | 0.6404 | 63.7% | 0.6401 |
| 0.3 | 63.3% | 0.6406 | 63.3% | 0.6403 |
| 1.0 | 63.2% | 0.6406 | 63.3% | 0.6403 |
| 3.0 | 63.1% | 0.6407 | 63.1% | 0.6404 |

Coeficientes de las interacciones (estandarizados, en el mejor C con cruces):

- inter_pase: +0.000
- inter_carrera: -0.019
- Referencia, elo_diff: +0.259

## Boosting (100 árboles, profundidad 2)

Puede descubrir interacciones por su cuenta si ve la forma de cada equipo por separado.

| Variables | Acierto | Log loss |
|---|---|---|
| Base (18) | 64.0% | 0.6356 |
| Base + forma por equipo (26) | 64.3% | 0.6370 |
| Base + interacciones (20) | 63.8% | 0.6353 |

## Veredicto

- Mejor log loss base: 0.6382
- Mejor log loss con cruces: 0.6380
- Mejora: +0.0002 (se pedía más de 0.002)
- **NO ADOPTAR**
