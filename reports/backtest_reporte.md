# Backtest semana por semana

Generado: 2026-11-11T20:08:38

- Modelo: `{"tipo": "logistica", "C": 0.003, "calibrar": false}`
- Cada semana se predijo entrenando solo con partidos anteriores.
- 2021–2023 es optimista: con esos años se eligió el modelo.
- **2024–2025 es la prueba honesta.**

## Veredicto (solo 2024–2025)

| Criterio | Resultado |
|---|---|
| Supera a 'siempre el local' en log loss | Sí |
| Supera al Elo en log loss | Sí |
| El rango 70–80% acierta entre 65% y 85% | Sí |
| Supera a Las Vegas en log loss (no era objetivo) | No |

## Comparación

### Prueba honesta 2024–2025 (569 partidos)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 54.1% | 0.6906 | 0.2487 |
| Elo | 65.7% | 0.6236 | 0.2171 |
| Modelo | 64.3% | 0.6182 | 0.2150 |
| Las Vegas | 68.4% | 0.5981 | 0.2059 |

### Optimista 2021–2023 (851 partidos)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 55.0% | 0.6885 | 0.2477 |
| Elo | 61.6% | 0.6505 | 0.2291 |
| Modelo | 63.1% | 0.6381 | 0.2242 |
| Las Vegas | 65.0% | 0.6184 | 0.2152 |

## Por temporada

### 2021 (284 partidos)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 51.8% | 0.6966 | 0.2517 |
| Elo | 60.2% | 0.6575 | 0.2318 |
| Modelo | 62.0% | 0.6470 | 0.2283 |
| Las Vegas | 62.3% | 0.6232 | 0.2177 |

### 2022 (282 partidos)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 56.7% | 0.6841 | 0.2455 |
| Elo | 64.2% | 0.6351 | 0.2230 |
| Modelo | 64.2% | 0.6233 | 0.2175 |
| Las Vegas | 65.6% | 0.6050 | 0.2093 |

### 2023 (285 partidos)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 56.5% | 0.6847 | 0.2458 |
| Elo | 60.4% | 0.6589 | 0.2326 |
| Modelo | 63.2% | 0.6439 | 0.2267 |
| Las Vegas | 67.0% | 0.6270 | 0.2186 |

### 2024 (285 partidos)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 54.7% | 0.6891 | 0.2480 |
| Elo | 67.0% | 0.6102 | 0.2109 |
| Modelo | 68.1% | 0.6084 | 0.2101 |
| Las Vegas | 70.9% | 0.5892 | 0.2010 |

### 2025 (284 partidos)

| Método | Acierto | Log loss | Brier |
|---|---|---|---|
| Siempre el local | 53.5% | 0.6922 | 0.2495 |
| Elo | 64.4% | 0.6370 | 0.2232 |
| Modelo | 60.6% | 0.6281 | 0.2199 |
| Las Vegas | 65.8% | 0.6070 | 0.2109 |

## Aciertos por rango de confianza

### Prueba honesta 2024–2025

| Confianza | Partidos | Confianza media | Acierto real |
|---|---|---|---|
| 50–60% | 224 | 55.0% | 53.1% |
| 60–70% | 165 | 64.6% | 63.6% |
| 70–80% | 131 | 74.2% | 76.3% |
| 80–100% | 49 | 84.7% | 85.7% |

### Todo 2021–2025

| Confianza | Partidos | Confianza media | Acierto real |
|---|---|---|---|
| 50–60% | 574 | 54.9% | 54.7% |
| 60–70% | 435 | 64.7% | 61.8% |
| 70–80% | 295 | 74.4% | 74.9% |
| 80–100% | 116 | 83.8% | 85.3% |

## Modelo contra Las Vegas (2024–2025)

### Cuando el modelo y Las Vegas eligen distinto ganador

- Partidos: 83 de 569
- Acertó el modelo: 30. Acertó Las Vegas: 53.

Los 15 desacuerdos más grandes en probabilidad:

| Partido | Modelo (local) | Las Vegas (local) | Ganó | ¿Quién estuvo más cerca? |
|---|---|---|---|---|
| 2024 sem 18: SEA en LA | 61.2% | 24.9% | SEA | Las Vegas |
| 2024 sem 18: KC en DEN | 48.3% | 79.6% | DEN | Las Vegas |
| 2025 sem 9: BAL en MIA | 49.0% | 21.3% | BAL | Las Vegas |
| 2024 sem 18: NYG en PHI | 84.9% | 58.3% | PHI | Modelo |
| 2025 sem 18: GB en MIN | 61.7% | 83.2% | MIN | Las Vegas |
| 2024 sem 18: BUF en NE | 18.7% | 40.0% | NE | Las Vegas |
| 2025 sem 16: SF en IND | 54.5% | 33.6% | SF | Las Vegas |
| 2024 sem 7: NYJ en PIT | 60.6% | 40.0% | PIT | Modelo |
| 2024 sem 1: GB en PHI | 33.9% | 54.3% | PHI | Las Vegas |
| 2025 sem 15: IND en SEA | 67.2% | 87.2% | SEA | Las Vegas |
| 2025 sem 18: LAC en DEN | 71.6% | 90.4% | DEN | Las Vegas |
| 2024 sem 8: ARI en MIA | 45.2% | 63.7% | ARI | Modelo |
| 2025 sem 17: DEN en KC | 29.9% | 12.0% | DEN | Las Vegas |
| 2025 sem 19: SF en PHI | 51.7% | 69.6% | SF | Modelo |
| 2025 sem 17: JAX en IND | 51.4% | 33.6% | JAX | Las Vegas |
