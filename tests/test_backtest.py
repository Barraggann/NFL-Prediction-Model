"""Pruebas de la Fase 5: el backtest nunca entrena con el futuro."""

from datetime import date, timedelta

import numpy as np

from nfl_model.backtest import escribir_reporte, recorrer
from nfl_model.features import VARIABLES


def _filas_falsas(temporadas=(2019, 2020, 2021, 2022), semanas=6, por_semana=8):
    rng = np.random.default_rng(0)
    filas = []
    for temporada in temporadas:
        inicio = date(temporada, 9, 10)
        for semana in range(1, semanas + 1):
            for i in range(por_semana):
                fila = {v: float(rng.normal()) for v in VARIABLES}
                fila["elo_diff"] = float(rng.normal(0, 150))
                p = 1 / (1 + 10 ** (-fila["elo_diff"] / 400))
                fila.update(
                    {
                        "game_id": f"{temporada}_{semana:02d}_{i}",
                        "season": temporada,
                        "week": semana,
                        # Jueves y domingo: la semana tiene más de un día de juego.
                        "gameday": inicio + timedelta(days=7 * (semana - 1) + (0 if i == 0 else 3)),
                        "home_team": f"L{i}",
                        "away_team": f"V{i}",
                        "home_win": int(rng.random() < p),
                        "p_home_elo": p,
                        "p_home_vegas": min(max(p + 0.02, 0.01), 0.99),
                    }
                )
                filas.append(fila)
    return filas


def test_nunca_entrena_con_partidos_de_la_misma_semana_o_posteriores():
    filas = _filas_falsas()
    registro = []

    def ajustar_espia(entrenamiento, config):
        return max(f["gameday"] for f in entrenamiento)

    def predecir_espia(ultima_fecha_vista, partidos, config):
        registro.append((ultima_fecha_vista, min(f["gameday"] for f in partidos)))
        return [0.5] * len(partidos)

    recorrer(filas, {}, 2021, 2022, ajustar_fn=ajustar_espia, predecir_fn=predecir_espia)

    assert len(registro) == 12  # 2 temporadas x 6 semanas
    for ultima_vista, primera_predicha in registro:
        assert ultima_vista < primera_predicha


def test_predice_cada_partido_una_sola_vez():
    filas = _filas_falsas()
    config = {"tipo": "logistica", "C": 0.1}
    predicciones = recorrer(filas, config, 2021, 2022)
    esperados = [f["game_id"] for f in filas if f["season"] >= 2021]
    assert sorted(p["game_id"] for p in predicciones) == sorted(esperados)
    assert all(0 < p["p_home_modelo"] < 1 for p in predicciones)


def test_el_reporte_se_arma_sin_errores():
    filas = _filas_falsas(temporadas=(2022, 2023, 2024, 2025))
    config = {"tipo": "logistica", "C": 0.1, "calibrar": False}
    predicciones = recorrer(filas, config, 2023, 2025)
    texto = escribir_reporte(predicciones, 0.56, config)
    assert "## Veredicto" in texto
    assert "Prueba honesta 2024–2025" in texto
