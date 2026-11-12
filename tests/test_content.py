"""Pruebas de la Fase 7: las imágenes se generan con el tamaño correcto."""

import matplotlib.image as mpimg
import pytest

from nfl_model.content import (
    ALTO,
    ANCHO,
    imagen_historial,
    imagen_picks,
    imagen_vegas,
    mayor_diferencia,
    picks_de_semana,
)


def _registro():
    base = {"temporada": "2026", "semana": "10", "generado": "2026-11-11T10:00:00",
            "gameday": "2026-11-15", "p_home_elo": "0.5", "qb_anunciado": "1"}
    return [
        {**base, "game_id": "g1", "visita": "BUF", "local": "NYJ", "pick": "BUF",
         "p_pick": "0.80", "p_home_modelo": "0.20", "p_home_vegas": "0.22"},
        {**base, "game_id": "g2", "visita": "LA", "local": "ARI", "pick": "LA",
         "p_pick": "0.72", "p_home_modelo": "0.28", "p_home_vegas": "0.20"},
        {**base, "game_id": "g3", "visita": "NE", "local": "DET", "pick": "DET",
         "p_pick": "0.53", "p_home_modelo": "0.53", "p_home_vegas": ""},
        {**base, "game_id": "g4", "visita": "KC", "local": "ATL", "pick": "KC",
         "p_pick": "0.60", "p_home_modelo": "0.40", "p_home_vegas": "0.38",
         "semana": "11"},
    ]


def _es_vertical(ruta):
    alto, ancho = mpimg.imread(ruta).shape[:2]
    return (ancho, alto) == (ANCHO, ALTO)


def test_picks_de_semana_filtra_y_ordena():
    picks = picks_de_semana(_registro(), 2026, 10)
    assert [p["pick"] for p in picks] == ["BUF", "LA", "DET"]


def test_mayor_diferencia_usa_el_equipo_del_pick():
    d = mayor_diferencia(picks_de_semana(_registro(), 2026, 10))
    # LA: modelo 72%, Las Vegas 80% (1 - 0.20). BUF: 80% contra 78%.
    assert d["pick"] == "LA"
    assert d["p_vegas_pick"] == pytest.approx(0.80)
    assert d["diferencia"] == pytest.approx(-0.08)


def test_mayor_diferencia_sin_lineas():
    solo_sin_linea = [f for f in _registro() if f["p_home_vegas"] == ""]
    assert mayor_diferencia(solo_sin_linea) is None


def test_imagen_de_picks(tmp_path):
    ruta = imagen_picks(picks_de_semana(_registro(), 2026, 10), 2026, 10, tmp_path / "a" / "picks.png")
    assert _es_vertical(ruta)


def test_imagen_de_picks_sin_ninguno_fuerte(tmp_path):
    flojos = [{**f, "p_pick": "0.55"} for f in picks_de_semana(_registro(), 2026, 10)]
    assert _es_vertical(imagen_picks(flojos, 2026, 10, tmp_path / "picks.png"))


def test_imagen_contra_vegas(tmp_path):
    ruta = imagen_vegas(picks_de_semana(_registro(), 2026, 10), 2026, 10, tmp_path / "vegas.png")
    assert _es_vertical(ruta)
    assert imagen_vegas([], 2026, 10, tmp_path / "nada.png") is None


def test_imagen_de_historial(tmp_path):
    tabla = [
        {"rango": "50–60%", "partidos": 224, "aciertos": 119, "confianza_media": 0.55},
        {"rango": "60–70%", "partidos": 165, "aciertos": 105, "confianza_media": 0.646},
        {"rango": "70–80%", "partidos": 131, "aciertos": 100, "confianza_media": 0.742},
        {"rango": "80–100%", "partidos": 0, "aciertos": 0, "confianza_media": None},
    ]
    assert _es_vertical(imagen_historial(tabla, "Backtest 2024–2025 · 520 partidos", tmp_path / "h.png"))
    assert imagen_historial([], "", tmp_path / "vacio.png") is None
