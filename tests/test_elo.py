"""Pruebas de la Fase 2. Usan datos inventados, no necesitan descarga."""

import pytest

from nfl_model.elo import calcular_elo, metricas, prob_local, prob_vegas


def _partido(game_id, temporada, local, visita, resultado, neutral=False, jugado=True):
    return {
        "game_id": game_id,
        "season": temporada,
        "home_team": local,
        "away_team": visita,
        "result": resultado,
        "neutral": neutral,
        "played": jugado,
    }


def test_probabilidad_basica():
    assert prob_local(1500, 1500, 0) == pytest.approx(0.5)
    assert prob_local(1500, 1500, 50) > 0.5
    assert prob_local(1600, 1500, 0) + prob_local(1500, 1600, 0) == pytest.approx(1.0)


def test_lo_que_gana_uno_lo_pierde_el_otro():
    _, ratings = calcular_elo([_partido("g1", 2020, "A", "B", 7)], k=20, ventaja_local=50)
    assert ratings["A"] > 1500 > ratings["B"]
    assert ratings["A"] + ratings["B"] == pytest.approx(3000)


def test_sede_neutral_no_da_ventaja():
    salida, _ = calcular_elo(
        [_partido("g1", 2020, "A", "B", 3, neutral=True)], k=20, ventaja_local=50
    )
    assert salida[0]["p_home_elo"] == pytest.approx(0.5)


def test_ganar_por_mas_puntos_sube_mas():
    _, por_poco = calcular_elo([_partido("g1", 2020, "A", "B", 1)], k=20, ventaja_local=0)
    _, por_mucho = calcular_elo([_partido("g1", 2020, "A", "B", 28)], k=20, ventaja_local=0)
    assert por_mucho["A"] > por_poco["A"]


def test_regresion_entre_temporadas():
    partidos = [
        _partido("g1", 2020, "A", "B", 14),
        _partido("g2", 2021, "A", "B", 0, jugado=False),
    ]
    salida, _ = calcular_elo(partidos, k=20, ventaja_local=0, regresion=0.5)
    subida_2020 = salida[0]["elo_home_pre"]  # 1500
    _, tras_g1 = calcular_elo(partidos[:1], k=20, ventaja_local=0)
    esperado = 1500 + (tras_g1["A"] - 1500) * 0.5
    assert subida_2020 == pytest.approx(1500)
    assert salida[1]["elo_home_pre"] == pytest.approx(esperado)


def test_la_probabilidad_no_usa_el_resultado_del_mismo_partido():
    base = [_partido("g1", 2020, "A", "B", 10), _partido("g2", 2020, "B", "A", 3)]
    alterado = [base[0], _partido("g2", 2020, "B", "A", -30)]
    p_base = calcular_elo(base, k=20, ventaja_local=50)[0][1]["p_home_elo"]
    p_alterado = calcular_elo(alterado, k=20, ventaja_local=50)[0][1]["p_home_elo"]
    assert p_base == pytest.approx(p_alterado)


def test_partido_no_jugado_no_cambia_ratings():
    _, ratings = calcular_elo(
        [_partido("g1", 2020, "A", "B", None, jugado=False)], k=20, ventaja_local=50
    )
    assert ratings == {"A": 1500, "B": 1500}


def test_probabilidad_de_vegas_quita_la_comision():
    assert prob_vegas(-110, -110) == pytest.approx(0.5)
    assert prob_vegas(-200, 170) == pytest.approx(0.6429, abs=1e-4)
    assert prob_vegas(170, -200) == pytest.approx(1 - 0.6429, abs=1e-4)
    assert prob_vegas(None, -110) is None


def test_metricas():
    m = metricas([(0.8, 1), (0.6, 0), (0.3, 0), (0.4, 1)])
    assert m["partidos"] == 4
    assert m["acierto"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx((0.04 + 0.36 + 0.09 + 0.36) / 4)
