"""Pruebas del experimento de cruces ataque-defensa."""

import pytest

from nfl_model.exp_cruces import agregar_interacciones, calcular_medias
from nfl_model.features import METRICAS_CRUCE


def _fila(**valores):
    base = {f"{lado}_{m}": 0.0 for lado in ("home", "away") for m in METRICAS_CRUCE}
    base.update(valores)
    return base


MEDIAS_CERO = {m: 0.0 for m in METRICAS_CRUCE}


def test_buen_ataque_local_contra_mala_defensa_visitante_es_positivo():
    f = _fila(home_off_pass_epa=0.2, away_def_pass_epa=0.1)  # def alta = permite mucho
    nueva = agregar_interacciones([f], MEDIAS_CERO)[0]
    assert nueva["inter_pase"] == pytest.approx(0.02)
    assert nueva["inter_carrera"] == pytest.approx(0.0)


def test_es_simetrica_al_intercambiar_local_y_visitante():
    ida = _fila(home_off_rush_epa=0.3, away_def_rush_epa=0.2,
                away_off_rush_epa=-0.1, home_def_rush_epa=0.05)
    vuelta = _fila(away_off_rush_epa=0.3, home_def_rush_epa=0.2,
                   home_off_rush_epa=-0.1, away_def_rush_epa=0.05)
    a = agregar_interacciones([ida], MEDIAS_CERO)[0]["inter_carrera"]
    b = agregar_interacciones([vuelta], MEDIAS_CERO)[0]["inter_carrera"]
    assert a == pytest.approx(-b)


def test_centra_con_las_medias_y_no_modifica_las_filas_originales():
    f = _fila(home_off_pass_epa=0.3, away_def_pass_epa=0.2)
    medias = {**MEDIAS_CERO, "off_pass_epa": 0.1, "def_pass_epa": 0.1}
    nueva = agregar_interacciones([f], medias)[0]
    # (0.3-0.1)*(0.2-0.1) - (0-0.1)*(0-0.1) = 0.02 - 0.01
    assert nueva["inter_pase"] == pytest.approx(0.01)
    assert "inter_pase" not in f


def test_medias_promedian_local_y_visitante():
    filas = [_fila(home_off_pass_epa=0.4), _fila(away_off_pass_epa=0.2)]
    assert calcular_medias(filas)["off_pass_epa"] == pytest.approx(0.15)
