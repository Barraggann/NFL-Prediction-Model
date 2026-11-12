"""Pruebas de la Fase 4 con datos inventados."""

import numpy as np
import pytest

from nfl_model.features import VARIABLES
from nfl_model.train import (
    ajustar,
    ajustar_platt,
    aplicar_platt,
    con_ganador,
    platt_cruzado,
    predecir,
    tabla_confianza,
)


def _filas_falsas(n=600, semilla=0):
    """El local gana más cuando elo_diff es alto; el resto es ruido."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n):
        fila = {v: float(rng.normal()) for v in VARIABLES}
        fila["elo_diff"] = float(rng.normal(0, 150))
        p = 1 / (1 + 10 ** (-fila["elo_diff"] / 400))
        fila["home_win"] = int(rng.random() < p)
        fila["season"] = 2006 + i % 3
        filas.append(fila)
    return filas


@pytest.mark.parametrize(
    "config",
    [
        {"tipo": "logistica", "C": 0.1},
        {"tipo": "boosting", "max_iter": 50, "max_depth": 2},
    ],
)
def test_el_modelo_aprende_que_mas_elo_es_mas_probabilidad(config):
    filas = _filas_falsas()
    modelo = ajustar(filas, config)

    neutro = {v: 0.0 for v in VARIABLES}
    favorito = {**neutro, "elo_diff": 300.0}
    desvalido = {**neutro, "elo_diff": -300.0}
    p_fav, p_des = predecir(modelo, [favorito, desvalido], config)

    assert 0 < p_des < 0.5 < p_fav < 1


def test_con_ganador_descarta_empates_y_nulos():
    filas = _filas_falsas(n=6)
    filas[0]["home_win"] = None  # empate o partido futuro
    filas[1]["diff_off_epa"] = None  # sin forma previa
    filas[2]["season"] = 1999  # fuera del rango
    assert len(con_ganador(filas, 2006, 2020)) == 3


def test_platt_identidad_no_cambia_nada():
    p = np.array([0.1, 0.5, 0.9])
    assert aplicar_platt(p, 1.0, 0.0) == pytest.approx(p)


def test_platt_corrige_un_modelo_sobreconfiado():
    rng = np.random.default_rng(1)
    verdad = rng.uniform(0.3, 0.7, 4000)
    y = (rng.random(4000) < verdad).astype(int)
    logit = np.log(verdad / (1 - verdad))
    exagerado = 1 / (1 + np.exp(-2 * logit))  # el doble de confianza de la real
    a, _ = ajustar_platt(exagerado, y)
    assert a == pytest.approx(0.5, abs=0.15)


def test_predecir_aplica_calibracion_si_se_pide():
    filas = _filas_falsas()
    base = {"tipo": "logistica", "C": 0.1}
    modelo = ajustar(filas, base)
    crudo = predecir(modelo, filas[:5], base)
    encogido = predecir(
        modelo, filas[:5], {**base, "calibrar": True, "platt": {"a": 0.5, "b": 0.0}}
    )
    for c, e in zip(crudo, encogido):
        assert abs(e - 0.5) < abs(c - 0.5)


def test_platt_cruzado_conserva_tamano_y_rango():
    filas = _filas_falsas()
    config = {"tipo": "logistica", "C": 0.1}
    p = predecir(ajustar(filas, config), filas, config)
    calibrado = platt_cruzado(filas, p)
    assert len(calibrado) == len(filas)
    assert all(0 < x < 1 for x in calibrado)


def test_tabla_de_confianza():
    filas = [{"home_win": 1}, {"home_win": 0}, {"home_win": 0}, {"home_win": 1}]
    p = [0.75, 0.25, 0.55, 0.45]  # confianzas: 75, 75, 55, 55
    tabla = {r["rango"]: r for r in tabla_confianza(filas, p)}
    assert tabla["70–80%"]["partidos"] == 2
    assert tabla["70–80%"]["acierto"] == pytest.approx(1.0)
    assert tabla["50–60%"]["partidos"] == 2
    assert tabla["50–60%"]["acierto"] == pytest.approx(0.0)
    assert tabla["80–100%"]["partidos"] == 0
