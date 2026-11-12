"""Pruebas de la Fase 6: picks semanales y su calificación."""

from datetime import date, timedelta

import numpy as np

from nfl_model.features import VARIABLES
from nfl_model.predict import (
    agregar_al_registro,
    elegir_semana,
    generar_picks,
    leer_registro,
    reporte_semana,
)
from nfl_model.track import calificar, comparar_con_vegas, escribir_historial, por_rango

HOY = date(2026, 11, 11)
CONFIG = {"tipo": "logistica", "C": 0.1, "calibrar": False}


def _filas_falsas():
    """Historia jugada (2024-2025) más dos semanas futuras de 2026."""
    rng = np.random.default_rng(0)
    filas = []

    def fila(gid, temporada, semana, dia, jugado, i):
        f = {v: float(rng.normal()) for v in VARIABLES}
        f["elo_diff"] = float(rng.normal(0, 150))
        p = 1 / (1 + 10 ** (-f["elo_diff"] / 400))
        f.update(
            {
                "game_id": gid,
                "season": temporada,
                "week": semana,
                "gameday": dia,
                "home_team": f"L{i}",
                "away_team": f"V{i}",
                "played": jugado,
                "home_win": int(rng.random() < p) if jugado else None,
                "p_home_elo": p,
                "p_home_vegas": min(max(p - 0.03, 0.01), 0.99),
            }
        )
        return f

    for temporada in (2024, 2025):
        for semana in range(1, 11):
            dia = date(temporada, 9, 8) + timedelta(days=7 * (semana - 1))
            for i in range(8):
                filas.append(fila(f"{temporada}_{semana:02d}_{i}", temporada, semana, dia, True, i))
    for semana, dia in ((10, date(2026, 11, 15)), (11, date(2026, 11, 22))):
        for i in range(4):
            filas.append(fila(f"2026_{semana:02d}_{i}", 2026, semana, dia, False, i))
    return filas


def test_elige_la_proxima_semana_pendiente():
    assert elegir_semana(_filas_falsas(), HOY) == (2026, 10)
    assert elegir_semana(_filas_falsas(), date(2026, 11, 16)) == (2026, 11)
    assert elegir_semana(_filas_falsas(), date(2027, 3, 1)) is None


def test_genera_un_pick_por_partido_pendiente():
    picks = generar_picks(_filas_falsas(), CONFIG, 2026, 10, HOY)
    assert len(picks) == 4
    for p in picks:
        assert p["pick"] in (p["local"], p["visita"])
        assert 0.5 <= p["p_pick"] < 1
        gana_local = p["pick"] == p["local"]
        assert (p["p_home_modelo"] >= 0.5) == gana_local


def test_no_genera_picks_de_partidos_que_ya_empezaron():
    filas = _filas_falsas()
    assert generar_picks(filas, CONFIG, 2026, 10, date(2026, 11, 16)) == []
    assert generar_picks(filas, CONFIG, 2025, 3, HOY) == []


def test_el_registro_solo_agrega_y_nunca_reescribe(tmp_path):
    ruta = tmp_path / "picks" / "picks_log.csv"
    picks = generar_picks(_filas_falsas(), CONFIG, 2026, 10, HOY)

    assert len(agregar_al_registro(picks, ruta)) == 4
    original = ruta.read_text("utf-8")

    # Segundo intento con picks "cambiados": no se escribe nada.
    cambiados = [{**p, "pick": "OTRO", "p_pick": 0.99} for p in picks]
    assert agregar_al_registro(cambiados, ruta) == []
    assert ruta.read_text("utf-8") == original

    # Una semana nueva sí se agrega, debajo de lo anterior.
    semana_11 = generar_picks(_filas_falsas(), CONFIG, 2026, 11, HOY)
    assert len(agregar_al_registro(semana_11, ruta)) == 4
    assert ruta.read_text("utf-8").startswith(original)
    assert len(leer_registro(ruta)) == 8


def test_reporte_de_la_semana(tmp_path):
    ruta = tmp_path / "picks_log.csv"
    agregar_al_registro(generar_picks(_filas_falsas(), CONFIG, 2026, 10, HOY), ruta)
    texto = reporte_semana(leer_registro(ruta), 2026, 10)
    assert "semana 10" in texto
    assert "No es consejo de apuestas" in texto
    partidos = [
        linea for linea in texto.splitlines()
        if linea.startswith("| V") and not linea.startswith("| Visita")
    ]
    assert len(partidos) == 4


def _registro_falso():
    base = {"temporada": "2026", "semana": "10", "generado": "2026-11-11T10:00:00",
            "gameday": "2026-11-15", "p_home_elo": "0.5", "qb_anunciado": "1"}
    return [
        # pick local, 75%, gana local -> acierto
        {**base, "game_id": "g1", "visita": "V1", "local": "L1", "pick": "L1",
         "p_pick": "0.75", "p_home_modelo": "0.75", "p_home_vegas": "0.70"},
        # pick visita, 65%, gana local -> fallo (Las Vegas iba con el local)
        {**base, "game_id": "g2", "visita": "V2", "local": "L2", "pick": "V2",
         "p_pick": "0.65", "p_home_modelo": "0.35", "p_home_vegas": "0.55"},
        # empate
        {**base, "game_id": "g3", "visita": "V3", "local": "L3", "pick": "L3",
         "p_pick": "0.55", "p_home_modelo": "0.55", "p_home_vegas": "0.52"},
        # no jugado
        {**base, "game_id": "g4", "visita": "V4", "local": "L4", "pick": "V4",
         "p_pick": "0.80", "p_home_modelo": "0.20", "p_home_vegas": ""},
    ]


RESULTADOS = {
    "g1": {"played": True, "home_win": 1, "tie": False},
    "g2": {"played": True, "home_win": 1, "tie": False},
    "g3": {"played": True, "home_win": None, "tie": True},
    "g4": {"played": False, "home_win": None, "tie": False},
}


def test_calificar():
    estados = {f["game_id"]: f["estado"] for f in calificar(_registro_falso(), RESULTADOS)}
    assert estados == {"g1": "acierto", "g2": "fallo", "g3": "empate", "g4": "pendiente"}


def test_calificar_no_modifica_el_registro():
    registro = _registro_falso()
    copia = [dict(f) for f in registro]
    calificar(registro, RESULTADOS)
    assert registro == copia


def test_por_rango_solo_cuenta_partidos_decididos():
    tabla = {r["rango"]: r for r in por_rango(calificar(_registro_falso(), RESULTADOS))}
    assert (tabla["70–80%"]["partidos"], tabla["70–80%"]["aciertos"]) == (1, 1)
    assert (tabla["60–70%"]["partidos"], tabla["60–70%"]["aciertos"]) == (1, 0)
    assert tabla["50–60%"]["partidos"] == 0  # el empate no cuenta
    assert tabla["80–100%"]["partidos"] == 0  # el pendiente tampoco


def test_comparacion_con_vegas():
    v = comparar_con_vegas(calificar(_registro_falso(), RESULTADOS))
    assert v["partidos"] == 2
    assert v["aciertos_modelo"] == 1
    assert v["aciertos_vegas"] == 2
    assert v["eligieron_distinto"] == 1
    assert v["modelo_acerto_en_distintos"] == 0
    assert v["log_loss_vegas"] < v["log_loss_modelo"]


def test_historial_se_arma_y_avisa_de_muestra_chica():
    texto = escribir_historial(calificar(_registro_falso(), RESULTADOS))
    assert "1 de 2 (50.0%)" in texto
    assert "la muestra es chica" in texto
    assert texto.count("## Por semana") == 1
