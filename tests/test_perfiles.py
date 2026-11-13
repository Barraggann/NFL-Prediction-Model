"""Pruebas de los perfiles de equipo y sus rankings."""

from datetime import date

import matplotlib.image as mpimg
import pytest

from nfl_model.content import ALTO, ANCHO, duelo_de_la_semana, imagen_duelo
from nfl_model.perfiles import calcular_perfiles, duelos, escribir_reporte, texto_duelo
from nfl_model.predict import reporte_semana


def _stat(pase, carrera, pase_perm, carrera_perm, capturas=0, robos=0, perdidas=0):
    jugadas = 60
    return {
        "off_yards": pase + carrera, "off_pass_yards": pase, "off_rush_yards": carrera,
        "off_plays": jugadas, "off_epa": 0.1, "off_sacks": 0, "off_turnovers": perdidas,
        "def_yards": pase_perm + carrera_perm, "def_pass_yards": pase_perm,
        "def_rush_yards": carrera_perm, "def_plays": jugadas, "def_epa": -0.1,
        "def_sacks": capturas, "def_turnovers": robos,
    }


def _partido(gid, dia, local, visita, pts_local, pts_visita, tipo="REG", jugado=True, temporada=2026):
    return {
        "game_id": gid, "season": temporada, "game_type": tipo, "gameday": dia,
        "home_team": local, "away_team": visita, "played": jugado,
        "home_score": pts_local, "away_score": pts_visita,
    }


def _liga():
    """A pasa mucho, B corre mucho; C y D son el reflejo defensivo."""
    partidos = [
        _partido("g1", date(2026, 9, 13), "A", "C", 30, 10),
        _partido("g2", date(2026, 9, 13), "B", "D", 20, 17),
        _partido("g3", date(2026, 9, 20), "A", "D", 28, 14),
        _partido("g4", date(2026, 9, 20), "B", "C", 24, 21),
        # No deben contar: playoffs, no jugado y otra temporada.
        _partido("p1", date(2027, 1, 16), "A", "B", 99, 0, tipo="WC"),
        _partido("f1", date(2026, 9, 27), "A", "B", None, None, jugado=False),
        _partido("v1", date(2025, 9, 14), "A", "B", 99, 0, temporada=2025),
    ]
    stats = {
        ("g1", "A"): _stat(300, 80, 150, 90, capturas=4, robos=2),
        ("g1", "C"): _stat(150, 90, 300, 80, perdidas=2),
        ("g2", "B"): _stat(180, 170, 200, 100),
        ("g2", "D"): _stat(200, 100, 180, 170),
        ("g3", "A"): _stat(320, 100, 210, 70, capturas=2),
        ("g3", "D"): _stat(210, 70, 320, 100),
        ("g4", "B"): _stat(160, 190, 220, 110, robos=1),
        ("g4", "C"): _stat(220, 110, 160, 190, perdidas=1),
        ("p1", "A"): _stat(999, 999, 0, 0),
        ("p1", "B"): _stat(0, 0, 999, 999),
        ("v1", "A"): _stat(999, 999, 0, 0),
        ("v1", "B"): _stat(0, 0, 999, 999),
    }
    return partidos, stats


def test_promedios_por_partido():
    perfiles = calcular_perfiles(*_liga(), 2026)
    a = perfiles["A"]
    assert a["partidos"] == 2
    assert a["pts"] == pytest.approx(29.0)
    assert a["pts_perm"] == pytest.approx(12.0)
    assert a["pase"] == pytest.approx(310.0)
    assert a["carrera"] == pytest.approx(90.0)
    assert a["yds"] == pytest.approx(400.0)
    assert a["pase_perm"] == pytest.approx(180.0)
    assert a["yds_jugada"] == pytest.approx(800 / 120)
    assert a["capturas"] == pytest.approx(3.0)
    assert a["dif_balones"] == pytest.approx(1.0)
    assert perfiles["C"]["dif_balones"] == pytest.approx(-1.5)


def test_solo_cuenta_temporada_regular_jugada():
    perfiles = calcular_perfiles(*_liga(), 2026)
    assert set(perfiles) == {"A", "B", "C", "D"}
    assert all(p["partidos"] == 2 for p in perfiles.values())


def test_ranking_respeta_el_sentido_de_cada_estadistica():
    perfiles = calcular_perfiles(*_liga(), 2026)
    # Ataque: más es mejor.
    assert perfiles["A"]["rank"]["pase"] == 1
    assert perfiles["B"]["rank"]["carrera"] == 1
    # Defensa: menos permitido es mejor. A permite 180 por aire; D, 250.
    assert perfiles["A"]["rank"]["pase_perm"] == 1
    assert perfiles["D"]["rank"]["pase_perm"] == 4
    assert perfiles["A"]["rank"]["pts_perm"] == 1
    # Perder menos balones es mejor: C pierde 1.5 por partido.
    assert perfiles["C"]["rank"]["perdidas"] == 4


def test_los_empates_comparten_lugar():
    perfiles = calcular_perfiles(*_liga(), 2026)
    # A, B y D no pierden balones: los tres son #1.
    assert [perfiles[e]["rank"]["perdidas"] for e in "ABD"] == [1, 1, 1]


def test_antes_de_corta_por_fecha():
    perfiles = calcular_perfiles(*_liga(), 2026, antes_de=date(2026, 9, 20))
    assert all(p["partidos"] == 1 for p in perfiles.values())
    assert perfiles["A"]["pase"] == pytest.approx(300.0)


def test_duelos_ordena_por_ventaja():
    perfiles = calcular_perfiles(*_liga(), 2026)
    cruces = duelos(perfiles, "D", "A")  # A visita a D
    assert len(cruces) == 4
    mayor = cruces[0]
    # El pase de A (#1) contra la defensa aérea de D (#4).
    assert (mayor["ataque"], mayor["defensa"], mayor["tipo"]) == ("A", "D", "aire")
    assert mayor["ventaja"] == 3
    assert [c["ventaja"] for c in cruces] == sorted((c["ventaja"] for c in cruces), reverse=True)
    assert "Ataque aéreo de A (#1" in texto_duelo(mayor)
    assert duelos(perfiles, "A", "ZZZ") == []


def test_reporte_de_perfiles():
    texto = escribir_reporte(calcular_perfiles(*_liga(), 2026), 2026)
    for seccion in ("## Ofensiva", "## Defensiva", "## Presión y balones"):
        assert seccion in texto
    assert "310.0 (#1)" in texto


def _picks():
    base = {"temporada": "2026", "semana": "4", "generado": "2026-09-23T10:00:00",
            "gameday": "2026-09-27", "p_home_elo": "0.5", "qb_anunciado": "1",
            "p_home_vegas": "0.40"}
    return [
        {**base, "game_id": "x1", "visita": "A", "local": "D", "pick": "A",
         "p_pick": "0.74", "p_home_modelo": "0.26"},
        {**base, "game_id": "x2", "visita": "C", "local": "B", "pick": "B",
         "p_pick": "0.58", "p_home_modelo": "0.58"},
    ]


def test_el_reporte_semanal_incluye_duelos_si_hay_perfiles():
    perfiles = calcular_perfiles(*_liga(), 2026)
    sin = reporte_semana(_picks(), 2026, 4)
    con = reporte_semana(_picks(), 2026, 4, perfiles)
    assert "Duelo más desigual" not in sin
    assert "Duelo más desigual" in con
    assert "Ataque aéreo de A" in con


def test_imagen_del_duelo(tmp_path):
    perfiles = calcular_perfiles(*_liga(), 2026)
    duelo = duelo_de_la_semana(_picks(), perfiles)
    assert duelo["partido"]["game_id"] == "x1"
    ruta = imagen_duelo(duelo, len(perfiles), 2026, 4, tmp_path / "duelo.png")
    alto, ancho = mpimg.imread(ruta).shape[:2]
    assert (ancho, alto) == (ANCHO, ALTO)
    assert imagen_duelo(None, 4, 2026, 4, tmp_path / "nada.png") is None
