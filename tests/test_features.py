"""Pruebas de la Fase 3. La más importante es la anti-fuga."""

from datetime import date, timedelta

import polars as pl
import pytest

from nfl_model import DATA_PROCESSED
from nfl_model.features import (
    ALFA,
    METRICAS_FORMA,
    VARIABLES,
    CalidadQB,
    construir_variables,
)

# ---------- datos inventados ----------

EQUIPOS = ["A", "B", "C", "D"]
CRUCES = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]


def _liga_falsa(temporadas=(2020, 2021), qb_extra=None):
    """Mini liga de 4 equipos; cada temporada juegan todos contra todos."""
    partidos, stats, elo = [], {}, {}
    dia = date(2020, 9, 1)
    n = 0
    for temporada in temporadas:
        for semana, (local, visita) in enumerate(CRUCES, 1):
            n += 1
            gid = f"{temporada}_{semana:02d}_{visita}_{local}"
            partidos.append(
                {
                    "game_id": gid,
                    "season": temporada,
                    "week": semana,
                    "game_type": "REG",
                    "gameday": dia + timedelta(days=7 * n),
                    "home_team": local,
                    "away_team": visita,
                    "played": True,
                    "home_win": n % 2,
                    "neutral": False,
                    "div_game": 1,
                    "home_rest": 7,
                    "away_rest": 14 if n == 3 else 7,
                    "home_qb_id": f"qb_{local}",
                    "away_qb_id": (qb_extra or {}).get(gid, f"qb_{visita}"),
                }
            )
            for i, equipo in enumerate((local, visita)):
                stats[(gid, equipo)] = {m: 0.01 * n + 0.1 * i + 0.001 * j
                                        for j, m in enumerate(METRICAS_FORMA)}
            elo[gid] = {
                "elo_home_pre": 1500.0 + n,
                "elo_away_pre": 1500.0 - n,
                "p_home_elo": 0.55,
                "p_home_vegas": None,
            }
    return partidos, stats, elo


def _por_id(filas):
    return {f["game_id"]: f for f in filas}


def test_una_fila_por_partido_con_todas_las_variables():
    partidos, stats, elo = _liga_falsa()
    filas = construir_variables(partidos, stats, elo)
    assert len(filas) == len(partidos)
    for f in filas:
        assert set(VARIABLES) <= set(f)


def test_primer_partido_no_tiene_forma_previa():
    partidos, stats, elo = _liga_falsa()
    primera = construir_variables(partidos, stats, elo)[0]
    assert primera["diff_off_epa"] is None
    assert primera["elo_diff"] == pytest.approx(2.0)


def test_anti_fuga_cambiar_el_futuro_no_cambia_el_pasado():
    partidos, stats, elo = _liga_falsa()
    original = construir_variables(partidos, stats, elo)

    corte = 7  # se altera todo desde el partido 8 en adelante
    stats_alterado = dict(stats)
    for p in partidos[corte:]:
        for equipo in (p["home_team"], p["away_team"]):
            stats_alterado[(p["game_id"], equipo)] = {m: 99.0 for m in METRICAS_FORMA}
    alterado = construir_variables(partidos, stats_alterado, elo)

    # Hasta el partido 8 inclusive (cuyas variables son previas a su kickoff) nada cambia.
    for a, b in zip(original[: corte + 1], alterado[: corte + 1]):
        for v in VARIABLES:
            assert a[v] == b[v], (a["game_id"], v)
    # Y sí cambia después: la prueba no pasa por casualidad.
    assert any(
        a[v] != b[v] for a, b in zip(original[corte + 1 :], alterado[corte + 1 :]) for v in VARIABLES
    )


def test_anti_fuga_truncar_el_calendario_da_lo_mismo():
    partidos, stats, elo = _liga_falsa()
    completo = construir_variables(partidos, stats, elo)
    truncado = construir_variables(partidos[:9], stats, elo)
    for a, b in zip(completo[:9], truncado):
        for v in VARIABLES:
            assert a[v] == b[v], (a["game_id"], v)


def test_promedio_movil_exponencial():
    partidos, stats, elo = _liga_falsa(temporadas=(2020,))
    filas = _por_id(construir_variables(partidos, stats, elo))
    # A juega los partidos 1 (vs B), 3 (vs C) y 5 (vs D), siempre de local.
    # D juega el 2 (visita en C) y el 4 (visita en B).
    a1, a3 = stats[("2020_01_B_A", "A")], stats[("2020_03_C_A", "A")]
    d2, d4 = stats[("2020_02_D_C", "D")], stats[("2020_04_D_B", "D")]
    forma_a = (1 - ALFA) * a1["off_epa"] + ALFA * a3["off_epa"]
    forma_d = (1 - ALFA) * d2["off_epa"] + ALFA * d4["off_epa"]
    assert filas["2020_05_D_A"]["diff_off_epa"] == pytest.approx(forma_a - forma_d)


def test_descanso_y_bye():
    partidos, stats, elo = _liga_falsa(temporadas=(2020,))
    tercero = construir_variables(partidos, stats, elo)[2]
    assert tercero["rest_diff"] == -7
    assert tercero["away_bye"] == 1
    assert tercero["home_bye"] == 0


def test_cambio_de_qb():
    cambio = {"2020_04_D_B": "qb_suplente"}
    partidos, stats, elo = _liga_falsa(temporadas=(2020,), qb_extra=cambio)
    filas = _por_id(construir_variables(partidos, stats, elo))
    assert filas["2020_02_D_C"]["away_qb_change"] == 0  # primer partido de D
    assert filas["2020_04_D_B"]["away_qb_change"] == 1  # entra el suplente
    assert filas["2020_05_D_A"]["away_qb_change"] == 1  # vuelve el titular
    assert filas["2020_03_C_A"]["home_qb_change"] == 0


def test_partido_futuro_no_actualiza_la_forma():
    partidos, stats, elo = _liga_falsa(temporadas=(2020,))
    base = _por_id(construir_variables(partidos, stats, elo))
    partidos[2] = {**partidos[2], "played": False, "home_win": None}
    con_pendiente = _por_id(construir_variables(partidos, stats, elo))
    # Sin el partido 3, la forma de A en el partido 5 es solo la del partido 1.
    assert con_pendiente["2020_05_D_A"]["diff_off_epa"] != base["2020_05_D_A"]["diff_off_epa"]


# ---------- quarterbacks ----------


def test_qb_sin_historial_vale_menos_que_uno_probado():
    calidad = CalidadQB()
    for _ in range(30):
        calidad.registrar("bueno", 35, 35 * 0.25)
        calidad.registrar("malo", 35, 35 * -0.15)
    assert calidad.valor("bueno") > calidad.valor("novato") > calidad.valor("malo")
    assert calidad.valor(None) == pytest.approx(calidad.valor("novato"))


def test_qb_con_pocas_jugadas_se_acerca_a_la_previa():
    calidad = CalidadQB()
    for _ in range(30):
        calidad.registrar("veterano", 35, 35 * 0.05)
    previa = calidad.valor("novato")
    calidad.registrar("racha", 10, 10 * 1.0)  # diez jugadas espectaculares
    assert previa < calidad.valor("racha") < previa + 0.1


def _qb_stats_falsos(partidos, valor_por_qb):
    """Cada titular lanza 35 pases por partido con su EPA por jugada."""
    return {
        p["game_id"]: [
            {"qb_id": qb, "dropbacks": 35, "qb_epa_sum": 35 * valor_por_qb.get(qb, 0.0)}
            for qb in (p["home_qb_id"], p["away_qb_id"])
        ]
        for p in partidos
    }


def test_diff_qb_favorece_al_equipo_con_mejor_quarterback():
    partidos, stats, elo = _liga_falsa()
    qb_stats = _qb_stats_falsos(partidos, {"qb_A": 0.3, "qb_D": -0.2})
    filas = _por_id(construir_variables(partidos, stats, elo, qb_stats=qb_stats))
    # Segunda temporada, A recibe a D: ambos ya tienen historial.
    assert filas["2021_05_D_A"]["diff_qb"] > 0.05
    # Sin cambios de titular, el ajuste por cambio es pequeño.
    assert abs(filas["2021_05_D_A"]["diff_qb_cambio"]) < 0.05


def test_cambio_a_un_suplente_baja_el_ajuste_del_equipo():
    cambio = {"2021_05_D_A": "qb_suplente"}  # D llega con suplente sin historial
    partidos, stats, elo = _liga_falsa(qb_extra=cambio)
    qb_stats = _qb_stats_falsos(partidos, {"qb_D": 0.3})
    filas = _por_id(construir_variables(partidos, stats, elo, qb_stats=qb_stats))
    # D es visitante: perder a su buen QB favorece al local.
    assert filas["2021_05_D_A"]["diff_qb_cambio"] > 0.02


def test_anti_fuga_quarterbacks():
    partidos, stats, elo = _liga_falsa()
    qb_stats = _qb_stats_falsos(partidos, {"qb_A": 0.3, "qb_D": -0.2})
    original = construir_variables(partidos, stats, elo, qb_stats=qb_stats)

    corte = 7
    alterado_stats = dict(qb_stats)
    for p in partidos[corte:]:
        alterado_stats[p["game_id"]] = [
            {"qb_id": q["qb_id"], "dropbacks": 35, "qb_epa_sum": 999.0}
            for q in qb_stats[p["game_id"]]
        ]
    alterado = construir_variables(partidos, stats, elo, qb_stats=alterado_stats)

    for a, b in zip(original[: corte + 1], alterado[: corte + 1]):
        assert a["diff_qb"] == b["diff_qb"], a["game_id"]
        assert a["diff_qb_cambio"] == b["diff_qb_cambio"], a["game_id"]
    assert any(
        a["diff_qb"] != b["diff_qb"]
        for a, b in zip(original[corte + 1 :], alterado[corte + 1 :])
    )


# ---------- datos reales ----------

ARCHIVO = DATA_PROCESSED / "features.parquet"
datos_reales = pytest.mark.skipif(
    not ARCHIVO.exists(), reason="Falta correr: python -m nfl_model.features"
)


@datos_reales
def test_una_fila_por_partido_real():
    variables = pl.read_parquet(ARCHIVO)
    partidos = pl.read_parquet(DATA_PROCESSED / "games.parquet")
    assert variables.height == partidos.height
    assert variables["game_id"].n_unique() == variables.height


@datos_reales
def test_sin_nulos_en_partidos_jugados_desde_2006():
    jugados = pl.read_parquet(ARCHIVO).filter(pl.col("played") & (pl.col("season") >= 2006))
    nulos = {v: jugados[v].null_count() for v in VARIABLES}
    assert sum(nulos.values()) == 0, nulos
