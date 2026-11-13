"""Pruebas de la Fase 1.

Las primeras usan datos inventados y corren siempre.
Las de "datos reales" se saltan si todavía no se ha corrido la descarga.
"""

import polars as pl
import pytest

from nfl_model import DATA_PROCESSED
from nfl_model.ingest import (
    COLUMNAS_PARTIDO,
    jugadas_utiles,
    limpiar_partidos,
    resumir_por_equipo,
    resumir_qbs,
)

# ---------- datos inventados ----------


def _calendario_falso() -> pl.DataFrame:
    base = {c: [None, None, None, None] for c in COLUMNAS_PARTIDO}
    base.update(
        {
            "game_id": ["2015_01_SD_OAK", "2015_02_STL_DAL", "2015_03_KC_DEN", "2026_18_NE_BUF"],
            "season": [2015, 2015, 2015, 2026],
            "game_type": ["REG"] * 4,
            "week": [1, 2, 3, 18],
            "gameday": ["2015-09-13", "2015-09-20", "2015-09-27", "2027-01-10"],
            "home_team": ["OAK", "DAL", "DEN", "BUF"],
            "away_team": ["SD", "STL", "KC", "NE"],
            "home_score": [20, 17, 24, None],
            "away_score": [27, 17, 10, None],
            "result": [-7, 0, 14, None],
            "location": ["Home", "Neutral", "Home", "Home"],
        }
    )
    return pl.DataFrame(base, infer_schema_length=None)


def test_limpieza_normaliza_equipos_y_marca_resultados():
    partidos = limpiar_partidos(_calendario_falso()).sort("game_id")
    fila = {f["game_id"]: f for f in partidos.to_dicts()}

    assert fila["2015_01_SD_OAK"]["home_team"] == "LV"
    assert fila["2015_01_SD_OAK"]["away_team"] == "LAC"
    assert fila["2015_02_STL_DAL"]["away_team"] == "LA"

    assert fila["2015_01_SD_OAK"]["home_win"] == 0
    assert fila["2015_03_KC_DEN"]["home_win"] == 1

    # Empate: jugado, sin ganador.
    assert fila["2015_02_STL_DAL"]["tie"] is True
    assert fila["2015_02_STL_DAL"]["home_win"] is None
    assert fila["2015_02_STL_DAL"]["neutral"] is True

    # Partido futuro: no jugado, sin ganador.
    assert fila["2026_18_NE_BUF"]["played"] is False
    assert fila["2026_18_NE_BUF"]["home_win"] is None


def _pbp_falso() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["g1"] * 5,
            "season": [2020] * 5,
            "week": [1] * 5,
            "season_type": ["REG"] * 5,
            "posteam": ["KC", "KC", "OAK", "OAK", None],
            "defteam": ["OAK", "OAK", "KC", "KC", None],
            "pass": [1, 0, 1, 0, 0],
            "rush": [0, 1, 0, 1, 0],
            "epa": [1.0, 0.0, -0.5, 0.5, 2.0],
            "success": [1.0, 0.0, 0.0, 1.0, 1.0],
            # El pase de OAK solo trae el id de respaldo y no trae qb_epa.
            "passer_id": ["q_kc", None, None, None, None],
            "passer_player_id": ["q_kc", None, "q_lv", None, None],
            "qb_epa": [1.2, 0.0, None, 0.5, 2.0],
            # KC: pase de 12 y carrera de 3. OAK: captura de -7 con balón perdido y carrera de 5.
            "play_type": ["pass", "run", "pass", "run", "no_play"],
            "yards_gained": [12, 3, -7, 5, 0],
            "sack": [0, 0, 1, 0, 0],
            "interception": [0, 0, 0, 0, 0],
            "fumble_lost": [0, 0, 1, 0, None],
        },
        schema_overrides={"passer_id": pl.String, "passer_player_id": pl.String},
    )


def test_resumen_por_quarterback():
    qbs = {f["qb_id"]: f for f in resumir_qbs(jugadas_utiles(_pbp_falso())).to_dicts()}
    assert set(qbs) == {"q_kc", "q_lv"}
    assert qbs["q_kc"]["team"] == "KC"
    assert qbs["q_kc"]["dropbacks"] == 1
    assert qbs["q_kc"]["qb_epa_sum"] == pytest.approx(1.2)
    # Sin qb_epa se usa el epa de la jugada; el equipo queda normalizado.
    assert qbs["q_lv"]["team"] == "LV"
    assert qbs["q_lv"]["qb_epa_sum"] == pytest.approx(-0.5)


def test_resumen_por_equipo_cruza_ofensiva_y_defensiva():
    equipos = resumir_por_equipo(jugadas_utiles(_pbp_falso()))
    fila = {f["team"]: f for f in equipos.to_dicts()}

    assert set(fila) == {"KC", "LV"}  # OAK normalizado, jugada sin equipo fuera
    assert fila["KC"]["off_plays"] == 2
    assert fila["KC"]["off_epa"] == pytest.approx(0.5)
    assert fila["KC"]["off_pass_epa"] == pytest.approx(1.0)
    assert fila["KC"]["off_rush_epa"] == pytest.approx(0.0)
    # La defensiva de KC es la ofensiva de LV.
    assert fila["KC"]["def_epa"] == pytest.approx(fila["LV"]["off_epa"])
    assert fila["LV"]["def_epa"] == pytest.approx(0.5)

    # Yardas, capturas y balones: lo que uno hace en ataque es lo que el otro permite.
    assert fila["KC"]["off_pass_yards"] == pytest.approx(12)
    assert fila["KC"]["off_rush_yards"] == pytest.approx(3)
    assert fila["KC"]["off_yards"] == pytest.approx(15)
    assert fila["LV"]["off_pass_yards"] == pytest.approx(-7)  # la captura resta al pase
    assert fila["LV"]["off_sacks"] == pytest.approx(1)
    assert fila["LV"]["off_turnovers"] == pytest.approx(1)
    assert fila["KC"]["def_sacks"] == pytest.approx(1)
    assert fila["KC"]["def_turnovers"] == pytest.approx(1)
    assert fila["KC"]["def_rush_yards"] == pytest.approx(5)


# ---------- datos reales ----------

PARTIDOS = DATA_PROCESSED / "games.parquet"
EQUIPOS = DATA_PROCESSED / "team_games.parquet"

datos_reales = pytest.mark.skipif(
    not (PARTIDOS.exists() and EQUIPOS.exists()),
    reason="Falta correr: python -m nfl_model.ingest",
)


def _partidos_esperados(temporada: int) -> set[int]:
    if temporada <= 2001:
        return {248}  # 31 equipos, 16 partidos
    if temporada <= 2020:
        return {256}  # 32 equipos, 16 partidos
    if temporada == 2022:
        return {271, 272}  # un partido cancelado
    return {272}  # 32 equipos, 17 partidos


@datos_reales
def test_sin_partidos_duplicados():
    partidos = pl.read_parquet(PARTIDOS)
    assert partidos["game_id"].n_unique() == partidos.height


@datos_reales
def test_partidos_por_temporada_regular():
    partidos = pl.read_parquet(PARTIDOS).filter(pl.col("game_type") == "REG")
    conteo = partidos.group_by("season").agg(pl.len().alias("n")).sort("season")
    ultima = conteo["season"].max()
    for fila in conteo.to_dicts():
        if fila["season"] == ultima:
            continue  # la temporada en curso puede estar incompleta
        assert fila["n"] in _partidos_esperados(fila["season"]), fila


@datos_reales
def test_partidos_jugados_tienen_marcador_y_ganador_o_empate():
    jugados = pl.read_parquet(PARTIDOS).filter(pl.col("played"))
    assert jugados["home_score"].null_count() == 0
    assert jugados["away_score"].null_count() == 0
    sin_definir = jugados.filter(pl.col("home_win").is_null() & ~pl.col("tie"))
    assert sin_definir.height == 0


@datos_reales
def test_32_equipos_desde_2002():
    partidos = pl.read_parquet(PARTIDOS).filter(pl.col("season") >= 2002)
    equipos = set(partidos["home_team"].to_list()) | set(partidos["away_team"].to_list())
    assert len(equipos) == 32, sorted(equipos)


@datos_reales
def test_cada_partido_jugado_tiene_dos_filas_de_estadisticas():
    partidos = pl.read_parquet(PARTIDOS)
    equipos = pl.read_parquet(EQUIPOS)
    ultima = partidos["season"].max()
    primera = equipos["season"].min()

    filas = equipos.group_by("game_id").agg(pl.len().alias("n"))
    assert filas.filter(pl.col("n") != 2).height == 0

    # Temporadas terminadas: ningún partido jugado sin estadísticas.
    jugados = partidos.filter(
        pl.col("played") & (pl.col("season") < ultima) & (pl.col("season") >= primera)
    )
    faltan = jugados.join(filas, on="game_id", how="anti")

    # nflverse no tiene el jugada por jugada de 3 partidos de 1999-2000.
    # El modelo entrena desde 2006, así que no le afecta.
    antiguos = faltan.filter(pl.col("season") <= 2000)
    recientes = faltan.filter(pl.col("season") > 2000)
    assert antiguos.height <= 3, antiguos["game_id"].to_list()
    assert recientes.height == 0, recientes["game_id"].to_list()[:10]


@datos_reales
def test_equipos_de_estadisticas_coinciden_con_el_calendario():
    partidos = pl.read_parquet(PARTIDOS)
    equipos = pl.read_parquet(EQUIPOS)
    validos = pl.concat(
        [
            partidos.select("game_id", pl.col("home_team").alias("team")),
            partidos.select("game_id", pl.col("away_team").alias("team")),
        ]
    )
    huerfanos = equipos.join(validos, on=["game_id", "team"], how="anti")
    assert huerfanos.height == 0, huerfanos.select("game_id", "team").head(10).to_dicts()
