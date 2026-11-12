"""Fase 1: descarga datos de nflverse y deja dos tablas limpias.

Salidas:
  data/raw/schedules.parquet            calendario completo tal como llega
  data/raw/pbp_slim_<temporada>.parquet jugadas de pase/carrera, pocas columnas
  data/processed/games.parquet          una fila por partido
  data/processed/team_games.parquet     una fila por equipo y partido (EPA, éxito)
  data/processed/qb_games.parquet       una fila por quarterback y partido
  data/processed/ingest_summary.json    resumen para revisar sin abrir los parquet

Uso:
  python -m nfl_model.ingest
  python -m nfl_model.ingest --desde 2015
  python -m nfl_model.ingest --refrescar     # vuelve a bajar todas las temporadas
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import nflreadpy as nfl
import polars as pl

from . import DATA_PROCESSED, DATA_RAW

PRIMERA_TEMPORADA = 1999

# Franquicias que se mudaron o cuya abreviatura cambió: todo a la actual.
ABREVIATURAS = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAC": "JAX",
    "WSH": "WAS",
}

COLUMNAS_PARTIDO = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "location",
    "div_game",
    "home_rest",
    "away_rest",
    "home_moneyline",
    "away_moneyline",
    "spread_line",
    "home_qb_id",
    "away_qb_id",
    "home_qb_name",
    "away_qb_name",
    "roof",
]

COLUMNAS_JUGADA = [
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "pass",
    "rush",
    "epa",
    "success",
    # Para la calidad del QB. passer_id incluye capturas y scrambles;
    # passer_player_id es el respaldo cuando viene vacío.
    "passer_id",
    "passer_player_id",
    "qb_epa",
]

# Columnas que deben traer los archivos pbp_slim ya guardados.
COLUMNAS_SLIM = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "pass", "rush", "epa", "success", "passer_id", "qb_epa",
]

METRICAS = ["plays", "epa", "pass_epa", "rush_epa", "success"]


def normalizar_equipo(columna: str) -> pl.Expr:
    """Expresión que lleva una columna de equipo a la abreviatura actual."""
    return pl.col(columna).replace(ABREVIATURAS).alias(columna)


def limpiar_partidos(calendario: pl.DataFrame) -> pl.DataFrame:
    """Calendario crudo -> una fila por partido con las columnas del modelo."""
    faltantes = [c for c in COLUMNAS_PARTIDO if c not in calendario.columns]
    if faltantes:
        raise ValueError(f"El calendario no trae estas columnas: {faltantes}")

    return (
        calendario.select(COLUMNAS_PARTIDO)
        .filter(pl.col("season") >= PRIMERA_TEMPORADA)
        .with_columns(
            normalizar_equipo("home_team"),
            normalizar_equipo("away_team"),
            pl.col("gameday").cast(pl.String).str.to_date(strict=False),
        )
        .with_columns(
            (pl.col("location") == "Neutral").fill_null(False).alias("neutral"),
            (
                pl.col("home_score").is_not_null()
                & pl.col("away_score").is_not_null()
            ).alias("played"),
        )
        .with_columns(
            # Nulo si el partido no se ha jugado o terminó en empate.
            pl.when(pl.col("played") & (pl.col("result") != 0))
            .then((pl.col("result") > 0).cast(pl.Int8))
            .otherwise(None)
            .alias("home_win"),
            (pl.col("played") & (pl.col("result") == 0)).alias("tie"),
        )
        .unique(subset="game_id", keep="first")
        .sort(["gameday", "game_id"])
    )


def jugadas_utiles(pbp: pl.DataFrame) -> pl.DataFrame:
    """Deja solo pases y carreras con EPA, y las columnas que se usan."""
    return (
        pbp.select(COLUMNAS_JUGADA)
        .filter(
            ((pl.col("pass") == 1) | (pl.col("rush") == 1))
            & pl.col("epa").is_not_null()
            & pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
        )
        .with_columns(
            normalizar_equipo("posteam"),
            normalizar_equipo("defteam"),
            pl.col("pass").cast(pl.Int8),
            pl.col("rush").cast(pl.Int8),
            pl.col("epa").cast(pl.Float64),
            pl.col("success").cast(pl.Float64),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.coalesce(
                pl.col("passer_id").cast(pl.String),
                pl.col("passer_player_id").cast(pl.String),
            ).alias("passer_id"),
            pl.coalesce(
                pl.col("qb_epa").cast(pl.Float64), pl.col("epa").cast(pl.Float64)
            ).alias("qb_epa"),
        )
        .select(COLUMNAS_SLIM)
    )


def resumir_qbs(jugadas: pl.DataFrame) -> pl.DataFrame:
    """Jugadas -> una fila por quarterback y partido (solo jugadas de pase)."""
    return (
        jugadas.filter((pl.col("pass") == 1) & pl.col("passer_id").is_not_null())
        .group_by(["game_id", "season", "week", "posteam", "passer_id"])
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("qb_epa").sum().alias("qb_epa_sum"),
        )
        .rename({"posteam": "team", "passer_id": "qb_id"})
        .sort(["season", "week", "game_id", "team", "qb_id"])
    )


def resumir_por_equipo(jugadas: pl.DataFrame) -> pl.DataFrame:
    """Jugadas -> una fila por equipo y partido, con ofensiva y defensiva."""
    ofensiva = jugadas.group_by(
        ["game_id", "season", "week", "posteam", "defteam"]
    ).agg(
        pl.len().alias("off_plays"),
        pl.col("epa").mean().alias("off_epa"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("off_pass_epa"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("off_rush_epa"),
        pl.col("success").mean().alias("off_success"),
    )

    # La defensiva de un equipo es la ofensiva de su rival en ese partido.
    defensiva = ofensiva.select(
        "game_id",
        pl.col("defteam").alias("team"),
        *[pl.col(f"off_{m}").alias(f"def_{m}") for m in METRICAS],
    )

    return (
        ofensiva.rename({"posteam": "team", "defteam": "opponent"})
        .join(defensiva, on=["game_id", "team"], how="left")
        .sort(["season", "week", "game_id", "team"])
    )


def descargar_jugadas(desde: int, refrescar: bool) -> pl.DataFrame:
    """Baja el jugada por jugada temporada por temporada (cabe en memoria).

    Las temporadas ya terminadas se bajan una sola vez; la actual, siempre.
    """
    actual = nfl.get_current_season()
    partes = []
    for temporada in range(desde, actual + 1):
        archivo = DATA_RAW / f"pbp_slim_{temporada}.parquet"
        if archivo.exists() and temporada < actual and not refrescar:
            guardado = pl.read_parquet(archivo)
            if set(COLUMNAS_SLIM) <= set(guardado.columns):
                partes.append(guardado.select(COLUMNAS_SLIM))
                print(f"  {temporada}: ya estaba en disco")
                continue
            print(f"  {temporada}: el archivo guardado no trae columnas nuevas, se vuelve a bajar")
        jugadas = jugadas_utiles(nfl.load_pbp(temporada))
        jugadas.write_parquet(archivo)
        partes.append(jugadas)
        print(f"  {temporada}: {jugadas.height:,} jugadas descargadas")
    return pl.concat(partes, how="vertical_relaxed")


def _resumen_qbs(partidos: pl.DataFrame, qbs: pl.DataFrame) -> dict:
    """Comprueba que los titulares del calendario aparecen en las jugadas."""
    vistos = set(zip(qbs["game_id"].to_list(), qbs["qb_id"].to_list()))
    jugados = partidos.filter(pl.col("played") & (pl.col("season") >= 2006)).to_dicts()
    total = encontrados = 0
    ejemplos = []
    for p in jugados:
        for qb in (p["home_qb_id"], p["away_qb_id"]):
            if qb is None:
                continue
            total += 1
            if (p["game_id"], qb) in vistos:
                encontrados += 1
            elif len(ejemplos) < 5:
                ejemplos.append({"game_id": p["game_id"], "qb_id": qb})
    return {
        "filas_qb_partido": qbs.height,
        "qbs_distintos": qbs["qb_id"].n_unique(),
        "titulares_desde_2006": total,
        "titulares_encontrados_en_jugadas": encontrados,
        "ejemplos_no_encontrados": ejemplos,
        "ejemplo_id_en_jugadas": qbs["qb_id"].head(3).to_list(),
    }


def construir_resumen(partidos: pl.DataFrame, equipos: pl.DataFrame) -> dict:
    regular = partidos.filter(pl.col("game_type") == "REG")
    por_temporada = (
        regular.group_by("season")
        .agg(
            pl.len().alias("programados"),
            pl.col("played").sum().alias("jugados"),
            pl.col("tie").sum().alias("empates"),
            pl.col("home_moneyline").is_null().sum().alias("sin_moneyline"),
            pl.col("home_qb_id").is_null().sum().alias("sin_qb"),
        )
        .sort("season")
    )
    con_stats = equipos.group_by("season").agg(
        pl.col("game_id").n_unique().alias("partidos_con_jugadas")
    )
    por_temporada = por_temporada.join(con_stats, on="season", how="left")

    jugados = partidos.filter(pl.col("played"))
    return {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "partidos_total": partidos.height,
        "partidos_jugados": jugados.height,
        "game_id_duplicados": partidos.height - partidos["game_id"].n_unique(),
        "equipos_distintos": sorted(
            set(partidos["home_team"].to_list())
            | set(partidos["away_team"].to_list())
        ),
        "sedes_neutrales": int(partidos["neutral"].sum()),
        "ultimo_partido_jugado": str(jugados["gameday"].max()),
        "filas_equipo_partido": equipos.height,
        "nulos_en_team_games": {
            c: int(equipos[c].null_count()) for c in equipos.columns
        },
        "temporada_regular": por_temporada.to_dicts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y limpia datos NFL.")
    parser.add_argument("--desde", type=int, default=PRIMERA_TEMPORADA)
    parser.add_argument("--refrescar", action="store_true")
    args = parser.parse_args()
    if args.desde < PRIMERA_TEMPORADA:
        parser.error(f"--desde no puede ser menor a {PRIMERA_TEMPORADA}")
    ejecutar(args.desde, args.refrescar)


def ejecutar(desde: int = PRIMERA_TEMPORADA, refrescar: bool = False) -> None:
    """Descarga y limpia. La usan también predict.py y track.py."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Calendario y resultados...")
    calendario = nfl.load_schedules()
    calendario.write_parquet(DATA_RAW / "schedules.parquet")
    partidos = limpiar_partidos(calendario)
    partidos.write_parquet(DATA_PROCESSED / "games.parquet")
    print(f"  {partidos.height:,} partidos")

    print("Jugada por jugada...")
    jugadas = descargar_jugadas(desde, refrescar)
    equipos = resumir_por_equipo(jugadas)
    equipos.write_parquet(DATA_PROCESSED / "team_games.parquet")
    print(f"  {equipos.height:,} filas equipo-partido")
    qbs = resumir_qbs(jugadas)
    qbs.write_parquet(DATA_PROCESSED / "qb_games.parquet")
    print(f"  {qbs.height:,} filas quarterback-partido")

    resumen = construir_resumen(partidos, equipos)
    resumen["quarterbacks"] = _resumen_qbs(partidos, qbs)
    ruta = DATA_PROCESSED / "ingest_summary.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), "utf-8")
    print(f"Listo. Resumen en {ruta}")


if __name__ == "__main__":
    main()
