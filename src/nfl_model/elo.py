"""Fase 2: ratings Elo y comparación con Las Vegas.

Salidas:
  data/processed/elo_games.parquet  Elo de cada equipo ANTES de cada partido
  data/processed/elo_params.json    K y ventaja de local elegidos
  reports/elo_reporte.md            métricas, ranking actual y picks de la semana

Uso:
  python -m nfl_model.elo

Disciplina: los parámetros se ajustan con 2006-2020 y se reportan también en
2021-2023. Las temporadas 2024-2025 no se miran aquí; quedan para la Fase 5.
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import polars as pl

from . import DATA_PROCESSED, REPORTS_DIR

ELO_INICIAL = 1500.0
REGRESION = 1 / 3  # cuánto vuelve cada equipo hacia 1500 entre temporadas

AJUSTE = (2006, 2020)
VALIDACION = (2021, 2023)
VEGAS_DESDE = 2010  # antes hay temporadas sin moneyline

REJILLA_K = range(8, 42, 2)
REJILLA_VENTAJA = range(0, 90, 10)


# ---------- Elo ----------


def prob_local(elo_local: float, elo_visita: float, ventaja: float) -> float:
    """Probabilidad de que gane el local."""
    return 1 / (1 + 10 ** (-(elo_local - elo_visita + ventaja) / 400))


def calcular_elo(
    filas: list[dict],
    k: float,
    ventaja_local: float,
    regresion: float = REGRESION,
) -> tuple[list[dict], dict[str, float]]:
    """Recorre los partidos en orden cronológico.

    `filas` necesita: game_id, season, home_team, away_team, result
    (local menos visita), neutral, played.

    Devuelve (una fila por partido con los Elo previos y la probabilidad,
    ratings al final). La probabilidad de un partido solo usa partidos anteriores.
    """
    elo: dict[str, float] = {}
    ultima_temporada: dict[str, int] = {}
    salida = []

    for f in filas:
        local, visita, temporada = f["home_team"], f["away_team"], f["season"]

        for equipo in (local, visita):
            if equipo not in elo:
                elo[equipo] = ELO_INICIAL
            elif ultima_temporada[equipo] != temporada:
                elo[equipo] = ELO_INICIAL + (elo[equipo] - ELO_INICIAL) * (1 - regresion)
            ultima_temporada[equipo] = temporada

        ventaja = 0.0 if f["neutral"] else ventaja_local
        elo_l, elo_v = elo[local], elo[visita]
        p = prob_local(elo_l, elo_v, ventaja)
        salida.append(
            {
                "game_id": f["game_id"],
                "elo_home_pre": elo_l,
                "elo_away_pre": elo_v,
                "p_home_elo": p,
            }
        )

        if not f["played"]:
            continue

        margen = f["result"]
        resultado = 1.0 if margen > 0 else 0.0 if margen < 0 else 0.5
        # Ventaja previa del que ganó: los favoritos que ganan por mucho suben menos.
        dif_ganador = (elo_l + ventaja - elo_v) if margen >= 0 else (elo_v - elo_l - ventaja)
        multiplicador = math.log(max(abs(margen), 1) + 1) * 2.2 / (dif_ganador * 0.001 + 2.2)
        cambio = k * multiplicador * (resultado - p)
        elo[local] = elo_l + cambio
        elo[visita] = elo_v - cambio

    return salida, elo


# ---------- Las Vegas ----------


def _prob_cruda(moneyline: float) -> float:
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    return 100 / (moneyline + 100)


def prob_vegas(ml_local: float | None, ml_visita: float | None) -> float | None:
    """Probabilidad implícita del local, sin la comisión de la casa."""
    if ml_local is None or ml_visita is None or ml_local == 0 or ml_visita == 0:
        return None
    local, visita = _prob_cruda(ml_local), _prob_cruda(ml_visita)
    return local / (local + visita)


# ---------- métricas ----------


def metricas(pares: list[tuple[float, int]]) -> dict:
    """pares = (probabilidad del local, 1 si ganó el local)."""
    n = len(pares)
    if n == 0:
        return {"partidos": 0, "acierto": None, "log_loss": None, "brier": None}
    aciertos = sum(1 for p, y in pares if (p >= 0.5) == (y == 1))
    log_loss = 0.0
    brier = 0.0
    for p, y in pares:
        p = min(max(p, 1e-6), 1 - 1e-6)
        log_loss -= math.log(p) if y == 1 else math.log(1 - p)
        brier += (p - y) ** 2
    return {
        "partidos": n,
        "acierto": aciertos / n,
        "log_loss": log_loss / n,
        "brier": brier / n,
    }


def _con_ganador(filas: list[dict], periodo: tuple[int, int]) -> list[dict]:
    return [
        f
        for f in filas
        if f["home_win"] is not None and periodo[0] <= f["season"] <= periodo[1]
    ]


def ajustar_parametros(filas: list[dict]) -> dict:
    """Elige K y ventaja de local con el menor log loss en el periodo de ajuste."""
    mejor = None
    for k in REJILLA_K:
        for ventaja in REJILLA_VENTAJA:
            salida, _ = calcular_elo(filas, k, ventaja)
            prob = {s["game_id"]: s["p_home_elo"] for s in salida}
            pares = [(prob[f["game_id"]], f["home_win"]) for f in _con_ganador(filas, AJUSTE)]
            perdida = metricas(pares)["log_loss"]
            if mejor is None or perdida < mejor["log_loss"]:
                mejor = {"k": k, "ventaja_local": ventaja, "log_loss": perdida}
    return mejor


# ---------- reporte ----------


def _fmt(valor, decimales=3, pct=False) -> str:
    if valor is None:
        return "—"
    return f"{valor * 100:.1f}%" if pct else f"{valor:.{decimales}f}"


def _tabla_metricas(filas: list[dict], periodo: tuple[int, int], p_base: float) -> list[str]:
    """Compara los tres métodos sobre los MISMOS partidos (los que tienen moneyline)."""
    partidos = [f for f in _con_ganador(filas, periodo) if f["p_home_vegas"] is not None]
    metodos = {
        "Siempre el local": [(p_base, f["home_win"]) for f in partidos],
        "Elo": [(f["p_home_elo"], f["home_win"]) for f in partidos],
        "Las Vegas": [(f["p_home_vegas"], f["home_win"]) for f in partidos],
    }
    lineas = [
        f"### {periodo[0]}–{periodo[1]} ({len(partidos)} partidos con moneyline)",
        "",
        "| Método | Acierto | Log loss | Brier |",
        "|---|---|---|---|",
    ]
    for nombre, pares in metodos.items():
        m = metricas(pares)
        lineas.append(
            f"| {nombre} | {_fmt(m['acierto'], pct=True)} | {_fmt(m['log_loss'], 4)} | {_fmt(m['brier'], 4)} |"
        )
    return lineas + [""]


def escribir_reporte(filas: list[dict], ratings: dict[str, float], params: dict) -> str:
    ajuste = _con_ganador(filas, AJUSTE)
    p_base = sum(f["home_win"] for f in ajuste) / len(ajuste)

    lineas = [
        "# Elo: reporte",
        "",
        f"Generado: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Parámetros elegidos",
        "",
        f"- K = {params['k']}",
        f"- Ventaja de local = {params['ventaja_local']} puntos Elo",
        f"- Regresión entre temporadas = {REGRESION:.2f}",
        f"- Log loss en {AJUSTE[0]}–{AJUSTE[1]} (todos los partidos): {params['log_loss']:.4f}",
        f"- Frecuencia de victoria del local en ese periodo: {p_base * 100:.1f}%",
        "",
        "## Comparación",
        "",
        "Las temporadas 2024–2025 no se evalúan aquí a propósito.",
        "",
    ]
    lineas += _tabla_metricas(filas, (max(AJUSTE[0], VEGAS_DESDE), AJUSTE[1]), p_base)
    lineas += _tabla_metricas(filas, VALIDACION, p_base)

    lineas += ["## Ranking Elo actual", "", "| # | Equipo | Elo |", "|---|---|---|"]
    orden = sorted(ratings.items(), key=lambda x: -x[1])
    for i, (equipo, valor) in enumerate(orden, 1):
        lineas.append(f"| {i} | {equipo} | {valor:.0f} |")
    lineas.append("")

    pendientes = [f for f in filas if not f["played"]]
    if pendientes:
        temporada = min(f["season"] for f in pendientes)
        pendientes = [f for f in pendientes if f["season"] == temporada]
        semana = min(f["week"] for f in pendientes)
        proximos = [f for f in pendientes if f["week"] == semana]
        proximos.sort(key=lambda f: -max(f["p_home_elo"], 1 - f["p_home_elo"]))
        lineas += [
            f"## Picks por Elo: temporada {temporada}, semana {semana}",
            "",
            "| Visita | Local | Pick | Elo | Las Vegas (mismo equipo) |",
            "|---|---|---|---|---|",
        ]
        for f in proximos:
            gana_local = f["p_home_elo"] >= 0.5
            pick = f["home_team"] if gana_local else f["away_team"]
            p_elo = f["p_home_elo"] if gana_local else 1 - f["p_home_elo"]
            p_vegas = f["p_home_vegas"]
            if p_vegas is not None and not gana_local:
                p_vegas = 1 - p_vegas
            lineas.append(
                f"| {f['away_team']} | {f['home_team']} | {pick} | "
                f"{_fmt(p_elo, pct=True)} | {_fmt(p_vegas, pct=True)} |"
            )
        lineas.append("")

    return "\n".join(lineas)


def main() -> None:
    partidos = pl.read_parquet(DATA_PROCESSED / "games.parquet").sort(["gameday", "game_id"])
    filas = partidos.to_dicts()

    print("Ajustando K y ventaja de local...")
    params = ajustar_parametros(filas)
    print(f"  K={params['k']}, ventaja={params['ventaja_local']}, log loss={params['log_loss']:.4f}")

    salida, ratings = calcular_elo(filas, params["k"], params["ventaja_local"])
    por_partido = {s["game_id"]: s for s in salida}
    for f in filas:
        f.update(por_partido[f["game_id"]])
        f["p_home_vegas"] = prob_vegas(f["home_moneyline"], f["away_moneyline"])

    tabla = pl.DataFrame(
        [
            {
                "game_id": f["game_id"],
                "elo_home_pre": f["elo_home_pre"],
                "elo_away_pre": f["elo_away_pre"],
                "p_home_elo": f["p_home_elo"],
                "p_home_vegas": f["p_home_vegas"],
            }
            for f in filas
        ],
        schema={
            "game_id": pl.String,
            "elo_home_pre": pl.Float64,
            "elo_away_pre": pl.Float64,
            "p_home_elo": pl.Float64,
            "p_home_vegas": pl.Float64,
        },
    )
    tabla.write_parquet(DATA_PROCESSED / "elo_games.parquet")

    params_ruta = DATA_PROCESSED / "elo_params.json"
    params_ruta.write_text(
        json.dumps({**params, "regresion": REGRESION, "ajuste": AJUSTE}, indent=2), "utf-8"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reporte = REPORTS_DIR / "elo_reporte.md"
    reporte.write_text(escribir_reporte(filas, ratings, params), "utf-8")
    print(f"Listo. Reporte en {reporte}")


if __name__ == "__main__":
    main()
