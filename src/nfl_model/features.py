"""Fase 3: variables por partido.

Regla única: cada variable usa solo información anterior al kickoff.
Todas las de "forma" se expresan como local menos visitante.

Salidas:
  data/processed/features.parquet       una fila por partido
  data/processed/features_summary.json  resumen para revisar

Uso:
  python -m nfl_model.features
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import polars as pl

from . import DATA_PROCESSED

# Peso del partido más reciente en el promedio móvil exponencial.
# 0.15 equivale a que un partido pierda la mitad de su peso en ~4 partidos.
ALFA = 0.15
# Cuánto se acerca la forma de un equipo al promedio de la liga entre temporadas.
REGRESION_FORMA = 0.4
DESCANSO_NORMAL = 7
DESCANSO_BYE = 13

METRICAS_FORMA = [
    "off_epa",
    "off_pass_epa",
    "off_rush_epa",
    "off_success",
    "def_epa",
    "def_pass_epa",
    "def_rush_epa",
    "def_success",
]

# Lista oficial de variables del modelo (la usa la Fase 4).
VARIABLES = (
    ["elo_diff", "neutral", "div_game", "rest_diff", "home_bye", "away_bye",
     "home_qb_change", "away_qb_change", "diff_qb", "diff_qb_cambio"]
    + [f"diff_{m}" for m in METRICAS_FORMA]
)

# --- Calidad del quarterback (EPA por jugada de pase, con prudencia) ---
# Un QB sin historial vale "promedio de la liga menos esto": los novatos y
# suplentes rinden, en promedio, por debajo del titular típico.
QB_PENALIZACION_NOVATO = 0.05
# Jugadas ficticias al valor anterior: con pocas jugadas reales manda la previa.
QB_JUGADAS_PREVIAS = 250.0
# Cada partido nuevo le quita 2% de peso a lo anterior del mismo QB.
QB_OLVIDO = 0.98
LIGA_OLVIDO = 0.999


class CalidadQB:
    """Lleva el valor de cada quarterback usando solo partidos ya jugados."""

    def __init__(self) -> None:
        self.epa: dict[str, float] = {}
        self.jugadas: dict[str, float] = {}
        self.liga_epa = 0.0
        self.liga_jugadas = 0.0

    def valor(self, qb: str | None) -> float:
        media = self.liga_epa / self.liga_jugadas if self.liga_jugadas > 0 else 0.0
        previa = media - QB_PENALIZACION_NOVATO
        if qb is None:
            return previa
        return (self.epa.get(qb, 0.0) + QB_JUGADAS_PREVIAS * previa) / (
            self.jugadas.get(qb, 0.0) + QB_JUGADAS_PREVIAS
        )

    def registrar(self, qb: str, jugadas: float, epa: float) -> None:
        self.epa[qb] = self.epa.get(qb, 0.0) * QB_OLVIDO + epa
        self.jugadas[qb] = self.jugadas.get(qb, 0.0) * QB_OLVIDO + jugadas
        self.liga_epa = self.liga_epa * LIGA_OLVIDO + epa
        self.liga_jugadas = self.liga_jugadas * LIGA_OLVIDO + jugadas

COLUMNAS_META = [
    "game_id", "season", "week", "game_type", "gameday",
    "home_team", "away_team", "played", "home_win",
]


def _promedio_liga(forma: dict[str, dict | None], metrica: str) -> float | None:
    valores = [f[metrica] for f in forma.values() if f is not None]
    return sum(valores) / len(valores) if valores else None


def construir_variables(
    partidos: list[dict],
    stats: dict[tuple[str, str], dict],
    elo: dict[str, dict],
    alfa: float = ALFA,
    regresion: float = REGRESION_FORMA,
    qb_stats: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Recorre los partidos en orden cronológico y arma una fila por partido.

    partidos: filas de games.parquet, ordenadas por fecha.
    stats:    (game_id, equipo) -> métricas de ese equipo en ese partido.
    elo:      game_id -> fila de elo_games.parquet.
    qb_stats: game_id -> filas de qb_games.parquet de ese partido.
    """
    forma: dict[str, dict | None] = {}  # estado de cada equipo ANTES del partido
    ultima_temporada: dict[str, int] = {}
    ultimo_qb: dict[str, str] = {}
    calidad = CalidadQB()
    # Por equipo: qué QBs han sido titulares últimamente y con cuánto peso.
    qb_reciente: dict[str, dict[str | None, float]] = {}
    qb_stats = qb_stats or {}
    salida = []

    for p in partidos:
        gid, temporada = p["game_id"], p["season"]
        local, visita = p["home_team"], p["away_team"]

        # 1. Poner al día el estado de ambos equipos (cambio de temporada).
        for equipo in (local, visita):
            if equipo not in forma:
                forma[equipo] = None
            elif ultima_temporada[equipo] != temporada and forma[equipo] is not None:
                nueva = {}
                for m in METRICAS_FORMA:
                    liga = _promedio_liga(forma, m)
                    nueva[m] = liga + (forma[equipo][m] - liga) * (1 - regresion)
                forma[equipo] = nueva
            ultima_temporada[equipo] = temporada

        # 2. Variables con lo que se sabía antes del kickoff.
        e = elo[gid]
        descanso_l = p["home_rest"] if p["home_rest"] is not None else DESCANSO_NORMAL
        descanso_v = p["away_rest"] if p["away_rest"] is not None else DESCANSO_NORMAL
        fila = {c: p[c] for c in COLUMNAS_META}
        fila.update(
            {
                "p_home_elo": e["p_home_elo"],
                "p_home_vegas": e["p_home_vegas"],
                "elo_diff": e["elo_home_pre"] - e["elo_away_pre"],
                "neutral": int(bool(p["neutral"])),
                "div_game": int(p["div_game"] or 0),
                "rest_diff": descanso_l - descanso_v,
                "home_bye": int(descanso_l >= DESCANSO_BYE),
                "away_bye": int(descanso_v >= DESCANSO_BYE),
                "home_qb_change": _cambio_qb(ultimo_qb, local, p["home_qb_id"]),
                "away_qb_change": _cambio_qb(ultimo_qb, visita, p["away_qb_id"]),
            }
        )
        f_l, f_v = forma[local], forma[visita]
        for m in METRICAS_FORMA:
            fila[f"diff_{m}"] = None if f_l is None or f_v is None else f_l[m] - f_v[m]

        # Quarterback: si aún no se anuncia titular, se asume el del último partido.
        titular = {}
        valor_qb = {}
        ajuste_qb = {}
        for equipo, anunciado in ((local, p["home_qb_id"]), (visita, p["away_qb_id"])):
            titular[equipo] = anunciado if anunciado is not None else ultimo_qb.get(equipo)
            valor_qb[equipo] = calidad.valor(titular[equipo])
            # Cuánto mejor o peor es este titular que los que el equipo venía usando,
            # todos medidos con lo que se sabe hoy. Mismo QB de siempre -> 0.
            pesos = qb_reciente.get(equipo)
            if pesos:
                referencia = sum(w * calidad.valor(q) for q, w in pesos.items()) / sum(
                    pesos.values()
                )
                ajuste_qb[equipo] = valor_qb[equipo] - referencia
            else:
                ajuste_qb[equipo] = 0.0
        fila["diff_qb"] = valor_qb[local] - valor_qb[visita]
        fila["diff_qb_cambio"] = ajuste_qb[local] - ajuste_qb[visita]
        salida.append(fila)

        # 3. Solo ahora se incorpora lo que pasó en este partido.
        if not p["played"]:
            continue
        for equipo in (local, visita):
            pesos = qb_reciente.get(equipo)
            if not pesos:
                qb_reciente[equipo] = {titular[equipo]: 1.0}
            else:
                nuevos = {q: w * (1 - alfa) for q, w in pesos.items() if w > 0.001}
                nuevos[titular[equipo]] = nuevos.get(titular[equipo], 0.0) + alfa
                qb_reciente[equipo] = nuevos
        for q in qb_stats.get(gid, []):
            calidad.registrar(q["qb_id"], q["dropbacks"], q["qb_epa_sum"])
        for equipo, qb in ((local, p["home_qb_id"]), (visita, p["away_qb_id"])):
            if qb is not None:
                ultimo_qb[equipo] = qb
            observado = stats.get((gid, equipo))
            if observado is None or any(observado[m] is None for m in METRICAS_FORMA):
                continue
            previo = forma[equipo]
            if previo is None:
                forma[equipo] = {m: observado[m] for m in METRICAS_FORMA}
            else:
                forma[equipo] = {
                    m: (1 - alfa) * previo[m] + alfa * observado[m] for m in METRICAS_FORMA
                }

    return salida


def _cambio_qb(ultimo_qb: dict[str, str], equipo: str, qb: str | None) -> int:
    """1 si el titular anunciado es distinto al del partido anterior del equipo."""
    anterior = ultimo_qb.get(equipo)
    if qb is None or anterior is None:
        return 0
    return int(qb != anterior)


# ---------- resumen ----------


def _correlacion(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def construir_resumen(filas: list[dict]) -> dict:
    # Solo 2006-2020: el resumen no debe mirar validación ni temporadas guardadas.
    base = [f for f in filas if 2006 <= f["season"] <= 2020 and f["home_win"] is not None]
    desde_2006 = [f for f in filas if f["season"] >= 2006 and f["played"]]
    por_variable = {}
    for v in VARIABLES:
        pares = [(f[v], f["home_win"]) for f in base if f[v] is not None]
        xs = [float(x) for x, _ in pares]
        ys = [float(y) for _, y in pares]
        media = sum(xs) / len(xs) if xs else None
        desv = math.sqrt(sum((x - media) ** 2 for x in xs) / len(xs)) if xs else None
        por_variable[v] = {
            "nulos_jugados_desde_2006": sum(1 for f in desde_2006 if f[v] is None),
            "media_2006_2020": media,
            "desv_2006_2020": desv,
            "correlacion_con_victoria_local_2006_2020": _correlacion(xs, ys),
        }
    pendientes = [f for f in filas if not f["played"]]
    return {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "alfa": ALFA,
        "regresion_forma": REGRESION_FORMA,
        "filas": len(filas),
        "filas_jugadas_desde_2006": len(desde_2006),
        "partidos_base_2006_2020": len(base),
        "pendientes": len(pendientes),
        "pendientes_con_nulos": sum(
            1 for f in pendientes if any(f[v] is None for v in VARIABLES)
        ),
        "variables": por_variable,
    }


def main() -> None:
    partidos = (
        pl.read_parquet(DATA_PROCESSED / "games.parquet")
        .sort(["gameday", "game_id"])
        .to_dicts()
    )
    stats = {
        (f["game_id"], f["team"]): f
        for f in pl.read_parquet(DATA_PROCESSED / "team_games.parquet").to_dicts()
    }
    elo = {
        f["game_id"]: f
        for f in pl.read_parquet(DATA_PROCESSED / "elo_games.parquet").to_dicts()
    }

    qb_stats: dict[str, list[dict]] = {}
    for f in pl.read_parquet(DATA_PROCESSED / "qb_games.parquet").to_dicts():
        qb_stats.setdefault(f["game_id"], []).append(f)

    filas = construir_variables(partidos, stats, elo, qb_stats=qb_stats)
    pl.DataFrame(filas, infer_schema_length=None).write_parquet(
        DATA_PROCESSED / "features.parquet"
    )

    ruta = DATA_PROCESSED / "features_summary.json"
    ruta.write_text(
        json.dumps(construir_resumen(filas), indent=2, ensure_ascii=False), "utf-8"
    )
    print(f"{len(filas):,} partidos, {len(VARIABLES)} variables. Resumen en {ruta}")


if __name__ == "__main__":
    main()
