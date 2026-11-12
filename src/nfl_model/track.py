"""Fase 6: califica los picks ya jugados.

Lee el registro (sin modificarlo), lo cruza con los resultados y escribe el
historial acumulado por rango de confianza.

Salidas:
  picks/resultados.csv   cada pick con su resultado (se regenera completo)
  reports/historial.md   aciertos por semana y por rango de confianza

Uso:
  python -m nfl_model.track
  python -m nfl_model.track --sin-actualizar
"""

from __future__ import annotations

import argparse
import csv
import math

import polars as pl

from . import DATA_PROCESSED, PICKS_DIR, REPORTS_DIR
from .predict import AVISO, leer_registro

RANGOS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def calificar(registro: list[dict], resultados: dict[str, dict]) -> list[dict]:
    """Agrega a cada pick su estado: acierto, fallo, empate o pendiente.

    resultados: game_id -> {"played": bool, "home_win": 0/1/None, "tie": bool}
    """
    salida = []
    for fila in registro:
        r = resultados.get(fila["game_id"])
        if r is None or not r["played"]:
            estado = "pendiente"
        elif r["tie"] or r["home_win"] is None:
            estado = "empate"
        else:
            gano_local = r["home_win"] == 1
            estado = "acierto" if (fila["pick"] == fila["local"]) == gano_local else "fallo"
        salida.append({**fila, "estado": estado})
    return salida


def _decididos(filas: list[dict]) -> list[dict]:
    return [f for f in filas if f["estado"] in ("acierto", "fallo")]


def _marca(filas: list[dict]) -> str:
    decididos = _decididos(filas)
    if not decididos:
        return "—"
    aciertos = sum(1 for f in decididos if f["estado"] == "acierto")
    return f"{aciertos} de {len(decididos)} ({aciertos / len(decididos) * 100:.1f}%)"


def por_rango(filas: list[dict]) -> list[dict]:
    tabla = []
    for bajo, alto in RANGOS:
        dentro = [f for f in _decididos(filas) if bajo <= float(f["p_pick"]) < alto]
        aciertos = sum(1 for f in dentro if f["estado"] == "acierto")
        tabla.append(
            {
                "rango": f"{bajo * 100:.0f}–{min(alto, 1) * 100:.0f}%",
                "partidos": len(dentro),
                "aciertos": aciertos,
                "confianza_media": (
                    sum(float(f["p_pick"]) for f in dentro) / len(dentro) if dentro else None
                ),
            }
        )
    return tabla


def comparar_con_vegas(filas: list[dict]) -> dict:
    """Modelo y Las Vegas sobre los mismos partidos decididos con moneyline."""
    comunes = [f for f in _decididos(filas) if f["p_home_vegas"] not in ("", None)]
    aciertos_modelo = aciertos_vegas = distintos = modelo_en_distintos = 0
    ll_modelo = ll_vegas = 0.0
    for f in comunes:
        acierto = f["estado"] == "acierto"
        pick_local = f["pick"] == f["local"]
        gano_local = pick_local == acierto
        p_modelo = min(max(float(f["p_home_modelo"]), 1e-6), 1 - 1e-6)
        p_vegas = min(max(float(f["p_home_vegas"]), 1e-6), 1 - 1e-6)
        vegas_local = p_vegas >= 0.5
        aciertos_modelo += acierto
        aciertos_vegas += vegas_local == gano_local
        if vegas_local != pick_local:
            distintos += 1
            modelo_en_distintos += acierto
        ll_modelo -= math.log(p_modelo if gano_local else 1 - p_modelo)
        ll_vegas -= math.log(p_vegas if gano_local else 1 - p_vegas)
    n = len(comunes)
    return {
        "partidos": n,
        "aciertos_modelo": aciertos_modelo,
        "aciertos_vegas": aciertos_vegas,
        "log_loss_modelo": ll_modelo / n if n else None,
        "log_loss_vegas": ll_vegas / n if n else None,
        "eligieron_distinto": distintos,
        "modelo_acerto_en_distintos": modelo_en_distintos,
    }


def escribir_historial(filas: list[dict]) -> str:
    decididos = _decididos(filas)
    pendientes = sum(1 for f in filas if f["estado"] == "pendiente")
    empates = sum(1 for f in filas if f["estado"] == "empate")
    lineas = [
        "# Historial de picks",
        "",
        AVISO,
        "",
        f"- Picks registrados: {len(filas)}",
        f"- Calificados: {len(decididos)}. Pendientes: {pendientes}. Empates (no cuentan): {empates}.",
        f"- **Marca total: {_marca(filas)}**",
        "",
        "## Por rango de confianza",
        "",
        "| Confianza | Partidos | Aciertos | Acierto real | Confianza media |",
        "|---|---|---|---|---|",
    ]
    for r in por_rango(filas):
        real = f"{r['aciertos'] / r['partidos'] * 100:.1f}%" if r["partidos"] else "—"
        media = f"{r['confianza_media'] * 100:.1f}%" if r["partidos"] else "—"
        lineas.append(f"| {r['rango']} | {r['partidos']} | {r['aciertos']} | {real} | {media} |")

    lineas += ["", "## Por semana", "", "| Temporada | Semana | Marca | 70% o más |", "|---|---|---|---|"]
    semanas = sorted({(int(f["temporada"]), int(f["semana"])) for f in filas})
    for temporada, semana in semanas:
        de_semana = [
            f for f in filas if int(f["temporada"]) == temporada and int(f["semana"]) == semana
        ]
        fuertes = [f for f in de_semana if float(f["p_pick"]) >= 0.70]
        lineas.append(f"| {temporada} | {semana} | {_marca(de_semana)} | {_marca(fuertes)} |")

    v = comparar_con_vegas(filas)
    lineas += ["", "## Contra Las Vegas (mismos partidos)", ""]
    if v["partidos"]:
        lineas += [
            f"- Partidos comparables: {v['partidos']}",
            f"- Aciertos del modelo: {v['aciertos_modelo']}. Aciertos de Las Vegas: {v['aciertos_vegas']}.",
            f"- Log loss del modelo: {v['log_loss_modelo']:.4f}. De Las Vegas: {v['log_loss_vegas']:.4f}. (Menor es mejor.)",
            f"- Eligieron distinto ganador en {v['eligieron_distinto']}; el modelo acertó {v['modelo_acerto_en_distintos']}.",
        ]
    else:
        lineas.append("- Todavía no hay partidos calificados con línea de Las Vegas.")

    if len(decididos) < 50:
        lineas += [
            "",
            f"Con {len(decididos)} partidos calificados la muestra es chica: "
            "una racha buena o mala todavía no dice nada sobre el modelo.",
        ]
    return "\n".join(lineas) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Califica los picks registrados.")
    parser.add_argument("--sin-actualizar", action="store_true")
    args = parser.parse_args()

    registro = leer_registro()
    if not registro:
        print("El registro está vacío. Primero corre: python -m nfl_model.predict")
        return

    if not args.sin_actualizar:
        from . import ingest

        ingest.ejecutar()

    resultados = {
        p["game_id"]: p
        for p in pl.read_parquet(DATA_PROCESSED / "games.parquet")
        .select("game_id", "played", "home_win", "tie")
        .to_dicts()
    }
    filas = calificar(registro, resultados)

    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    with (PICKS_DIR / "resultados.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = REPORTS_DIR / "historial.md"
    ruta.write_text(escribir_historial(filas), "utf-8")
    print(f"Marca total: {_marca(filas)}. Historial en {ruta}")


if __name__ == "__main__":
    main()
