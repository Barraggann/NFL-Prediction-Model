"""Fase 5: backtest semana por semana (walk-forward).

Para cada semana de 2021-2025 se entrena con todos los partidos anteriores
y se predice esa semana. Así se imita el uso real.

Cómo leer el resultado:
  2021-2023  optimista: con esos años se eligieron la regularización y el modelo.
  2024-2025  la prueba honesta: nada se ajustó mirándolos. SOLO SE ABRE UNA VEZ.

Salidas:
  data/processed/backtest_predicciones.parquet
  reports/backtest_reporte.md

Uso:
  python -m nfl_model.backtest
  python -m nfl_model.backtest --repetir   # solo si de verdad hace falta
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import polars as pl

from . import DATA_PROCESSED, REPORTS_DIR
from .elo import metricas
from .train import ajustar, con_ganador, cargar_filas, predecir, tabla_confianza

PRIMERA_ENTRENAMIENTO = 2006
OPTIMISTA = (2021, 2023)
HONESTO = (2024, 2025)


def recorrer(
    filas: list[dict],
    config: dict,
    desde: int,
    hasta: int,
    ajustar_fn=ajustar,
    predecir_fn=predecir,
) -> list[dict]:
    """Predice cada semana entrenando solo con partidos de fechas anteriores.

    Devuelve las filas evaluables (jugadas, sin empate) con `p_home_modelo`.
    """
    utiles = con_ganador(filas, PRIMERA_ENTRENAMIENTO, 9999)
    semanas: dict[tuple[int, int], list[dict]] = {}
    for f in utiles:
        if desde <= f["season"] <= hasta:
            semanas.setdefault((f["season"], f["week"]), []).append(f)

    salida = []
    for clave in sorted(semanas, key=lambda k: min(f["gameday"] for f in semanas[k])):
        partidos = semanas[clave]
        corte = min(f["gameday"] for f in partidos)
        entrenamiento = [f for f in utiles if f["gameday"] < corte]
        modelo = ajustar_fn(entrenamiento, config)
        for f, p in zip(partidos, predecir_fn(modelo, partidos, config)):
            salida.append({**f, "p_home_modelo": p})
    return salida


# ---------- reporte ----------


def _fmt(valor, decimales=4, pct=False) -> str:
    if valor is None:
        return "—"
    return f"{valor * 100:.1f}%" if pct else f"{valor:.{decimales}f}"


def _en(filas: list[dict], periodo: tuple[int, int]) -> list[dict]:
    return [f for f in filas if periodo[0] <= f["season"] <= periodo[1]]


def _comparacion(filas: list[dict], p_base: float) -> dict[str, dict]:
    """Los cuatro métodos sobre los mismos partidos (los que tienen moneyline)."""
    con_linea = [f for f in filas if f["p_home_vegas"] is not None]
    return {
        "Siempre el local": metricas([(p_base, f["home_win"]) for f in con_linea]),
        "Elo": metricas([(f["p_home_elo"], f["home_win"]) for f in con_linea]),
        "Modelo": metricas([(f["p_home_modelo"], f["home_win"]) for f in con_linea]),
        "Las Vegas": metricas([(f["p_home_vegas"], f["home_win"]) for f in con_linea]),
    }


def _tabla_comparacion(titulo: str, filas: list[dict], p_base: float) -> list[str]:
    comp = _comparacion(filas, p_base)
    n = comp["Modelo"]["partidos"]
    lineas = [f"### {titulo} ({n} partidos)", "", "| Método | Acierto | Log loss | Brier |",
              "|---|---|---|---|"]
    for nombre, m in comp.items():
        lineas.append(
            f"| {nombre} | {_fmt(m['acierto'], pct=True)} | {_fmt(m['log_loss'])} | {_fmt(m['brier'])} |"
        )
    return lineas + [""]


def _tabla_confianza(titulo: str, filas: list[dict]) -> list[str]:
    lineas = [f"### {titulo}", "", "| Confianza | Partidos | Confianza media | Acierto real |",
              "|---|---|---|---|"]
    for r in tabla_confianza(filas, [f["p_home_modelo"] for f in filas]):
        lineas.append(
            f"| {r['rango']} | {r['partidos']} | {_fmt(r['confianza_media'], pct=True)} | {_fmt(r['acierto'], pct=True)} |"
        )
    return lineas + [""]


def _desacuerdos(filas: list[dict]) -> list[str]:
    con_linea = [f for f in filas if f["p_home_vegas"] is not None]
    distinto = [
        f for f in con_linea if (f["p_home_modelo"] >= 0.5) != (f["p_home_vegas"] >= 0.5)
    ]
    gano_modelo = sum(
        1 for f in distinto if (f["p_home_modelo"] >= 0.5) == (f["home_win"] == 1)
    )
    lineas = [
        "### Cuando el modelo y Las Vegas eligen distinto ganador",
        "",
        f"- Partidos: {len(distinto)} de {len(con_linea)}",
        f"- Acertó el modelo: {gano_modelo}. Acertó Las Vegas: {len(distinto) - gano_modelo}.",
        "",
        "Los 15 desacuerdos más grandes en probabilidad:",
        "",
        "| Partido | Modelo (local) | Las Vegas (local) | Ganó | ¿Quién estuvo más cerca? |",
        "|---|---|---|---|---|",
    ]
    mayores = sorted(
        con_linea, key=lambda f: -abs(f["p_home_modelo"] - f["p_home_vegas"])
    )[:15]
    for f in mayores:
        y = f["home_win"]
        cerca = "Modelo" if abs(f["p_home_modelo"] - y) < abs(f["p_home_vegas"] - y) else "Las Vegas"
        ganador = f["home_team"] if y == 1 else f["away_team"]
        lineas.append(
            f"| {f['season']} sem {f['week']}: {f['away_team']} en {f['home_team']} | "
            f"{_fmt(f['p_home_modelo'], pct=True)} | {_fmt(f['p_home_vegas'], pct=True)} | "
            f"{ganador} | {cerca} |"
        )
    return lineas + [""]


def _veredicto(honesto: list[dict], p_base: float) -> list[str]:
    comp = _comparacion(honesto, p_base)
    modelo = comp["Modelo"]["log_loss"]
    rango_70 = next(
        r
        for r in tabla_confianza(honesto, [f["p_home_modelo"] for f in honesto])
        if r["rango"].startswith("70")
    )
    pruebas = [
        ("Supera a 'siempre el local' en log loss", modelo < comp["Siempre el local"]["log_loss"]),
        ("Supera al Elo en log loss", modelo < comp["Elo"]["log_loss"]),
        (
            "El rango 70–80% acierta entre 65% y 85%",
            rango_70["acierto"] is not None and 0.65 <= rango_70["acierto"] <= 0.85,
        ),
        ("Supera a Las Vegas en log loss (no era objetivo)", modelo < comp["Las Vegas"]["log_loss"]),
    ]
    lineas = ["## Veredicto (solo 2024–2025)", "", "| Criterio | Resultado |", "|---|---|"]
    lineas += [f"| {texto} | {'Sí' if ok else 'No'} |" for texto, ok in pruebas]
    return lineas + [""]


def escribir_reporte(predicciones: list[dict], p_base: float, config: dict) -> str:
    optimista, honesto = _en(predicciones, OPTIMISTA), _en(predicciones, HONESTO)
    visible = {k: v for k, v in config.items() if k in ("tipo", "C", "max_iter", "max_depth", "calibrar")}
    lineas = [
        "# Backtest semana por semana",
        "",
        f"Generado: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Modelo: `{json.dumps(visible)}`",
        "- Cada semana se predijo entrenando solo con partidos anteriores.",
        f"- {OPTIMISTA[0]}–{OPTIMISTA[1]} es optimista: con esos años se eligió el modelo.",
        f"- **{HONESTO[0]}–{HONESTO[1]} es la prueba honesta.**",
        "",
    ]
    lineas += _veredicto(honesto, p_base)
    lineas += ["## Comparación", ""]
    lineas += _tabla_comparacion(f"Prueba honesta {HONESTO[0]}–{HONESTO[1]}", honesto, p_base)
    lineas += _tabla_comparacion(f"Optimista {OPTIMISTA[0]}–{OPTIMISTA[1]}", optimista, p_base)
    lineas += ["## Por temporada", ""]
    for temporada in sorted({f["season"] for f in predicciones}):
        lineas += _tabla_comparacion(str(temporada), _en(predicciones, (temporada, temporada)), p_base)
    lineas += ["## Aciertos por rango de confianza", ""]
    lineas += _tabla_confianza(f"Prueba honesta {HONESTO[0]}–{HONESTO[1]}", honesto)
    lineas += _tabla_confianza(f"Todo {OPTIMISTA[0]}–{HONESTO[1]}", predicciones)
    lineas += [f"## Modelo contra Las Vegas ({HONESTO[0]}–{HONESTO[1]})", ""]
    lineas += _desacuerdos(honesto)
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest walk-forward.")
    parser.add_argument("--repetir", action="store_true")
    args = parser.parse_args()

    reporte = REPORTS_DIR / "backtest_reporte.md"
    if reporte.exists() and not args.repetir:
        sys.exit(
            "El backtest ya se corrió. Las temporadas 2024-2025 solo valen como prueba\n"
            "honesta la primera vez: si cambias el modelo después de verlas y lo repites,\n"
            "el resultado deja de ser confiable. Para repetirlo de todos modos: --repetir"
        )

    config = json.loads((DATA_PROCESSED / "model_config.json").read_text("utf-8"))
    filas = cargar_filas()
    base = con_ganador(filas, 2006, 2020)
    p_base = sum(f["home_win"] for f in base) / len(base)

    print(f"Recorriendo {OPTIMISTA[0]}–{HONESTO[1]} semana por semana...")
    predicciones = recorrer(filas, config, OPTIMISTA[0], HONESTO[1])
    print(f"  {len(predicciones)} partidos predichos")

    columnas = ["game_id", "season", "week", "gameday", "home_team", "away_team",
                "home_win", "p_home_modelo", "p_home_elo", "p_home_vegas"]
    pl.DataFrame(
        [{c: f[c] for c in columnas} for f in predicciones], infer_schema_length=None
    ).write_parquet(DATA_PROCESSED / "backtest_predicciones.parquet")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reporte.write_text(escribir_reporte(predicciones, p_base, config), "utf-8")
    print(f"Listo. Reporte en {reporte}")


if __name__ == "__main__":
    main()
