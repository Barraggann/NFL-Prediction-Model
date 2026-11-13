"""Experimento: ¿ayudan las variables de cruce ataque-defensa?

No cambia el modelo ni su configuración; solo escribe un reporte.

Por qué no basta "sumar" el cruce: el modelo ya tiene diff_off_pass_epa y
diff_def_pass_epa. El cruce aditivo (mi ataque aéreo + lo que permite tu defensa
aérea, menos lo mismo al revés) es exactamente diff_off_pass_epa - diff_def_pass_epa,
una combinación de variables que la regresión logística ya puede formar sola.
Lo único nuevo que un cruce puede aportar es una INTERACCIÓN: que un buen ataque
le saque a una mala defensa más de lo que predice la suma. Eso se prueba aquí.

Variables probadas (cada una, local menos visitante):
  inter_pase     (ataque aéreo propio centrado) x (defensa aérea rival centrada)
  inter_carrera  lo mismo por tierra

Regla fijada antes de ver el resultado: se adoptan solo si mejoran el log loss
de validación (2021-2023) por más de 0.002. No se mira 2024-2025.

Salida:
  reports/exp_cruces.md

Uso:
  python -m nfl_model.features      # una vez, para guardar la forma por equipo
  python -m nfl_model.exp_cruces
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import REPORTS_DIR
from .elo import metricas
from .features import METRICAS_CRUCE, VARIABLES
from .train import ENTRENAMIENTO, REJILLA_C, VALIDACION, cargar_filas, con_ganador

MARGEN = 0.002
INTERACCIONES = ["inter_pase", "inter_carrera"]
CRUDAS = [f"{lado}_{m}" for lado in ("home", "away") for m in METRICAS_CRUCE]


def agregar_interacciones(filas: list[dict], medias: dict[str, float]) -> list[dict]:
    """Devuelve copias de las filas con inter_pase e inter_carrera.

    `medias` centra cada métrica (clave sin el prefijo home_/away_), para que el
    producto sea positivo cuando un ataque sobre el promedio enfrenta a una
    defensa que permite más que el promedio.
    """
    salida = []
    for f in filas:
        def c(lado: str, metrica: str) -> float:
            return f[f"{lado}_{metrica}"] - medias[metrica]

        nueva = dict(f)
        nueva["inter_pase"] = (
            c("home", "off_pass_epa") * c("away", "def_pass_epa")
            - c("away", "off_pass_epa") * c("home", "def_pass_epa")
        )
        nueva["inter_carrera"] = (
            c("home", "off_rush_epa") * c("away", "def_rush_epa")
            - c("away", "off_rush_epa") * c("home", "def_rush_epa")
        )
        salida.append(nueva)
    return salida


def calcular_medias(filas: list[dict]) -> dict[str, float]:
    """Media de cada métrica entre local y visitante, solo con las filas dadas."""
    medias = {}
    for m in METRICAS_CRUCE:
        valores = [f[f"{lado}_{m}"] for f in filas for lado in ("home", "away")]
        medias[m] = sum(valores) / len(valores)
    return medias


def _completas(filas: list[dict]) -> list[dict]:
    return [f for f in filas if all(f.get(c) is not None for c in CRUDAS)]


def _matriz(filas: list[dict], variables: list[str]) -> np.ndarray:
    return np.array([[f[v] for v in variables] for f in filas], dtype=float)


def _y(filas: list[dict]) -> np.ndarray:
    return np.array([f["home_win"] for f in filas], dtype=int)


def evaluar(modelo, entrenamiento, validacion, variables) -> tuple[dict, object]:
    modelo.fit(_matriz(entrenamiento, variables), _y(entrenamiento))
    p = modelo.predict_proba(_matriz(validacion, variables))[:, 1]
    return metricas([(float(pi), f["home_win"]) for pi, f in zip(p, validacion)]), modelo


def _logistica(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def _boosting():
    return HistGradientBoostingClassifier(
        max_iter=100, max_depth=2, learning_rate=0.05, early_stopping=False, random_state=0
    )


def main() -> None:
    filas = cargar_filas()
    if not filas or "home_off_pass_epa" not in filas[0]:
        print("Falta la forma por equipo. Corre primero: python -m nfl_model.features")
        return

    entrenamiento = _completas(con_ganador(filas, *ENTRENAMIENTO))
    validacion = _completas(con_ganador(filas, *VALIDACION))
    medias = calcular_medias(entrenamiento)  # solo con entrenamiento: sin fuga
    entrenamiento = agregar_interacciones(entrenamiento, medias)
    validacion = agregar_interacciones(validacion, medias)

    con_cruces = VARIABLES + INTERACCIONES
    lineas = [
        "# Experimento: variables de cruce ataque-defensa",
        "",
        f"Generado: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Entrenamiento {ENTRENAMIENTO[0]}–{ENTRENAMIENTO[1]}: {len(entrenamiento)} partidos",
        f"- Validación {VALIDACION[0]}–{VALIDACION[1]}: {len(validacion)} partidos",
        f"- Regla: adoptar solo si el log loss de validación mejora más de {MARGEN}.",
        "- El cruce aditivo ya está en el modelo; aquí se prueba la interacción (producto).",
        "",
        "## Regresión logística",
        "",
        "| C | Base: acierto | Base: log loss | Con cruces: acierto | Con cruces: log loss |",
        "|---|---|---|---|---|",
    ]
    mejor_base = mejor_cruce = None
    modelo_cruce = None
    for c in REJILLA_C:
        base, _ = evaluar(_logistica(c), entrenamiento, validacion, VARIABLES)
        cruce, ajustado = evaluar(_logistica(c), entrenamiento, validacion, con_cruces)
        lineas.append(
            f"| {c} | {base['acierto'] * 100:.1f}% | {base['log_loss']:.4f} | "
            f"{cruce['acierto'] * 100:.1f}% | {cruce['log_loss']:.4f} |"
        )
        if mejor_base is None or base["log_loss"] < mejor_base["log_loss"]:
            mejor_base = base
        if mejor_cruce is None or cruce["log_loss"] < mejor_cruce["log_loss"]:
            mejor_cruce, modelo_cruce = cruce, ajustado

    coef = dict(zip(con_cruces, modelo_cruce[-1].coef_[0]))
    lineas += [
        "",
        "Coeficientes de las interacciones (estandarizados, en el mejor C con cruces):",
        "",
        f"- inter_pase: {coef['inter_pase']:+.3f}",
        f"- inter_carrera: {coef['inter_carrera']:+.3f}",
        f"- Referencia, elo_diff: {coef['elo_diff']:+.3f}",
        "",
        "## Boosting (100 árboles, profundidad 2)",
        "",
        "Puede descubrir interacciones por su cuenta si ve la forma de cada equipo por separado.",
        "",
        "| Variables | Acierto | Log loss |",
        "|---|---|---|",
    ]
    for nombre, variables in (
        ("Base (18)", VARIABLES),
        ("Base + forma por equipo (26)", VARIABLES + CRUDAS),
        ("Base + interacciones (20)", con_cruces),
    ):
        m, _ = evaluar(_boosting(), entrenamiento, validacion, variables)
        lineas.append(f"| {nombre} | {m['acierto'] * 100:.1f}% | {m['log_loss']:.4f} |")

    mejora = mejor_base["log_loss"] - mejor_cruce["log_loss"]
    adoptar = mejora > MARGEN
    lineas += [
        "",
        "## Veredicto",
        "",
        f"- Mejor log loss base: {mejor_base['log_loss']:.4f}",
        f"- Mejor log loss con cruces: {mejor_cruce['log_loss']:.4f}",
        f"- Mejora: {mejora:+.4f} (se pedía más de {MARGEN})",
        f"- **{'ADOPTAR' if adoptar else 'NO ADOPTAR'}**",
        "",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = REPORTS_DIR / "exp_cruces.md"
    ruta.write_text("\n".join(lineas), "utf-8")
    print(f"{'ADOPTAR' if adoptar else 'NO ADOPTAR'} (mejora {mejora:+.4f}). Reporte en {ruta}")


if __name__ == "__main__":
    main()
