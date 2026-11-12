"""Fase 4: elegir modelo y revisar calibración.

Entrena con 2006-2020 y elige con 2021-2023. No toca 2024-2025.

Salidas:
  data/processed/model_config.json  configuración elegida (la usan Fases 5 y 6)
  reports/modelo_reporte.md         comparación, coeficientes y calibración
  reports/calibracion.png           gráfica de calibración en validación

Uso:
  python -m nfl_model.train
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import DATA_PROCESSED, REPORTS_DIR
from .elo import metricas
from .features import VARIABLES

ENTRENAMIENTO = (2006, 2020)
VALIDACION = (2021, 2023)

REJILLA_C = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
REJILLA_BOOSTING = [
    {"max_iter": n, "max_depth": d} for d in (2, 3) for n in (50, 100, 200, 400)
]
# Lo complejo solo gana si mejora con claridad.
MARGEN_BOOSTING = 0.002
MARGEN_CALIBRACION = 0.001
# Decisión manual del 2026-11-11: quedarse con la logística aunque el boosting
# gane por poco. En validación el boosting ganó por 0.0026 de log loss (ruido con
# 851 partidos) pero su rango de 70-80% acertó 64.2%; la logística, 72.1%.
# Poner en False para volver a la regla automática.
PREFERIR_LOGISTICA = True

RANGOS_CONFIANZA = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


# ---------- piezas reutilizables (Fases 5 y 6) ----------


def cargar_filas() -> list[dict]:
    return (
        pl.read_parquet(DATA_PROCESSED / "features.parquet")
        .sort(["gameday", "game_id"])
        .to_dicts()
    )


def con_ganador(filas: list[dict], desde: int, hasta: int) -> list[dict]:
    """Partidos jugados, sin empate, con todas las variables, en el rango."""
    return [
        f
        for f in filas
        if f["home_win"] is not None
        and desde <= f["season"] <= hasta
        and all(f[v] is not None for v in VARIABLES)
    ]


def matriz(filas: list[dict]) -> np.ndarray:
    return np.array([[f[v] for v in VARIABLES] for f in filas], dtype=float)


def etiquetas(filas: list[dict]) -> np.ndarray:
    return np.array([f["home_win"] for f in filas], dtype=int)


def crear_modelo(config: dict):
    if config["tipo"] == "logistica":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=config["C"], max_iter=2000)
        )
    if config["tipo"] == "boosting":
        return HistGradientBoostingClassifier(
            max_iter=config["max_iter"],
            max_depth=config["max_depth"],
            learning_rate=0.05,
            early_stopping=False,
            random_state=0,
        )
    raise ValueError(f"Tipo de modelo desconocido: {config['tipo']}")


def aplicar_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))


def ajustar(filas_entrenamiento: list[dict], config: dict):
    modelo = crear_modelo(config)
    modelo.fit(matriz(filas_entrenamiento), etiquetas(filas_entrenamiento))
    return modelo


def predecir(modelo, filas: list[dict], config: dict) -> list[float]:
    """Probabilidad de que gane el local en cada fila."""
    if not filas:
        return []
    p = modelo.predict_proba(matriz(filas))[:, 1]
    platt = config.get("platt")
    if config.get("calibrar") and platt:
        p = aplicar_platt(p, platt["a"], platt["b"])
    return [float(x) for x in p]


# ---------- calibración ----------


def ajustar_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    reg = LogisticRegression(C=1e6, max_iter=2000).fit(logit, y)
    return float(reg.coef_[0][0]), float(reg.intercept_[0])


def platt_cruzado(filas: list[dict], p: list[float]) -> list[float]:
    """Calibra cada temporada con las otras, para medir sin trampa."""
    temporadas = sorted({f["season"] for f in filas})
    p_arr = np.array(p)
    y = etiquetas(filas)
    de = np.array([f["season"] for f in filas])
    salida = p_arr.copy()
    for t in temporadas:
        dentro = de == t
        a, b = ajustar_platt(p_arr[~dentro], y[~dentro])
        salida[dentro] = aplicar_platt(p_arr[dentro], a, b)
    return [float(x) for x in salida]


def tabla_confianza(filas: list[dict], p: list[float]) -> list[dict]:
    """Aciertos del pick por rango de confianza (como en el video)."""
    tabla = []
    for bajo, alto in RANGOS_CONFIANZA:
        n = aciertos = 0
        suma = 0.0
        for f, p_local in zip(filas, p):
            confianza = max(p_local, 1 - p_local)
            if not (bajo <= confianza < alto):
                continue
            n += 1
            suma += confianza
            aciertos += int((p_local >= 0.5) == (f["home_win"] == 1))
        tabla.append(
            {
                "rango": f"{bajo * 100:.0f}–{min(alto, 1) * 100:.0f}%",
                "partidos": n,
                "confianza_media": suma / n if n else None,
                "acierto": aciertos / n if n else None,
            }
        )
    return tabla


def graficar_calibracion(filas: list[dict], p: list[float], ruta) -> None:
    p_arr = np.array(p)
    y = etiquetas(filas)
    orden = np.argsort(p_arr)
    grupos = np.array_split(orden, 10)
    x = [p_arr[g].mean() for g in grupos if len(g)]
    obs = [y[g].mean() for g in grupos if len(g)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    ax.plot(x, obs, marker="o", label="Modelo (deciles)")
    ax.set_xlabel("Probabilidad que dio el modelo (gana el local)")
    ax.set_ylabel("Frecuencia real de victoria del local")
    ax.set_title(f"Calibración en validación {VALIDACION[0]}–{VALIDACION[1]}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------- selección ----------


def _pares(filas: list[dict], p: list[float]) -> list[tuple[float, int]]:
    return [(pi, f["home_win"]) for f, pi in zip(filas, p)]


def _fmt(valor, decimales=4, pct=False) -> str:
    if valor is None:
        return "—"
    return f"{valor * 100:.1f}%" if pct else f"{valor:.{decimales}f}"


def _fila_metricas(nombre: str, pares) -> str:
    m = metricas(pares)
    return f"| {nombre} | {_fmt(m['acierto'], pct=True)} | {_fmt(m['log_loss'])} | {_fmt(m['brier'])} |"


def main() -> None:
    filas = cargar_filas()
    entrenamiento = con_ganador(filas, *ENTRENAMIENTO)
    validacion = con_ganador(filas, *VALIDACION)
    print(f"Entrenamiento: {len(entrenamiento)} partidos. Validación: {len(validacion)}.")

    # Modelo A: regresión logística.
    resultados_a = []
    for c in REJILLA_C:
        config = {"tipo": "logistica", "C": c}
        p = predecir(ajustar(entrenamiento, config), validacion, config)
        resultados_a.append((config, p, metricas(_pares(validacion, p))))
    mejor_a = min(resultados_a, key=lambda r: r[2]["log_loss"])

    # Modelo B: gradient boosting poco profundo.
    resultados_b = []
    for hiper in REJILLA_BOOSTING:
        config = {"tipo": "boosting", **hiper}
        p = predecir(ajustar(entrenamiento, config), validacion, config)
        resultados_b.append((config, p, metricas(_pares(validacion, p))))
    mejor_b = min(resultados_b, key=lambda r: r[2]["log_loss"])

    gana_b = (
        not PREFERIR_LOGISTICA
        and mejor_b[2]["log_loss"] < mejor_a[2]["log_loss"] - MARGEN_BOOSTING
    )
    config, p_elegido, m_elegido = mejor_b if gana_b else mejor_a
    config = dict(config)
    print(f"Elegido: {config}  log loss validación = {m_elegido['log_loss']:.4f}")

    # ¿Hace falta calibrar? Se mide calibrando cada temporada con las otras dos.
    p_cruzado = platt_cruzado(validacion, p_elegido)
    ll_crudo = m_elegido["log_loss"]
    ll_calibrado = metricas(_pares(validacion, p_cruzado))["log_loss"]
    a, b = ajustar_platt(np.array(p_elegido), etiquetas(validacion))
    config["calibrar"] = bool(ll_calibrado < ll_crudo - MARGEN_CALIBRACION)
    config["platt"] = {"a": a, "b": b}
    p_final = p_cruzado if config["calibrar"] else p_elegido

    (DATA_PROCESSED / "model_config.json").write_text(
        json.dumps(
            {**config, "variables": VARIABLES, "entrenamiento": ENTRENAMIENTO,
             "validacion": VALIDACION},
            indent=2,
        ),
        "utf-8",
    )

    # ---------- reporte ----------
    p_base = sum(f["home_win"] for f in entrenamiento) / len(entrenamiento)
    con_linea = [i for i, f in enumerate(validacion) if f["p_home_vegas"] is not None]

    def sub(lista):
        return [lista[i] for i in con_linea]

    val_linea = sub(validacion)
    lineas = [
        "# Modelo: reporte de selección",
        "",
        f"Generado: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Entrenamiento {ENTRENAMIENTO[0]}–{ENTRENAMIENTO[1]}: {len(entrenamiento)} partidos",
        f"- Validación {VALIDACION[0]}–{VALIDACION[1]}: {len(validacion)} partidos",
        f"- Variables: {len(VARIABLES)}",
        f"- **Modelo elegido:** `{json.dumps({k: v for k, v in config.items() if k != 'platt'})}`",
        "",
        "Las temporadas 2024–2025 no se evalúan aquí a propósito.",
        "",
        f"## Validación, todos los partidos ({len(validacion)})",
        "",
        "| Método | Acierto | Log loss | Brier |",
        "|---|---|---|---|",
        _fila_metricas("Siempre el local", [(p_base, f["home_win"]) for f in validacion]),
        _fila_metricas("Elo", [(f["p_home_elo"], f["home_win"]) for f in validacion]),
        _fila_metricas("Logística (mejor C)", _pares(validacion, mejor_a[1])),
        _fila_metricas("Boosting (mejor)", _pares(validacion, mejor_b[1])),
        "",
        f"## Validación, partidos con moneyline ({len(val_linea)})",
        "",
        "| Método | Acierto | Log loss | Brier |",
        "|---|---|---|---|",
        _fila_metricas("Elo", [(f["p_home_elo"], f["home_win"]) for f in val_linea]),
        _fila_metricas("Logística (mejor C)", _pares(val_linea, sub(mejor_a[1]))),
        _fila_metricas("Boosting (mejor)", _pares(val_linea, sub(mejor_b[1]))),
        _fila_metricas("Las Vegas", [(f["p_home_vegas"], f["home_win"]) for f in val_linea]),
        "",
        "## Rejilla: regresión logística",
        "",
        "| C | Acierto | Log loss |",
        "|---|---|---|",
    ]
    for cfg, _, m in resultados_a:
        lineas.append(f"| {cfg['C']} | {_fmt(m['acierto'], pct=True)} | {_fmt(m['log_loss'])} |")
    lineas += ["", "## Rejilla: boosting", "", "| Árboles | Profundidad | Acierto | Log loss |",
               "|---|---|---|---|"]
    for cfg, _, m in resultados_b:
        lineas.append(
            f"| {cfg['max_iter']} | {cfg['max_depth']} | {_fmt(m['acierto'], pct=True)} | {_fmt(m['log_loss'])} |"
        )

    modelo_a = ajustar(entrenamiento, mejor_a[0])
    coeficientes = sorted(
        zip(VARIABLES, modelo_a[-1].coef_[0]), key=lambda x: -abs(x[1])
    )
    lineas += [
        "",
        "## Coeficientes de la logística (variables estandarizadas)",
        "",
        "Positivo favorece al local.",
        "",
        "| Variable | Coeficiente |",
        "|---|---|",
    ]
    lineas += [f"| {v} | {c:+.3f} |" for v, c in coeficientes]

    lineas += [
        "",
        "## Calibración del modelo elegido",
        "",
        f"- Log loss sin calibrar: {_fmt(ll_crudo)}",
        f"- Log loss calibrando cada temporada con las otras dos: {_fmt(ll_calibrado)}",
        f"- Platt sobre toda la validación: pendiente a = {a:.3f}, desplazamiento b = {b:+.3f} "
        "(a = 1 y b = 0 significa que ya estaba calibrado)",
        f"- ¿Se aplica calibración? **{'Sí' if config['calibrar'] else 'No'}**",
        "",
        "### Aciertos por rango de confianza (validación)",
        "",
        "| Confianza | Partidos | Confianza media | Acierto real |",
        "|---|---|---|---|",
    ]
    for r in tabla_confianza(validacion, p_final):
        lineas.append(
            f"| {r['rango']} | {r['partidos']} | {_fmt(r['confianza_media'], pct=True)} | {_fmt(r['acierto'], pct=True)} |"
        )
    # Los dos candidatos lado a lado, para decidir viendo también la calibración.
    for nombre, p_candidato in (
        ("logística (mejor C)", mejor_a[1]),
        ("boosting (mejor)", mejor_b[1]),
    ):
        lineas += [
            "",
            f"### Rangos de confianza: {nombre}",
            "",
            "| Confianza | Partidos | Confianza media | Acierto real |",
            "|---|---|---|---|",
        ]
        for r in tabla_confianza(validacion, p_candidato):
            lineas.append(
                f"| {r['rango']} | {r['partidos']} | {_fmt(r['confianza_media'], pct=True)} | {_fmt(r['acierto'], pct=True)} |"
            )
    lineas += ["", "Gráfica: `reports/calibracion.png`", ""]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    graficar_calibracion(validacion, p_final, REPORTS_DIR / "calibracion.png")
    ruta = REPORTS_DIR / "modelo_reporte.md"
    ruta.write_text("\n".join(lineas), "utf-8")
    print(f"Listo. Reporte en {ruta}")


if __name__ == "__main__":
    main()
