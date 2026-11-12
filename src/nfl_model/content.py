"""Fase 7: imágenes verticales (1080x1920) para video.

Tres imágenes, armadas desde el registro de picks:
  reports/contenido/semana_<temporada>_<NN>_picks.png   los picks más fuertes
  reports/contenido/semana_<temporada>_<NN>_vegas.png   mayor diferencia con Las Vegas
  reports/contenido/historial.png                       aciertos por rango de confianza

El historial usa los picks en vivo ya calificados; mientras no haya ninguno,
usa el backtest 2024-2025 y lo dice en la imagen.

Uso:
  python -m nfl_model.content
  python -m nfl_model.content --semana 10 --temporada 2026

Diseño: todo se dibuja en pixeles sobre un lienzo de 1080x1920. El contenido
evita los bordes donde TikTok/Reels ponen sus botones (arriba, abajo, derecha).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import DATA_PROCESSED, PICKS_DIR, REPORTS_DIR
from .predict import AVISO, leer_registro
from .track import por_rango

ANCHO, ALTO = 1080, 1920
IZQ, DER = 70, 930  # zona segura horizontal
LARGO = DER - IZQ

# Paleta para fondo oscuro (valores de referencia validados para ese fondo).
FONDO = "#1a1a19"
TINTA = "#f0efec"
TINTA_2 = "#c3c2b7"
TENUE = "#898781"
PISTA = "#2c2c2a"
AZUL = "#3987e5"  # serie 1: el modelo
NARANJA = "#d95926"  # serie 2: Las Vegas

# Tipografía del sistema; si no está, matplotlib usa la siguiente de la lista.
FUENTE = ["Segoe UI", "DejaVu Sans"]

FUERTE = 0.70
MAX_PICKS = 5


# ---------- utilidades de dibujo ----------


def _lienzo():
    fig = plt.figure(figsize=(ANCHO / 100, ALTO / 100), dpi=100, facecolor=FONDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ANCHO)
    ax.set_ylim(ALTO, 0)  # y crece hacia abajo, como en pixeles
    ax.axis("off")
    return fig, ax


def _texto(ax, x, y, texto, px, color=TINTA, peso="normal", alinear="left"):
    ax.text(
        x, y, texto,
        fontsize=px * 0.72,  # pixeles -> puntos a 100 dpi
        color=color, fontweight=peso, ha=alinear, va="baseline",
        fontfamily=FUENTE,
    )


def _barra(ax, y, fraccion, color, alto=26):
    """Pista de 0 a 100% con relleno hasta `fraccion`; extremo redondeado."""
    radio = alto / 2
    ax.add_patch(
        FancyBboxPatch(
            (IZQ, y), LARGO, alto,
            boxstyle=f"round,pad=0,rounding_size={radio}",
            facecolor=PISTA, edgecolor="none",
        )
    )
    largo = max(LARGO * max(min(fraccion, 1.0), 0.0), 0)
    if largo <= 0:
        return
    largo = max(largo, alto)
    ax.add_patch(
        FancyBboxPatch(
            (IZQ, y), largo, alto,
            boxstyle=f"round,pad=0,rounding_size={radio}",
            facecolor=color, edgecolor="none",
        )
    )


def _encabezado(ax, sobretitulo, titulo, subtitulo):
    _texto(ax, IZQ, 300, sobretitulo.upper(), 30, TENUE, "bold")
    _texto(ax, IZQ, 390, titulo, 72, TINTA, "bold")
    _texto(ax, IZQ, 452, subtitulo, 34, TINTA_2)


def _pie(ax):
    _texto(ax, IZQ, 1500, AVISO, 26, TENUE)


def _guardar(fig, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=100, facecolor=FONDO)
    plt.close(fig)
    return ruta


def _pct(valor: float, decimales: int = 0) -> str:
    return f"{valor * 100:.{decimales}f}%"


# ---------- datos ----------


def picks_de_semana(registro: list[dict], temporada: int, semana: int) -> list[dict]:
    filas = [
        f for f in registro
        if int(f["temporada"]) == temporada and int(f["semana"]) == semana
    ]
    return sorted(filas, key=lambda f: -float(f["p_pick"]))


def mayor_diferencia(picks: list[dict]) -> dict | None:
    """El partido donde modelo y Las Vegas difieren más (sobre el equipo del pick)."""
    mejor = None
    for f in picks:
        if f["p_home_vegas"] in ("", None):
            continue
        p_vegas_local = float(f["p_home_vegas"])
        p_vegas = p_vegas_local if f["pick"] == f["local"] else 1 - p_vegas_local
        diferencia = float(f["p_pick"]) - p_vegas
        if mejor is None or abs(diferencia) > abs(mejor["diferencia"]):
            mejor = {**f, "p_vegas_pick": p_vegas, "diferencia": diferencia}
    return mejor


def datos_historial() -> tuple[list[dict], str]:
    """(tabla por rango, texto de la fuente). En vivo si hay; si no, backtest."""
    ruta = PICKS_DIR / "resultados.csv"
    if ruta.exists():
        with ruta.open(newline="", encoding="utf-8") as archivo:
            filas = list(csv.DictReader(archivo))
        decididos = [f for f in filas if f["estado"] in ("acierto", "fallo")]
        if decididos:
            return por_rango(filas), f"Picks en vivo · {len(decididos)} partidos"

    ruta = DATA_PROCESSED / "backtest_predicciones.parquet"
    if not ruta.exists():
        return [], ""
    filas = []
    for f in pl.read_parquet(ruta).filter(pl.col("season") >= 2024).to_dicts():
        p = f["p_home_modelo"]
        acierto = (p >= 0.5) == (f["home_win"] == 1)
        filas.append({"p_pick": max(p, 1 - p), "estado": "acierto" if acierto else "fallo"})
    return por_rango(filas), f"Backtest 2024–2025 · {len(filas)} partidos"


# ---------- las tres imágenes ----------


def imagen_picks(picks: list[dict], temporada: int, semana: int, ruta: Path) -> Path:
    fuertes = [f for f in picks if float(f["p_pick"]) >= FUERTE][:MAX_PICKS]
    hay_fuertes = bool(fuertes)
    mostrar = fuertes or picks[:3]

    fig, ax = _lienzo()
    _encabezado(
        ax,
        f"NFL · Semana {semana} · {temporada}",
        "Picks de 70% o más" if hay_fuertes else "Los picks más fuertes",
        "Probabilidad de victoria según el modelo"
        if hay_fuertes
        else "Esta semana ninguno llega a 70%",
    )
    # El partido va pegado a su barra y lejos del siguiente equipo, para que
    # no se lea como título de la fila de abajo.
    y = 610
    for f in mostrar:
        p = float(f["p_pick"])
        _texto(ax, IZQ, y, f["pick"], 76, TINTA, "bold")
        _texto(ax, DER, y, _pct(p), 76, TINTA, "bold", "right")
        _barra(ax, y + 26, p, AZUL)
        _texto(ax, IZQ, y + 86, f"{f['visita']} en {f['local']}", 30, TENUE)
        y += 185
    _pie(ax)
    return _guardar(fig, ruta)


def imagen_vegas(picks: list[dict], temporada: int, semana: int, ruta: Path) -> Path | None:
    d = mayor_diferencia(picks)
    if d is None:
        return None
    p_modelo, p_vegas = float(d["p_pick"]), d["p_vegas_pick"]
    mismo_ganador = p_vegas >= 0.5
    rival = d["visita"] if d["pick"] == d["local"] else d["local"]

    fig, ax = _lienzo()
    _encabezado(
        ax,
        f"NFL · Semana {semana} · {temporada}",
        "Modelo vs Las Vegas",
        "El partido donde más difieren",
    )
    _texto(ax, IZQ, 690, f"{d['visita']} en {d['local']}", 104, TINTA, "bold")
    _texto(
        ax, IZQ, 760,
        f"Los dos eligen a {d['pick']}; cambia cuánta confianza le dan"
        if mismo_ganador
        else f"El modelo va con {d['pick']}; Las Vegas, con {rival}",
        32, TINTA_2,
    )

    _texto(ax, IZQ, 880, f"Probabilidad de que gane {d['pick']}", 30, TENUE, "bold")
    y = 990
    for nombre, valor, color in (("Modelo", p_modelo, AZUL), ("Las Vegas", p_vegas, NARANJA)):
        ax.add_patch(Rectangle((IZQ, y - 36), 36, 36, facecolor=color, edgecolor="none"))
        _texto(ax, IZQ + 54, y, nombre, 48, TINTA)
        _texto(ax, DER, y, _pct(valor, 1), 76, TINTA, "bold", "right")
        _barra(ax, y + 30, valor, color)
        y += 190

    _texto(ax, IZQ, 1380, f"{abs(d['diferencia']) * 100:.1f} puntos de diferencia", 44, TINTA, "bold")
    _texto(
        ax, IZQ, 1430,
        "En el backtest 2024–2025, cuando eligieron distinto, Las Vegas acertó 53 de 83.",
        24, TENUE,
    )
    _pie(ax)
    return _guardar(fig, ruta)


def imagen_historial(tabla: list[dict], fuente: str, ruta: Path) -> Path | None:
    if not tabla:
        return None
    fig, ax = _lienzo()
    _encabezado(ax, "NFL · Historial del modelo", "¿Acierta lo que promete?", fuente)

    # Leyenda: el relleno es lo que pasó; la marca, lo que el modelo prometía.
    ax.add_patch(Rectangle((IZQ, 520), 30, 30, facecolor=AZUL, edgecolor="none"))
    _texto(ax, IZQ + 44, 546, "Acierto real", 28, TINTA_2)
    ax.add_patch(Rectangle((IZQ + 280, 512), 6, 46, facecolor=TINTA, edgecolor="none"))
    _texto(ax, IZQ + 302, 546, "Lo que prometía el modelo", 28, TINTA_2)

    y = 720
    for r in tabla:
        _texto(ax, IZQ, y, f"Confianza {r['rango']}", 44, TINTA, "bold")
        if r["partidos"]:
            real = r["aciertos"] / r["partidos"]
            _texto(ax, DER, y, _pct(real), 76, TINTA, "bold", "right")
            _barra(ax, y + 28, real, AZUL)
            x = IZQ + LARGO * r["confianza_media"]
            # Marca con anillo del color del fondo para que se lea sobre el relleno.
            ax.add_patch(Rectangle((x - 6, y + 16), 12, 50, facecolor=FONDO, edgecolor="none"))
            ax.add_patch(Rectangle((x - 3, y + 18), 6, 46, facecolor=TINTA, edgecolor="none"))
            _texto(ax, IZQ, y + 96, f"{r['aciertos']} de {r['partidos']} partidos", 30, TENUE)
        else:
            _barra(ax, y + 28, 0, AZUL)
            _texto(ax, IZQ, y + 96, "Sin partidos calificados todavía", 30, TENUE)
        y += 188
    _pie(ax)
    return _guardar(fig, ruta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera las imágenes para video.")
    parser.add_argument("--semana", type=int)
    parser.add_argument("--temporada", type=int)
    args = parser.parse_args()

    registro = leer_registro()
    if not registro:
        print("El registro está vacío. Primero corre: python -m nfl_model.predict")
        return
    ultima = max((int(f["temporada"]), int(f["semana"])) for f in registro)
    temporada = args.temporada or ultima[0]
    semana = args.semana or ultima[1]
    picks = picks_de_semana(registro, temporada, semana)
    if not picks:
        print(f"No hay picks registrados para la temporada {temporada}, semana {semana}.")
        return

    carpeta = REPORTS_DIR / "contenido"
    base = f"semana_{temporada}_{semana:02d}"
    hechas = [
        imagen_picks(picks, temporada, semana, carpeta / f"{base}_picks.png"),
        imagen_vegas(picks, temporada, semana, carpeta / f"{base}_vegas.png"),
        imagen_historial(*datos_historial(), carpeta / "historial.png"),
    ]
    for ruta in hechas:
        if ruta is not None:
            print(f"  {ruta}")


if __name__ == "__main__":
    main()
