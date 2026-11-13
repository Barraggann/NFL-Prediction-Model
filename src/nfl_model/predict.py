"""Fase 6: picks de la semana.

Actualiza los datos, entrena con todo lo anterior a la semana y guarda los picks
en un registro al que solo se le agregan filas. Un partido ya registrado nunca
se vuelve a escribir, aunque el modelo cambie de opinión.

Salidas:
  picks/picks_log.csv                     registro permanente
  reports/semana_<temporada>_<NN>.md      tabla de la semana

Uso:
  python -m nfl_model.predict                 # la próxima semana por jugarse
  python -m nfl_model.predict --semana 11
  python -m nfl_model.predict --sin-actualizar  # no vuelve a descargar datos
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

from . import DATA_PROCESSED, PICKS_DIR, REPORTS_DIR
from .features import VARIABLES
from .train import ajustar, cargar_filas, con_ganador, predecir

REGISTRO = PICKS_DIR / "picks_log.csv"
COLUMNAS_REGISTRO = [
    "generado", "temporada", "semana", "game_id", "gameday", "visita", "local",
    "pick", "p_pick", "p_home_modelo", "p_home_vegas", "p_home_elo",
    "qb_anunciado",
]
PRIMERA_ENTRENAMIENTO = 2006
AVISO = "Experimento de ciencia de datos. No es consejo de apuestas."


def actualizar_datos() -> None:
    """Descarga resultados nuevos y recalcula Elo y variables."""
    from . import elo, features, ingest

    ingest.ejecutar()
    elo.main()
    features.main()


def elegir_semana(filas: list[dict], hoy: date) -> tuple[int, int] | None:
    """La semana más próxima con partidos que aún no empiezan."""
    pendientes = [f for f in filas if not f["played"] and f["gameday"] >= hoy]
    if not pendientes:
        return None
    primero = min(pendientes, key=lambda f: (f["gameday"], f["week"]))
    return primero["season"], primero["week"]


def generar_picks(
    filas: list[dict],
    config: dict,
    temporada: int,
    semana: int,
    hoy: date,
    qb_anunciado: dict[str, bool] | None = None,
) -> list[dict]:
    """Picks de los partidos de esa semana que todavía no se juegan."""
    objetivo = [
        f
        for f in filas
        if f["season"] == temporada
        and f["week"] == semana
        and not f["played"]
        and f["gameday"] >= hoy
        and all(f[v] is not None for v in VARIABLES)
    ]
    if not objetivo:
        return []

    corte = min(f["gameday"] for f in objetivo)
    entrenamiento = [
        f for f in con_ganador(filas, PRIMERA_ENTRENAMIENTO, 9999) if f["gameday"] < corte
    ]
    modelo = ajustar(entrenamiento, config)
    ahora = datetime.now().isoformat(timespec="seconds")
    qb_anunciado = qb_anunciado or {}

    picks = []
    for f, p in zip(objetivo, predecir(modelo, objetivo, config)):
        gana_local = p >= 0.5
        picks.append(
            {
                "generado": ahora,
                "temporada": temporada,
                "semana": semana,
                "game_id": f["game_id"],
                "gameday": f["gameday"].isoformat(),
                "visita": f["away_team"],
                "local": f["home_team"],
                "pick": f["home_team"] if gana_local else f["away_team"],
                "p_pick": round(p if gana_local else 1 - p, 4),
                "p_home_modelo": round(p, 4),
                "p_home_vegas": None if f["p_home_vegas"] is None else round(f["p_home_vegas"], 4),
                "p_home_elo": round(f["p_home_elo"], 4),
                "qb_anunciado": int(qb_anunciado.get(f["game_id"], True)),
            }
        )
    return picks


def leer_registro(ruta: Path = REGISTRO) -> list[dict]:
    if not ruta.exists():
        return []
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def agregar_al_registro(picks: list[dict], ruta: Path = REGISTRO) -> list[dict]:
    """Agrega solo los partidos que no estaban. Devuelve los que agregó."""
    ya = {fila["game_id"] for fila in leer_registro(ruta)}
    nuevos = [p for p in picks if p["game_id"] not in ya]
    if not nuevos:
        return []
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escribir_encabezado = not ruta.exists()
    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_REGISTRO)
        if escribir_encabezado:
            escritor.writeheader()
        for p in nuevos:
            escritor.writerow({c: "" if p[c] is None else p[c] for c in COLUMNAS_REGISTRO})
    return nuevos


def _num(texto: str) -> float | None:
    return float(texto) if texto not in ("", None) else None


def reporte_semana(
    registro: list[dict], temporada: int, semana: int, perfiles: dict | None = None
) -> str:
    """Tabla de la semana, armada desde el registro (lo que quedó escrito).

    Con `perfiles` agrega el duelo ataque-defensa más desigual de cada partido.
    """
    filas = [
        f for f in registro
        if int(f["temporada"]) == temporada and int(f["semana"]) == semana
    ]
    filas.sort(key=lambda f: -float(f["p_pick"]))

    lineas = [
        f"# Picks: temporada {temporada}, semana {semana}",
        "",
        f"{AVISO}",
        "",
        "| Visita | Local | Pick | Modelo | Las Vegas (mismo equipo) | Diferencia | Generado |",
        "|---|---|---|---|---|---|---|",
    ]
    desacuerdos = []
    for f in filas:
        p_pick = float(f["p_pick"])
        p_vegas_local = _num(f["p_home_vegas"])
        gana_local = f["pick"] == f["local"]
        if p_vegas_local is None:
            vegas, diferencia = "—", "—"
        else:
            p_vegas = p_vegas_local if gana_local else 1 - p_vegas_local
            vegas = f"{p_vegas * 100:.1f}%"
            diferencia = f"{(p_pick - p_vegas) * 100:+.1f}"
            desacuerdos.append((abs(p_pick - p_vegas), p_vegas < 0.5, f, p_vegas))
        nota = "" if f["qb_anunciado"] == "1" else " *"
        lineas.append(
            f"| {f['visita']} | {f['local']} | **{f['pick']}**{nota} | {p_pick * 100:.1f}% | "
            f"{vegas} | {diferencia} | {f['generado'][:10]} |"
        )
    if any(f["qb_anunciado"] != "1" for f in filas):
        lineas += ["", "\\* Sin quarterback titular anunciado al generar el pick; se asumió el del partido anterior."]

    fuertes = [f for f in filas if float(f["p_pick"]) >= 0.70]
    lineas += ["", "## Picks de 70% o más", ""]
    lineas += [f"- {f['pick']} ({float(f['p_pick']) * 100:.0f}%)" for f in fuertes] or ["- Ninguno esta semana."]

    contrarios = [d for d in desacuerdos if d[1]]
    lineas += ["", "## Contra Las Vegas", ""]
    if contrarios:
        for _, _, f, p_vegas in sorted(contrarios, key=lambda d: -d[0]):
            lineas.append(
                f"- {f['visita']} en {f['local']}: el modelo va con **{f['pick']}** "
                f"({float(f['p_pick']) * 100:.0f}%); Las Vegas le da {p_vegas * 100:.0f}%."
            )
        lineas += ["", "En el backtest 2024–2025, cuando no coincidieron, Las Vegas acertó 53 de 83."]
    else:
        lineas.append("- El modelo y Las Vegas eligen al mismo ganador en todos los partidos.")
    if desacuerdos:
        _, _, f, p_vegas = max(desacuerdos, key=lambda d: d[0])
        lineas += [
            "",
            f"Mayor diferencia de confianza: {f['visita']} en {f['local']} "
            f"(modelo {float(f['p_pick']) * 100:.0f}% con {f['pick']}, Las Vegas {p_vegas * 100:.0f}%).",
        ]

    if perfiles:
        from .perfiles import duelos, texto_duelo

        lineas += [
            "",
            "## Duelo más desigual de cada partido",
            "",
            "Ranking por partido entre los equipos de la liga, antes de esta semana. "
            "Es contexto para explicar el pick: el modelo no usa estos rankings.",
            "",
            "| Partido | Duelo | Lugares de ventaja |",
            "|---|---|---|",
        ]
        for f in filas:
            cruces = duelos(perfiles, f["local"], f["visita"])
            if not cruces:
                continue
            mayor = cruces[0]
            lineas.append(
                f"| {f['visita']} en {f['local']} | {texto_duelo(mayor)} | {mayor['ventaja']:+d} |"
            )
    return "\n".join(lineas) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera los picks de la semana.")
    parser.add_argument("--semana", type=int)
    parser.add_argument("--temporada", type=int)
    parser.add_argument("--sin-actualizar", action="store_true")
    args = parser.parse_args()

    if not args.sin_actualizar:
        actualizar_datos()

    import polars as pl

    filas = cargar_filas()
    hoy = date.today()
    proxima = elegir_semana(filas, hoy)
    if proxima is None and args.semana is None:
        print("No hay partidos pendientes en el calendario.")
        return
    temporada = args.temporada or (proxima[0] if proxima else max(f["season"] for f in filas))
    semana = args.semana or proxima[1]

    partidos = pl.read_parquet(DATA_PROCESSED / "games.parquet").to_dicts()
    qb_anunciado = {
        p["game_id"]: p["home_qb_id"] is not None and p["away_qb_id"] is not None
        for p in partidos
    }
    config = json.loads((DATA_PROCESSED / "model_config.json").read_text("utf-8"))

    picks = generar_picks(filas, config, temporada, semana, hoy, qb_anunciado)
    if not picks:
        print(f"Temporada {temporada}, semana {semana}: no quedan partidos por jugarse.")
    nuevos = agregar_al_registro(picks)
    omitidos = len(picks) - len(nuevos)
    print(f"Temporada {temporada}, semana {semana}: {len(nuevos)} picks nuevos en el registro.")
    if omitidos:
        print(f"  {omitidos} partidos ya estaban registrados y no se tocaron.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = REPORTS_DIR / f"semana_{temporada}_{semana:02d}.md"
    ruta.write_text(
        reporte_semana(leer_registro(), temporada, semana, perfiles_de_semana(temporada, semana)),
        "utf-8",
    )
    print(f"Tabla en {ruta}")


def perfiles_de_semana(temporada: int, semana: int) -> dict:
    """Perfiles con los partidos anteriores a esa semana. Vacío si faltan datos."""
    import polars as pl

    from .perfiles import cargar

    partidos = pl.read_parquet(DATA_PROCESSED / "games.parquet").to_dicts()
    fechas = [p["gameday"] for p in partidos if p["season"] == temporada and p["week"] == semana]
    if not fechas:
        return {}
    try:
        return cargar(temporada, antes_de=min(fechas))[0]
    except KeyError:
        # Datos descargados antes de que existieran las columnas de yardas.
        print("  Sin perfiles: falta volver a correr python -m nfl_model.ingest")
        return {}


if __name__ == "__main__":
    main()
