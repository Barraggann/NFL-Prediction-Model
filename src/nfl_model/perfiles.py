"""Perfiles de equipo: estadísticas por partido y su ranking en la liga.

Sirve para explicar los picks, no para predecir: el modelo no usa estos rankings.
Solo cuenta partidos de temporada regular ya jugados.

Salida:
  reports/perfiles_<temporada>.md   los 32 equipos, ofensiva, defensiva y balones

Uso:
  python -m nfl_model.perfiles
  python -m nfl_model.perfiles --temporada 2025

Nota: las yardas salen del jugada por jugada, con la convención oficial (las
capturas restan al pase, los scrambles suman a la carrera). Pueden diferir por
unas pocas yardas de las cifras oficiales (hincadas, laterales, correcciones).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

import polars as pl

from . import DATA_PROCESSED, REPORTS_DIR

# clave -> (etiqueta, True si más es mejor, decimales)
ESTADISTICAS = {
    # Ofensiva
    "pts": ("Puntos", True, 1),
    "yds": ("Yardas totales", True, 1),
    "pase": ("Yardas por aire", True, 1),
    "carrera": ("Yardas por tierra", True, 1),
    "yds_jugada": ("Yardas por jugada", True, 2),
    "epa": ("EPA por jugada", True, 3),
    # Defensiva (lo que permite: menos es mejor)
    "pts_perm": ("Puntos permitidos", False, 1),
    "yds_perm": ("Yardas permitidas", False, 1),
    "pase_perm": ("Yardas por aire permitidas", False, 1),
    "carrera_perm": ("Yardas por tierra permitidas", False, 1),
    "yds_jugada_perm": ("Yardas por jugada permitidas", False, 2),
    "epa_perm": ("EPA por jugada permitido", False, 3),
    # Balones y presión
    "capturas": ("Capturas logradas", True, 1),
    "capturas_perm": ("Capturas permitidas", False, 1),
    "robos": ("Balones recuperados", True, 1),
    "perdidas": ("Balones perdidos", False, 1),
    "dif_balones": ("Diferencial de balones", True, 1),
}

OFENSIVA = ["pts", "yds", "pase", "carrera", "yds_jugada", "epa"]
DEFENSIVA = ["pts_perm", "yds_perm", "pase_perm", "carrera_perm", "yds_jugada_perm", "epa_perm"]
BALONES = ["capturas", "capturas_perm", "robos", "perdidas", "dif_balones"]

# Los cuatro cruces de un partido: (tipo, estadística del ataque, de la defensa).
CRUCES = [("aire", "pase", "pase_perm"), ("tierra", "carrera", "carrera_perm")]


def calcular_perfiles(
    partidos: list[dict],
    stats: dict[tuple[str, str], dict],
    temporada: int,
    antes_de: date | None = None,
) -> dict[str, dict]:
    """equipo -> {"partidos": n, <estadística>: valor por partido, "rank": {...}}.

    Con `antes_de` solo cuentan los partidos anteriores a esa fecha.
    """
    sumas: dict[str, dict[str, float]] = {}
    for p in partidos:
        if p["season"] != temporada or p["game_type"] != "REG" or not p["played"]:
            continue
        if antes_de is not None and p["gameday"] >= antes_de:
            continue
        lados = (
            (p["home_team"], p["home_score"], p["away_score"]),
            (p["away_team"], p["away_score"], p["home_score"]),
        )
        for equipo, a_favor, en_contra in lados:
            s = stats.get((p["game_id"], equipo))
            if s is None:
                continue
            t = sumas.setdefault(equipo, {})

            def sumar(clave, valor):
                t[clave] = t.get(clave, 0.0) + float(valor or 0.0)

            sumar("n", 1)
            sumar("pts", a_favor)
            sumar("pts_perm", en_contra)
            sumar("yds", s["off_yards"])
            sumar("pase", s["off_pass_yards"])
            sumar("carrera", s["off_rush_yards"])
            sumar("jugadas", s["off_plays"])
            sumar("epa_total", s["off_epa"] * s["off_plays"])
            sumar("yds_perm", s["def_yards"])
            sumar("pase_perm", s["def_pass_yards"])
            sumar("carrera_perm", s["def_rush_yards"])
            sumar("jugadas_perm", s["def_plays"])
            sumar("epa_total_perm", s["def_epa"] * s["def_plays"])
            sumar("capturas", s["def_sacks"])
            sumar("capturas_perm", s["off_sacks"])
            sumar("robos", s["def_turnovers"])
            sumar("perdidas", s["off_turnovers"])

    perfiles: dict[str, dict] = {}
    for equipo, t in sumas.items():
        n = t["n"]
        perfil = {"partidos": int(n)}
        for clave in ("pts", "yds", "pase", "carrera", "pts_perm", "yds_perm",
                      "pase_perm", "carrera_perm", "capturas", "capturas_perm",
                      "robos", "perdidas"):
            perfil[clave] = t[clave] / n
        perfil["dif_balones"] = (t["robos"] - t["perdidas"]) / n
        perfil["yds_jugada"] = t["yds"] / t["jugadas"] if t["jugadas"] else 0.0
        perfil["epa"] = t["epa_total"] / t["jugadas"] if t["jugadas"] else 0.0
        perfil["yds_jugada_perm"] = t["yds_perm"] / t["jugadas_perm"] if t["jugadas_perm"] else 0.0
        perfil["epa_perm"] = t["epa_total_perm"] / t["jugadas_perm"] if t["jugadas_perm"] else 0.0
        perfiles[equipo] = perfil

    # Ranking: 1 es el mejor. Los empates comparten lugar.
    for equipo, perfil in perfiles.items():
        perfil["rank"] = {}
        for clave, (_, mas_es_mejor, _) in ESTADISTICAS.items():
            propio = perfil[clave]
            mejores = sum(
                1
                for otro in perfiles.values()
                if (otro[clave] > propio if mas_es_mejor else otro[clave] < propio)
            )
            perfil["rank"][clave] = mejores + 1
    return perfiles


def duelos(perfiles: dict[str, dict], local: str, visita: str) -> list[dict]:
    """Los cruces ataque contra defensa de un partido, del más desigual al menos.

    `ventaja` es cuántos lugares de ranking le saca el ataque a la defensa rival:
    positiva y grande = buen ataque contra mala defensa.
    """
    if local not in perfiles or visita not in perfiles:
        return []
    salida = []
    for ataque, defensa in ((local, visita), (visita, local)):
        for tipo, clave_ataque, clave_defensa in CRUCES:
            a, d = perfiles[ataque], perfiles[defensa]
            salida.append(
                {
                    "tipo": tipo,
                    "ataque": ataque,
                    "defensa": defensa,
                    "rank_ataque": a["rank"][clave_ataque],
                    "valor_ataque": a[clave_ataque],
                    "rank_defensa": d["rank"][clave_defensa],
                    "valor_defensa": d[clave_defensa],
                    "ventaja": d["rank"][clave_defensa] - a["rank"][clave_ataque],
                }
            )
    return sorted(salida, key=lambda x: -x["ventaja"])


def texto_duelo(d: dict) -> str:
    modo = "aéreo" if d["tipo"] == "aire" else "terrestre"
    contra = "el pase" if d["tipo"] == "aire" else "la carrera"
    return (
        f"Ataque {modo} de {d['ataque']} (#{d['rank_ataque']}, {d['valor_ataque']:.0f} yds) "
        f"contra la defensa de {d['defensa']} ante {contra} "
        f"(#{d['rank_defensa']}, {d['valor_defensa']:.0f} permitidas)"
    )


def cargar(temporada: int | None = None, antes_de: date | None = None) -> tuple[dict, int]:
    """Lee los datos procesados y devuelve (perfiles, temporada usada)."""
    partidos = pl.read_parquet(DATA_PROCESSED / "games.parquet").to_dicts()
    stats = {
        (f["game_id"], f["team"]): f
        for f in pl.read_parquet(DATA_PROCESSED / "team_games.parquet").to_dicts()
    }
    if temporada is None:
        temporada = max(p["season"] for p in partidos if p["played"])
    return calcular_perfiles(partidos, stats, temporada, antes_de), temporada


# ---------- reporte ----------


def _celda(perfil: dict, clave: str) -> str:
    decimales = ESTADISTICAS[clave][2]
    signo = "+" if clave == "dif_balones" else ""
    return f"{perfil[clave]:{signo}.{decimales}f} (#{perfil['rank'][clave]})"


def _tabla(perfiles: dict[str, dict], claves: list[str], ordenar_por: str) -> list[str]:
    encabezados = ["Equipo", "PJ"] + [ESTADISTICAS[c][0] for c in claves]
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    orden = sorted(perfiles, key=lambda e: (perfiles[e]["rank"][ordenar_por], e))
    for equipo in orden:
        p = perfiles[equipo]
        celdas = [equipo, str(p["partidos"])] + [_celda(p, c) for c in claves]
        lineas.append("| " + " | ".join(celdas) + " |")
    return lineas + [""]


def escribir_reporte(perfiles: dict[str, dict], temporada: int) -> str:
    lineas = [
        f"# Perfiles de equipo: temporada {temporada}",
        "",
        f"Generado: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Promedios por partido, solo temporada regular.",
        "- Entre paréntesis, el lugar entre los equipos de la liga (#1 es el mejor).",
        "- En defensa, #1 es quien menos permite.",
        "- Las yardas pueden diferir por poco de las cifras oficiales.",
        "",
        "## Ofensiva",
        "",
    ]
    lineas += _tabla(perfiles, OFENSIVA, "yds")
    lineas += ["## Defensiva", ""]
    lineas += _tabla(perfiles, DEFENSIVA, "yds_perm")
    lineas += ["## Presión y balones", ""]
    lineas += _tabla(perfiles, BALONES, "dif_balones")
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perfiles y rankings de equipo.")
    parser.add_argument("--temporada", type=int)
    args = parser.parse_args()

    perfiles, temporada = cargar(args.temporada)
    if not perfiles:
        print(f"No hay partidos jugados con estadísticas en la temporada {temporada}.")
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = REPORTS_DIR / f"perfiles_{temporada}.md"
    ruta.write_text(escribir_reporte(perfiles, temporada), "utf-8")
    print(f"{len(perfiles)} equipos. Reporte en {ruta}")


if __name__ == "__main__":
    main()
