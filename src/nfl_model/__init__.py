"""Modelo de probabilidad de victoria para partidos de la NFL."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PICKS_DIR = ROOT / "picks"
REPORTS_DIR = ROOT / "reports"
