# nfl-model

Modelo que estima la probabilidad de victoria de cada partido de la NFL, la compara con la línea de Las Vegas y lleva un historial de aciertos por rango de confianza.

Experimento de ciencia de datos. No es consejo de apuestas.

## Instalación (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest
```

Requiere Python 3.10 o superior.

## Rutina semanal

Con el entorno activado:

```powershell
python -m nfl_model.track      # martes: califica la semana que terminó
python -m nfl_model.predict    # miércoles: picks de la semana que viene
python -m nfl_model.content    # imágenes 1080x1920 en reports/contenido/
```

- `picks/picks_log.csv` es el registro permanente: solo se le agregan filas.
- `reports/semana_<temporada>_<NN>.md` trae la tabla de la semana.
- `reports/historial.md` trae la marca acumulada por rango de confianza.

## Estructura

```
src/nfl_model/   código del modelo
tests/           pruebas
data/            datos descargados (fuera de git)
picks/           historial de picks
reports/         tablas y gráficas semanales
```

## Datos

nflverse, vía `nflreadpy`. Públicos, sin llave de API.
