"""Utilidades del proyecto de reconocimiento de dígitos."""

from pathlib import Path

# Raíz del proyecto (carpeta que contiene app.py, train.py, models/, results/ ...)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "modelo_final.keras"
METADATA_PATH = MODELS_DIR / "metadata.json"
CARACTERISTICAS_PATH = MODELS_DIR / "calibracion.json"
MODEL_PATH_V1 = MODELS_DIR / "modelo_final_v1_load_digits.keras"
HISTORIAL_PATH = DATA_DIR / "historial.csv"
BANCO_DIR = DATA_DIR / "banco_dibujos"

# Constantes compartidas entre entrenamiento y aplicación.
N_PIXELES = 64
N_CLASES = 10
ESCALA_PIXELES = 16.0       # load_digits usa enteros de 0 a 16
FACTOR_NORMALIZACION = 16.0
UMBRAL_CONFIANZA = 0.70
SEMILLA = 42
