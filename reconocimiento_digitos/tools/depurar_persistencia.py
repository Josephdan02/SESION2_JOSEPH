"""Depuración manual: ejecuta la prueba de persistencia en primer plano.

Uso:  python tools/depurar_persistencia.py
Es una herramienta de diagnóstico; no forma parte del flujo normal.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

from train import prueba_persistencia  # noqa: E402
from utils import RESULTS_DIR  # noqa: E402


def main() -> int:
    modelo = __import__("tensorflow").keras.models.load_model(RAIZ / "models" / "modelo_final.keras")
    prueba = np.load(RESULTS_DIR / "muestras_depuracion.npz")
    X = prueba["X"]
    y = prueba["y"]
    idx = prueba["idx"]
    probs = modelo.predict(X, verbose=0)
    resultado = prueba_persistencia(X, y, idx, probs, RESULTS_DIR / "prueba_persistencia.json")
    print(resultado["aprobada"], resultado["diferencia_maxima_absoluta"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
