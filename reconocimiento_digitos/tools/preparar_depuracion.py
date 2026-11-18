"""Prepara las muestras de prueba que usa tools/depurar_persistencia.py."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from utils.evaluation import particionar_datos  # noqa: E402

digits = load_digits()
X = (digits.data / 16.0).astype("float32")
y = digits.target
_, _, (X_prueba, y_prueba, idx_prueba) = particionar_datos(X, y, 42)
np.savez(RAIZ / "results" / "muestras_depuracion.npz", X=X_prueba, y=y_prueba, idx=idx_prueba)
print("Muestras preparadas:", X_prueba.shape)
