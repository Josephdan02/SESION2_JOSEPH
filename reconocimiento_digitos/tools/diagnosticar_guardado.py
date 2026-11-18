"""¿Por qué las probabilidades cambian tras guardar y recargar el modelo?

Compara, con las mismas muestras:
  A. modelo en memoria (justo tras entrenar),
  B. el mismo objeto tras ``save()`` + ``load_model()`` (mismo proceso),
  C. el modelo cargado en un proceso distinto.

Uso:  python tools/diagnosticar_guardado.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
from sklearn.datasets import load_digits
from tensorflow import keras

from utils.evaluation import particionar_datos
from train import CONFIG_FUNCTIONAL, CONFIG_SEQUENTIAL, construir_functional, construir_sequential, entrenar

digits = load_digits()
X = (digits.data / 16.0).astype("float32")
y = digits.target
(Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _) = particionar_datos(X, y, 42)
ytr_cat = keras.utils.to_categorical(ytr, 10)
yv_cat = keras.utils.to_categorical(yv, 10)
muestras = Xte[:20]

for nombre, constructor, config in (("Sequential", construir_sequential, CONFIG_SEQUENTIAL),
                                    ("Functional", construir_functional, CONFIG_FUNCTIONAL)):
    modelo = constructor(config)
    entrenar(modelo, Xtr, ytr_cat, Xv, yv_cat, nombre)
    p_mem = modelo.predict(muestras, verbose=0)
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "m.keras"
        modelo.save(ruta)
        p_rec = keras.models.load_model(ruta).predict(muestras, verbose=0)
    dif = np.abs(p_mem - p_rec).max()
    print(f"{nombre}: diferencia memoria vs recarga = {dif:.3e}")
    print(f"  ¿el modelo tiene BatchNorm? {any('Batch' in type(l).__name__ for l in modelo.layers)}")
    print(f"  tipo de capas: {[type(l).__name__ for l in modelo.layers]}")
