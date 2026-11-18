"""Compara modelos (v1 load_digits vs v2 aumentado) sobre los MISMOS dibujos.

Uso:  python tools/comparar_modelos_dibujos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))

# Parámetros que train.py puede importar SIN que se ejecute el experimento.
MODELOS = {
    "v1_solo_load_digits": RAIZ / "models" / "modelo_final_v1_load_digits.keras",
    "v2_aumentado": RAIZ / "models" / "modelo_final.keras",
}


def main():
    import numpy as np
    import tensorflow as tf

    from benchmark_dibujos import dibujo
    from utils.preprocessing import preprocesar_imagen

    for nombre, ruta in MODELOS.items():
        if not ruta.exists():
            print(f"(falta {ruta})")
            continue
        modelo = tf.keras.models.load_model(ruta)
        for pincel in (12, 18, 24):
            salida, ok = [], 0
            for k in range(10):
                v, _ = preprocesar_imagen(dibujo(k, pincel=pincel))
                p = int(np.argmax(modelo.predict(v.reshape(1, -1), verbose=0)[0]))
                ok += p == k
                salida.append(f"{k}" if p == k else f"{k}->{p}")
            print(f"{nombre:18s} pincel {pincel:2d}: {ok}/10  " + " ".join(salida))


if __name__ == "__main__":
    main()
