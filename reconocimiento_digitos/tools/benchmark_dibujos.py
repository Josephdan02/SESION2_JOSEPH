"""Benchmark objetivo: trazo tipo ratón -> preprocesamiento -> modelo.

Genera dibujos con trazo GRUESO y redondeado (como el de la app) de los diez
dígitos, los pasa por ``utils.preprocessing.preprocesar_imagen`` y mide aciertos
con el modelo guardado. Sirve para comparar versiones del preprocesamiento de
forma reproducible; NO es la prueba manual de 20 dibujos.

Uso:  python tools/benchmark_dibujos.py [ruta_modelo]
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

PISTAS = {
    0: [[(0.50, 0.12), (0.22, 0.28), (0.18, 0.55), (0.28, 0.84), (0.50, 0.90),
         (0.72, 0.84), (0.82, 0.55), (0.78, 0.28), (0.50, 0.12)]],
    1: [[(0.40, 0.30), (0.55, 0.10), (0.55, 0.90)]],
    2: [[(0.25, 0.25), (0.48, 0.12), (0.68, 0.22), (0.66, 0.40), (0.45, 0.58),
         (0.28, 0.76), (0.26, 0.84), (0.75, 0.86)]],
    3: [[(0.22, 0.16), (0.62, 0.10), (0.76, 0.28), (0.50, 0.46), (0.76, 0.62),
         (0.66, 0.88), (0.24, 0.84)]],
    4: [[(0.66, 0.10), (0.22, 0.62), (0.80, 0.62)], [(0.66, 0.10), (0.66, 0.90)]],
    5: [[(0.76, 0.10), (0.28, 0.10), (0.26, 0.44), (0.58, 0.40), (0.78, 0.60),
         (0.70, 0.86), (0.22, 0.86)]],
    6: [[(0.70, 0.10), (0.40, 0.32), (0.26, 0.66), (0.36, 0.88), (0.62, 0.88),
         (0.74, 0.66), (0.56, 0.50), (0.30, 0.58)]],
    7: [[(0.20, 0.12), (0.80, 0.12), (0.44, 0.90)]],
    8: [[(0.50, 0.08), (0.34, 0.16), (0.30, 0.30), (0.38, 0.42), (0.50, 0.47), (0.62, 0.42),
         (0.70, 0.30), (0.66, 0.16), (0.50, 0.08)], [(0.50, 0.47), (0.32, 0.56),
         (0.27, 0.72), (0.36, 0.86), (0.50, 0.91), (0.64, 0.86), (0.73, 0.72), (0.68, 0.56),
         (0.50, 0.47)]],
    9: [[(0.74, 0.30), (0.50, 0.10), (0.26, 0.26), (0.32, 0.46), (0.56, 0.50),
         (0.74, 0.36), (0.70, 0.62), (0.62, 0.90)]],
}


def dibujo(digito: int, lado: int = 320, pincel: int = 18) -> Image.Image:
    """Dibujo en modo L (fondo blanco, trazo negro) imitando el lienzo."""
    imagen = Image.new("L", (lado, lado), 255)
    d = ImageDraw.Draw(imagen)
    for trazo in PISTAS[digito]:
        puntos = [(x * lado, y * lado) for x, y in trazo]
        d.line(puntos, fill=0, width=pincel, joint="curve")
        for x, y in puntos:  # extremos redondeados como capstyle=ROUND
            d.ellipse([x - pincel / 2, y - pincel / 2, x + pincel / 2, y + pincel / 2], fill=0)
    return imagen


def evaluar(modelo, preprocesar, pincel: int = 18, lado: int = 320) -> dict:
    detalle = {}
    for digito in range(10):
        vector, matriz8 = preprocesar(dibujo(digito, lado, pincel))
        probs = modelo.predict(vector.reshape(1, -1), verbose=0)[0]
        detalle[digito] = {
            "predicho": int(np.argmax(probs)),
            "probabilidad": float(probs.max()),
            "pixeles_activos": int((matriz8 > 8).sum()),
            "matriz": matriz8.astype(int).tolist(),
        }
    aciertos = sum(1 for d, v in detalle.items() if v["predicho"] == d)
    return {"aciertos": aciertos, "total": 10, "detalle": detalle}


def main() -> int:
    import tensorflow as tf

    from utils.preprocessing import preprocesar_imagen

    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "models" / "modelo_final.keras"
    modelo = tf.keras.models.load_model(ruta)
    print(f"Modelo: {ruta.name}")
    for pincel in (12, 18, 24):
        res = evaluar(modelo, preprocesar_imagen, pincel=pincel)
        pares = " ".join(f"{d}->{v['predicho']}" for d, v in res["detalle"].items())
        print(f"pincel {pincel:2d}px: {res['aciertos']}/10  [{pares}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
