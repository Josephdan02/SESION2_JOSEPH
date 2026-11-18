"""Herramienta de diagnóstico visual y numérico (punto 3 del enunciado).

Para cada dígito (por defecto 2, 4, 8 y 9) muestra, en una sola figura:

  1. El original de load_digits (vecino más cercano, escala de grises).
  2. El dibujo de ratón sintético tal como llega al preprocesamiento.
  3. El 8x8 final que recibe la red (ampliado con vecino más cercano).
  4. Un 8x8 de referencia del dataset, para comparar grosor y centrado.

Además imprime los valores numéricos de las matrices y la predicción real.

Uso:  python tools/depurar_dibujos.py [0 1 2 ...]
Salida: results/diagnostico_visual.png
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from benchmark_dibujos import dibujo  # noqa: E402
from utils import RESULTS_DIR, UMBRAL_CONFIANZA  # noqa: E402
from utils.preprocessing import preprocesar_imagen  # noqa: E402


def matriz_a_imagen(matriz: np.ndarray):
    """8x8 0-16 -> imagen ampliada con vecino más cercano (píxeles visibles)."""
    return np.kron(np.asarray(matriz, dtype=float), np.ones((10, 10)))


def main(digitos: list[int]) -> int:
    import tensorflow as tf

    modelo = tf.keras.models.load_model(RAIZ / "models" / "modelo_final.keras")
    d = load_digits()
    filas = len(digitos)
    fig, ejes = plt.subplots(filas, 4, figsize=(11, 2.6 * filas))
    ejes = np.atleast_2d(ejes)

    for fila, k in enumerate(digitos):
        original = d.images[int(np.where(d.target == k)[0][0])]
        lienzo = dibujo(k)
        vector, procesado = preprocesar_imagen(lienzo)
        probs = modelo.predict(vector.reshape(1, -1), verbose=0)[0]
        pred = int(np.argmax(probs))

        print(f"\n=== dígito {k} | predicho {pred} ({probs.max():.2%}) | "
              f"activos(>8) original={int((original > 8).sum())} procesado={int((procesado > 8).sum())} ===")
        print("8x8 procesado:")
        print(procesado.astype(int))
        print("top-3:", [(int(i), round(float(probs[i]), 4)) for i in np.argsort(probs)[::-1][:3]])

        ejes[fila, 0].imshow(original, cmap="gray_r", vmin=0, vmax=16)
        ejes[fila, 0].set_title(f"load_digits ({k})")
        ejes[fila, 1].imshow(np.asarray(lienzo), cmap="gray", vmin=0, vmax=255)
        ejes[fila, 1].set_title("dibujo con ratón")
        ejes[fila, 2].imshow(matriz_a_imagen(procesado), cmap="gray_r", vmin=0, vmax=16)
        ejes[fila, 2].set_title(f"8x8 que ve la red -> {pred}")
        ejes[fila, 3].imshow(np.kron(original, np.ones((10, 10))), cmap="gray_r", vmin=0, vmax=16)
        ejes[fila, 3].set_title("referencia del dataset")
        if probs.max() < UMBRAL_CONFIANZA:
            ejes[fila, 2].set_xlabel("baja confianza", fontsize=8)
        for eje in ejes[fila]:
            eje.set_xticks([])
            eje.set_yticks([])

    fig.suptitle("Diagnóstico: dibujo del lienzo vs. imágenes de load_digits", fontsize=12)
    fig.tight_layout()
    ruta = RESULTS_DIR / "diagnostico_visual.png"
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    print(f"\nFigura guardada en {ruta}")
    return 0


if __name__ == "__main__":
    argumentos = [int(a) for a in sys.argv[1:]] or [2, 4, 8, 9]
    sys.exit(main(argumentos))
