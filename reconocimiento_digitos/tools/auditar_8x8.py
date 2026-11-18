"""Auditoría numérica: compara el 8x8 de los dibujos con las imágenes de load_digits.

Uso:  python tools/auditar_8x8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))

import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from benchmark_dibujos import dibujo  # noqa: E402
from utils.preprocessing import preprocesar_imagen  # noqa: E402

np.set_printoptions(linewidth=200)


def stats(nombre, matrices):
    matrices = np.asarray(matrices)
    print(f"{nombre}: activos(>8)={float((matrices > 8).sum(axis=(1, 2)).mean()):.1f} "
          f"| suma media={matrices.sum(axis=(1, 2)).mean():.1f} "
          f"| píxeles=0: {float((matrices == 0).mean()) * 100:.0f}% "
          f"| fila superior media={matrices[:, 0].mean():.2f} "
          f"| fila inferior media={matrices[:, 7].mean():.2f} "
          f"| col izquierda={matrices[:, :, 0].mean():.2f}")


d = load_digits()
stats("load_digits   ", d.images)
stats("nuestros dibujos", [preprocesar_imagen(dibujo(k))[1] for k in range(10)])

for k in (2, 4, 8, 9):
    idx = np.where(d.target == k)[0][0]
    print(f"\n=== dígito {k} ===")
    print("load_digits:")
    print(d.images[idx].astype(int))
    v, m = preprocesar_imagen(dibujo(k))
    print("nuestro dibujo procesado:")
    print(m.astype(int))
    print("centro de masa dataset:", np.round(np.array(np.nonzero(d.images[idx] > 8)).mean(axis=1), 2))
    filas, cols = np.nonzero(m > 8)
    print("centro de masa dibujo :", (round(filas.mean(), 2), round(cols.mean(), 2)))
