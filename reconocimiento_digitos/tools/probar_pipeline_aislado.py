"""¿El pipeline actual (v2) conserva la información de un dígito del dataset?

Convierte cada dígito de load_digits al estilo del lienzo (ampliado, trazo negro,
grande) y lo vuelve a pasar por ``preprocesar_imagen``. Si el pipeline es correcto,
el 8x8 resultante debe parecerse al original. Si no, la pérdida está en el
preprocesamiento y no en el modelo.

Uso:  python tools/probar_pipeline_aislado.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
from sklearn.datasets import load_digits

from utils.preprocessing import preprocesar_imagen
from utils.rendering import convertir_matriz_a_lienzo, _expandir

np.set_printoptions(linewidth=200)
d = load_digits()

for k in (2, 4, 8, 9):
    i = int(np.where(d.target == k)[0][0])
    original = d.images[i]
    lienzo = _expandir(convertir_matriz_a_lienzo(original, lado=320, margen=0.18), 1.35)
    _, procesada = preprocesar_imagen(lienzo)
    corr = float(np.corrcoef(original.ravel(), procesada.ravel())[0, 1])
    print(f"=== dígito {k} ===")
    print("original:")
    print(original.astype(int))
    print("tras lienzo + pipeline:")
    print(procesada.astype(int))
    print(f"activos(>8): original={int((original > 8).sum())} procesado={int((procesada > 8).sum())} "
          f"| correlación={corr:.3f}")
    print()
