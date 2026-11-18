"""Banco de dibujos sintéticos multi-estilo, SEPARADO de los conjuntos oficiales.

Conjunto de trabajo (nunca se usa para reportar métricas de load_digits):

  * 10 trazos base por dígito (``TRAZOS_ESTANDAR``), dibujados con pincel de
    9, 12, 16, 20 y 26 px, con pequeñas variaciones de posición y tamaño.
  * 6 variantes por dígito de dígitos de load_digits «re-dibujados» con estilo
    de lienzo (``utils.rendering``), con giro, escala y pincel aleatorios.

Estas imágenes solo se pueden usar como datos ADICIONALES de entrenamiento en un
experimento documentado (``train.py --usar-banco-dibujos``) y como evaluación
diagnóstica interna. La partición oficial train/validación/prueba de
``load_digits`` se mantiene intacta y sus métricas se reportan por separado.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from utils.rendering import _escalar, _expandir, _girar, convertir_matriz_a_lienzo

TRAZOS_ESTANDAR = {
    0: [[(0.50, 0.12), (0.22, 0.28), (0.18, 0.55), (0.28, 0.84), (0.50, 0.90),
         (0.72, 0.84), (0.82, 0.55), (0.78, 0.28), (0.50, 0.12)]],
    1: [[(0.40, 0.30), (0.55, 0.10), (0.55, 0.90)]],
    2: [[(0.25, 0.25), (0.48, 0.12), (0.68, 0.22), (0.66, 0.40), (0.48, 0.55),
         (0.30, 0.66), (0.26, 0.78), (0.26, 0.84), (0.75, 0.86)]],
    3: [[(0.22, 0.16), (0.62, 0.10), (0.76, 0.28), (0.50, 0.46), (0.76, 0.62),
         (0.66, 0.88), (0.24, 0.84)]],
    4: [[(0.66, 0.10), (0.22, 0.62), (0.80, 0.62)], [(0.66, 0.10), (0.66, 0.90)]],
    5: [[(0.76, 0.10), (0.28, 0.10), (0.26, 0.44), (0.58, 0.40), (0.78, 0.60),
         (0.70, 0.86), (0.22, 0.86)]],
    6: [[(0.70, 0.10), (0.40, 0.32), (0.26, 0.66), (0.36, 0.88), (0.62, 0.88),
         (0.74, 0.66), (0.56, 0.50), (0.30, 0.58)]],
    7: [[(0.20, 0.12), (0.80, 0.12), (0.44, 0.90)]],
    8: [[(0.50, 0.10), (0.28, 0.26), (0.34, 0.48), (0.50, 0.50), (0.68, 0.48),
         (0.74, 0.26), (0.50, 0.10)], [(0.50, 0.50), (0.26, 0.64), (0.32, 0.86),
         (0.50, 0.90), (0.70, 0.86), (0.74, 0.64), (0.50, 0.50)]],
    9: [[(0.74, 0.30), (0.50, 0.10), (0.26, 0.26), (0.32, 0.46), (0.56, 0.50),
         (0.74, 0.36), (0.70, 0.62), (0.62, 0.90)]],
}

PINCELES = (9, 12, 16, 20, 26)


def dibujar_pistas(trazos, lado: int = 320, pincel: int = 18, dx: float = 0.0,
                   dy: float = 0.0, escala: float = 1.0) -> Image.Image:
    """Dibuja trazos relativos (0-1) en un lienzo modo L blanco con trazo negro."""
    imagen = Image.new("L", (lado, lado), 255)
    d = ImageDraw.Draw(imagen)
    for trazo in trazos:
        puntos = [((x + dx) * lado, (y + dy) * lado) for x, y in trazo]
        if escala != 1.0:
            cx, cy = lado / 2, lado / 2
            puntos = [(cx + (x - cx) * escala, cy + (y - cy) * escala) for x, y in puntos]
        d.line(puntos, fill=0, width=pincel, joint="curve")
        for x, y in puntos:
            d.ellipse([x - pincel / 2, y - pincel / 2, x + pincel / 2, y + pincel / 2], fill=0)
    return imagen


def banco_sintetico(semilla: int = 1234, variantes_por_pincel: int = 1,
                    variantes_redibujadas: int = 6) -> list[tuple[int, Image.Image, str]]:
    """Devuelve (etiqueta, lienzo L, origen). Origen: 'pista' o 'redibujado'."""
    from sklearn.datasets import load_digits

    generador = np.random.default_rng(semilla)
    muestras = []
    for k, trazos in TRAZOS_ESTANDAR.items():
        for pincel in PINCELES:
            for _ in range(variantes_por_pincel):
                lienzo = dibujar_pistas(
                    trazos,
                    pincel=pincel,
                    dx=float(generador.uniform(-0.05, 0.05)),
                    dy=float(generador.uniform(-0.05, 0.05)),
                    escala=float(generador.uniform(0.85, 1.10)),
                )
                muestras.append((k, lienzo, "pista"))

    d = load_digits()
    for k in range(10):
        indices = np.where(d.target == k)[0]
        elegidos = generador.choice(indices, size=variantes_redibujadas, replace=False)
        for i in elegidos:
            lienzo = convertir_matriz_a_lienzo(
                d.images[i], lado=320, margen=float(generador.uniform(0.15, 0.30))
            )
            lienzo = _expandir(lienzo, float(generador.uniform(0.6, 1.8)))
            lienzo = _girar(lienzo, float(generador.uniform(-8, 8)))
            lienzo = _escalar(lienzo, float(generador.uniform(0.88, 1.10)))
            muestras.append((k, lienzo, "redibujado"))
    return muestras


def matrices_del_banco(semilla: int = 1234):
    """Procesa el banco y devuelve X (n,64), y (n,) y el origen de cada muestra."""
    from utils.preprocessing import preprocesar_imagen, recortar_trazo

    X, y, origen = [], [], []
    for etiqueta, lienzo, tipo in banco_sintetico(semilla):
        if recortar_trazo(lienzo) is None:
            continue
        vector, _ = preprocesar_imagen(lienzo)
        X.append(vector)
        y.append(etiqueta)
        origen.append(tipo)
    return np.array(X, dtype="float32"), np.array(y, dtype=np.int64), np.array(origen)
