"""Preprocesamiento de los dibujos hechos con el ratón.

El modelo se entrenó con ``load_digits``:
    - imágenes de 8x8 píxeles,
    - dígito CLARO sobre fondo OSCURO,
    - valores enteros de 0 a 16.

El lienzo de Tkinter, en cambio, es grande, tiene fondo BLANCO y trazo OSCURO.
Por eso el dibujo se transforma con el mismo procedimiento en todas las
predicciones:

    1. Obtener la imagen del lienzo (Pillow).
    2. Detectar el área con trazo y recortar los márgenes vacíos.
    3. Invertir colores (dígito claro sobre fondo oscuro).
    4. Redimensionar proporcionalmente para que ocupe ~6x6 píxeles.
    5. Centrar en un lienzo de 8x8.
    6. Convertir a grises y ajustar la escala a 0-16.
    7. Dividir entre 16.0 y aplanar a un vector de 64 valores.

El remuestreo se hace con NumPy (promedio de cajas, equivalente a
``Image.Resampling.BOX``). Se implementa así para que el resultado sea el mismo
en cualquier versión o compilación de Pillow, sin depender de que el filtro
C esté disponible.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from utils import ESCALA_PIXELES, FACTOR_NORMALIZACION, N_PIXELES

TAMANO_MODELO = 8          # lado de la imagen que espera la red
TAMANO_DIGITO = 6          # caja máxima (6x6) que ocupa el dígito dentro de 8x8
UMBRAL_TRAZO = 200         # cualquier píxel más oscuro que esto cuenta como trazo


def lienzo_vacio(imagen: Image.Image, umbral: int = UMBRAL_TRAZO) -> bool:
    """Devuelve True si la imagen (modo L) no tiene ningún píxel de trazo."""
    arreglo = np.asarray(imagen.convert("L"), dtype=np.uint8)
    return bool(arreglo.min(initial=255) >= umbral)


def recortar_trazo(imagen: Image.Image, umbral: int = UMBRAL_TRAZO) -> Image.Image | None:
    """Recorta la caja que contiene el trazo. Devuelve None si no hay trazo."""
    arreglo = np.asarray(imagen.convert("L"), dtype=np.uint8)
    mascara = arreglo < umbral
    if not mascara.any():
        return None
    filas = np.where(mascara.any(axis=1))[0]
    columnas = np.where(mascara.any(axis=0))[0]
    caja = (int(columnas[0]), int(filas[0]), int(columnas[-1]) + 1, int(filas[-1]) + 1)
    return imagen.convert("L").crop(caja)


def redimensionar_promedio(arreglo: np.ndarray, nuevo_alto: int, nuevo_ancho: int) -> np.ndarray:
    """Reduce una imagen promediando los píxeles de cada bloque (tipo BOX).

    A diferencia de tomar un solo píxel (vecino más cercano), el promedio
    conserva la información de trazos que caen entre dos píxeles nuevos.
    """
    alto, ancho = arreglo.shape
    filas = np.linspace(0, alto, nuevo_alto + 1).round().astype(int)
    columnas = np.linspace(0, ancho, nuevo_ancho + 1).round().astype(int)
    filas[-1], columnas[-1] = alto, ancho
    salida = np.zeros((nuevo_alto, nuevo_ancho), dtype=np.float32)
    for i in range(nuevo_alto):
        for j in range(nuevo_ancho):
            bloque = arreglo[filas[i]:filas[i + 1], columnas[j]:columnas[j + 1]]
            salida[i, j] = bloque.mean() if bloque.size else 0.0
    return salida


def preprocesar_imagen(imagen: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Convierte el dibujo del lienzo en el vector (64,) que consume la red.

    Devuelve ``(vector, imagen_8x8)`` donde ``vector`` son 64 valores
    normalizados (0-1) e ``imagen_8x8`` es la matriz de 8x8 en la escala 0-16
    (se usa para mostrar en la interfaz los píxeles reales que recibe la red).
    """
    recorte = recortar_trazo(imagen)
    if recorte is None:
        raise ValueError("El lienzo está vacío: no hay ningún trazo para reconocer.")

    # Invertir: trazo oscuro sobre blanco -> dígito claro sobre fondo oscuro.
    arreglo = 255.0 - np.asarray(recorte, dtype=np.float32)

    # Redimensionar conservando la proporción dentro de la caja de 6x6.
    alto, ancho = arreglo.shape
    escala = TAMANO_DIGITO / max(alto, ancho)
    nuevo_ancho = max(1, min(TAMANO_DIGITO - 1, int(round(ancho * escala))))
    nuevo_alto = max(1, min(TAMANO_DIGITO - 1, int(round(alto * escala))))
    reducido = redimensionar_promedio(arreglo, nuevo_alto, nuevo_ancho)

    # Centrar en 8x8 (el resto del lienzo queda negro = 0).
    lienzo_8 = np.zeros((TAMANO_MODELO, TAMANO_MODELO), dtype=np.float32)
    fila = (TAMANO_MODELO - nuevo_alto) // 2
    columna = (TAMANO_MODELO - nuevo_ancho) // 2
    lienzo_8[fila:fila + nuevo_alto, columna:columna + nuevo_ancho] = reducido

    # Escala real del dataset: enteros de 0 a 16.
    matriz_16 = np.clip(np.round(lienzo_8 * ESCALA_PIXELES / 255.0), 0, ESCALA_PIXELES)

    vector = (matriz_16 / FACTOR_NORMALIZACION).reshape(N_PIXELES).astype("float32")
    return vector, matriz_16


def vector_a_imagen_grande(vector: np.ndarray, factor: int = 24) -> np.ndarray:
    """Expande el vector (64,) a una matriz 8x8 ampliada para visualizarla."""
    return np.reshape(vector, (TAMANO_MODELO, TAMANO_MODELO)) * factor
