"""Preprocesamiento de los dibujos hechos con el ratón.

El modelo de ``load_digits`` espera:
    - imágenes de 8x8 píxeles,
    - dígito CLARO sobre fondo OSCURO (0 = sin trazo, 16 = trazo intenso),
    - poco margen: el dígito ocupa de 5 a 8 filas/columnas y ~19 píxeles activos.

El lienzo de Tkinter es grande, tiene fondo BLANCO y trazo OSCURO grueso.

La auditoría (``tools/auditar_8x8.py``) encontró dos causas reales:

  1. **Reducción que perdía el trazo**: al promediar bloques, un trazo de 18 px
     sobre 320 queda en ~2 de 16 (por debajo del umbral activo). El resultado
     tenía 0.7 píxeles activos frente a 18.7 del dataset.
  2. **Escala fija**: el dígito se encogía a 6 px y dejaba márgenes que el
     dataset no tiene; el "grosor relativo" quedaba muy fino.

Procedimiento (idéntico en TODAS las predicciones):

    1. Recortar la caja del trazo.
    2. Invertir colores (dígito claro sobre fondo oscuro).
    3. Normalizar el grosor: se engrosa el trazo hasta el grosor relativo típico
       del dataset (``grosor_relativo`` = pluma / lado de la caja).
    4. Redimensionar de forma proporcional a la caja típica del dataset.
    5. Centrar con el centro de masa típico del dataset.
    6. Convertir a grises, llevar a 0-16 y aplanar a 64 valores.

``tools/probar_pipeline_aislado.py`` comprueba que un dígito del dataset, pasado
por este pipeline como si se hubiera dibujado, se conserve (misma cantidad de
píxeles activos y forma similar). Los parámetros viven en
``models/calibracion.json`` y se calculan SOLO con el conjunto de entrenamiento.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from utils import CARACTERISTICAS_PATH, ESCALA_PIXELES, FACTOR_NORMALIZACION, N_PIXELES

TAMANO_MODELO = 8
UMBRAL_TRAZO = 200          # píxel más oscuro que esto = trazo del usuario
UMBRAL_ACTIVO = 8           # valor >= 8 en la escala 0-16 = píxel activo

# Valores de partida (se sustituyen por los medidos en entrenamiento).
DEFAULTS = {
    "lado_caja": 7.0,            # lado medio de la caja del dígito en load_digits
    "grosor_relativo": 0.42,     # pluma / lado de la caja
    "centro_masa_fila": 3.5,
    "centro_masa_columna": 3.5,
}

_calibracion: dict | None = None


def cargar_calibracion() -> dict:
    """Lee models/calibracion.json (calculado SOLO con entrenamiento) si existe."""
    global _calibracion
    if _calibracion is None:
        valores = dict(DEFAULTS)
        if CARACTERISTICAS_PATH.exists():
            import json

            datos = json.loads(CARACTERISTICAS_PATH.read_text(encoding="utf-8"))
            for clave in DEFAULTS:
                if clave in datos:
                    valores[clave] = float(datos[clave])
        _calibracion = valores
    return _calibracion


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
    """Reduce una imagen promediando bloques (equivalente a Image.Resampling.BOX).

    Se implementa en NumPy porque el filtro de Pillow de esta instalación no
    promedia al reducir. El promedio conserva la energía del trazo y produce los
    grises intermedios que caracterizan al dataset.
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


def grosor_trazo(arreglo: np.ndarray) -> float:
    """Grosor aproximado del trazo (px) = área del trazo / longitud de su eje.

    La longitud se aproxima con el perímetro de la caja dividido entre 2, que es
    razonable para trazos abiertos (1, 2, 4, 7) y algo menor para curvas.
    """
    binaria = arreglo >= 128
    area = float(binaria.sum())
    if area <= 0:
        return 0.0
    filas = np.where(binaria.any(axis=1))[0]
    columnas = np.where(binaria.any(axis=0))[0]
    alto = float(filas[-1] - filas[0] + 1)
    ancho = float(columnas[-1] - columnas[0] + 1)
    longitud = alto + ancho
    return area / longitud if longitud > 0 else float(np.sqrt(area))


def engrosar_trazo(arreglo: np.ndarray, extra_px: float) -> np.ndarray:
    """Engrosa el trazo ``extra_px`` píxeles en la resolución original.

    Se dilata la máscara del trazo con un núcleo circular (distancia euclídea),
    lo que aproxima el dibujo con un pincel más grueso sin cambiar la forma.
    """
    if extra_px <= 0:
        return arreglo
    radio = int(np.ceil(extra_px / 2.0))
    if radio <= 0:
        return arreglo
    binaria = arreglo >= 128
    acumulada = binaria.copy()
    for dy in range(-radio, radio + 1):
        for dx in range(-radio, radio + 1):
            if dy * dy + dx * dx > radio * radio:
                continue
            if dy == 0 and dx == 0:
                continue
            acumulada |= np.roll(np.roll(binaria, dy, axis=0), dx, axis=1)
    return np.where(acumulada, 255.0, arreglo)


def preprocesar_imagen(imagen: Image.Image, calibracion: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Convierte el dibujo del lienzo en el vector (64,) que consume la red.

    Devuelve ``(vector, imagen_8x8)``: 64 valores normalizados (0-1) y la matriz
    8x8 en escala 0-16 (la que se muestra en la interfaz).
    """
    cal = calibracion or cargar_calibracion()
    recorte = recortar_trazo(imagen)
    if recorte is None:
        raise ValueError("El lienzo está vacío: no hay ningún trazo para reconocer.")

    # Invertir: fondo blanco + trazo oscuro  ->  fondo oscuro + trazo claro.
    arreglo = 255.0 - np.asarray(recorte, dtype=np.float32)
    alto, ancho = arreglo.shape
    lado_mayor = max(alto, ancho)

    # 1) Normalizar el GROSOR RELATIVO: en load_digits la pluma mide ~0.42 del
    #    lado de la caja. Si el usuario dibuja un trazo más fino, se engrosa.
    grosor_relativo = float(cal["grosor_relativo"])
    grosor_actual = grosor_trazo(arreglo)
    objetivo = grosor_relativo * lado_mayor
    if 0 < grosor_actual < objetivo * 0.85:
        arreglo = engrosar_trazo(arreglo, objetivo - grosor_actual)

    # 2) Redimensionar de forma proporcional a la caja típica del dataset.
    lado_objetivo = float(np.clip(cal["lado_caja"], 3.0, TAMANO_MODELO))
    escala = lado_objetivo / lado_mayor
    nuevo_ancho = int(np.clip(round(ancho * escala), 1, TAMANO_MODELO))
    nuevo_alto = int(np.clip(round(alto * escala), 1, TAMANO_MODELO))
    reducido = redimensionar_promedio(arreglo, nuevo_alto, nuevo_ancho)

    # 3) Centrar con el centro de masa típico del dataset (mismo encuadre).
    lienzo_8 = np.zeros((TAMANO_MODELO, TAMANO_MODELO), dtype=np.float32)
    fila_cm, col_cm = _centro_masa(reducido)
    fila0 = int(round(cal["centro_masa_fila"] - fila_cm))
    col0 = int(round(cal["centro_masa_columna"] - col_cm))
    fila0 = int(np.clip(fila0, 0, TAMANO_MODELO - nuevo_alto))
    col0 = int(np.clip(col0, 0, TAMANO_MODELO - nuevo_ancho))
    lienzo_8[fila0:fila0 + nuevo_alto, col0:col0 + nuevo_ancho] = reducido

    matriz_16 = np.clip(np.round(lienzo_8 * ESCALA_PIXELES / 255.0), 0, ESCALA_PIXELES)
    vector = (matriz_16 / FACTOR_NORMALIZACION).reshape(N_PIXELES).astype("float32")
    return vector, matriz_16


def _centro_masa(matriz: np.ndarray) -> tuple[float, float]:
    total = float(matriz.sum())
    if total <= 0:
        return (matriz.shape[0] - 1) / 2.0, (matriz.shape[1] - 1) / 2.0
    filas, columnas = np.indices(matriz.shape)
    return float((filas * matriz).sum() / total), float((columnas * matriz).sum() / total)


def vector_a_imagen_grande(vector: np.ndarray, factor: int = 24) -> np.ndarray:
    """Expande el vector (64,) a una matriz 8x8 ampliada para visualizarla."""
    return np.reshape(vector, (TAMANO_MODELO, TAMANO_MODELO)) * factor


# ---------------------------------------------------------------------------
# Calibración con el conjunto de ENTRENAMIENTO (nunca validación ni prueba)
# ---------------------------------------------------------------------------

def medir_dataset(imagenes: np.ndarray) -> dict:
    """Medidas de caja, grosor y centro de masa de las imágenes 8x8 del dataset."""
    lados, grosores, centros = [], [], []
    for img in imagenes:
        activo = img >= UMBRAL_ACTIVO
        if not activo.any():
            continue
        filas = np.where(activo.any(axis=1))[0]
        columnas = np.where(activo.any(axis=0))[0]
        lado = max(filas[-1] - filas[0] + 1, columnas[-1] - columnas[0] + 1)
        lados.append(lado)
        # Grosor en la escala del 8x8: área / (alto + ancho) de la caja.
        alto = filas[-1] - filas[0] + 1
        ancho = columnas[-1] - columnas[0] + 1
        grosores.append(float(activo.sum()) / float(alto + ancho))
        centros.append(_centro_masa(img.astype(np.float32)))
    centros = np.array(centros)
    return {
        "lado_caja": float(np.median(lados)),
        "grosor_relativo": float(np.median(grosores) / np.median(lados)),
        "centro_masa_fila": float(centros[:, 0].mean()),
        "centro_masa_columna": float(centros[:, 1].mean()),
        "n_imagenes": int(len(lados)),
    }


def calibrar_preprocesamiento(imagenes_entrenamiento: np.ndarray) -> dict:
    """Calcula y guarda models/calibracion.json usando SOLO el conjunto de entrenamiento."""
    global _calibracion
    valores = medir_dataset(imagenes_entrenamiento)
    import json

    CARACTERISTICAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CARACTERISTICAS_PATH.write_text(json.dumps(valores, indent=2), encoding="utf-8")
    _calibracion = {k: float(v) for k, v in valores.items() if k in DEFAULTS}
    return valores
