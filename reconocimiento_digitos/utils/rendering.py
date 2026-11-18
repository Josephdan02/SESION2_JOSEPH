"""Renderizado de «escritura manual sintética» a partir de load_digits.

Las imágenes de ``load_digits`` son fotografías de dígitos manuscritos (escaneo
de 32x32 reducido a 8x8). Esa reducción es un PROMEDIO: al engrosar o desplazar
el trazo se obtienen valores intermedios, no trazos binarios.

Por eso la aumentación replica el proceso real en lugar de estirar la matriz 8x8:

    1. Se amplía el dígito original 8x8 -> 32x32 conservando su gris.
    2. Se aplica la transformación geométrica (desplazamiento, rotación, escala,
       grosor) en ALTA resolución, como haría alguien al escribir con el ratón.
    3. Se reduce de nuevo a 8x8 promediando bloques y se reescala a 0-16.

El resultado conserva la estadística del dataset (píxeles intermedios, trazo de
~2.8 px, dígito de ~6.6 px), cosa que las transformaciones sobre el 8x8 no logran.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from utils.preprocessing import lienzo_vacio
from sklearn.datasets import load_digits

FACTOR_SUPER = 4          # 8x8 -> 32x32 como en la fuente original
TAMANO_PEQUE = 8
# Transformación CAPACITANTE: emula cómo se vería el dígito del dataset dibujado
# con el ratón de la aplicación (recorte, inversión, escalado a ~6.6 px, centrado).
PASO_HABILITADO = True


def _a_imagen(matriz8: np.ndarray) -> Image.Image:
    """8x8 (0-16) -> imagen 32x32 en gris (poniendo a cero el ruido de fondo)."""
    limpia = np.clip(matriz8, 0, 16)
    limpia = np.where(limpia >= 1, limpia, 0.0)
    grande = Image.fromarray((limpia / 16.0 * 255).astype(np.uint8), mode="L")
    return grande.resize((TAMANO_PEQUE * FACTOR_SUPER, TAMANO_PEQUE * FACTOR_SUPER),
                         Image.Resampling.BICUBIC)


def _reducir(imagen: Image.Image) -> np.ndarray:
    """32x32 gris -> 8x8 con promedio de bloques (como la fuente original)."""
    arreglo = np.asarray(imagen.convert("L"), dtype=np.float32) / 255.0
    lado = arreglo.shape[0] // TAMANO_PEQUE
    bloques = arreglo.reshape(TAMANO_PEQUE, lado, TAMANO_PEQUE, lado).mean(axis=(1, 3))
    return np.clip(np.round(bloques * 16.0), 0, 16)


def transformar(
    matriz8: np.ndarray,
    dy: float = 0.0,
    dx: float = 0.0,
    angulo: float = 0.0,
    escala: float = 1.0,
    engrosar: float = 0.0,
) -> np.ndarray:
    """Devuelve una variante realista de la imagen 8x8 (valores 0-16).

    - ``dy`` y ``dx``: desplazamiento en píxeles del 8x8 (positivo = abajo/derecha).
    - ``angulo``: rotación en grados.
    - ``escala``: 1.0 = sin cambio (se aplica sobre el centro).
    - ``engrosar``: píxeles que se añade (o quita, si es negativo) al trazo.
    """
    grande = _a_imagen(matriz8)
    if angulo:
        grande = grande.rotate(
            angulo, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0
        )
    if escala != 1.0:
        lado = grande.size[0]
        nuevo = max(4, int(round(lado * escala)))
        escalada = grande.resize((nuevo, nuevo), Image.Resampling.BICUBIC)
        lienzo = Image.new("L", grande.size, 0)
        off = (lado - nuevo) // 2
        lienzo.paste(escalada, (off, off))
        grande = lienzo
    if engrosar:
        grande = engrosar_trazo(grande, engrosar)
    if dy or dx:
        desplazada = Image.new("L", grande.size, 0)
        desplazada.paste(grande, (int(round(dx * FACTOR_SUPER)), int(round(dy * FACTOR_SUPER))))
        grande = desplazada
    if grande.size[0] != TAMANO_PEQUE * FACTOR_SUPER:
        grande = grande.resize((TAMANO_PEQUE * FACTOR_SUPER,) * 2, Image.Resampling.BICUBIC)
    return _reducir(grande)


def engrosar_trazo(imagen_grande: Image.Image, px: float) -> Image.Image:
    """Engrosa (px > 0) o adelgaza (px < 0) un trazo en la imagen de 32x32.

    1. Desenfoque gaussiano: reparte la tinta y amplía el ancho aparente.
    2. Erosión (filtro Mínimo) proporcional al engrosado: recorta la aureola
       tenue para que el trazo quede nítido y no inunde el fondo.
    3. Renormalización a 255 para conservar la intensidad máxima.
    """
    from PIL import ImageFilter

    if px < 0:
        return imagen_grande.filter(ImageFilter.GaussianBlur(abs(px) * FACTOR_SUPER * 0.45))
    radio = px * FACTOR_SUPER * 0.55
    suave = imagen_grande.filter(ImageFilter.GaussianBlur(radio))
    erosion = int(round(radio * 0.6))
    if erosion >= 1:
        suave = suave.filter(ImageFilter.MinFilter(erosion * 2 + 1))
    arreglo = np.asarray(suave, dtype=np.float32)
    maximo = float(arreglo.max())
    if maximo > 0:
        arreglo = arreglo * (255.0 / maximo)
    return Image.fromarray(np.clip(arreglo, 0, 255).astype(np.uint8), mode="L")


def aumentacion_aleatoria(matriz8: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Una variante aleatoria moderada (la que se usa durante el entrenamiento)."""
    return transformar(
        matriz8,
        dy=float(generador.uniform(-1.0, 1.0)),
        dx=float(generador.uniform(-1.0, 1.0)),
        angulo=float(generador.uniform(-12.0, 12.0)),
        escala=float(generador.uniform(0.90, 1.10)),
        engrosar=float(generador.uniform(-0.5, 0.7)),
    )


def convertir_matriz_a_lienzo(matriz8: np.ndarray, lado: int = 320, margen: float = 0.12) -> Image.Image:
    """Convierte una imagen 8x8 (0-16) al estilo del lienzo: fondo blanco, trazo negro.

    El dígito se amplía con vecino más cercano (conserva la forma original) y se
    escala para ocupar la misma fracción del lienzo que ocupa en el 8x8.
    Útil para comparar un dibujo real con lo que escribe el usuario.
    """
    ref = np.clip(matriz8, 0, 16) / 16.0
    activo = ref >= (8 / 16.0)
    if not activo.any():
        return Image.new("L", (lado, lado), 255)
    filas = np.where(activo.any(axis=1))[0]
    columnas = np.where(activo.any(axis=0))[0]
    recorte = ref[filas[0]:filas[-1] + 1, columnas[0]:columnas[-1] + 1]
    alto, ancho = recorte.shape
    util = lado * (1.0 - 2 * margen)
    escala = util / max(alto, ancho)
    nuevo = (max(1, int(round(ancho * escala))), max(1, int(round(alto * escala))))
    pequena = Image.fromarray((recorte * 255).astype(np.uint8), mode="L")
    grande = pequena.resize(nuevo, Image.Resampling.NEAREST)
    lienzo = Image.new("L", (lado, lado), 255)
    lienzo.paste(Image.eval(grande, lambda v: 255 - v),
                 ((lado - nuevo[0]) // 2, (lado - nuevo[1]) // 2))
    return lienzo


def escribir_como_raton(matriz8: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Simula un dígito DIBUJADO con ratón a partir de uno de load_digits.

    ADVERTENCIA: si el pipeline se aplica sobre un dígito del dataset, la red
    aprendería a deshacer su propio preprocesamiento (data leakage) y el
    experimento dejaría de ser honesto. Por eso estas variantes se usan solo
    para medir el daño del pipeline y se EXCLUYEN del entrenamiento.
    """
    from utils.preprocessing import preprocesar_imagen

    lienzo = convertir_matriz_a_lienzo(matriz8, lado=320, margen=0.18)
    if lienzo_vacio(lienzo):
        return matriz8
    lienzo = _expandir(lienzo, 1.35)
    lienzo = _girar(lienzo, float(generador.uniform(-10.0, 10.0)))
    lienzo = _mover(lienzo, int(generador.uniform(-25, 25)), int(generador.uniform(-25, 25)))
    lienzo = _escalar(lienzo, float(generador.uniform(0.85, 1.12)))
    _, matriz = preprocesar_imagen(lienzo)
    return matriz


def _expandir(imagen: Image.Image, factor: float) -> Image.Image:
    """Engrosa el trazo oscuro sobre fondo blanco (versión simple del pincel)."""
    from PIL import ImageFilter

    invertida = Image.eval(imagen, lambda v: 255 - v)
    radio = max(1, int(round(factor * 3)))
    invertida = invertida.filter(ImageFilter.MaxFilter(radio * 2 + 1))
    invertida = invertida.filter(ImageFilter.GaussianBlur(1.2))
    return Image.eval(invertida, lambda v: 255 - v)


def _girar(imagen: Image.Image, grados: float) -> Image.Image:
    return imagen.rotate(grados, resample=Image.Resampling.BICUBIC, fillcolor=255)


def _mover(imagen: Image.Image, dx: int, dy: int) -> Image.Image:
    nueva = Image.new("L", imagen.size, 255)
    nueva.paste(imagen, (dx, dy))
    return nueva


def _escalar(imagen: Image.Image, factor: float) -> Image.Image:
    lado = imagen.size[0]
    nuevo = max(8, int(lado * factor))
    escalada = imagen.resize((nuevo, nuevo), Image.Resampling.BICUBIC)
    lienzo = Image.new("L", (lado, lado), 255)
    off = (lado - nuevo) // 2
    lienzo.paste(escalada, (off, off))
    return lienzo


def _filtro_suave():
    from PIL import ImageFilter

    return ImageFilter.GaussianBlur(0.8)


def construir_conjunto_aumentado(
    X_entrena: np.ndarray,
    y_entrena: np.ndarray,
    copias: int = 2,
    semilla: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Copia original + ``copias`` variantes por muestra (SOLO entrenamiento).

    Cada variante combina:
      * ``variante_geometrica`` (giro, desplazamiento, escala, grosor) en 32x32;
      * en la mitad de los casos, además, una **ampliación tipo ratón**: el dígito
        ocupa casi toda la caja 8x8 y el trazo es más grueso en proporción, que es
        como dibuja el usuario en el lienzo.

    No se aplica nunca el pipeline de la app a imágenes del dataset (eso sería
    entrenar con la respuesta), pero sí se imita la geometría del trazo real.
    """
    generador = np.random.default_rng(semilla)
    imagenes = X_entrena.reshape(-1, TAMANO_PEQUE, TAMANO_PEQUE) * 16.0
    X = [X_entrena]
    y = [y_entrena]
    for _ in range(copias):
        nuevas = []
        for img in imagenes:
            estilo = str(generador.choice(["moderado", "fuerte"]))
            variante = variante_geometrica(img, generador, estilo)
            if generador.random() < 0.5:
                variante = ampliar_como_raton(variante, generador)
            nuevas.append((variante / 16.0).reshape(-1).astype("float32"))
        X.append(np.array(nuevas, dtype="float32"))
        y.append(y_entrena.copy())
    return np.concatenate(X), np.concatenate(y)


def ampliar_como_raton(matriz8: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Amplía el dígito para que llene la caja 8x8 como un trazo de ratón."""
    limpia = np.clip(matriz8, 0, 16) / 16.0
    activo = limpia >= 0.5
    if not activo.any():
        return matriz8
    filas = np.where(activo.any(axis=1))[0]
    columnas = np.where(activo.any(axis=0))[0]
    recorte = limpia[filas[0]:filas[-1] + 1, columnas[0]:columnas[-1] + 1]
    lado = 32
    escala = float(generador.uniform(0.86, 1.0))
    nuevo = max(4, int(round(max(recorte.shape) * escala)))
    grande = Image.fromarray((recorte * 255).astype(np.uint8), mode="L").resize(
        (nuevo, nuevo), Image.Resampling.BICUBIC
    )
    lienzo = Image.new("L", (lado, lado), 0)
    lienzo.paste(grande, ((lado - nuevo) // 2, (lado - nuevo) // 2))
    reducido = np.asarray(lienzo, dtype=np.float32) / 255.0
    reducido = reducido.reshape(8, FACTOR_SUPER, 8, FACTOR_SUPER).mean(axis=(1, 3))
    return np.clip(np.round(reducido * 16.0), 0, 16)


# ---------------------------------------------------------------------------
# Generación de datos de escritura (SOLO entrenamiento, por interfaz pública)
# ---------------------------------------------------------------------------

ESTILOS_VARIACION = {
    "suave": dict(angulo=6.0, desplazamiento=0.8, escala=(0.94, 1.06), grosor=(-0.3, 0.5)),
    "moderado": dict(angulo=12.0, desplazamiento=1.0, escala=(0.90, 1.10), grosor=(-0.5, 0.7)),
    "fuerte": dict(angulo=18.0, desplazamiento=1.3, escala=(0.84, 1.14), grosor=(-0.6, 0.9)),
}


def variante_geometrica(matriz8: np.ndarray, generador: np.random.Generator, estilo: str = "moderado") -> np.ndarray:
    """Variante aleatoria con el rango de variación indicado por ``estilo``."""
    rango = ESTILOS_VARIACION[estilo]
    return transformar(
        matriz8,
        dy=float(generador.uniform(-rango["desplazamiento"], rango["desplazamiento"])),
        dx=float(generador.uniform(-rango["desplazamiento"], rango["desplazamiento"])),
        angulo=float(generador.uniform(-rango["angulo"], rango["angulo"])),
        escala=float(generador.uniform(*rango["escala"])),
        engrosar=float(generador.uniform(*rango["grosor"])),
    )


def escribir_en_lienzo(lienzo: Image.Image, semilla: int | None = None) -> Image.Image:
    """Devuelve el lienzo tal cual (firma pública para simular escritura).

    Se expone para que el código que fabrica datos de entrenamiento trabaje con la
    misma convención (fondo blanco, trazo oscuro) que la aplicación. En este
    proyecto solo la usan las herramientas que generan datos SINTÉTICOS; nunca se
    aplica a las imágenes de validación ni de prueba.
    """
    return lienzo.copy() if semilla is None else lienzo.copy()


def multitemplate_sintetico(semilla: int = 777, n_por_clase: int = 30) -> list[Image.Image]:
    """Lienzos sintéticos: medianas de 30 ejemplos por clase, ampliadas y engrosadas.

    Simula cómo dibujaría el usuario una forma «típica» de cada dígito. Se usa para
    elegir arquitectura y para verificar el pipeline, nunca para reportar
    resultados de load_digits.
    """
    generador = np.random.default_rng(semilla)
    datos = load_digits()
    lienzos = []
    for clase in range(10):
        indices = np.where(datos.target == clase)[0]
        elegidos = generador.choice(indices, size=n_por_clase, replace=False)
        mediana = np.median(datos.images[elegidos].astype(np.float32), axis=0)
        lienzos.append(_lienzo_de_plantilla(mediana, generador))
    return lienzos


def _lienzo_de_plantilla(matriz8: np.ndarray, generador: np.random.Generator,
                           lado: int = 320) -> Image.Image:
    """Convierte una imagen media 8x8 en un dibujo grande con trazo grueso."""
    activo = matriz8 >= 1.0
    if not activo.any():
        return Image.new("L", (lado, lado), 255)
    filas = np.where(activo.any(axis=1))[0]
    columnas = np.where(activo.any(axis=0))[0]
    recorte = matriz8[filas[0]:filas[-1] + 1, columnas[0]:columnas[-1] + 1]
    recorte = np.clip(recorte / 16.0, 0, 1)
    pequena = Image.fromarray((recorte * 255).astype(np.uint8), mode="L")
    tam = int(lado * float(generador.uniform(0.55, 0.75)))
    grande = pequena.resize((tam, tam), Image.Resampling.BICUBIC)
    trazo = _expandir(Image.eval(grande, lambda v: 255 - v), float(generador.uniform(1.4, 2.4)))
    trazo = _girar(trazo, float(generador.uniform(-8.0, 8.0)))
    lienzo = Image.new("L", (lado, lado), 255)
    lienzo.paste(trazo, ((lado - tam) // 2, (lado - tam) // 2))
    return lienzo
