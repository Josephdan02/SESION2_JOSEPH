"""Pruebas automatizadas de las funciones principales.

Ejecutar con:  python -m pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from utils import FACTOR_NORMALIZACION, N_PIXELES, UMBRAL_CONFIANZA  # noqa: E402
from utils import history as hist  # noqa: E402
from utils.preprocessing import cargar_calibracion, lienzo_vacio, preprocesar_imagen  # noqa: E402
from utils.rendering import transformar  # noqa: E402


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def dibujo_siete(lado=300) -> Image.Image:
    imagen = Image.new("L", (lado, lado), 255)
    d = ImageDraw.Draw(imagen)
    d.line([(lado * 0.25, lado * 0.25), (lado * 0.75, lado * 0.25)], fill=0, width=20)
    d.line([(lado * 0.75, lado * 0.25), (lado * 0.55, lado * 0.8)], fill=0, width=20)
    return imagen


def test_lienzo_vacio_detecta_blanco():
    assert lienzo_vacio(Image.new("L", (200, 200), 255)) is True


def test_lienzo_vacio_detecta_trazo():
    assert lienzo_vacio(dibujo_siete()) is False


def test_preprocesar_devuelve_64_valores_normalizados():
    vector, matriz = preprocesar_imagen(dibujo_siete())
    assert vector.shape == (N_PIXELES,)
    assert vector.dtype == np.float32
    assert vector.min() >= 0.0 and vector.max() <= 1.0
    assert matriz.shape == (8, 8)
    assert matriz.max() <= 16.0


def test_preprocesar_invierte_colores():
    """Tras la inversión el dígito debe ser CLARO sobre fondo OSCURO."""
    imagen = Image.new("L", (300, 300), 255)
    ImageDraw.Draw(imagen).ellipse([90, 90, 210, 210], outline=0, width=18)
    _, matriz = preprocesar_imagen(imagen)
    esquinas = [matriz[0, 0], matriz[0, -1], matriz[-1, 0], matriz[-1, -1]]
    assert max(esquinas) == 0.0                        # fondo oscuro
    assert matriz.max() >= 14.0                        # trazo claro y nítido


def test_pipeline_orientacion_no_se_invierte():
    """Un trazo vertical en el centro debe seguir siendo vertical (sin giro de 90°)."""
    imagen = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(imagen)
    d.line([(150, 60), (150, 240)], fill=0, width=20)
    _, matriz = preprocesar_imagen(imagen)
    filas, columnas = np.nonzero(matriz > 8)
    assert (filas.max() - filas.min()) > (columnas.max() - columnas.min())


def test_pipeline_horizontal_se_centra():
    """Un trazo simétrico debe quedar centrado en el 8x8 (sin sesgo de encuadre)."""
    imagen = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(imagen)
    d.line([(70, 150), (230, 150)], fill=0, width=20)
    _, matriz = preprocesar_imagen(imagen)
    filas, columnas = np.nonzero(matriz > 8)
    assert abs(filas.mean() - 3.5) <= 1.0
    assert abs(columnas.mean() - 3.5) <= 1.0


def test_preprocesar_centra_sin_deformar():
    """Un trazo desplazado al borde debe quedar centrado y con proporción."""
    imagen = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(imagen)
    d.line([(10, 250), (40, 250)], fill=0, width=10)  # aspa pequeña abajo-izquierda
    _, matriz = preprocesar_imagen(imagen)
    activos = np.argwhere(matriz > 0)
    assert activos.size > 0
    filas, columnas = activos[:, 0], activos[:, 1]
    centro_fila = (filas.min() + filas.max()) / 2
    centro_col = (columnas.min() + columnas.max()) / 2
    assert abs(centro_fila - 3.5) <= 1.0
    assert abs(centro_col - 3.5) <= 1.0


def test_preprocesar_lienzo_vacio_lanza_error():
    with pytest.raises(ValueError):
        preprocesar_imagen(Image.new("L", (100, 100), 255))


def test_preprocesar_no_altera_proporcion():
    """Un rectángulo alargado (horizontal o vertical) debe conservar su proporción."""
    for ancho_px, alto_px in ((100, 20), (20, 100)):
        imagen = Image.new("L", (300, 300), 255)
        d = ImageDraw.Draw(imagen)
        x0, y0 = 150 - ancho_px // 2, 150 - alto_px // 2
        d.rectangle([x0, y0, x0 + ancho_px, y0 + alto_px], fill=0)
        _, matriz = preprocesar_imagen(imagen)
        activos = np.argwhere(matriz > 8)
        assert activos.size > 0
        alto = activos[:, 0].max() - activos[:, 0].min() + 1
        ancho = activos[:, 1].max() - activos[:, 1].min() + 1
        if ancho_px > alto_px:
            assert ancho > alto
        else:
            assert alto > ancho


def test_pipeline_conserva_pixeles_activos():
    """La regresión que causaba el fallo: el 8x8 debe tener píxeles activos reales.

    La versión anterior dejaba 0.7 píxeles activos; el dataset tiene ~18.7.
    """
    _, matriz = preprocesar_imagen(dibujo_siete())
    assert int((matriz > 8).sum()) >= 8


def test_calibracion_coherente():
    cal = cargar_calibracion()
    assert 6.0 <= cal["lado_caja"] <= 8.0
    assert 0.05 <= cal["grosor_relativo"] <= 0.25


def test_aumentacion_conserva_forma_y_etiqueta():
    """Una transformación no debe destruir el dígito ni salirse de 0-16."""
    from sklearn.datasets import load_digits

    img = load_digits().images[int(np.where(load_digits().target == 5)[0][0])]
    variante = transformar(img, dy=0.5, dx=-0.5, angulo=10, escala=1.05, engrosar=0.3)
    assert variante.shape == (8, 8)
    assert variante.min() >= 0 and variante.max() <= 16
    assert (variante > 8).sum() >= 8


def test_factor_normalizacion_documentado():
    assert FACTOR_NORMALIZACION == 16.0
    assert UMBRAL_CONFIANZA == 0.70


# ---------------------------------------------------------------------------
# Validación de etiquetas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valor,esperado", [("0", 0), (9, 9), ("7", 7), (" 5 ", 5)])
def test_validar_digito_acepta_rango(valor, esperado):
    assert hist.validar_digito(valor) == esperado


@pytest.mark.parametrize("valor", ["-1", "10", "abc", "", None, 3.5, True])
def test_validar_digito_rechaza_fuera_de_rango(valor):
    with pytest.raises(ValueError):
        hist.validar_digito(valor)


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------

def test_historial_crea_csv_y_persiste(tmp_path):
    ruta = tmp_path / "data" / "historial.csv"
    assert not ruta.exists()
    hist.registrar_resultado(7, 7, 0.92, ruta=ruta)
    assert ruta.exists()
    df = hist.leer_historial(ruta)
    assert len(df) == 1
    assert df.iloc[0]["resultado"] == "correcto"
    assert float(df.iloc[0]["probabilidad_maxima"]) == pytest.approx(0.92)

    hist.registrar_resultado(3, 8, 0.55, ruta=ruta)
    df = hist.leer_historial(ruta)
    assert len(df) == 2
    assert df.iloc[1]["resultado"] == "incorrecto"


def test_historial_rechaza_probabilidad_invalida(tmp_path):
    with pytest.raises(ValueError):
        hist.registrar_resultado(1, 1, 1.5, ruta=tmp_path / "h.csv")


def test_prueba_20_no_permite_tres_dibujos_del_mismo_digito(tmp_path):
    ruta = tmp_path / "prueba_20.csv"
    hist.registrar_prueba_20(4, 4, 0.9, False, ruta=ruta)
    hist.registrar_prueba_20(4, 9, 0.6, True, ruta=ruta)
    with pytest.raises(ValueError):
        hist.registrar_prueba_20(4, 4, 0.9, False, ruta=ruta)
    assert len(hist.leer_prueba_20(ruta)) == 2


def test_prueba_20_resumen_y_faltantes(tmp_path):
    ruta = tmp_path / "prueba_20.csv"
    hist.registrar_prueba_20(1, 1, 0.95, False, ruta=ruta)
    hist.registrar_prueba_20(2, 7, 0.55, True, ruta=ruta)
    resumen = hist.resumen_prueba_20(ruta)
    assert resumen["registrados"] == 2
    assert resumen["faltan"] == 18
    assert resumen["correctos"] == 1
    assert resumen["incorrectos"] == 1
    assert resumen["advertencias_baja_confianza"] == 1
    assert resumen["realizada"] is False


# ---------------------------------------------------------------------------
# Artefactos del entrenamiento (se omiten si train.py aún no se ejecutó)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "nombre",
    [
        "comparacion_modelos.csv",
        "metricas_prueba.json",
        "analisis_errores.csv",
        "prueba_persistencia.json",
        "prueba_ruido.json",
        "indices_datos.npz",
    ],
)
def test_artefactos_existen(nombre):
    ruta = RAIZ / "results" / nombre
    if not ruta.exists():
        pytest.skip("Ejecuta 'python train.py' para generar los artefactos.")
    assert ruta.stat().st_size > 0


def test_particiones_no_se_solapan():
    ruta = RAIZ / "results" / "indices_datos.npz"
    if not ruta.exists():
        pytest.skip("Ejecuta 'python train.py' para generar los índices.")
    datos = np.load(ruta)
    train, val, test = datos["train"], datos["val"], datos["test"]
    total = len(train) + len(val) + len(test)
    assert total == 1797
    assert len(np.intersect1d(train, val)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0
    assert abs(len(val) - len(test)) <= 2


def test_persistencia_aprobada():
    ruta = RAIZ / "results" / "prueba_persistencia.json"
    if not ruta.exists():
        pytest.skip("Ejecuta 'python train.py' para generar la prueba.")
    import json

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert datos["aprobada"] is True
    assert datos["diferencia_maxima_absoluta"] <= datos["tolerancia"]


def test_modelo_carga_y_predice():
    ruta = RAIZ / "models" / "modelo_final.keras"
    if not ruta.exists():
        pytest.skip("Ejecuta 'python train.py' para generar el modelo.")
    import tensorflow as tf

    modelo = tf.keras.models.load_model(ruta)
    salida = modelo.predict(np.zeros((1, 64), dtype="float32"), verbose=0)
    assert salida.shape == (1, 10)
    assert salida.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Banco de dibujos propios (conjunto independiente)
# ---------------------------------------------------------------------------

def test_banco_guarda_lienzo_y_8x8(tmp_path):
    from utils.preprocessing import preprocesar_imagen

    lienzo = dibujo_siete()
    _, matriz = preprocesar_imagen(lienzo)
    fila = hist.guardar_dibujo_banco(lienzo, matriz, 7, 7, 0.95, "Sequential", directorio=tmp_path)
    assert fila["digito_real"] == 7
    assert len(hist.leer_banco(tmp_path)) == 1
    assert (tmp_path / fila["archivo_lienzo"]).exists()
    assert (tmp_path / fila["archivo_8x8"]).exists()


def test_banco_separa_del_historial(tmp_path):
    """Registrar en el banco no debe escribir en historial.csv ni en la prueba de 20."""
    from utils.preprocessing import preprocesar_imagen

    lienzo = dibujo_siete()
    _, matriz = preprocesar_imagen(lienzo)
    hist.guardar_dibujo_banco(lienzo, matriz, 7, 3, 0.4, "Sequential", directorio=tmp_path)
    assert hist.leer_historial(tmp_path / "historial.csv").empty
    assert hist.leer_prueba_20(tmp_path / "prueba.csv").empty


def test_banco_validacion_de_etiquetas():
    with pytest.raises(ValueError):
        hist.guardar_dibujo_banco(None, np.zeros((8, 8)), 10, 1, 0.5, "x")
