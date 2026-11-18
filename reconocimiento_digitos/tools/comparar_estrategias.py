"""Compara pipelines candidatos del preprocesamiento sobre dibujos tipo ratón.

Cada candidato es una configuración de ``utils/preprocessing``. Se evalúan dos
familias de dibujos para no afinar solo a un estilo:

  A. Pistas sintéticas (``tools/benchmark_dibujos.py``) con pincel de 12, 18 y 24 px.
  B. Dígitos de load_digits «re-dibujados»: se convierten al estilo del lienzo
     (``utils.rendering.convertir_matriz_a_lienzo`` + engrosado de pincel), se les
     aplica ruido leve de posición/escala y se pasan por el pipeline.

Se usa únicamente para elegir la configuración; la validación y la prueba
oficiales no intervienen aquí.

Uso:  python tools/comparar_estrategias.py
Salida: results/comparacion_estrategias.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from benchmark_dibujos import dibujo  # noqa: E402
from utils import RESULTS_DIR, SEMILLA  # noqa: E402
from utils.multiples import PINCELES, TRAZOS_ESTANDAR, dibujar_pistas  # noqa: E402
from utils.preprocessing import preprocesar_imagen  # noqa: E402
from utils.rendering import _expandir, _escalar, _girar, convertir_matriz_a_lienzo  # noqa: E402

CANAL = [
    {"nombre": "v1 original (caja 6, pluma 2)", "lado_caja": 6.0, "grosor_relativo": 0.05},
    {"nombre": "v2 literal (caja 6, sin grosor)", "lado_caja": 6.0, "grosor_relativo": 0.05},
    {"nombre": "actual (caja 7, grosor 0.42)", "lado_caja": 7.0, "grosor_relativo": 0.42},
    {"nombre": "solo-grosor fino (caja 7, 0.13)", "lado_caja": 7.0, "grosor_relativo": 0.13},
    {"nombre": "solo-grosor medio (caja 7, 0.16)", "lado_caja": 7.0, "grosor_relativo": 0.16},
    {"nombre": "caja 7.5 grosor 0.16", "lado_caja": 7.5, "grosor_relativo": 0.16},
    {"nombre": "caja 6.6 grosor 0.16", "lado_caja": 6.6, "grosor_relativo": 0.16},
]


def set_b_dibujos(n_por_digito: int = 6) -> list[tuple[int, object]]:
    """Dígitos de load_digits redibujados con estilo de lienzo."""
    d = load_digits()
    generador = np.random.default_rng(SEMILLA)
    salida = []
    for k in range(10):
        indices = np.where(d.target == k)[0]
        elegidos = generador.choice(indices, size=min(n_por_digito, len(indices)), replace=False)
        for i in elegidos:
            # Las variantes reproducen EXACTAMENTE las de utils.multiples
            # (misma semilla y mismo orden) para no filtrar datos de evaluación.
            margen = float(generador.uniform(0.15, 0.30))
            factor = float(generador.uniform(0.6, 1.8))
            giro = float(generador.uniform(-8, 8))
            escala = float(generador.uniform(0.88, 1.10))
            lienzo = convertir_matriz_a_lienzo(d.images[i], lado=320, margen=margen)
            lienzo = _expandir(lienzo, factor)
            lienzo = _girar(lienzo, giro)
            lienzo = _escalar(lienzo, escala)
            salida.append((k, lienzo))
    return salida


def evaluar(modelo, cal: dict, lienzos_b) -> dict:
    a_aciertos = 0
    for pincel in (12, 18, 24):
        for k in range(10):
            v, _ = preprocesar_imagen(dibujo(k, pincel=pincel), calibracion=cal)
            a_aciertos += int(np.argmax(modelo.predict(v.reshape(1, -1), verbose=0)[0]) == k)
    b_aciertos = 0
    for k, lienzo in lienzos_b:
        v, _ = preprocesar_imagen(lienzo, calibracion=cal)
        b_aciertos += int(np.argmax(modelo.predict(v.reshape(1, -1), verbose=0)[0]) == k)
    pinceles = []
    for pincel in PINCELES:
        aciertos = 0
        for k, trazos in TRAZOS_ESTANDAR.items():
            v, _ = preprocesar_imagen(dibujar_pistas(trazos, pincel=pincel), calibracion=cal)
            aciertos += int(np.argmax(modelo.predict(v.reshape(1, -1), verbose=0)[0]) == k)
        pinceles.append(aciertos)
    return {
        "aciertos_pistas_30": a_aciertos,
        "aciertos_redibujados_60": b_aciertos,
        "aciertos_banco_estrecho_50": int(sum(pinceles)),
        "total": 140,
        "aciertos_totales": a_aciertos + b_aciertos + int(sum(pinceles)),
    }


def main() -> int:
    import tensorflow as tf

    modelo = tf.keras.models.load_model(RAIZ / "models" / "modelo_final.keras")
    b = set_b_dibujos()
    filas = []
    for cand in CANAL:
        cal = {k: float(v) for k, v in cand.items() if k != "nombre"}
        base = {"centro_masa_fila": 3.5, "centro_masa_columna": 3.5, **cal}
        met = evaluar(modelo, base, b)
        filas.append({"estrategia": cand["nombre"], **{k: v for k, v in cand.items() if k != "nombre"}, **met})
        print(f"{cand['nombre']:34s} {met['aciertos_totales']}/140 "
              f"(pistas {met['aciertos_pistas_30']}/30, redibujados {met['aciertos_redibujados_60']}/60, "
              f"banco {met['aciertos_banco_estrecho_50']}/50)")
    df = pd.DataFrame(filas)
    df.to_csv(RESULTS_DIR / "comparacion_estrategias.csv", index=False, encoding="utf-8")
    print("Guardado en results/comparacion_estrategias.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
