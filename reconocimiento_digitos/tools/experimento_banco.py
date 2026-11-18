"""Experimento: ¿ayuda añadir el banco de dibujos sintéticos al entrenamiento?

Compara dos variantes densas con la MISMA semilla y la MISMA partición que
``train.py``:

  * v2: load_digits (solo entrenamiento) + aumentación x3.
  * v3: v2 + banco sintético de pistas (50 muestras, semilla 7).

Evaluaciones (ninguna interviene en la selección):
  * VALIDACIÓN de load_digits  -> criterio de selección (F1 macro).
  * PRUEBA de load_digits      -> evaluación única.
  * BANCO DISJUNTO (semillas 555 pistas + 1234 redibujados) -> evaluación de
    escritura tipo ratón no vista. Se separa por origen para comparar.

Uso:  python tools/experimento_banco.py
Salida: results/experimento_banco_v3.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from tensorflow import keras  # noqa: E402

from utils import RESULTS_DIR, SEMILLA  # noqa: E402
from utils.evaluation import guardar_json, metricas_clasificacion, particionar_datos  # noqa: E402
from utils.multiples import PINCELES, TRAZOS_ESTANDAR, banco_sintetico, dibujar_pistas  # noqa: E402
from utils.preprocessing import preprocesar_imagen, recortar_trazo  # noqa: E402
from utils.rendering import construir_conjunto_aumentado  # noqa: E402
from train import (  # noqa: E402
    CONFIG_FUNCTIONAL,
    CONFIG_SEQUENTIAL,
    construir_functional,
    construir_sequential,
    entrenar,
)


def preparar_datos():
    digits = load_digits()
    X = (digits.data / 16.0).astype("float32")
    y = digits.target.astype(np.int64)
    (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _) = particionar_datos(X, y, SEMILLA)
    return Xtr, ytr, Xv, yv, Xte, yte


def banco_para_entrenamiento() -> tuple[np.ndarray, np.ndarray]:
    """Variantes de ENTRENAMIENTO (50 pistas semilla 7 + 10 mediáticas semilla 777).

    Las medianas se calculan con 30 ejemplos por clase de load_digits; esas 30
    imágenes NO se excluyen pero tampoco se usan sus versiones originales: solo la
    forma promediada y redibujada, que es un dato sintético nuevo.
    """
    from utils.rendering import multitemplate_sintetico

    X, y = [], []
    for etiqueta, lienzo, _tipo in banco_sintetico(semilla=7, variantes_por_pincel=1,
                                                   variantes_redibujadas=0):
        if recortar_trazo(lienzo) is None:
            continue
        vector, _ = preprocesar_imagen(lienzo)
        X.append(vector)
        y.append(etiqueta)
    for etiqueta, lienzo in enumerate(multitemplate_sintetico(semilla=777, n_por_clase=30)):
        vector, _ = preprocesar_imagen(lienzo)
        X.append(vector)
        y.append(etiqueta)
    return np.array(X, dtype="float32"), np.array(y, dtype=np.int64)


def banco_de_evaluacion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluación disjunta: semillas 555 (pistas), 1234 (redibujados) y 2025 (medianas)."""
    from utils.rendering import multitemplate_sintetico

    X, y, origen = [], [], []
    for etiqueta, lienzo, tipo in banco_sintetico(semilla=1234, variantes_por_pincel=1,
                                                  variantes_redibujadas=5):
        # La semilla 1234 aplica un jitter distinto al de la semilla 7 (la del
        # entrenamiento), así que estas pistas sirven como evaluación.
        if recortar_trazo(lienzo) is None:
            continue
        vector, _ = preprocesar_imagen(lienzo)
        X.append(vector)
        y.append(etiqueta)
        origen.append(tipo)

    generador = np.random.default_rng(555)
    for k, trazos in TRAZOS_ESTANDAR.items():
        for pincel in PINCELES:
            lienzo = dibujar_pistas(
                trazos, pincel=pincel,
                dx=float(generador.uniform(-0.05, 0.05)),
                dy=float(generador.uniform(-0.05, 0.05)),
                escala=float(generador.uniform(0.85, 1.10)),
            )
            vector, _ = preprocesar_imagen(lienzo)
            X.append(vector)
            y.append(k)
            origen.append("pista_555")
    for k, lienzo in enumerate(multitemplate_sintetico(semilla=2025, n_por_clase=30)):
        vector, _ = preprocesar_imagen(lienzo)
        X.append(vector)
        y.append(k)
        origen.append("mediana_2025")
    return np.array(X, dtype="float32"), np.array(y, dtype=np.int64), np.array(origen)


def evaluar_banco(modelo, Xb, yb, origen) -> dict:
    probs = np.concatenate([modelo(Xb[i:i + 1]) for i in range(len(Xb))])
    aciertos = np.argmax(probs, axis=1) == yb
    por_tipo = {}
    for tipo in np.unique(origen):
        mascara = origen == tipo
        por_tipo[str(tipo)] = {
            "accuracy": round(float(aciertos[mascara].mean()), 4),
            "n": int(mascara.sum()),
        }
    return {"accuracy": round(float(aciertos.mean()), 4), "n": int(len(yb)), "por_origen": por_tipo}


def predecir(modelo, X) -> np.ndarray:
    return np.concatenate([modelo(X[i:i + 1]) for i in range(len(X))])


def main() -> int:
    Xtr, ytr, Xv, yv, Xte, yte = preparar_datos()
    X_aug, y_aug = construir_conjunto_aumentado(Xtr, ytr, copias=2, semilla=SEMILLA)
    X_b, y_b = banco_para_entrenamiento()
    X_v3 = np.concatenate([X_aug, X_b])
    y_v3 = np.concatenate([y_aug, y_b])
    Xe, ye, origen_e = banco_de_evaluacion()
    yv_cat = keras.utils.to_categorical(yv, 10)
    print(f"Banco de entrenamiento: {len(X_b)} | banco de evaluación: {len(Xe)} "
          f"({dict(zip(*np.unique(origen_e, return_counts=True)))})")

    variantes = {
        "v2": (X_aug, y_aug, f"load_digits x3 = {len(X_aug)}"),
        "v3": (X_v3, y_v3, f"load_digits x3 + banco({len(X_b)}) = {len(X_v3)}"),
    }

    resultados = {}
    for nombre, (Xent, yent, descripcion) in variantes.items():
        yent_cat = keras.utils.to_categorical(yent, 10)
        for config, constructor in ((CONFIG_SEQUENTIAL, construir_sequential),
                                    (CONFIG_FUNCTIONAL, construir_functional)):
            modelo = constructor(config)
            historia = entrenar(modelo, Xent, yent_cat, Xv, yv_cat, f"{nombre}-{config['nombre']}")
            met_val = metricas_clasificacion(yv, predecir(modelo, Xv))
            met_test = metricas_clasificacion(yte, predecir(modelo, Xte))
            met_banco = evaluar_banco(modelo, Xe, ye, origen_e)
            resultados[f"{nombre}|{config['nombre']}"] = {
                "datos": descripcion,
                "n_entrenamiento": int(len(Xent)),
                "epocas_efectivas": len(historia.history["loss"]),
                "accuracy_validacion": met_val["accuracy"],
                "f1_macro_validacion": met_val["f1_macro"],
                "accuracy_prueba": met_test["accuracy"],
                "f1_macro_prueba": met_test["f1_macro"],
                "banco_dibujos": met_banco,
            }
            print(f"{nombre} {config['nombre']:10s} val={met_val['accuracy']:.4f} "
                  f"f1_val={met_val['f1_macro']:.4f} | prueba={met_test['accuracy']:.4f} | "
                  f"banco dibujos={met_banco['accuracy']:.4f} {met_banco['por_origen']}")

    guardar_json(RESULTS_DIR / "experimento_banco_v3.json", resultados)
    print("Guardado en results/experimento_banco_v3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
