"""Entrenamiento, evaluación, gráficos y persistencia del reconocedor de dígitos.

Ejecutar con:  python train.py

Flujo:
1. Carga de load_digits y partición 60/20/20 estratificada (semilla 42).
2. Entrenamiento de un modelo Sequential y de un modelo Functional.
3. Comparación en VALIDACIÓN (F1 macro como criterio principal).
4. Evaluación de ambos en PRUEBA (sin modificar nada a partir de ella).
5. Gráficos, análisis de errores, persistencia y prueba de ruido.
"""

from __future__ import annotations

import json  # usado por el script de persistencia (subproceso) y por sus pruebas
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Silenciar mensajes informativos de TensorFlow (no afectan al resultado).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_digits
from tensorflow import keras
from tensorflow.keras import layers

from utils import (
    FACTOR_NORMALIZACION,
    MODEL_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    SEMILLA,
    UMBRAL_CONFIANZA,
)
from utils.evaluation import (
    graficar_curvas,
    graficar_matriz_confusion,
    guardar_json,
    metricas_clasificacion,
    particionar_datos,
)

EPOCAS = 200
BATCH_SIZE = 32
PRECISION = "float32"

CONFIG_SEQUENTIAL = {
    "nombre": "Sequential",
    "capas": [
        "Input(shape=(64,))",
        "Dense(128, activation='relu', kernel_regularizer=l2(1e-4))",
        "Dropout(0.25)",
        "Dense(64, activation='relu', kernel_regularizer=l2(1e-4))",
        "Dense(10, activation='softmax')",
    ],
    "neuronas": [128, 64],
    "dropout": 0.25,
    "l2": 1e-4,
    "learning_rate": 0.001,
    "batch_size": BATCH_SIZE,
    "justificacion": (
        "Dos capas ocultas (128 y 64 neuronas) son suficientes para 64 entradas y "
        "1797 ejemplos: la primera combina píxeles en patrones y la segunda los "
        "resume por clase. Dropout 0.25 y L2 1e-4 frenan el sobreajuste; Adam con "
        "lr=0.001 converge de forma estable."
    ),
}

CONFIG_FUNCTIONAL = {
    "nombre": "Functional",
    "capas": [
        "Input(shape=(64,), name='pixeles')",
        "Dense(64, activation='relu')",
        "Dropout(0.20)",
        "Dense(32, activation='relu')",
        "Dense(10, activation='softmax', name='probabilidades')",
    ],
    "neuronas": [64, 32],
    "dropout": 0.20,
    "l2": 0.0,
    "learning_rate": 0.001,
    "batch_size": BATCH_SIZE,
    "justificacion": (
        "Arquitectura más compacta (64 y 32 neuronas) sin regularización L2 para "
        "comparar un diseño distinto con la misma API funcional. Menos parámetros "
        "reducen el riesgo de memorizar el pequeño conjunto de entrenamiento."
    ),
}


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def construir_sequential(config: dict, semilla: int = SEMILLA) -> keras.Model:
    keras.utils.set_random_seed(semilla)
    l2 = keras.regularizers.l2(config["l2"]) if config.get("l2") else None
    modelo = keras.Sequential(
        [
            keras.Input(shape=(64,), name="pixeles"),
            layers.Dense(128, activation="relu", kernel_regularizer=l2),
            layers.Dropout(config["dropout"]),
            layers.Dense(64, activation="relu", kernel_regularizer=l2),
            layers.Dense(10, activation="softmax", name="probabilidades"),
        ],
        name="modelo_sequential",
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def construir_functional(config: dict, semilla: int = SEMILLA) -> keras.Model:
    keras.utils.set_random_seed(semilla)
    entrada = keras.Input(shape=(64,), name="pixeles")
    x = layers.Dense(64, activation="relu", name="oculta_1")(entrada)
    x = layers.Dropout(config["dropout"], name="dropout")(x)
    x = layers.Dense(32, activation="relu", name="oculta_2")(x)
    salida = layers.Dense(10, activation="softmax", name="probabilidades")(x)
    modelo = keras.Model(inputs=entrada, outputs=salida, name="modelo_functional")
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def entrenar(modelo: keras.Model, X_entrena, y_entrena_cat, X_valida, y_valida_cat, etiqueta: str):
    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True, verbose=0
    )
    historia = modelo.fit(
        X_entrena,
        y_entrena_cat,
        validation_data=(X_valida, y_valida_cat),
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        callbacks=[parada],
        verbose=2,
    )
    print(f"  Épocas efectivas ({etiqueta}): {len(historia.history['loss'])}")
    return historia


# ---------------------------------------------------------------------------
# Análisis de errores
# ---------------------------------------------------------------------------

def explicar_confusion(real: int, predicho: int, vector: np.ndarray) -> str:
    filas = np.reshape(vector, (8, 8)) > 0.25
    fila_sup, fila_inf = float(filas[:2].mean()), float(filas[6:].mean())
    col_izq, col_der = float(filas[:, :2].mean()), float(filas[:, 6:].mean())
    pistas = [f"píxeles activos={int(filas.sum())}"]
    if fila_sup > 0.35 and fila_inf < 0.15:
        pistas.append("mucha tinta arriba y poca abajo")
    if fila_inf > 0.35 and fila_sup < 0.15:
        pistas.append("mucha tinta abajo")
    if col_izq > 0.40 and col_der < 0.10:
        pistas.append("trazo cargado a la izquierda")
    if col_der > 0.40 and col_izq < 0.10:
        pistas.append("trazo cargado a la derecha")
    if int(filas.sum()) <= 8:
        pistas.append("trazo muy fino (poca señal)")
    pistas.append("píxeles de 8x8 borrosos tras reducir el tamaño")
    return f"Se confundió el {real} con el {predicho}: " + "; ".join(pistas) + "."


# ---------------------------------------------------------------------------
# Prueba de ruido
# ---------------------------------------------------------------------------

def prueba_ruido(modelo: keras.Model, X_prueba, y_prueba, idx_prueba, ruta_json: Path) -> dict:
    generador = np.random.default_rng(SEMILLA)
    indices_locales = np.sort(generador.choice(len(X_prueba), size=30, replace=False))
    X_sel, y_sel = X_prueba[indices_locales], y_prueba[indices_locales]

    probs_originales = modelo.predict(X_sel, verbose=0)
    ruido = generador.normal(0.0, 0.15, size=X_sel.shape)
    X_ruido = np.clip(X_sel + ruido, 0.0, 1.0).astype("float32")
    probs_ruido = modelo.predict(X_ruido, verbose=0)

    m_orig = metricas_clasificacion(y_sel, probs_originales)
    m_ruido = metricas_clasificacion(y_sel, probs_ruido)

    resultados = {
        "descripcion": "Treinta imágenes de prueba con ruido gaussiano N(0, 0.15) y recorte a [0, 1].",
        "semilla": SEMILLA,
        "desviacion_estandar_ruido": 0.15,
        "n_muestras": int(len(y_sel)),
        "indices_originales_dataset": [int(idx_prueba[i]) for i in indices_locales],
        "exactitud_original": m_orig["accuracy"],
        "exactitud_con_ruido": m_ruido["accuracy"],
        "f1_macro_original": m_orig["f1_macro"],
        "f1_macro_con_ruido": m_ruido["f1_macro"],
        "diferencia_accuracy": round(m_orig["accuracy"] - m_ruido["accuracy"], 6),
        "detalle": [
            {
                "indice_local": int(i),
                "indice_original": int(idx_prueba[i]),
                "real": int(y_sel[i]),
                "predicha_original": int(np.argmax(probs_originales[i])),
                "predicha_con_ruido": int(np.argmax(probs_ruido[i])),
                "prob_max_original": round(float(probs_originales[i].max()), 6),
                "prob_max_con_ruido": round(float(probs_ruido[i].max()), 6),
            }
            for i in range(len(y_sel))
        ],
    }
    guardar_json(ruta_json, resultados)
    return resultados


# ---------------------------------------------------------------------------
# Prueba de persistencia (proceso independiente)
# ---------------------------------------------------------------------------

CODIGO_PRUEBA_PERSISTENCIA = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import tensorflow as tf
from tensorflow import keras

datos = np.load(sys.argv[2])
modelo = keras.models.load_model(sys.argv[3])
probs = modelo.predict(datos["ids"], verbose=0)
with open(sys.argv[4], "w", encoding="utf-8") as f:
    json.dump({"versiones": {"tensorflow": tf.__version__, "keras": keras.__version__},
               "probabilidades": probs.round(8).tolist()}, f)
'''


def prueba_persistencia(X_prueba, y_prueba, idx_prueba, probs_antes, ruta_json: Path, tolerancia=1e-6):
    generador = np.random.default_rng(SEMILLA)
    posiciones = np.sort(generador.choice(len(X_prueba), size=5, replace=False))
    muestras = X_prueba[posiciones]
    indices_originales = idx_prueba[posiciones]
    probs_antes_sel = probs_antes[posiciones]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "muestras.npz", ids=muestras)
        (tmp / "script_persistencia.py").write_text(CODIGO_PRUEBA_PERSISTENCIA, encoding="utf-8")
        salida = tmp / "salida.json"
        comando = [
            sys.executable,
            str(tmp / "script_persistencia.py"),
            str(Path(__file__).resolve().parent),
            str(tmp / "muestras.npz"),
            str(MODEL_PATH),
            str(salida),
        ]
        proceso = subprocess.run(
            comando, capture_output=True, text=True, timeout=600, cwd=str(MODEL_PATH.parent)
        )
        if proceso.returncode != 0:
            raise RuntimeError(f"El proceso independiente falló:\n{proceso.stderr[-2000:]}")
        recargado = json.loads(salida.read_text(encoding="utf-8"))

    probs_despues = np.array(recargado["probabilidades"], dtype=float)
    diferencias = np.abs(probs_despues - probs_antes_sel)
    max_dif = float(diferencias.max())
    predicciones_iguales = bool(np.array_equal(probs_despues.argmax(1), probs_antes_sel.argmax(1)))

    resultados = {
        "descripcion": "Predicciones antes de guardar el modelo vs. un proceso nuevo que carga el .keras.",
        "proceso_independiente": True,
        "versiones_proceso_nuevo": recargado["versiones"],
        "tolerancia": tolerancia,
        "diferencia_maxima_absoluta": max_dif,
        "predicciones_identicas": predicciones_iguales,
        "aprobada": bool(max_dif <= tolerancia and predicciones_iguales),
        "indices_originales_dataset": [int(i) for i in indices_originales],
        "digitos_reales": [int(v) for v in y_prueba[posiciones]],
        "probabilidades_antes": probs_antes_sel.round(8).tolist(),
        "probabilidades_despues": probs_despues.round(8).tolist(),
        "prediccion_antes": [int(v) for v in probs_antes_sel.argmax(1)],
        "prediccion_despues": [int(v) for v in probs_despues.argmax(1)],
    }
    guardar_json(ruta_json, resultados)
    return resultados


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | NumPy {np.__version__}")

    # 1) Datos -----------------------------------------------------------------
    digits = load_digits()
    X = (digits.data / FACTOR_NORMALIZACION).astype(PRECISION)
    y = digits.target.astype(np.int64)
    print(f"Dataset: {X.shape[0]} imágenes de {digits.images.shape[1]}x{digits.images.shape[2]} px")

    (X_entrena, y_entrena, idx_entrena), (X_valida, y_valida, idx_valida), (
        X_prueba,
        y_prueba,
        idx_prueba,
    ) = particionar_datos(X, y, SEMILLA)
    print(f"Entrenamiento {len(X_entrena)} | Validación {len(X_valida)} | Prueba {len(X_prueba)}")

    np.savez(
        RESULTS_DIR / "indices_datos.npz",
        train=idx_entrena,
        val=idx_valida,
        test=idx_prueba,
        semilla=SEMILLA,
    )

    y_entrena_cat = keras.utils.to_categorical(y_entrena, 10)
    y_valida_cat = keras.utils.to_categorical(y_valida, 10)

    # 2) Entrenamiento ---------------------------------------------------------
    print("\n=== Entrenamiento del modelo Sequential ===")
    modelo_seq = construir_sequential(CONFIG_SEQUENTIAL)
    modelo_seq.summary()
    hist_seq = entrenar(modelo_seq, X_entrena, y_entrena_cat, X_valida, y_valida_cat, "Sequential")

    print("\n=== Entrenamiento del modelo Functional ===")
    modelo_fun = construir_functional(CONFIG_FUNCTIONAL)
    modelo_fun.summary()
    hist_fun = entrenar(modelo_fun, X_entrena, y_entrena_cat, X_valida, y_valida_cat, "Functional")

    # 3) Comparación en VALIDACIÓN --------------------------------------------
    probs_val_seq = modelo_seq.predict(X_valida, verbose=0)
    probs_val_fun = modelo_fun.predict(X_valida, verbose=0)
    met_val_seq = metricas_clasificacion(y_valida, probs_val_seq)
    met_val_fun = metricas_clasificacion(y_valida, probs_val_fun)

    filas = []
    for config, modelo, met, hist in (
        (CONFIG_SEQUENTIAL, modelo_seq, met_val_seq, hist_seq),
        (CONFIG_FUNCTIONAL, modelo_fun, met_val_fun, hist_fun),
    ):
        filas.append(
            {
                "modelo": config["nombre"],
                "arquitectura": " -> ".join(config["capas"]),
                "neuronas_ocultas": str(config["neuronas"]),
                "dropout": config["dropout"],
                "regularizacion_l2": config["l2"],
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "parametros_entrenables": int(modelo.count_params()),
                "epocas_efectivas": len(hist.history["loss"]),
                "accuracy_validacion": met["accuracy"],
                "f1_macro_validacion": met["f1_macro"],
            }
        )
    comparacion = pd.DataFrame(filas)
    comparacion.to_csv(RESULTS_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8")
    print("\nComparación en validación:")
    print(comparacion[["modelo", "parametros_entrenables", "epocas_efectivas",
                       "accuracy_validacion", "f1_macro_validacion"]].to_string(index=False))

    # F1 macro principal, accuracy como desempate.
    ordenados = comparacion.sort_values(
        ["f1_macro_validacion", "accuracy_validacion"], ascending=False
    )
    nombre_ganador = ordenados.iloc[0]["modelo"]
    modelo_final, config_final = (
        (modelo_seq, CONFIG_SEQUENTIAL) if nombre_ganador == "Sequential" else (modelo_fun, CONFIG_FUNCTIONAL)
    )
    print(f"\nModelo seleccionado por validación: {nombre_ganador}")

    # 4) Evaluación final en PRUEBA (sin tocar la arquitectura) ---------------
    probs_test_seq = modelo_seq.predict(X_prueba, verbose=0)
    probs_test_fun = modelo_fun.predict(X_prueba, verbose=0)
    met_test_seq = metricas_clasificacion(y_prueba, probs_test_seq)
    met_test_fun = metricas_clasificacion(y_prueba, probs_test_fun)
    probs_test_final = probs_test_seq if nombre_ganador == "Sequential" else probs_test_fun
    met_test_final = met_test_seq if nombre_ganador == "Sequential" else met_test_fun

    metricas = {
        "descripcion": "Evaluación única sobre la partición de prueba (20 %), nunca usada para elegir.",
        "criterio_de_seleccion": "F1 macro en validación; accuracy como desempate.",
        "modelo_seleccionado": nombre_ganador,
        "semilla": SEMILLA,
        "tamano_prueba": int(len(y_prueba)),
        "modelos": {
            "Sequential": {
                "accuracy": met_test_seq["accuracy"],
                "f1_macro": met_test_seq["f1_macro"],
                "aciertos": met_test_seq["aciertos"],
            },
            "Functional": {
                "accuracy": met_test_fun["accuracy"],
                "f1_macro": met_test_fun["f1_macro"],
                "aciertos": met_test_fun["aciertos"],
            },
        },
        "modelo_final_en_prueba": {
            "nombre": nombre_ganador,
            "accuracy": met_test_final["accuracy"],
            "f1_macro": met_test_final["f1_macro"],
            "aciertos": met_test_final["aciertos"],
        },
        "umbral_baja_confianza": UMBRAL_CONFIANZA,
    }
    guardar_json(RESULTS_DIR / "metricas_prueba.json", metricas)
    print(
        f"Prueba -> Sequential acc={met_test_seq['accuracy']:.4f} F1={met_test_seq['f1_macro']:.4f} | "
        f"Functional acc={met_test_fun['accuracy']:.4f} F1={met_test_fun['f1_macro']:.4f}"
    )

    # 5) Gráficos --------------------------------------------------------------
    graficar_curvas(
        {"Sequential": hist_seq, "Functional": hist_fun},
        RESULTS_DIR / "curvas_entrenamiento.png",
        "Curvas de entrenamiento: Sequential vs Functional (validación)",
    )
    graficar_matriz_confusion(
        y_prueba,
        probs_test_final,
        RESULTS_DIR / "matriz_confusion.png",
        f"Matriz de confusión en prueba - modelo {nombre_ganador}",
    )

    # 6) Análisis de errores ---------------------------------------------------
    predicciones = met_test_final["predicciones"]
    erroneas = np.where(predicciones != y_prueba)[0]
    errores = []
    for pos in erroneas:
        probs_muestra = probs_test_final[pos]
        indices_top = np.argsort(probs_muestra)[::-1]
        errores.append(
            {
                "indice_muestra": int(idx_prueba[pos]),
                "posicion_en_prueba": int(pos),
                "digito_real": int(y_prueba[pos]),
                "digito_predicho": int(predicciones[pos]),
                "probabilidad_maxima": round(float(probs_muestra.max()), 6),
                "segunda_opcion": int(indices_top[1]),
                "probabilidad_segunda": round(float(probs_muestra[indices_top[1]]), 6),
                "baja_confianza": bool(probs_muestra.max() < UMBRAL_CONFIANZA),
                "explicacion": explicar_confusion(
                    int(y_prueba[pos]), int(predicciones[pos]), X_prueba[pos]
                ),
            }
        )
    pd.DataFrame(errores).to_csv(RESULTS_DIR / "analisis_errores.csv", index=False, encoding="utf-8")
    print(f"Errores en prueba: {len(errores)} de {len(y_prueba)} muestras.")

    # 7) Persistencia ----------------------------------------------------------
    modelo_final.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

    metadata = {
        "nombre_modelo": nombre_ganador,
        "arquitectura_seleccionada": f"Modelo {nombre_ganador} (red densa)",
        "api": "Sequential" if nombre_ganador == "Sequential" else "Functional (Input -> Dense -> Model)",
        "configuracion_capas": config_final["capas"],
        "neuronas_ocultas": config_final["neuronas"],
        "dropout": config_final["dropout"],
        "regularizacion_l2": config_final["l2"],
        "optimizador": "Adam",
        "learning_rate": config_final["learning_rate"],
        "funcion_perdida": "categorical_crossentropy",
        "batch_size": config_final["batch_size"],
        "epocas_maximas": EPOCAS,
        "epocas_efectivas": int(
            len(hist_seq.history["loss"]) if nombre_ganador == "Sequential" else len(hist_fun.history["loss"])
        ),
        "parametros_entrenables": int(modelo_final.count_params()),
        "num_entradas": 64,
        "forma_entrada": [8, 8],
        "num_clases": 10,
        "escala_original_pixeles": "0 a 16 (enteros de load_digits)",
        "factor_normalizacion": FACTOR_NORMALIZACION,
        "semilla": SEMILLA,
        "preprocesamiento": (
            "Píxeles de load_digits divididos entre 16.0 (rango 0-1). En la aplicación: "
            "recorte de márgenes del dibujo, inversión de colores (dígito claro sobre fondo "
            "oscuro), escalado proporcional a 6x6, centrado en 8x8, escala 0-16 y división "
            "entre 16.0, aplanado a 64 valores."
        ),
        "umbral_baja_confianza": UMBRAL_CONFIANZA,
        "tolerancia_persistencia": 1e-6,
        "fecha_entrenamiento": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versiones": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
        },
        "seleccion": {
            "criterio": "F1 macro en validación (desempate por accuracy)",
            "f1_macro_validacion": {
                "Sequential": met_val_seq["f1_macro"],
                "Functional": met_val_fun["f1_macro"],
            },
            "accuracy_validacion": {
                "Sequential": met_val_seq["accuracy"],
                "Functional": met_val_fun["accuracy"],
            },
        },
    }
    guardar_json(MODELS_DIR / "metadata.json", metadata)

    persistencia = prueba_persistencia(
        X_prueba, y_prueba, idx_prueba, probs_test_final, RESULTS_DIR / "prueba_persistencia.json"
    )
    print(
        f"Persistencia: diferencia máxima {persistencia['diferencia_maxima_absoluta']:.3e} "
        f"(tolerancia {persistencia['tolerancia']}) -> {'APROBADA' if persistencia['aprobada'] else 'FALLIDA'}"
    )

    # 8) Prueba de ruido -------------------------------------------------------
    ruido = prueba_ruido(modelo_final, X_prueba, y_prueba, idx_prueba, RESULTS_DIR / "prueba_ruido.json")
    print(
        f"Ruido: accuracy original {ruido['exactitud_original']:.4f} -> "
        f"con ruido {ruido['exactitud_con_ruido']:.4f}"
    )

    print("\nEntrenamiento y evaluación finalizados correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
