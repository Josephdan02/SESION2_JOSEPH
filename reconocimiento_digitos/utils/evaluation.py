"""Utilidades de partición, evaluación, gráficos y persistencia."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Gráficos sin ventana; la interfaz usa su propio backend.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def particionar_datos(X, y, semilla: int = 42):
    """Divide 60 % entrenamiento, 20 % validación y 20 % prueba, estratificado."""
    X_desarrollo, X_prueba, y_desarrollo, y_prueba, idx_dev, idx_prueba = train_test_split(
        X, y, np.arange(len(y)), test_size=0.20, random_state=semilla, stratify=y
    )
    X_entrena, X_valida, y_entrena, y_valida, idx_entrena, idx_valida = train_test_split(
        X_desarrollo,
        y_desarrollo,
        idx_dev,
        test_size=0.25,
        random_state=semilla,
        stratify=y_desarrollo,
    )
    return (
        (X_entrena, y_entrena, idx_entrena),
        (X_valida, y_valida, idx_valida),
        (X_prueba, y_prueba, idx_prueba),
    )


def metricas_clasificacion(y_real, probabilidades, n_clases: int = 10) -> dict:
    """Accuracy y F1 macro a partir de las probabilidades de la red."""
    y_real = np.asarray(y_real)
    predicciones = np.argmax(np.asarray(probabilidades), axis=1)
    acc = float((predicciones == y_real).mean())
    f1 = float(
        f1_score(y_real, predicciones, labels=list(range(n_clases)), average="macro", zero_division=0)
    )
    return {
        "accuracy": round(acc, 6),
        "f1_macro": round(f1, 6),
        "aciertos": int((predicciones == y_real).sum()),
        "total": int(len(y_real)),
        "predicciones": predicciones,
    }


def graficar_curvas(historiales: dict, ruta: Path, titulo: str = "Curvas de entrenamiento") -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nombre, historia in historiales.items():
        h = historia.history
        ejes[0].plot(h["loss"], label=f"{nombre} - entrenamiento")
        ejes[0].plot(h["val_loss"], linestyle="--", label=f"{nombre} - validación")
        ejes[1].plot(h.get("accuracy", []), label=f"{nombre} - entrenamiento")
        ejes[1].plot(h.get("val_accuracy", []), linestyle="--", label=f"{nombre} - validación")
    ejes[0].set_title("Pérdida (categorical_crossentropy)")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Pérdida")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(fontsize=8)
    ejes[1].set_title("Exactitud")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Accuracy")
    ejes[1].grid(alpha=0.3)
    ejes[1].legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_matriz_confusion(y_real, probabilidades, ruta: Path, titulo: str) -> None:
    predicciones = np.argmax(np.asarray(probabilidades), axis=1)
    matriz = confusion_matrix(y_real, predicciones, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(titulo)
    ax.set_xlabel("Dígito predicho")
    ax.set_ylabel("Dígito real")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_json(ruta: Path, datos) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar_json(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))
