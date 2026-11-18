"""Historial persistente de predicciones y validaciones asociadas.

Todos los CSV y JSON se guardan en UTF-8 con ``utf-8-sig`` cuando se abren
directamente con Excel, pero pandas escribe en UTF-8 normal; la aplicación
detecta el BOM al leer.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import RESULTS_DIR, HISTORIAL_PATH

COLUMNAS_HISTORIAL = [
    "fecha_hora",
    "digito_real",
    "digito_predicho",
    "probabilidad_maxima",
    "resultado",
]

COLUMNAS_PRUEBA_20 = [
    "id_dibujo",
    "fecha_hora",
    "digito_real",
    "digito_predicho",
    "probabilidad_maxima",
    "resultado",
    "advertencia_baja_confianza",
    "probabilidades",
]


def validar_digito(valor) -> int:
    """Comprueba que el valor sea un entero entre 0 y 9 y lo devuelve."""
    if isinstance(valor, bool):
        raise ValueError("La etiqueta debe ser un número entero entre 0 y 9.")
    try:
        numero = int(str(valor).strip())
    except (TypeError, ValueError):
        raise ValueError("La etiqueta debe ser un número entero entre 0 y 9.")
    if numero < 0 or numero > 9:
        raise ValueError("La etiqueta debe estar entre 0 y 9.")
    return numero


def _leer_csv(ruta: Path, columnas: list[str]) -> pd.DataFrame:
    if not ruta.exists() or ruta.stat().st_size == 0:
        return pd.DataFrame(columns=columnas)
    try:
        df = pd.read_csv(ruta, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columnas)
    for col in columnas:
        if col not in df.columns:
            df[col] = ""
    return df[columnas]


def _escribir_csv(ruta: Path, df: pd.DataFrame) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Historial cotidiano  (data/historial.csv)
# ---------------------------------------------------------------------------

def ruta_historial() -> Path:
    return HISTORIAL_PATH


def leer_historial(ruta: Path | None = None) -> pd.DataFrame:
    ruta = Path(ruta) if ruta else HISTORIAL_PATH
    return _leer_csv(ruta, COLUMNAS_HISTORIAL)


def registrar_resultado(
    digito_real,
    digito_predicho,
    probabilidad_maxima: float,
    ruta: Path | None = None,
    fecha_hora: str | None = None,
) -> pd.DataFrame:
    """Añade una fila al historial. Lanza ValueError si los datos no son válidos."""
    ruta = Path(ruta) if ruta else HISTORIAL_PATH
    real = validar_digito(digito_real)
    predicho = validar_digito(digito_predicho)
    prob = float(probabilidad_maxima)
    if not 0.0 <= prob <= 1.0:
        raise ValueError("La probabilidad debe estar entre 0 y 1.")

    df = leer_historial(ruta)
    nueva = pd.DataFrame(
        [
            {
                "fecha_hora": fecha_hora or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "digito_real": real,
                "digito_predicho": predicho,
                "probabilidad_maxima": round(prob, 6),
                "resultado": "correcto" if real == predicho else "incorrecto",
            }
        ],
        columns=COLUMNAS_HISTORIAL,
    )
    df = pd.concat([df, nueva], ignore_index=True)
    _escribir_csv(ruta, df)
    return df


# ---------------------------------------------------------------------------
# Prueba de 20 dibujos propios  (results/prueba_20_dibujos.csv)
# ---------------------------------------------------------------------------

def ruta_prueba_20(results_dir: Path | None = None) -> Path:
    base = Path(results_dir) if results_dir else RESULTS_DIR
    return base / "prueba_20_dibujos.csv"


def leer_prueba_20(ruta: Path | None = None) -> pd.DataFrame:
    ruta = Path(ruta) if ruta else ruta_prueba_20()
    return _leer_csv(ruta, COLUMNAS_PRUEBA_20)


def estado_prueba_20(ruta: Path | None = None, meta: int = 20) -> dict:
    """Cuenta cuántos dibujos por dígito hay y cuántos faltan."""
    df = leer_prueba_20(ruta)
    conteo = {d: int((df["digito_real"].astype(str) == str(d)).sum()) for d in range(10)}
    return {
        "registrados": int(len(df)),
        "faltan": max(0, meta - len(df)),
        "por_digito": conteo,
        "meta": meta,
    }


def registrar_prueba_20(
    digito_real,
    digito_predicho,
    probabilidad_maxima: float,
    advertencia_baja_confianza: bool,
    probabilidades: list[float] | np.ndarray | None = None,
    ruta: Path | None = None,
    max_por_digito: int = 2,
    meta_total: int = 20,
) -> pd.DataFrame:
    """Registra el PRIMER intento de un dibujo propio.

    Reglas implementadas:
    * La etiqueta debe estar entre 0 y 9.
    * No se puede superar ``max_por_digito`` (2) dibujos por dígito.
    * No se puede superar ``meta_total`` (20) registros.
    * Nunca se sobrescriben ni se borran errores: solo se añade.
    """
    ruta = Path(ruta) if ruta else ruta_prueba_20()
    real = validar_digito(digito_real)
    predicho = validar_digito(digito_predicho)
    prob = float(probabilidad_maxima)
    if not 0.0 <= prob <= 1.0:
        raise ValueError("La probabilidad debe estar entre 0 y 1.")

    df = leer_prueba_20(ruta)
    if len(df) >= meta_total:
        raise ValueError(f"La prueba de {meta_total} dibujos ya está completa.")
    repetidos = int((df["digito_real"].astype(str) == str(real)).sum())
    if repetidos >= max_por_digito:
        raise ValueError(
            f"Ya existen {max_por_digito} dibujos para el dígito {real}. "
            "La prueba admite dos dibujos por dígito."
        )

    probs = " ".join(f"{p:.6f}" for p in (probabilidades if probabilidades is not None else []))
    id_dibujo = len(df) + 1
    nueva = pd.DataFrame(
        [
            {
                "id_dibujo": id_dibujo,
                "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "digito_real": real,
                "digito_predicho": predicho,
                "probabilidad_maxima": round(prob, 6),
                "resultado": "correcto" if real == predicho else "incorrecto",
                "advertencia_baja_confianza": bool(advertencia_baja_confianza),
                "probabilidades": probs,
            }
        ],
        columns=COLUMNAS_PRUEBA_20,
    )
    df = pd.concat([df, nueva], ignore_index=True)
    _escribir_csv(ruta, df)
    return df


def resumen_prueba_20(ruta: Path | None = None) -> dict:
    """Resumen de la prueba de 20 dibujos (sin inventar datos si está vacía)."""
    ruta = Path(ruta) if ruta else ruta_prueba_20()
    df = leer_prueba_20(ruta)
    estado = estado_prueba_20(ruta)
    if df.empty:
        return {
            "realizada": False,
            "registrados": 0,
            "faltan": estado["faltan"],
            "accuracy": None,
            "correctos": 0,
            "incorrectos": 0,
            "advertencias_baja_confianza": 0,
            "por_digito": estado["por_digito"],
        }
    correctos = int((df["resultado"] == "correcto").sum())
    advertencias = df["advertencia_baja_confianza"].astype(str).str.lower().isin(["true", "1", "si", "sí"]).sum()
    return {
        "realizada": int(len(df)) >= estado["meta"],
        "registrados": int(len(df)),
        "faltan": estado["faltan"],
        "accuracy": round(correctos / len(df), 4),
        "correctos": correctos,
        "incorrectos": int(len(df) - correctos),
        "advertencias_baja_confianza": int(advertencias),
        "por_digito": estado["por_digito"],
    }


def guardar_json(ruta: Path, datos: dict) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Banco de escritura manual  (data/banco_dibujos/)
#
# Cada dibujo se guarda como PNG original del lienzo + su 8x8 procesado + fila
# en un índice CSV. Es un conjunto INDEPENDIENTE: no se mezcla con load_digits
# ni con la prueba de 20 dibujos hasta que se documente un experimento nuevo.
# ---------------------------------------------------------------------------

COLUMNAS_BANCO = [
    "id",
    "fecha_hora",
    "digito_real",
    "digito_predicho",
    "probabilidad_maxima",
    "modelo",
    "archivo_lienzo",
    "archivo_8x8",
]


def ruta_banco(directorio: Path | None = None) -> Path:
    from utils import BANCO_DIR

    base = Path(directorio) if directorio else BANCO_DIR
    return base / "indice.csv"


def leer_banco(directorio: Path | None = None) -> pd.DataFrame:
    return _leer_csv(ruta_banco(directorio), COLUMNAS_BANCO)


def guardar_dibujo_banco(
    lienzo,
    matriz8: np.ndarray,
    digito_real,
    digito_predicho,
    probabilidad_maxima: float,
    modelo: str,
    directorio: Path | None = None,
) -> dict:
    """Guarda el lienzo original y su 8x8 procesado. Nunca sobrescribe."""
    from utils import BANCO_DIR

    base = Path(directorio) if directorio else BANCO_DIR
    base.mkdir(parents=True, exist_ok=True)
    real = validar_digito(digito_real)
    predicho = validar_digito(digito_predicho)

    df = leer_banco(base)
    numero = len(df) + 1
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_lienzo = f"{numero:03d}_{real}_{marca}_lienzo.png"
    nombre_8 = f"{numero:03d}_{real}_{marca}_8x8.png"

    lienzo.convert("L").save(base / nombre_lienzo)
    np.save(base / nombre_8.replace(".png", ".npy"), np.asarray(matriz8, dtype=np.uint8))
    from PIL import Image as _Image

    _Image.fromarray((np.asarray(matriz8) / 16.0 * 255).astype(np.uint8), mode="L").save(base / nombre_8)

    fila = pd.DataFrame(
        [
            {
                "id": numero,
                "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "digito_real": real,
                "digito_predicho": predicho,
                "probabilidad_maxima": round(float(probabilidad_maxima), 6),
                "modelo": modelo,
                "archivo_lienzo": nombre_lienzo,
                "archivo_8x8": nombre_8,
            }
        ],
        columns=COLUMNAS_BANCO,
    )
    df = pd.concat([df, fila], ignore_index=True)
    _escribir_csv(ruta_banco(base), df)
    return fila.iloc[0].to_dict()
