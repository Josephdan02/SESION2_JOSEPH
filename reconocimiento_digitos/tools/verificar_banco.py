"""Verifica el guardado del banco de dibujos usando una carpeta temporal."""

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))

import numpy as np

from benchmark_dibujos import dibujo
from utils import history as hist
from utils.preprocessing import preprocesar_imagen

with tempfile.TemporaryDirectory() as tmp:
    lienzo = dibujo(9)
    _, matriz = preprocesar_imagen(lienzo)
    fila = hist.guardar_dibujo_banco(lienzo, matriz, 9, 9, 0.98, "Sequential", directorio=Path(tmp))
    df = hist.leer_banco(Path(tmp))
    archivos = sorted(p.name for p in Path(tmp).iterdir())
    print("fila:", fila)
    print("filas en indice:", len(df))
    print("archivos:", archivos)
    assert len(df) == 1
    assert any(a.endswith("_lienzo.png") for a in archivos)
    assert any(a.endswith("_8x8.png") for a in archivos)
    assert np.load(Path(tmp) / archivos[0].replace(".png", ".npy")) is not None
print("banco OK")
