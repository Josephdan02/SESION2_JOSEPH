"""Comprobación real de la interfaz gráfica (se ejecuta con una ventana visible).

Simula un dibujo en el lienzo, pulsa Reconocer, toma una captura de la ventana
y cierra la aplicación. Sirve para verificar que la GUI se construye y predice.

Uso:  python tools/verificar_treeview.py
Salida: results/captura_interfaz.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import tkinter as tk  # noqa: E402

from app import AplicacionDigitos  # noqa: E402

CAPTURA = RAIZ / "results" / "captura_interfaz.png"


def capturar_ventana(raiz: tk.Tk, ruta: Path) -> str:
    """Captura la ventana con la API de Windows (ctypes), sin dependencias extra."""
    import ctypes

    raiz.update()
    time.sleep(0.3)
    raiz.update()
    user32 = ctypes.windll.user32
    hwnd = user32.GetParent(raiz.winfo_id()) or raiz.winfo_id()
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    ancho, alto = rect.right - rect.left, rect.bottom - rect.top
    if ancho <= 0 or alto <= 0:
        return "No se pudo medir la ventana."

    gdi32 = ctypes.windll.gdi32
    hdc = user32.GetWindowDC(hwnd)
    hdc_mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, ancho, alto)
    gdi32.SelectObject(hdc_mem, bmp)
    user32.PrintWindow(hwnd, hdc_mem, 2)  # PW_RENDERFULLCONTENT

    from PIL import Image

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    clase_bmi = type("BITMAPINFO", (ctypes.Structure,), {"_fields_": [
        ("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]})
    bmi = clase_bmi()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = ancho
    bmi.bmiHeader.biHeight = -alto
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0
    buffer = ctypes.create_string_buffer(ancho * alto * 4)
    gdi32.GetDIBits(hdc_mem, bmp, 0, alto, buffer, ctypes.byref(bmi), 0)
    imagen = Image.frombuffer("RGBA", (ancho, alto), buffer, "raw", "BGRA", 0, 1).convert("RGB")
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    imagen.save(ruta)
    return str(ruta)


def guion(app: AplicacionDigitos, raiz: tk.Tk):
    app.raiz.update()
    app.variable_digito.set("7")

    x0, y0 = 90, 70
    puntos = [(x0, y0), (x0 + 160, y0), (x0 + 90, y0 + 190)]
    app.lienzo.create_line(*puntos, fill="black", width=18, capstyle=tk.ROUND)

    iniciar = time.time()
    app.reconocer()
    print(f"Reconocimiento: {time.time() - iniciar:.2f} s")
    if app.ultima_prediccion is None:
        print("ERROR: no se obtuvo predicción.")
    else:
        print(
            "Predicción:", app.ultima_prediccion["digito"],
            f"| probabilidad {app.ultima_prediccion['probabilidad']:.4f}",
            f"| aviso={'sí' if app.ultima_prediccion['probabilidad'] < 0.70 else 'no'}",
        )
    app.raiz.update()
    time.sleep(1.0)
    print("Captura:", capturar_ventana(app.raiz, CAPTURA))

    # Recorrer las pestañas para verificar que se dibujan sin errores.
    notebook = app.pestana_principal.master
    for i in range(3):
        notebook.select(i)
        app.raiz.update()
        time.sleep(0.4)
    notebook.select(0)

    # Prueba de lienzo vacío (no debe lanzar excepción ni predecir).
    app.limpiar_lienzo()
    app.raiz.update()
    print("Tras limpiar, ultima_prediccion =", app.ultima_prediccion)
    app.raiz.after(300, app.raiz.destroy)


def main():
    raiz = tk.Tk()
    app = AplicacionDigitos(raiz)
    app.refrescar_historial()
    raiz.after(900, lambda: guion(app, raiz))
    raiz.mainloop()
    print("Ventana cerrada correctamente.")


if __name__ == "__main__":
    main()
