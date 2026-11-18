"""Dibuja dígitos REALES en el lienzo de Tkinter (coordenadas relativas) y
muestra el 8x8 que llega al modelo, la predicción y el 8x8 guardado como PNG."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import tkinter as tk

from app import AplicacionDigitos
from utils.preprocessing import preprocesar_imagen

np.set_printoptions(linewidth=200)

FIGURAS = {
    9: [[(0.70, 0.25), (0.50, 0.12), (0.32, 0.28), (0.38, 0.45), (0.58, 0.48), (0.70, 0.35),
         (0.66, 0.60), (0.55, 0.85)]],
    8: [[(0.50, 0.12), (0.35, 0.22), (0.38, 0.42), (0.50, 0.46), (0.62, 0.42), (0.65, 0.22),
         (0.50, 0.12)], [(0.50, 0.46), (0.32, 0.60), (0.35, 0.82), (0.50, 0.88), (0.65, 0.82),
         (0.68, 0.60), (0.50, 0.46)]],
    2: [[(0.25, 0.28), (0.45, 0.12), (0.70, 0.20), (0.70, 0.40), (0.45, 0.60), (0.25, 0.82),
         (0.25, 0.88), (0.75, 0.88)]],
}


def main():
    raiz = tk.Tk()
    app = AplicacionDigitos(raiz)
    raiz.after(700, lambda: ejecutar(app, raiz))
    raiz.mainloop()


def ejecutar(app, raiz):
    ancho = app.lienzo.winfo_width()
    alto = app.lienzo.winfo_height()
    for esperado, trazos in FIGURAS.items():
        app.limpiar_lienzo()
        raiz.update()
        for trazo in trazos:
            app.lienzo.create_line(*[c for x, y in trazo for c in (x * ancho, y * alto)],
                                   fill="black", width=18, capstyle=tk.ROUND, joinstyle=tk.ROUND)
        raiz.update()
        lienzo = app.imagen_lienzo()
        vector, matriz = preprocesar_imagen(lienzo)
        probs = app.modelo.predict(vector.reshape(1, -1), verbose=0)[0]
        orden = np.argsort(probs)[::-1][:3]
        print(f"\n=== esperado {esperado}: predicho {int(np.argmax(probs))} ({probs.max():.4f}) "
              f"top3 {[(int(i), round(float(probs[i]), 3)) for i in orden]} ===")
        print(matriz.astype(int))
        lienzo.save(RAIZ / "results" / f"lienzo_real_{esperado}.png")
    raiz.destroy()


if __name__ == "__main__":
    main()
