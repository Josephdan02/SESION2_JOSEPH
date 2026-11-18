"""Aplicación de escritorio: Sistema interactivo de reconocimiento de dígitos.

Ejecutar con:  python app.py

La aplicación NUNCA entrena: solo carga models/modelo_final.keras.
Si el archivo no existe, avisa y deshabilita el reconocimiento.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from utils import (  # noqa: E402
    DATA_DIR,
    HISTORIAL_PATH,
    METADATA_PATH,
    MODEL_PATH,
    N_CLASES,
    UMBRAL_CONFIANZA,
)
from utils import history as hist_utils  # noqa: E402
from utils.preprocessing import lienzo_vacio, preprocesar_imagen  # noqa: E402

# Paleta de colores
COLOR_FONDO = "#f4f6fb"
COLOR_TARJETA = "#ffffff"
COLOR_PRIMARIO = "#2d3e70"
COLOR_ACENTO = "#3b82f6"
COLOR_EXITO = "#15803d"
COLOR_ERROR = "#b91c1c"
COLOR_ALERTA = "#b45309"
COLOR_TEXTO = "#1f2937"
COLOR_BORDE = "#d5dbe7"

ANCHO_LIENZO = 320
ALTO_LIENZO = 320
GRUESO_TRAZO = 18


class AplicacionDigitos:
    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        self.raiz.title("Sistema interactivo de reconocimiento de dígitos")
        self.raiz.geometry("1300x820")
        self.raiz.minsize(1120, 720)
        self.raiz.configure(bg=COLOR_FONDO)

        self.modelo = None
        self.metadata: dict = {}
        self.ultimo_vector: np.ndarray | None = None
        self.ultimas_probabilidades: np.ndarray | None = None
        self.ultima_prediccion: dict | None = None
        self.registro_realizado = False  # evita duplicados por doble clic
        self.guardando = False  # evita doble registro en la prueba de 20 dibujos
        self.trazo_ids: list[int] = []
        self.dibujo_actual = 0  # identifica cada dibujo nuevo del lienzo

        self._configurar_estilos()
        self._construir_interfaz()
        self.cargar_modelo()

    # ------------------------------------------------------------------
    # Estilos y construcción
    # ------------------------------------------------------------------
    def _configurar_estilos(self):
        estilo = ttk.Style(self.raiz)
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass
        estilo.configure("TFrame", background=COLOR_FONDO)
        estilo.configure("Tarjeta.TFrame", background=COLOR_TARJETA)
        estilo.configure("TLabel", background=COLOR_FONDO, foreground=COLOR_TEXTO, font=("Segoe UI", 10))
        estilo.configure("Tarjeta.TLabel", background=COLOR_TARJETA, foreground=COLOR_TEXTO, font=("Segoe UI", 10))
        estilo.configure("Titulo.TLabel", background=COLOR_PRIMARIO, foreground="white",
                         font=("Segoe UI", 16, "bold"))
        estilo.configure("Seccion.TLabel", background=COLOR_TARJETA, foreground=COLOR_PRIMARIO,
                         font=("Segoe UI", 11, "bold"))
        estilo.configure("Resultado.TLabel", background=COLOR_TARJETA, foreground=COLOR_ACENTO,
                         font=("Segoe UI", 22, "bold"))
        estilo.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        estilo.configure("Accion.TButton", background=COLOR_ACENTO, foreground="white",
                         font=("Segoe UI", 10, "bold"), padding=(14, 10))
        estilo.map("Accion.TButton", background=[("active", "#2563eb")])
        estilo.configure("Aviso.TLabel", background=COLOR_TARJETA, foreground=COLOR_ALERTA,
                         font=("Segoe UI", 9, "bold"), wraplength=330)
        estilo.configure("Info.TLabel", background=COLOR_TARJETA, foreground=COLOR_TEXTO,
                         font=("Segoe UI", 10), wraplength=330)
        estilo.configure("Notebook.TNotebook", background=COLOR_FONDO, borderwidth=0)
        estilo.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 6))
        estilo.configure("Treeview", font=("Segoe UI", 9), rowheight=24)
        estilo.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _construir_interfaz(self):
        encabezado = tk.Frame(self.raiz, bg=COLOR_PRIMARIO, height=58)
        encabezado.pack(fill="x")
        tk.Label(
            encabezado,
            text="Sistema interactivo de reconocimiento de dígitos",
            bg=COLOR_PRIMARIO,
            fg="white",
            font=("Segoe UI", 17, "bold"),
            padx=18,
            pady=14,
        ).pack(side="left")
        self.etiqueta_modelo = tk.Label(
            encabezado, text="", bg=COLOR_PRIMARIO, fg="#cbd5e1", font=("Segoe UI", 9), padx=18
        )
        self.etiqueta_modelo.pack(side="right")

        notebook = ttk.Notebook(self.raiz)
        notebook.pack(fill="both", expand=True, padx=14, pady=12)

        self.pestana_principal = ttk.Frame(notebook)
        self.pestana_historial = ttk.Frame(notebook)
        self.pestana_prueba = ttk.Frame(notebook)
        notebook.add(self.pestana_principal, text="Reconocer")
        notebook.add(self.pestana_historial, text="Historial")
        notebook.add(self.pestana_prueba, text="Prueba de 20 dibujos")

        self._construir_panel_principal(self.pestana_principal)
        self._construir_historial(self.pestana_historial)
        self._construir_prueba20(self.pestana_prueba)

    def _tarjeta(self, padre, titulo: str | None = None, **pack):
        marco = ttk.Frame(padre, style="Tarjeta.TFrame", padding=12)
        marco.pack(**pack)
        if titulo:
            ttk.Label(marco, text=titulo, style="Seccion.TLabel").pack(anchor="w", pady=(0, 8))
        return marco

    def _construir_panel_principal(self, padre):
        principal = ttk.Frame(padre)
        principal.pack(fill="both", expand=True, padx=8, pady=8)

        izquierda = self._tarjeta(principal, "Dibuja un número del 0 al 9",
                                  side="left", fill="y", padx=(0, 10))
        self.lienzo = tk.Canvas(
            izquierda, width=ANCHO_LIENZO, height=ALTO_LIENZO, bg="white",
            highlightthickness=2, highlightbackground=COLOR_BORDE, cursor="pencil",
        )
        self.lienzo.pack(pady=4)
        self.lienzo.bind("<ButtonPress-1>", self.iniciar_trazo)
        self.lienzo.bind("<B1-Motion>", self.dibujar)
        self.lienzo.bind("<ButtonRelease-1>", self.terminar_trazo)
        self.lienzo.bind("<Configure>", lambda _e: self._imagen_fondo())

        botones = ttk.Frame(izquierda, style="Tarjeta.TFrame")
        botones.pack(pady=(10, 4))
        ttk.Button(botones, text="Limpiar", command=self.limpiar_lienzo, width=12).pack(side="left", padx=4)
        ttk.Button(botones, text="Reconocer", style="Accion.TButton",
                   command=self.reconocer).pack(side="left", padx=4)
        ttk.Label(izquierda, text="Mantén presionado el botón izquierdo del ratón para dibujar.",
                  style="Tarjeta.TLabel", font=("Segoe UI", 9), wraplength=ANCHO_LIENZO).pack(pady=(4, 0))

        centro = ttk.Frame(principal)
        centro.pack(side="left", fill="both", expand=True, padx=5)

        tarjeta_imagen = self._tarjeta(centro, "Imagen procesada (8x8) -> entrada de la red",
                                       fill="x")
        self.lienzo_preprocesado = tk.Canvas(tarjeta_imagen, width=184, height=184, bg="white",
                                             highlightthickness=0)
        self.lienzo_preprocesado.pack(anchor="center", pady=4)
        self._imagen_fondo(preprocesado=True)

        tarjeta_resultado = self._tarjeta(centro, "Resultado", fill="x", pady=(10, 0))
        self.etiqueta_resultado = ttk.Label(tarjeta_resultado, text="Aún no hay predicción",
                                            style="Resultado.TLabel")
        self.etiqueta_resultado.pack(anchor="w")
        self.etiqueta_probabilidad = ttk.Label(tarjeta_resultado, style="Info.TLabel",
                                               text="Dibuja un número y pulsa Reconocer.")
        self.etiqueta_probabilidad.pack(anchor="w", pady=(2, 6))
        self.etiqueta_aviso = ttk.Label(tarjeta_resultado, style="Aviso.TLabel", text="")
        self.etiqueta_aviso.pack(anchor="w")
        self.etiqueta_top3 = ttk.Label(tarjeta_resultado, style="Info.TLabel", text="Top 3: -")
        self.etiqueta_top3.pack(anchor="w", pady=(6, 0))

        derecha = self._tarjeta(principal, "Probabilidades (0 al 9)",
                                side="left", fill="both", expand=True, padx=(10, 0))
        self.figura = plt.Figure(figsize=(4.3, 3.3), dpi=100)
        self.eje = self.figura.add_subplot(111)
        self.canvas_grafico = FigureCanvasTkAgg(self.figura, master=derecha)
        self.canvas_grafico.get_tk_widget().pack(fill="both", expand=True)
        self.dibujar_grafico_vacio()

        barra_registro = ttk.Frame(derecha, style="Tarjeta.TFrame")
        barra_registro.pack(fill="x", pady=(10, 0))
        ttk.Label(barra_registro, text="¿Qué número quisiste dibujar?",
                  style="Tarjeta.TLabel").pack(anchor="w")
        botones_registro = ttk.Frame(barra_registro, style="Tarjeta.TFrame")
        botones_registro.pack(anchor="w", pady=(6, 0))
        self.variable_digito = tk.StringVar(value="0")
        selector = ttk.Combobox(
            botones_registro, textvariable=self.variable_digito, width=3, state="readonly",
            values=[str(i) for i in range(N_CLASES)],
        )
        selector.pack(side="left", padx=(0, 6))
        ttk.Button(botones_registro, text="Registrar resultado",
                   command=self.registrar_resultado).pack(side="left", padx=(0, 6))
        ttk.Button(botones_registro, text="Registrar en prueba 20",
                   command=self.registrar_prueba20).pack(side="left", padx=(0, 6))
        ttk.Button(botones_registro, text="Guardar dibujo (banco)",
                   command=self.guardar_en_banco).pack(side="left")
        self.raiz.bind("<F11>", lambda _e: self.alternar_maximizado())

    def alternar_maximizado(self):
        """F11: alterna entre ventana normal y maximizada."""
        try:
            self.raiz.state("zoomed" if self.raiz.state() != "zoomed" else "normal")
        except tk.TclError:
            pass

    def _construir_historial(self, padre):
        marco = self._tarjeta(padre, "Historial de registros (data/historial.csv)", fill="both", expand=True)
        barra = ttk.Frame(marco, style="Tarjeta.TFrame")
        barra.pack(fill="x", pady=(0, 8))
        ttk.Button(barra, text="Actualizar", command=self.refrescar_historial).pack(side="left")
        self.resumen_historial = ttk.Label(barra, style="Tarjeta.TLabel", text="")
        self.resumen_historial.pack(side="left", padx=12)

        columnas = ("fecha_hora", "digito_real", "digito_predicho", "probabilidad_maxima", "resultado")
        contenedor = ttk.Frame(marco, style="Tarjeta.TFrame")
        contenedor.pack(fill="both", expand=True)
        self.tabla_historial = ttk.Treeview(contenedor, columns=columnas, show="headings", height=18)
        titulos = {
            "fecha_hora": ("Fecha y hora", 170),
            "digito_real": ("Dígito indicado", 120),
            "digito_predicho": ("Dígito predicho", 120),
            "probabilidad_maxima": ("Probabilidad", 110),
            "resultado": ("Resultado", 110),
        }
        for col, (texto, ancho) in titulos.items():
            self.tabla_historial.heading(col, text=texto)
            self.tabla_historial.column(col, width=ancho, anchor="center")
        barra_v = ttk.Scrollbar(contenedor, orient="vertical", command=self.tabla_historial.yview)
        self.tabla_historial.configure(yscrollcommand=barra_v.set)
        self.tabla_historial.pack(side="left", fill="both", expand=True)
        barra_v.pack(side="right", fill="y")

    def _construir_prueba20(self, padre):
        marco = self._tarjeta(padre, "Prueba de 20 dibujos propios (2 por cada dígito)", fill="both", expand=True)
        ttk.Label(
            marco,
            style="Tarjeta.TLabel",
            wraplength=1100,
            justify="left",
            text=(
                "Dibuja el número en la pestaña Reconocer, pulsa Reconocer y luego usa "
                "'Registrar en prueba 20'. Se guarda el PRIMER intento y nunca se sobrescribe. "
                "Si la probabilidad máxima es menor que 0.70 también queda marcada la "
                "advertencia de baja confianza. El archivo se guarda en results/prueba_20_dibujos.csv."
            ),
        ).pack(anchor="w", pady=(0, 8))
        self.etiqueta_estado_prueba = ttk.Label(marco, style="Info.TLabel", text="")
        self.etiqueta_estado_prueba.pack(anchor="w")
        self.etiqueta_resumen_prueba = ttk.Label(marco, style="Tarjeta.TLabel", text="")
        self.etiqueta_resumen_prueba.pack(anchor="w", pady=(4, 8))

        columnas = ("id_dibujo", "digito_real", "digito_predicho", "probabilidad_maxima",
                    "resultado", "advertencia_baja_confianza", "fecha_hora")
        contenedor = ttk.Frame(marco, style="Tarjeta.TFrame")
        contenedor.pack(fill="both", expand=True)
        self.tabla_prueba = ttk.Treeview(contenedor, columns=columnas, show="headings", height=14)
        titulos = {
            "id_dibujo": ("#", 45),
            "digito_real": ("Dibujé", 80),
            "digito_predicho": ("Predice", 80),
            "probabilidad_maxima": ("Prob. máx.", 90),
            "resultado": ("Resultado", 100),
            "advertencia_baja_confianza": ("Baja confianza", 120),
            "fecha_hora": ("Fecha y hora", 160),
        }
        for col, (texto, ancho) in titulos.items():
            self.tabla_prueba.heading(col, text=texto)
            self.tabla_prueba.column(col, width=ancho, anchor="center")
        self.tabla_prueba.pack(fill="both", expand=True)
        ttk.Button(marco, text="Actualizar", command=self.refrescar_prueba20).pack(anchor="w", pady=(8, 0))
        self.refrescar_prueba20()

    # ------------------------------------------------------------------
    # Modelo
    # ------------------------------------------------------------------
    def cargar_modelo(self):
        if not MODEL_PATH.exists():
            self.etiqueta_modelo.config(text="Modelo no encontrado")
            messagebox.showwarning(
                "Modelo no encontrado",
                "No se encontró models/modelo_final.keras.\n\n"
                "Primero ejecuta en una terminal:\n\n    python train.py\n\n"
                "Después vuelve a abrir esta aplicación.",
            )
            return
        try:
            import tensorflow as tf
            from tensorflow import keras

            self.modelo = keras.models.load_model(MODEL_PATH)
            if METADATA_PATH.exists():
                import json

                self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            nombre = self.metadata.get("nombre_modelo", "desconocido")
            self.etiqueta_modelo.config(text=f"Modelo cargado: {nombre} | TF {tf.__version__}")
        except Exception as error:  # noqa: BLE001
            self.modelo = None
            self.etiqueta_modelo.config(text="Error al cargar el modelo")
            messagebox.showerror(
                "Error al cargar el modelo",
                f"No fue posible cargar {MODEL_PATH}.\n\n{error}\n\n"
                "Vuelve a ejecutar 'python train.py'.",
            )

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------
    def _imagen_fondo(self, preprocesado: bool = False):
        if preprocesado:
            lienzo = self.lienzo_preprocesado
            lado = min(lienzo.winfo_width(), lienzo.winfo_height()) or 184
            lienzo.delete("pixel")
            lado = max(32, lado - 8)
            paso = lado / 8
            paleta = {0: "#111827", 1: "#3b82f6", 2: "#93c5fd", 3: "#eff6ff"}
            for f in range(8):
                for c in range(8):
                    valor = getattr(self, "_matriz_actual", np.zeros((8, 8)))[f, c]
                    nivel = min(3, int(valor / 16.0 * 3.99))
                    x0, y0 = c * paso + 4, f * paso + 4
                    lienzo.create_rectangle(
                        x0, y0, x0 + paso, y0 + paso,
                        fill=paleta[nivel] if valor > 0 else "#ffffff",
                        outline="#cbd5e1", tags="pixel", width=1,
                    )
        else:
            self.lienzo.delete("fondo")

    def iniciar_trazo(self, evento):
        self.registro_realizado = False
        ultimo = getattr(self, "_ultimo_punto", None)
        self._ultimo_punto = (evento.x, evento.y)
        if ultimo is None:
            self.trazo_ids.append(
                self.lienzo.create_line(evento.x, evento.y, evento.x + 1, evento.y + 1,
                                        fill="black", width=GRUESO_TRAZO,
                                        capstyle=tk.ROUND, smooth=True)
            )

    def dibujar(self, evento):
        ultimo = getattr(self, "_ultimo_punto", (evento.x, evento.y))
        self.trazo_ids.append(
            self.lienzo.create_line(ultimo[0], ultimo[1], evento.x, evento.y,
                                    fill="black", width=GRUESO_TRAZO,
                                    capstyle=tk.ROUND, smooth=True)
        )
        self._ultimo_punto = (evento.x, evento.y)

    def terminar_trazo(self, _evento):
        self._ultimo_punto = None

    def limpiar_lienzo(self):
        self.lienzo.delete("all")
        self.trazo_ids.clear()
        self._ultimo_punto = None
        self.ultimo_vector = None
        self.ultimas_probabilidades = None
        self.ultima_prediccion = None
        self.registro_realizado = False
        self.dibujo_actual += 1
        self._matriz_actual = np.zeros((8, 8))
        self._imagen_fondo(preprocesado=True)
        self.etiqueta_resultado.config(text="Aún no hay predicción")
        self.etiqueta_probabilidad.config(text="Dibuja un número y pulsa Reconocer.")
        self.etiqueta_aviso.config(text="")
        self.etiqueta_top3.config(text="Top 3: -")
        self.dibujar_grafico_vacio()

    def imagen_lienzo(self) -> Image.Image:
        """Captura el contenido del lienzo como imagen Pillow en escala de grises."""
        from PIL import ImageDraw

        ancho = self.lienzo.winfo_width() or ANCHO_LIENZO
        alto = self.lienzo.winfo_height() or ALTO_LIENZO
        imagen = Image.new("L", (ancho, alto), 255)
        dibujo = ImageDraw.Draw(imagen)
        for item in self.lienzo.find_all():
            if self.lienzo.type(item) != "line":
                continue
            coords = self.lienzo.coords(item)
            if len(coords) < 4:
                continue
            grosor = int(float(self.lienzo.itemcget(item, "width") or GRUESO_TRAZO))
            color = self.lienzo.itemcget(item, "fill") or "black"
            for i in range(0, len(coords) - 2, 2):
                x0, y0, x1, y1 = coords[i], coords[i + 1], coords[i + 2], coords[i + 3]
                dibujo.line([(x0, y0), (x1, y1)], fill=color, width=grosor)
                r = grosor / 2
                dibujo.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=color)
                dibujo.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color)
        return imagen

    # ------------------------------------------------------------------
    # Reconocimiento
    # ------------------------------------------------------------------
    def reconocer(self):
        if self.modelo is None:
            messagebox.showwarning(
                "Modelo no disponible",
                "Primero ejecuta 'python train.py' para generar models/modelo_final.keras.",
            )
            return
        imagen = self.imagen_lienzo()
        if lienzo_vacio(imagen, umbral=200):
            self.ultima_prediccion = None
            messagebox.showinfo(
                "Lienzo vacío",
                "El lienzo está vacío. Dibuja un número del 0 al 9 con el ratón antes de reconocer.",
            )
            return
        try:
            vector, matriz = preprocesar_imagen(imagen)
        except ValueError as error:
            messagebox.showinfo("Lienzo vacío", str(error))
            return

        probabilidades = np.asarray(self.modelo.predict(vector.reshape(1, -1), verbose=0)[0], dtype=float)
        prediccion = int(np.argmax(probabilidades))
        prob_max = float(probabilidades.max())
        top3 = np.argsort(probabilidades)[::-1][:3]

        self.ultimo_vector = vector
        self.ultimas_probabilidades = probabilidades
        self._matriz_actual = matriz
        self.registro_realizado = False
        self.ultima_prediccion = {
            "digito": prediccion,
            "probabilidad": prob_max,
            "vector": vector,
            "matriz": matriz,
            "imagen_lienzo": imagen,
            "probabilidades": probabilidades,
        }
        self._imagen_fondo(preprocesado=True)

        self.etiqueta_resultado.config(text=f"Dígito reconocido: {prediccion}")
        self.etiqueta_probabilidad.config(text=f"Probabilidad máxima: {prob_max * 100:.2f}%")
        if prob_max < UMBRAL_CONFIANZA:
            self.etiqueta_aviso.config(
                text="Advertencia: el modelo tiene baja confianza en esta predicción. "
                     "Intenta dibujar el número con mayor claridad."
            )
        else:
            self.etiqueta_aviso.config(text="")
        self.etiqueta_top3.config(
            text="Top 3: " + " | ".join(f"{d}: {probabilidades[d] * 100:.2f}%" for d in top3)
        )
        self.dibujar_grafico(probabilidades)

    # ------------------------------------------------------------------
    # Gráfico
    # ------------------------------------------------------------------
    def dibujar_grafico_vacio(self):
        self.eje.clear()
        self.eje.set_ylim(0, 1)
        self.eje.set_xlim(-0.6, 9.6)
        self.eje.set_xticks(range(10))
        self.eje.set_ylabel("Probabilidad")
        self.eje.set_xlabel("Dígito")
        self.eje.set_title("Aún sin predicción")
        self.eje.grid(axis="y", alpha=0.3)
        self.figura.tight_layout()
        self.canvas_grafico.draw_idle()

    def dibujar_grafico(self, probabilidades: np.ndarray):
        self.eje.clear()
        colores = [COLOR_ACENTO if i == int(np.argmax(probabilidades)) else "#c7d7f5" for i in range(10)]
        barras = self.eje.bar(range(10), probabilidades, color=colores, edgecolor="#334155")
        self.eje.set_ylim(0, 1.05)
        self.eje.set_xticks(range(10))
        self.eje.set_xlabel("Dígito")
        self.eje.set_ylabel("Probabilidad")
        self.eje.set_title("Probabilidades por clase", pad=18)
        self.eje.axhline(UMBRAL_CONFIANZA, color=COLOR_ALERTA, linestyle="--", linewidth=1,
                         label="Umbral 0.70")
        self.eje.legend(loc="upper left", fontsize=8, framealpha=0.9)
        self.eje.grid(axis="y", alpha=0.3)
        for barra, prob in zip(barras, probabilidades):
            if prob > 0.05:
                self.eje.text(barra.get_x() + barra.get_width() / 2, prob + 0.02,
                              f"{prob:.2f}", ha="center", fontsize=7)
        self.figura.tight_layout()
        self.canvas_grafico.draw_idle()

    # ------------------------------------------------------------------
    # Registro
    # ------------------------------------------------------------------
    def _validar_prediccion(self) -> bool:
        if self.ultima_prediccion is None:
            messagebox.showinfo(
                "Sin predicción",
                "Todavía no existe una predicción. Dibuja un número y pulsa Reconocer.",
            )
            return False
        return True

    def registrar_resultado(self):
        if not self._validar_prediccion():
            return
        if self.registro_realizado:
            messagebox.showinfo("Registro duplicado",
                                "Este resultado ya fue registrado. Dibuja otro número para registrar uno nuevo.")
            return
        try:
            digito_real = hist_utils.validar_digito(self.variable_digito.get())
            hist_utils.registrar_resultado(
                digito_real,
                self.ultima_prediccion["digito"],
                self.ultima_prediccion["probabilidad"],
            )
        except (ValueError, OSError) as error:
            messagebox.showerror("No se pudo registrar", str(error))
            return
        self.registro_realizado = True
        self.refrescar_historial()
        messagebox.showinfo(
            "Registro guardado",
            f"Se registró en {HISTORIAL_PATH.name}:\n\n"
            f"Dibujaste: {digito_real}\nPredicción: {self.ultima_prediccion['digito']} "
            f"({self.ultima_prediccion['probabilidad'] * 100:.2f}%)",
        )

    def registrar_prueba20(self):
        if not self._validar_prediccion() or self.guardando:
            return
        try:
            digito_real = hist_utils.validar_digito(self.variable_digito.get())
            hist_utils.registrar_prueba_20(
                digito_real,
                self.ultima_prediccion["digito"],
                self.ultima_prediccion["probabilidad"],
                self.ultima_prediccion["probabilidad"] < UMBRAL_CONFIANZA,
                self.ultima_prediccion["probabilidades"],
            )
        except (ValueError, OSError) as error:
            messagebox.showwarning("No se pudo registrar en la prueba", str(error))
            return
        finally:
            self.guardando = False
        self.guardando = True
        self.refrescar_prueba20()
        estado = hist_utils.estado_prueba_20()
        messagebox.showinfo(
            "Prueba de 20 dibujos",
            f"Registro guardado. Llevas {estado['registrados']} de {estado['meta']}. "
            f"Faltan {estado['faltan']} dibujos.\n\nRecuerda: limpia el lienzo antes del siguiente dibujo.",
        )

    def guardar_en_banco(self):
        """Guarda el lienzo y su 8x8 procesado en data/banco_dibujos (conjunto aparte)."""
        if not self._validar_prediccion():
            return
        try:
            digito_real = hist_utils.validar_digito(self.variable_digito.get())
            fila = hist_utils.guardar_dibujo_banco(
                self.ultima_prediccion["imagen_lienzo"],
                self.ultima_prediccion["matriz"],
                digito_real,
                self.ultima_prediccion["digito"],
                self.ultima_prediccion["probabilidad"],
                self.metadata.get("nombre_modelo", "desconocido"),
            )
        except (ValueError, OSError) as error:
            messagebox.showerror("No se pudo guardar", str(error))
            return
        total = len(hist_utils.leer_banco())
        messagebox.showinfo(
            "Dibujo guardado en el banco",
            f"Dibujo #{fila['id']} guardado (real {digito_real}, predicho "
            f"{self.ultima_prediccion['digito']}).\n\n"
            f"Total en el banco: {total}.\nCarpeta: data/banco_dibujos/\n\n"
            "Este conjunto está separado de entrenamiento, validación y prueba.",
        )

    # ------------------------------------------------------------------
    # Tablas
    # ------------------------------------------------------------------
    def refrescar_historial(self):
        for fila in self.tabla_historial.get_children():
            self.tabla_historial.delete(fila)
        df = hist_utils.leer_historial()
        for _, fila in df.iloc[::-1].iterrows():
            prob = float(fila["probabilidad_maxima"])
            self.tabla_historial.insert("", "end", values=(
                fila["fecha_hora"],
                int(float(fila["digito_real"])),
                int(float(fila["digito_predicho"])),
                f"{prob * 100:.2f}%",
                fila["resultado"],
            ))
        if df.empty:
            self.resumen_historial.config(text="Todavía no hay registros.")
        else:
            correctos = int((df["resultado"] == "correcto").sum())
            self.resumen_historial.config(
                text=f"Registros: {len(df)} | Correctos: {correctos} | "
                     f"Exactitud: {correctos / len(df) * 100:.1f}%"
            )

    def refrescar_prueba20(self):
        for fila in self.tabla_prueba.get_children():
            self.tabla_prueba.delete(fila)
        df = hist_utils.leer_prueba_20()
        for _, fila in df.iterrows():
            self.tabla_prueba.insert("", "end", values=(
                int(fila["id_dibujo"]),
                int(float(fila["digito_real"])),
                int(float(fila["digito_predicho"])),
                f"{float(fila['probabilidad_maxima']) * 100:.2f}%",
                fila["resultado"],
                "Sí" if str(fila["advertencia_baja_confianza"]).lower() in ("true", "1") else "No",
                fila["fecha_hora"],
            ))
        estado = hist_utils.estado_prueba_20()
        faltan_por_digito = {d: 2 - c for d, c in estado["por_digito"].items() if c < 2}
        detalle = ", ".join(f"{d}: falta(n) {c}" for d, c in faltan_por_digito.items()) or "completa"
        self.etiqueta_estado_prueba.config(
            text=f"Registrados: {estado['registrados']} de {estado['meta']} | "
                 f"Faltan: {estado['faltan']} | Por dígito -> {detalle}"
        )
        resumen = hist_utils.resumen_prueba_20()
        if resumen["registrados"] == 0:
            self.etiqueta_resumen_prueba.config(
                text="Aún no has realizado ningún dibujo de esta prueba. Debes hacerlos tú mismo en la aplicación."
            )
        else:
            self.etiqueta_resumen_prueba.config(
                text=(
                    f"Accuracy parcial: {resumen['accuracy'] * 100:.1f}% | "
                    f"Correctos: {resumen['correctos']} | Incorrectos: {resumen['incorrectos']} | "
                    f"Avisos de baja confianza: {resumen['advertencias_baja_confianza']}"
                    + ("" if resumen["realizada"] else " (prueba todavía incompleta)")
                )
            )


def _dibujar_tramo(imagen: Image.Image, x0, y0, x1, y1, grosor, color):
    """Dibuja un segmento grueso con extremos redondeados (función auxiliar de pruebas)."""
    from PIL import ImageDraw

    dibujo = ImageDraw.Draw(imagen)
    dibujo.line([(x0, y0), (x1, y1)], fill=color, width=grosor)
    r = grosor / 2
    dibujo.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=color)
    dibujo.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raiz = tk.Tk()
    app = AplicacionDigitos(raiz)
    app.refrescar_historial()
    raiz.mainloop()


if __name__ == "__main__":
    main()
