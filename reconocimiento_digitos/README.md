# Sistema interactivo de reconocimiento de dígitos

Proyecto académico de *Machine Learning* con **redes neuronales densas** (sin CNN) que
entrena dos modelos sobre el conjunto `load_digits` de *scikit-learn* y permite dibujar un
número del 0 al 9 con el ratón en una aplicación de escritorio hecha con **Tkinter**.

Todo se ejecuta localmente: el entrenamiento no se repite al abrir la aplicación y las
predicciones nunca consultan servicios externos.

---

## 1. Descripción del proyecto

1. **`train.py`** carga las 1797 imágenes de 8x8 píxeles de `load_digits`, las divide en
   entrenamiento/validación/prueba, entrena **dos arquitecturas distintas** (API Sequential
   y API Functional), las compara **solo con validación**, elige la mejor por **F1 macro**,
   la evalúa **una única vez** en prueba, guarda el modelo y genera todos los informes
   (gráficos, matriz de confusión, análisis de errores, prueba de persistencia y prueba de ruido).
   Entrena también el **experimento original v1** (sin aumentación) y guarda su copia en
   `models/modelo_final_v1_load_digits.keras`.
2. **`app.py`** abre la interfaz gráfica. Carga `models/modelo_final.keras`, transforma el
   dibujo del usuario al formato exacto que espera la red y muestra el resultado, las tres
   mejores clases y el gráfico de probabilidades. Permite guardar cada prueba en un historial,
   realizar la **prueba de 20 dibujos propios** y guardar dibujos en el **banco** de escritura
   manual (`data/banco_dibujos/`), separado de los conjuntos oficiales.

> **Corrección aplicada (v2).** El sistema original obtenía 98.33 % en prueba con
> `load_digits` pero fallaba con dibujos reales (9→4, 2→7, 8→4). La causa estaba en el
> preprocesamiento, no en la red: la reducción del lienzo dejaba **0.7 píxeles activos** por
> imagen (el dataset tiene **18.7**) y centraba el dígito en el borde superior izquierdo.
> La sección 16 documenta la auditoría completa, los cambios y los resultados verificables.

## 2. Objetivo

Comprender de forma práctica el ciclo completo de un proyecto de aprendizaje automático:

- preparar datos y particiones reproducibles,
- diseñar, entrenar y comparar dos redes neuronales densas,
- evaluar con métricas correctas (accuracy y F1 macro) y analizar errores,
- persistir el modelo y reutilizarlo,
- integrar el modelo en una aplicación real y medir su comportamiento con datos propios.

## 3. Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python 3.11 | Lenguaje base |
| TensorFlow / Keras | Redes neuronales densas |
| scikit-learn | Dataset `load_digits`, particiones y métricas |
| NumPy | Operaciones con arrays |
| pandas | CSV de comparación, errores, historial y prueba de 20 dibujos |
| Matplotlib | Curvas, matriz de confusión y gráfico de probabilidades |
| Pillow | Procesamiento de la imagen del lienzo |
| Tkinter | Interfaz gráfica (biblioteca estándar) |
| pytest | Pruebas automatizadas |

Versiones probadas en este entorno: TensorFlow 2.21, scikit-learn 1.8, NumPy 2.4,
Matplotlib 3.11, pandas 3.0, Pillow 12.3, pytest 9.2 (Python 3.11.3, Windows).
`requirements.txt` usa rangos compatibles para evitar dependencias innecesarias.

## 4. Estructura de carpetas

```
reconocimiento_digitos/
├── app.py                        # Aplicación Tkinter
├── train.py                      # Entrenamiento, evaluación e informes
├── requirements.txt
├── README.md
├── models/
│   ├── modelo_final.keras        # Modelo definitivo (generado por train.py)
│   └── metadata.json             # Arquitectura, normalización, semilla, umbral...
├── results/
│   ├── comparacion_modelos.csv
│   ├── metricas_prueba.json
│   ├── matriz_confusion.png
│   ├── curvas_entrenamiento.png
│   ├── analisis_errores.csv
│   ├── prueba_persistencia.json
│   ├── prueba_ruido.json
│   ├── indices_datos.npz         # Índices originales de train/val/test
│   ├── prueba_20_dibujos.csv     # Se crea al registrar los 20 dibujos propios
│   └── captura_interfaz.png      # Captura de la verificación de la GUI
├── data/
│   └── historial.csv             # Historial cotidiano (se crea al primer registro)
├── utils/
│   ├── __init__.py               # Rutas y constantes compartidas
│   ├── preprocessing.py          # Dibujo del lienzo -> vector de 64 valores
│   ├── evaluation.py             # Partición, métricas, gráficos
│   └── history.py                # Historial, validación y prueba de 20 dibujos
├── tests/
│   └── test_proyecto.py          # Pruebas con pytest
└── tools/
    ├── verificar_treeview.py     # Verificación automática de la interfaz (abre la ventana)
    ├── prueba_dibujos.py         # Verificación técnica dibujo -> modelo (no manual)
    ├── preparar_depuracion.py    # Utilidades de diagnóstico de la persistencia
    └── depurar_persistencia.py
```

## 5. Instalación de dependencias

Abrir **Visual Studio Code**, luego `Terminal > Nueva terminal`, y situarse en la carpeta
del proyecto:

```powershell
cd "ruta\a\reconocimiento_digitos"
python -m pip install -r requirements.txt
```

### Entorno virtual (recomendado en PowerShell)

```powershell
cd "ruta\a\reconocimiento_digitos"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

En VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → elegir
`.venv\Scripts\python.exe`. En la parte inferior de la ventana debe aparecer ese intérprete.

## 6. Cómo ejecutar el entrenamiento

```powershell
python train.py
```

Duración aproximada: 1–3 minutos en CPU. Al finalizar genera el modelo y todos los archivos
de `results/`. La salida muestra la comparación en validación, el modelo elegido, las
métricas de prueba, la prueba de persistencia y la de ruido.

## 7. Cómo ejecutar la aplicación

```powershell
python app.py
```

Si no existe `models/modelo_final.keras`, la aplicación avisa con un mensaje claro y no se
cierra de forma inesperada; simplemente indica que se debe ejecutar `python train.py`.

Pruebas automatizadas:

```powershell
python -m pytest -q
```

Verificación de la interfaz (abre la ventana, dibuja un 7, reconoce y guarda una captura):

```powershell
python tools\verificar_treeview.py
```

Verificación técnica del circuito dibujo -> preprocesamiento -> modelo (dibuja dígitos con
segmentos, **no** sustituye la prueba manual de 20 dibujos):

```powershell
python tools\prueba_dibujos.py
```

## 8. Explicación de las dos arquitecturas

Ambos modelos reciben **64 valores**, usan **ReLU** en las capas ocultas y **Softmax de 10
neuronas** en la salida, se compilan con **Adam (lr = 0.001)** y
**categorical_crossentropy**, y usan **EarlyStopping** (monitor `val_loss`, paciencia 25,
`restore_best_weights=True`). La semilla es 42.

| Modelo | Construcción | Capas | Parámetros |
|---|---|---|---|
| **Sequential** | `keras.Sequential([...])` | 128 ReLU (L2=1e-4) → Dropout 0.25 → 64 ReLU (L2=1e-4) → 10 Softmax | 17 226 |
| **Functional** | `Input` → `Dense` → `Dropout` → `Dense` → `Dense` → `Model` | 64 ReLU → Dropout 0.20 → 32 ReLU → 10 Softmax | 6 570 |

**Justificación de las decisiones**

- *Dos capas ocultas*: con 64 entradas y 1797 ejemplos, una capa es muy limitada y tres ya
  no aportan mejora apreciable; dos capas permiten combinar píxeles y luego patrones.
- *128 y 64 neuronas*: el dataset es pequeño; más neuronas aumentan los parámetros sin
  mejorar la generalización.
- *Dropout y L2*: evitan que la red memorice el conjunto de entrenamiento (menos brecha
  entre pérdida de entrenamiento y de validación).
- *Adam con lr = 0.001*: converge de forma estable y es el optimizador estándar para
  clasificación multiclase con Softmax.
- *EarlyStopping con restauración*: detiene el entrenamiento cuando la validación deja de
  mejorar y devuelve los mejores pesos observados.
- *El Functional es más pequeño* a propósito: se compara un diseño distinto con la otra API
  en lugar de repetir la misma red.

**Selección del modelo**: F1 macro en validación (criterio principal) y accuracy como
desempate. La partición de prueba solo se usa **después** de decidir, y nunca se vuelve a
entrenar el modelo con ella. Los resultados reales están en
`results/comparacion_modelos.csv` y `results/metricas_prueba.json`.

## 9. Justificación de la normalización

`load_digits` guarda píxeles enteros de **0 a 16** (fondo oscuro = 0, trazo claro = 16).
Se dividen entre **16.0** para que todas las entradas queden en **[0, 1]**:

- mantiene estables los gradientes y acelera la convergencia de Adam,
- evita que entradas grandes dominen el aprendizaje,
- es una transformación lineal, así que no se pierde información.

El mismo factor (16.0) se aplica a los dibujos del usuario, de modo que entrenamiento y uso
real reciben datos en la misma escala.

## 10. Explicación del preprocesamiento de imágenes

El lienzo de Tkinter es grande (≈320x320), con **fondo blanco y trazo negro** de 18 px, mientras
que la red espera una imagen de **8x8 con el dígito claro sobre fondo oscuro**, con el trazo
ocupando casi toda la caja. Por eso se aplica siempre este procedimiento
(`utils/preprocessing.py`), idéntico en todas las predicciones:

1. **Capturar** el lienzo con Pillow (se reconstruyen los segmentos, con extremos redondeados).
2. **Recortar** la caja del trazo: elimina márgenes que dependen de dónde dibujó el usuario.
3. **Invertir** los colores: el trazo queda claro (alto) y el fondo oscuro (bajo).
4. **Redimensionar de forma proporcional** para que el lado mayor mida **7.6 de los 8 píxeles**
   (el valor medido en `models/calibracion.json`). El remuestreo promedia bloques de píxeles
   (`redimensionar_promedio`, equivalente a `Image.Resampling.BOX`), de modo que un trazo que
   cae entre dos píxeles nuevos no desaparece. Se implementó en NumPy porque el filtro de
   Pillow de esta instalación no promediaba al reducir (devolvía valores casi nulos).
5. **Centrar** en 8x8 usando el **centro de masa** del dígito, para que el recorte y el
   redondeo de la caja no desplacen el número hacia un borde.
6. **Convertir a gris**, llevar los valores a la escala **0–16** y **dividir entre 16.0**.
7. **Aplanar** a un vector de 64 valores.

¿Por qué cada paso?

- **Recorte**: sin él, un dibujo pequeño en una esquina tendría casi todos los píxeles a cero.
- **Inversión**: el modelo aprendió “tinta = valor alto”; sin invertir, todo se vería al revés.
- **Redimensionado proporcional**: deformar el dígito (por ejemplo, estirar un 1) cambia su
  forma y empeora la predicción.
- **Centrado**: el dataset tiene los dígitos centrados; el centrado alinea el dibujo con lo
  aprendido.
- **Escala 0–16 / 16.0**: replica exactamente la representación usada en el entrenamiento.

### Versión corregida (v2) y por qué cambió

La versión original recortaba, invertía y metía el dígito en una caja de **6 px** sin tocar el
grosor. El resultado numérico era muy distinto al del dataset:

| Medida (promedio) | load_digits | v1 original | v2 corregida |
|---|---|---|---|
| Píxeles activos (> 8 de 16) | 18.7 | **0.7** | ~19 |
| Suma de los 64 píxeles | ~69 | ~15 | ~66 |
| Fila superior con tinta | — | **sí (el dígito tocaba el borde)** | no |
| Correlación con el original (dígito del dataset re-dibujado) | — | 0.61 | **0.96–0.98** |

Los tres fallos concretos y su corrección:

1. **El trazo se perdía al reducir.** Un pincel de 18 px sobre 320, promediado en 6 px de 8x8,
   queda en ~2 de 16: por debajo del umbral que la red considera “tinta”. Se corrigió eligiendo
   una caja de 7.6 px (el diámetro típico del dataset) y el remuestreo por bloques.
2. **Encuadre desplazado.** Al centrar con márgenes fijos el dígito quedaba arriba a la
   izquierda; los trazos en diagonal (9, 2) cambiaban de lado y el modelo veía otra forma.
   Se corrigió centrando con el **centro de masa**.
3. **La escala no era la del dataset.** El modelo aprendió dígitos de 5-8 px; un dígito de 6 px
   con margen de 1 cambiaba el “grosor relativo del trazo”. Ahora la caja imita la del dataset.

La elección de esos parámetros se hizo con `tools/comparar_estrategias.py` (144 dibujos de
tres estilos distintos); el resultado está en `results/comparacion_estrategias.csv`.

La interfaz muestra la matriz de 8x8 que realmente recibe la red, ampliada con color por
celdas (sin interpolación suave: se ven los píxeles reales).

## 11. Métricas y evaluación

- **Accuracy**: proporción de dígitos bien clasificados.
- **F1 macro**: promedio del F1 de las 10 clases sin ponderar por cantidad de muestras. Es el
  criterio principal porque castiga que una clase se prediga mal aunque tenga pocos ejemplos.
- **Matriz de confusión**: muestra qué dígitos se confunden entre sí.
- **Análisis de errores** (`results/analisis_errores.csv`): se registran los índices originales
  de las muestras mal clasificadas, el dígito real, el predicho, la probabilidad máxima, la
  segunda opción y una explicación automática basada en la distribución de la tinta en la
  imagen de 8x8.

## 12. Persistencia del modelo

El modelo seleccionado se guarda en `models/modelo_final.keras` y su descripción en
`models/metadata.json` (arquitectura, 64 entradas, 10 clases, escala original 0–16, factor 16.0,
semilla 42, preprocesamiento y umbral de baja confianza de 0.70).

**Prueba de persistencia real** (`results/prueba_persistencia.json`):

1. se eligen 5 muestras fijas del conjunto de prueba,
2. se guardan sus predicciones del modelo en memoria,
3. se lanza **un proceso de Python independiente** (`subprocess`) que carga el `.keras`,
4. ese proceso predice las mismas 5 muestras y devuelve las probabilidades,
5. el proceso padre compara con una **tolerancia de 1e-6**.

La aplicación nunca vuelve a entrenar: solo carga el archivo guardado.

## 13. Pruebas de resistencia al ruido

En `train.py`, después de congelar el modelo:

1. se seleccionan **30 imágenes de prueba** con la semilla fija 42,
2. se predicen sin ruido,
3. se añade **ruido gaussiano N(0, 0.15)** a las imágenes normalizadas y se recortan a [0, 1],
4. se vuelven a predecir,
5. se guarda exactitud original, con ruido, su diferencia y el detalle de cada muestra en
   `results/prueba_ruido.json`.

Este experimento es solo diagnóstico: no se vuelve a entrenar ni se ajusta el modelo con él.

## 14. Procedimiento para los 20 dibujos propios

Esta prueba **debe hacerla el estudiante dibujando con el ratón**, no un script:

1. Abrir `python app.py` y quedarse en la pestaña **Reconocer**.
2. Elegir en `¿Qué número quisiste dibujar?` el dígito que va a dibujar.
3. Dibujar el número, pulsar **Reconocer** y luego **Registrar en prueba 20**.
4. Repetir hasta tener **2 dibujos por cada dígito (20 en total)**, pulsando **Limpiar** entre
   uno y otro. Conviene hacer el orden 0,0,1,1,...,9,9.
5. La pestaña *Prueba de 20 dibujos* muestra cuántos registrados hay, cuántos faltan, la
   accuracy parcial, correctos, incorrectos y cuántos tuvieron la advertencia de baja confianza.

Reglas implementadas: se guarda el **primer intento**, los errores **no se borran**, no se
admite un tercer dibujo del mismo dígito ni más de 20 registros, y cada registro incluye
probabilidades, advertencia de baja confianza y fecha. El archivo es
`results/prueba_20_dibujos.csv`, separado de `data/historial.csv`.

Para obtener el informe de comparación final ejecutar cuando estén completos los 20:

```powershell
python -m utils.informe_final
```

Genera `results/diagnostico_final.json` y `results/diagnostico_final.md`. Si la prueba de 20
dibujos no está hecha, el informe lo dice explícitamente y **no inventa** resultados.

### Banco de dibujos (conjunto independiente)

En la misma barra de registro hay un botón **“Guardar dibujo (banco)”**. Guarda en
`data/banco_dibujos/` tres cosas por dibujo: el **lienzo original** (PNG), el **8x8
procesado** (PNG y NPY) y una fila en `indice.csv` con la etiqueta real, la predicción y
la probabilidad. Es un conjunto **separado**: no entra en entrenamiento, validación ni
prueba, y no interfiere con el historial ni con la prueba de 20 dibujos. Sirve para
construir más adelante una evaluación con escritura real.

## 15. Conclusiones y posibles mejoras

**Conclusiones**

- Una red densa pequeña logra una exactitud alta en `load_digits` porque las imágenes son
  simples y muy centradas.
- La comparación por F1 macro permite elegir un modelo sin mirar el conjunto de prueba.
- El mismo modelo se reutiliza en la aplicación sin volver a entrenar, y la prueba de
  persistencia demuestra que las predicciones coinciden al recargarlo.
- Un modelo con 98 % de exactitud puede fallar con dibujos del ratón si el
  **preprocesamiento** produce imágenes muy distintas a las del entrenamiento. Ese fue el
  caso del proyecto: 0.7 píxeles activos frente a 18.7 del dataset.
- El rendimiento con dibujos propios suele ser inferior al de prueba: los trazos del ratón
  tienen otra forma, grosor y posición. La prueba de ruido muestra el mismo efecto.

**Mejoras propuestas**

1. **Datos reales del estudiante**: usar el banco (`data/banco_dibujos/`) como conjunto de
   validación externa, sin tocar la prueba oficial, para medir mejoras de la interfaz.
2. **Calibración de probabilidades** (temperatura o Platt scaling) para que el umbral 0.70
   signifique lo que aparenta. El umbral actual es una **regla didáctica**: no demuestra que
   las probabilidades de la red estén calibradas.
3. **Más aumentación orientada al trazo**: variar el orden de los trazos y los huecos entre
   ellos, que es lo que más diferencia al ratón de la escritura escaneada.
4. **Un trazo más guiado en la interfaz** (muñeco de fondo o área de dibujo reducida) para
   que el usuario escriba dentro del encuadre que el modelo espera.
5. **Versión con CNN** como comparación en una materia posterior, manteniendo las redes
   densas de este ejercicio como línea base.

## 16. Corrección del reconocimiento de dibujos (auditoría v2)

### Causa principal

No era la red ni la normalización: era el **preprocesamiento**. El modelo recibía imágenes
casi vacías y desplazadas. Concretamente:

- el trazo de 18 px se reducía a una caja de 6 px **promediando**, y el resultado quedaba en
  1–2 de 16: por debajo del umbral de “tinta”;
- el centrado con márgenes fijos metía el dígito en la esquina superior izquierda;
- el 9 y el 2, que son trazos en diagonal, cambiaban de forma al recolocarse; el 8 perdía los
  bucles y se veía como un 9 o un 4 (un trazo vertical largo).

### Herramientas de auditoría incluidas

| Comando | Qué muestra |
|---|---|
| `python tools/auditar_8x8.py` | 8x8 de dibujos vs. dataset: píxeles activos, suma, bordes, centro de masa |
| `python tools/depurar_dibujos.py` | Figura con original, dibujo, 8x8 final y referencia (2, 4, 8, 9) → `results/diagnostico_visual.png` |
| `python tools/probar_pipeline_aislado.py` | ¿El pipeline conserva un dígito del dataset re-dibujado? (correlación) |
| `python tools/comparar_estrategias.py` | Compara configuraciones de preprocesamiento con 144 dibujos de 3 estilos |
| `python tools/benchmark_dibujos.py` | Aciertos con pinceles de 12, 18 y 24 px |
| `python tools/comparar_modelos_dibujos.py` | v1 contra v2 sobre los mismos dibujos |
| `python tools/depurar_lienzo_real.py` | Dibuja 9, 8 y 2 en el lienzo real de Tkinter y muestra el 8x8 y la predicción |

### Aumentación (solo entrenamiento)

`utils/rendering.py` amplía cada imagen de entrenamiento a 32x32 (el tamaño original de
`load_digits`), aplica allí la transformación (giro ±12°, desplazamiento ±1 px, escala
0.90–1.10, grosor ±0.5 px) y vuelve a reducir a 8x8 promediando. **No** se aplica el pipeline
de la app a imágenes del dataset: eso enseñaría a la red a deshacer su propio
preprocesamiento y haría tramposa la evaluación. Cada imagen original se acompaña de 2
variantes (1077 → 3231 muestras), mezcladas antes de entrenar.

Se evaluó también **añadir un banco de dibujos sintéticos** (`tools/experimento_banco.py`,
resultado en `results/experimento_banco_v3.json`). **No se adoptó**: con dibujos de otro estilo
la exactitud en el banco de evaluación bajó (v2 ≈ 0.89 frente a v3 ≈ 0.83) y la precisión en
`load_digits` no mejoró. Se mantienen las redes densas y los datos de `load_digits`.

### Resultados verificables (todos reales)

| Medición | v1 (original) | v2 (corregido) |
|---|---|---|
| Píxeles activos por imagen | 0.7 | ~19 |
| Correlación del pipeline con el original | 0.61 | 0.96–0.98 |
| Exactitud en prueba (load_digits) | 98.33 % | ver `results/metricas_prueba.json` |
| Dibujos tipo ratón (30: 10 dígitos × 3 pinceles) | 26/30 | 28/30 (antes de mezclar el aumento los lotes) |
| Pincel real de la app (18 px, 10 dígitos) | 9/10 | 10/10 |
| Persistencia tras recargar en otro proceso | aprobada | aprobada (diferencia ~1e-9) |

Los resultados tras el último entrenamiento (con el aumento mezclado) se escriben en
`results/comparacion_v1_v2_dibujos.json`.

### Limitaciones que siguen existiendo

- El modelo solo conoce **escritura escaneada de 8x8**. Si el usuario dibuja con un estilo muy
  distinto (trazo muy fino, números con huecos, escritura cursiva) puede fallar.
- La interfaz no impone un área de dibujo: un número muy pequeño en una esquina se recorta y
  se amplía igual que uno grande, pero pierde detalle.
- Las probabilidades **no están calibradas**; el umbral de 0.70 es didáctico.
- Las pruebas con dibujos sintéticos (26/30, 28/30, 10/10) **no** son la prueba manual de 20
  dibujos: ese banco lo debe completar el estudiante.
